"""Forecast ledger, historical replay: an incrementally learning volatility forecaster vs the options market.

Rules fixed before running (agreed with the user 2026-10-07; do not change after seeing results):
  Data       NIFTY, BANKNIFTY daily closes (data/raw/underlying), r = log close-to-close return.
             India VIX close; daily option smiles (src/backtest.Market); Stage 4 GJR-GARCH-t
             (monthly parameters fitted on data before the month; pstate sigma2_next).
  Origin     every session t; a forecast uses data up to and including t's close only.
  Targets    Y_h(t) = sum of r^2 over the next h sessions, h = 5 and 21.
  Forecasters (total variance over the h sessions)
    implied   NIFTY: India VIX, per-session variance (VIX/100)^2 x (30/365) / 21 (the formal
              baseline). BANKNIFTY: ATM IV (smile at log(K/F)=0, non-stale, spanning 0) of every
              expiry with >= 2 sessions left, total variance iv^2 x T_cal interpolated linearly
              in sessions-to-expiry (flat vol outside the range). The NIFTY ATM version is
              reported as implied_atm.
    garch     h-step GJR-GARCH-t sum (sigma2_next, month-of-t parameters, persistence
              alpha + gamma/2 + beta, + h mu^2) x past-only calibration (sum Y / sum forecast over
              windows completed by t, expanding, >= 250 windows).
    har       log-HAR: log mean r^2 over the next h on log mean r^2 over the last 5 / 22 / 66
              sessions; exponentially weighted least squares refitted every session on windows
              completed by t (half-life 3 years = 756 sessions); forecast exp(fit + s^2/2) x h;
              >= 500 windows.
    blend     sum_i w_i f_i over (implied, garch, har); weights in [0,1] summing to 1 that minimise
              mean QLIKE over the last 250 windows completed by t (>= 125 windows), re-fitted
              every session (user's choice after the synthetic test showed inverse-loss weights
              stay near equal).
    Sensitivity (reported, not tested): blend_inv = weights proportional to 1 / mean QLIKE
              (same window); har_fast / blend_fast = half-life 1 year (252), blend window 120
              (>= 60).
    Report-only comparators: implied_sc = implied x past-only calibration (implied with its
              average premium removed); garch_raw.
  Loss       QLIKE(Y, f) = Y/f - ln(Y/f) - 1 (Y floored at 1e-10); also RMSE of log vol, bias.
  Periods    warm-up to 2012; discovery 2013-2017 (design checks only); holdout 2018 onwards
             (origins whose target is complete) decides.
  PRIMARY    blend beats implied in the holdout: mean QLIKE difference > 0 with Newey-West
             (lag h) p < 0.05 after Holm over 4 tests (2 symbols x 2 horizons).
  Secondary  (Holm over 4: 2 symbols x 2 events), next monthly expiry m (last expiry of its
             calendar month, settling after t), a = ATM IV_m x sqrt(T_m), n = sessions to m:
    range     event |ln(S_settle / F_m)| < a. Market probability: P(|x| < a), x ~ N(-a^2/2, a^2).
              Model: same with variance v = blend total variance for n sessions (linear in n
              between the h=5 and h=21 forecasts, proportional outside). Also past base rate.
    seller    event: realised variance t -> settlement < a^2 (a variance seller wins).
              Baseline: past base rate (expanding, settled by t, >= 250). Model:
              Phi((ln a^2 - ln v - mu_e) / s_e), mu_e, s_e = mean / sd of ln(RV / v) over the
              last 500 rows settled by t (>= 250).
              Scored by Brier score (and log loss); NW lag 25; model vs baseline.
  Control    direction: P(r over next 5 sessions > 0). Online logistic regression (refit every
             session, half-life 3y) on ret5, ret22, ln(implied / har); vs 0.5 and the base rate.
             Expected: no skill.
  Money      (secondary) Stage 5 monthly straddles at 1 Cr through the realistic near-close
             simulator (FUT round, extra slippage S0, settlement-day hedge allowed), size x factor,
             factor = clip(g / median of past g of that symbol (>= 24 entries, else 1), 0, 2),
             g = 1 - v / a^2 (blend) vs constant size and vs the naive g' = 1 - trailing 22-session
             variance x n / a^2. Monthly book excess returns, holdout, NW lag 3.
  Checks     synthetic test (scripts/forecast_ledger_selftest.py) first; truncation look-ahead
             test (forecasts at t identical when all data after t is removed); shuffled-date
             placebo: targets circularly shifted by >= 1 year (200 shifts) - how much of the
             blend's win survives = the average premium, the rest = day-to-day tracking.
Writes data/processed/ledger/ (forecasts.parquet, prob.parquet, money.parquet, _report.txt)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src import backtest as B  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "data", "processed", "ledger")
UND = os.path.join(PROJECT_ROOT, "data", "raw", "underlying")
S4 = os.path.join(PROJECT_ROOT, "data", "processed", "stage4")
HS = (5, 21)
CFG = {"": (756, 250), "_fast": (252, 120)}
MIN_CAL, MIN_HAR = 250, 500
DISC0, HOLD0 = pd.Timestamp("2013-01-01"), pd.Timestamp("2018-01-01")
NW_PROB = 25
SYMS = ("NIFTY", "BANKNIFTY")


# ---------------------------------------------------------------- generic learners (arrays, origin-indexed)

def target(r: np.ndarray, h: int) -> np.ndarray:
    """Y[t] = sum r[t+1..t+h]^2 (nan if incomplete)."""
    c = np.r_[0.0, np.cumsum(r ** 2)]
    n = len(r)
    y = np.full(n, np.nan)
    t = np.arange(n - h)
    y[t] = c[t + h + 1] - c[t + 1]
    return y


def qlike(y, f):
    x = np.maximum(y, 1e-10) / f
    return x - np.log(x) - 1.0


def har(r: np.ndarray, h: int, hl: float, min_n: int = MIN_HAR) -> np.ndarray:
    """Exponentially weighted log-HAR, refitted at every origin on windows completed by t."""
    n = len(r)
    r2 = r ** 2
    c = np.r_[0.0, np.cumsum(r2)]

    def back(L):
        out = np.full(n, np.nan)
        t = np.arange(L - 1, n)
        out[t] = (c[t + 1] - c[t + 1 - L]) / L
        return out

    X = np.column_stack([np.ones(n)] + [np.log(np.maximum(back(L), 1e-10)) for L in (5, 22, 66)])
    yt = target(r, h)
    Y = np.log(np.maximum(yt / h, 1e-10))
    okX = np.isfinite(X).all(axis=1)
    lam = 0.5 ** (1.0 / hl)
    A = np.zeros((4, 4))
    bvec = np.zeros(4)
    yy = sw = 0.0
    cnt = 0
    f = np.full(n, np.nan)
    for t in range(n):
        i = t - h                       # the window starting at origin i completes at t
        if i >= 0 and okX[i] and np.isfinite(Y[i]):
            A, bvec, yy, sw = lam * A, lam * bvec, lam * yy, lam * sw
            x = X[i]
            A += np.outer(x, x)
            bvec += x * Y[i]
            yy += Y[i] ** 2
            sw += 1.0
            cnt += 1
        elif i >= 0:
            A, bvec, yy, sw = lam * A, lam * bvec, lam * yy, lam * sw
        if cnt >= min_n and okX[t]:
            beta = np.linalg.solve(A, bvec)
            rss = yy - 2 * beta @ bvec + beta @ A @ beta
            s2 = max(rss / sw, 0.0)
            f[t] = np.exp(X[t] @ beta + s2 / 2) * h
    return f


def calibrate(f: np.ndarray, y: np.ndarray, h: int, min_n: int = MIN_CAL) -> np.ndarray:
    """f x (sum y / sum f over windows completed by t), expanding."""
    ok = np.isfinite(f) & np.isfinite(y)
    cy, cf, cn = (np.cumsum(np.where(ok, v, 0.0)) for v in (y, f, ok.astype(float)))
    out = np.full(len(f), np.nan)
    for t in range(h, len(f)):
        k = t - h
        if cn[k] >= min_n:
            out[t] = f[t] * cy[k] / cf[k]
    return out


def blend(fs: list[np.ndarray], y: np.ndarray, h: int, W: int, method: str = "opt") -> tuple[np.ndarray, np.ndarray]:
    """Combination weights learned from the last W windows completed by t (all forecasters present).
    opt: weights in [0,1] summing to 1 that minimise mean QLIKE over those windows (SLSQP, started
    from the previous session's weights); inv: proportional to 1 / mean QLIKE."""
    from scipy.optimize import minimize
    F = np.column_stack(fs)
    k = F.shape[1]
    ok = np.isfinite(F).all(axis=1) & np.isfinite(y)
    L = np.where(ok[:, None], qlike(np.where(ok, y, 1.0)[:, None], np.where(ok[:, None], F, 1.0)), 0.0)
    cL = np.vstack([np.zeros(k), np.cumsum(L, axis=0)])
    cn = np.r_[0.0, np.cumsum(ok)]
    n = len(y)
    out = np.full(n, np.nan)
    wts = np.full((n, k), np.nan)
    w0 = np.full(k, 1.0 / k)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    for t in range(h, n):
        hi = t - h + 1                  # windows 0..t-h are complete
        lo = max(0, hi - W)
        m = cn[hi] - cn[lo]
        if m < W / 2 or not np.isfinite(F[t]).all():
            continue
        if method == "inv":
            ml = (cL[hi] - cL[lo]) / m
            w = (1 / ml) / (1 / ml).sum()
        else:
            sel = ok[lo:hi]
            Y, X = y[lo:hi][sel], F[lo:hi][sel]
            res = minimize(lambda w: qlike(Y, X @ w).mean(), w0, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                           constraints=cons, options={"ftol": 1e-10, "maxiter": 200})
            w = np.clip(res.x, 0.0, 1.0)
            w = w / w.sum()
            w0 = w
        wts[t] = w
        out[t] = F[t] @ w
    return out, wts


def garch_h(s2next: np.ndarray, months: pd.DatetimeIndex, params: pd.DataFrame, h: int) -> np.ndarray:
    p = params.set_index("month").reindex(months)
    phi = np.minimum((p["alpha"] + p["gamma"] / 2 + p["beta"]).to_numpy(float), 0.999)
    VL = (p["omega"].to_numpy(float) / (1 - phi)) / 1e4
    mu2 = (p["mu"].to_numpy(float) / 100) ** 2
    geo = (1 - phi ** h) / (1 - phi)
    return h * VL + geo * (s2next - VL) + h * mu2


def nw_t(d: np.ndarray, lag: int) -> tuple[float, float, int]:
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return np.nan, np.nan, n
    m = d.mean()
    e = d - m
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    t = m / np.sqrt(v / n)
    return m, t, n


def p2(t):
    return float(2 * norm.sf(abs(t))) if np.isfinite(t) else np.nan


def holm(ps: list[float]) -> list[float]:
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return list(adj)


# ---------------------------------------------------------------- data

def load_closes(sym: str) -> pd.Series:
    d = pd.read_csv(os.path.join(UND, f"{sym.lower()}.csv"), parse_dates=["date"])
    return d.set_index("date")["close"].astype(float).sort_index()


def implied_table(sym: str, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Per session: per-session implied variance interpolated to h = 5, 21; next monthly expiry info."""
    mk = B.Market(sym)
    pos = {d: i for i, d in enumerate(sessions)}
    fes = pd.Series(sorted(mk.settled))
    monthly = set(fes.groupby([fes.dt.year, fes.dt.month]).max())
    per = {}
    for (d, fe), (k, iv) in mk.smiles.items():
        st = mk.settled.get(fe)
        if st is None or d not in pos or st not in pos or k.min() > 0 or k.max() < 0:
            continue
        n_e = pos[st] - pos[d]
        if n_e < 2:
            continue
        a = float(np.interp(0.0, k, iv))
        T = (st - d).days / 365.0
        per.setdefault(d, []).append((n_e, a * a * T, fe, a, T, st))
    rows = []
    for d, lst in per.items():
        lst.sort()
        n = np.array([x[0] for x in lst], float)
        w = np.array([x[1] for x in lst], float)
        row = {"date": d}
        for h in HS:
            if h <= n[0]:
                tot = w[0] * h / n[0]
            elif h >= n[-1]:
                tot = w[-1] * h / n[-1]
            else:
                tot = np.interp(h, n, w)
            row[f"imp_atm{h}"] = tot
        mm = [x for x in lst if x[2] in monthly]
        if mm:
            n_m, _, fe, a, T, st = mm[0]
            F = mk.fwd.get((d, fe))
            row.update(m_fe=fe, m_iv=a, m_T=T, m_n=n_m, m_settle=st, m_F=F)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date").sort_index()


def build(sym: str, closes: pd.Series, vix: pd.Series, imp: pd.DataFrame) -> pd.DataFrame:
    sess = closes.index
    r = np.r_[np.nan, np.diff(np.log(closes.to_numpy()))]
    r[0] = 0.0
    df = pd.DataFrame(index=sess)
    df["close"] = closes.to_numpy()
    df["r"] = r
    df = df.join(imp)
    ps = pd.read_parquet(os.path.join(S4, "pstate.parquet"))
    ps = ps[ps["symbol"] == sym].set_index("date")
    s2n = ps["sigma2_next"].reindex(sess).to_numpy(float)
    prm = pd.read_parquet(os.path.join(S4, f"garch_params_{sym}.parquet"))
    months = sess.to_period("M").to_timestamp()
    if sym == "NIFTY":
        vv = vix.reindex(sess).to_numpy(float)
        per_vix = (vv / 100) ** 2 * (30 / 365) / 21
    for h in HS:
        y = target(r, h)
        df[f"y{h}"] = y
        imp_h = per_vix * h if sym == "NIFTY" else df[f"imp_atm{h}"].to_numpy(float)
        df[f"imp{h}"] = imp_h
        df[f"imp_sc{h}"] = calibrate(imp_h, y, h)
        g = garch_h(s2n, months, prm, h)
        df[f"garch_raw{h}"] = g
        df[f"garch{h}"] = calibrate(g, y, h)
        for suf, (hl, W) in CFG.items():
            df[f"har{suf}{h}"] = har(r, h, hl)
            comps = [imp_h, df[f"garch{h}"].to_numpy(), df[f"har{suf}{h}"].to_numpy()]
            b, w = blend(comps, y, h, W)
            df[f"blend{suf}{h}"] = b
            if suf == "":
                for j, nm in enumerate(("imp", "garch", "har")):
                    df[f"w_{nm}{h}"] = w[:, j]
                df[f"blend_inv{h}"] = blend(comps, y, h, W, method="inv")[0]
    df["symbol"] = sym
    return df


# ---------------------------------------------------------------- probability forecasts

def var_n(df: pd.DataFrame, n: np.ndarray, col: str = "blend") -> np.ndarray:
    f5, f21 = df[f"{col}5"].to_numpy(float), df[f"{col}21"].to_numpy(float)
    return np.where(n <= 5, f5 * n / 5, np.where(n >= 21, f21 * n / 21, f5 + (f21 - f5) * (n - 5) / 16))


def p_inside(a, v):
    s = np.sqrt(v)
    return norm.cdf((a + v / 2) / s) - norm.cdf((-a + v / 2) / s)


def past_stats(t_idx, s_idx, vals, window=None, min_n=MIN_CAL):
    """For each origin in t_idx, mean / sd of vals (rows settling at s_idx) over rows settled at or
    before that origin (the last `window` of them by settlement)."""
    order = np.argsort(s_idx, kind="stable")
    ss, vv = s_idx[order], vals[order]
    c1, c2 = np.r_[0.0, np.cumsum(vv)], np.r_[0.0, np.cumsum(vv ** 2)]
    hi = np.searchsorted(ss, t_idx, side="right")
    lo = np.maximum(0, hi - window) if window else np.zeros_like(hi)
    m = hi - lo
    mean = np.where(m >= min_n, (c1[hi] - c1[lo]) / np.maximum(m, 1), np.nan)
    var = np.where(m >= min_n, (c2[hi] - c2[lo]) / np.maximum(m, 1) - mean ** 2, np.nan)
    return mean, np.sqrt(np.maximum(var, 1e-12))


def prob_table(df: pd.DataFrame) -> pd.DataFrame:
    sess = df.index
    pos = {d: i for i, d in enumerate(sess)}
    x = df[df["m_fe"].notna() & df["m_F"].notna()].copy()
    x = x[x["m_settle"].map(lambda d: d in pos)]
    t_idx = np.array([pos[d] for d in x.index])
    s_idx = np.array([pos[d] for d in x["m_settle"]])
    n = (s_idx - t_idx).astype(float)
    a = x["m_iv"].to_numpy(float) * np.sqrt(x["m_T"].to_numpy(float))
    S = df["close"].to_numpy(float)[s_idx]
    c2 = np.r_[0.0, np.cumsum(df["r"].to_numpy(float) ** 2)]
    rv = c2[s_idx + 1] - c2[t_idx + 1]
    v = var_n(x, n)
    hit = (np.abs(np.log(S / x["m_F"].to_numpy(float))) < a).astype(float)
    win = (rv < a * a).astype(float)
    done = s_idx < len(sess)
    out = pd.DataFrame(index=x.index)
    out["n"], out["a"], out["v"] = n, a, v
    out["range_y"], out["seller_y"] = hit, win
    out["range_mkt"] = p_inside(a, a * a)
    out["range_model"] = p_inside(a, v)
    out["range_base"] = past_stats(t_idx, s_idx, hit)[0]
    e = np.log(np.maximum(rv, 1e-10) / v)
    ok = np.isfinite(e)
    mu, sd = past_stats(t_idx, s_idx[ok], e[ok], window=500)
    out["seller_model"] = norm.cdf((np.log(a * a) - np.log(v) - mu) / sd)
    out["seller_base"] = past_stats(t_idx, s_idx, win)[0]
    out["t_idx"], out["s_idx"] = t_idx, s_idx
    return out[done]


def direction(df: pd.DataFrame, hl: float = 756, min_n: int = MIN_HAR) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    lc = np.log(df["close"].to_numpy(float))
    n = len(lc)
    y = np.full(n, np.nan)
    y[: n - 5] = (lc[5:] > lc[:-5]).astype(float)
    ret5 = np.r_[np.full(5, np.nan), lc[5:] - lc[:-5]]
    ret22 = np.r_[np.full(22, np.nan), lc[22:] - lc[:-22]]
    vr = np.log(df["imp5"].to_numpy(float) / df["har5"].to_numpy(float))
    X = np.column_stack([ret5, ret22, vr])
    okX = np.isfinite(X).all(axis=1)
    p = np.full(n, np.nan)
    base = np.full(n, np.nan)
    lr = LogisticRegression(C=1.0, max_iter=200)
    for t in range(n):
        idx = np.arange(0, max(t - 5 + 1, 0))
        idx = idx[okX[idx] & np.isfinite(y[idx])]
        if len(idx):
            base[t] = y[idx].mean() if len(idx) >= MIN_CAL else np.nan
        if len(idx) < min_n or not okX[t]:
            continue
        w = 0.5 ** ((t - idx) / hl)
        lr.fit(X[idx], y[idx], sample_weight=w)
        p[t] = lr.predict_proba(X[t:t + 1])[0, 1]
    return pd.DataFrame({"dir_y": y, "dir_model": p, "dir_base": base}, index=df.index)


# ---------------------------------------------------------------- money test

def money(fc: dict, prob: dict) -> pd.DataFrame:
    import realistic_engine as R
    import realistic_nearclose as NC
    pos = pd.read_parquet(os.path.join(R.D5, "positions.parquet"))
    pos = pos[pos["strategy"] == "eng_straddle_bs"].sort_values(["symbol", "entry"])
    st = B.spread_table(os.path.join(R.D5, "_spread_table.parquet"))
    cms = [NC.SlipCM(R.CostModel(B.Costs(st), False), 0, 0), NC.SlipCM(R.CostModel(B.Costs(st, stress=True), True), 0, 0)]
    C = 1e7
    rows = []
    for sym in SYMS:
        df = fc[sym]
        rd = R.RealData(sym)
        gs, gn = [], []
        for p in pos[pos["symbol"] == sym].itertuples():
            c = rd.cycle(p)
            d = p.entry
            g = gnaive = np.nan
            if d in df.index and df.at[d, "m_fe"] == p.final_expiry:
                a2 = df.at[d, "m_iv"] ** 2 * df.at[d, "m_T"]
                n = float(df.at[d, "m_n"])
                v = float(var_n(df.loc[[d]], np.array([n]))[0])
                g = 1 - v / a2
                i = df.index.get_loc(d)
                tr = df["r"].to_numpy()[max(0, i - 21): i + 1]
                gnaive = 1 - (tr ** 2).mean() * n / a2
            facs = {}
            for nm, val, hist in (("blend", g, gs), ("naive", gnaive, gn)):
                past = [x for x in hist if np.isfinite(x)]
                if np.isfinite(val) and len(past) >= 24 and np.median(past) > 0:
                    facs[nm] = float(np.clip(val / np.median(past), 0, 2))
                else:
                    facs[nm] = 1.0
                hist.append(val)
            facs["const"] = 1.0
            if c is None:
                continue
            ncy = NC.nearclose(c)
            for nm, fct in facs.items():
                if fct <= 0:
                    rows.append(dict(symbol=sym, pid=p.pid, entry=d, sizing=nm, factor=0.0, date=d, pnl=0.0, cost=0.0))
                    continue
                cc = R.Cycle(ncy.dates, ncy.K, ncy.lot, ncy.fut_lot, c.units * C * fct, ncy.opt_close, ncy.opt_vwap,
                             ncy.delta, ncy.fut_close, ncy.fut_vwap, ncy.DF)
                r = R.run_cycle(cc, "round", "FUT", cms, False, last_hedge=True)
                if r is None:
                    rows.append(dict(symbol=sym, pid=p.pid, entry=d, sizing=nm, factor=fct, date=d, pnl=0.0, cost=0.0))
                    continue
                for k, dd in enumerate(c.dates):
                    rows.append(dict(symbol=sym, pid=p.pid, entry=d, sizing=nm, factor=fct, date=dd,
                                     pnl=r["pnl"][k] / C, cost=r["cost"][0, k] / C))
        print(f"  money test {sym} done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report

def vol_report(fc: dict, L: list) -> list:
    names = ["imp", "imp_atm", "imp_sc", "garch_raw", "garch", "har", "blend", "blend_inv", "har_fast", "blend_fast"]
    prim = []
    for lab, lo, hi in (("DISCOVERY 2013-2017", DISC0, HOLD0), ("HOLDOUT 2018+", HOLD0, pd.Timestamp("2100-01-01"))):
        L.append(f"\n--- {lab}: mean QLIKE (lower = better); log-vol RMSE; bias = mean ln(forecast/realised); "
                 f"DM = blend or X vs implied, NW t (positive = better than implied)")
        for sym in SYMS:
            df = fc[sym]
            for h in HS:
                x = df[(df.index >= lo) & (df.index < hi)]
                y = x[f"y{h}"].to_numpy(float)
                cols = [c for c in names if f"{c}{h}" in x]
                ok = np.isfinite(y)
                for c in cols:
                    ok &= np.isfinite(x[f"{c}{h}"].to_numpy(float))
                y = y[ok]
                parts = []
                L_imp = qlike(y, x[f"imp{h}"].to_numpy(float)[ok])
                for c in cols:
                    f = x[f"{c}{h}"].to_numpy(float)[ok]
                    q = qlike(y, f)
                    lv = 0.5 * (np.log(f) - np.log(np.maximum(y, 1e-10)))
                    m, t, n = nw_t(L_imp - q, h)
                    parts.append(f"{c:10s} QLIKE {q.mean():.4f} rmse {np.sqrt((lv ** 2).mean()):.3f} bias {lv.mean():+.3f}"
                                 + ("" if c == "imp" else f"  DM t {t:+.2f}"))
                    if c == "blend" and lo == HOLD0:
                        prim.append((sym, h, m, t, p2(t), n))
                L.append(f"  {sym} h={h}  (n={ok.sum()}; mean blend weights imp/garch/har "
                         f"{x[f'w_imp{h}'].mean():.2f}/{x[f'w_garch{h}'].mean():.2f}/{x[f'w_har{h}'].mean():.2f})")
                L += ["    " + s for s in parts]
    adj = holm([p for *_, p, _ in prim])
    L.append("\nPRIMARY TEST (holdout, blend vs implied, Holm over 4):")
    passed = []
    for (sym, h, m, t, p, n), pa in zip(prim, adj):
        ok = m > 0 and pa < 0.05
        passed.append(ok)
        L.append(f"  {sym} h={h}: mean QLIKE gain {m:+.4f}, NW t {t:+.2f}, p {p:.4f}, Holm p {pa:.4f}, n {n} -> "
                 f"{'PASS' if ok else 'fail'}")
    L.append(f"  VERDICT: {'PASS' if all(passed) else 'PASS on ' + str(sum(passed)) + ' of 4' if any(passed) else 'FAIL'}")
    return prim


def placebo(fc: dict, L: list, n_shift: int = 200):
    rng = np.random.default_rng(0)
    L.append("\nSHUFFLED-DATE PLACEBO (holdout): blend's QLIKE gain over implied with targets circularly shifted "
             ">= 1 year; share of the real gain that survives = average-premium part")
    for sym in SYMS:
        df = fc[sym]
        x = df[df.index >= HOLD0]
        for h in HS:
            y = x[f"y{h}"].to_numpy(float)
            fb, fi = x[f"blend{h}"].to_numpy(float), x[f"imp{h}"].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(fb) & np.isfinite(fi)
            y, fb, fi = y[ok], fb[ok], fi[ok]
            real = (qlike(y, fi) - qlike(y, fb)).mean()
            sh = []
            for _ in range(n_shift):
                k = int(rng.integers(252, len(y) - 252))
                ys = np.roll(y, k)
                sh.append((qlike(ys, fi) - qlike(ys, fb)).mean())
            sh = np.array(sh)
            L.append(f"  {sym} h={h}: real gain {real:+.4f}; shifted median {np.median(sh):+.4f} "
                     f"(5-95% {np.percentile(sh, 5):+.4f}..{np.percentile(sh, 95):+.4f}); share of shifts >= real "
                     f"{(sh >= real).mean():.2f}")


def prob_report(prob: dict, dirs: dict, L: list):
    L.append("\n--- PROBABILITY FORECASTS (next monthly expiry), Brier score (lower = better), NW lag 25")
    tests = []
    for lab, lo, hi in (("discovery", DISC0, HOLD0), ("holdout", HOLD0, pd.Timestamp("2100-01-01"))):
        for sym in SYMS:
            pr = prob[sym]
            x = pr[(pr.index >= lo) & (pr.index < hi)]
            for ev, base in (("range", "range_mkt"), ("seller", "seller_base")):
                cols = [f"{ev}_model", base] + ([f"{ev}_base"] if base != f"{ev}_base" else [])
                ok = np.isfinite(x[cols].to_numpy(float)).all(axis=1)
                yy = x[f"{ev}_y"].to_numpy(float)[ok]
                br = {c: (x[c].to_numpy(float)[ok] - yy) ** 2 for c in cols}
                ll = {c: -np.mean(yy * np.log(np.clip(x[c].to_numpy(float)[ok], 1e-4, 1 - 1e-4)) +
                                  (1 - yy) * np.log(np.clip(1 - x[c].to_numpy(float)[ok], 1e-4, 1 - 1e-4))) for c in cols}
                m, t, n = nw_t(br[base] - br[f"{ev}_model"], NW_PROB)
                s = ", ".join(f"{c} Brier {br[c].mean():.4f} logloss {ll[c]:.4f} meanp {x[c].to_numpy(float)[ok].mean():.3f}"
                              for c in cols)
                L.append(f"  {lab} {sym} {ev}: hit rate {yy.mean():.3f} (n {n}); {s}; model vs {base}: gain {m:+.4f} t {t:+.2f}")
                if lab == "holdout":
                    tests.append((f"{sym} {ev}", m, t, p2(t)))
    adj = holm([x[3] for x in tests])
    L.append("  SECONDARY (holdout, Holm over 4): " + "; ".join(
        f"{nm} gain {m:+.4f} t {t:+.2f} Holm p {pa:.3f}" for (nm, m, t, _), pa in zip(tests, adj)))
    L.append("\n--- CONTROL: direction over the next 5 sessions (expected: no skill)")
    for lab, lo, hi in (("discovery", DISC0, HOLD0), ("holdout", HOLD0, pd.Timestamp("2100-01-01"))):
        for sym in SYMS:
            d = dirs[sym]
            x = d[(d.index >= lo) & (d.index < hi)]
            ok = np.isfinite(x.to_numpy(float)).all(axis=1)
            yy = x["dir_y"].to_numpy(float)[ok]
            bm = (x["dir_model"].to_numpy(float)[ok] - yy) ** 2
            bb = (x["dir_base"].to_numpy(float)[ok] - yy) ** 2
            bh = (0.5 - yy) ** 2
            m, t, n = nw_t(bb - bm, 5)
            hit = ((x["dir_model"].to_numpy(float)[ok] > 0.5) == (yy > 0.5)).mean()
            L.append(f"  {lab} {sym}: up share {yy.mean():.3f}; Brier model {bm.mean():.4f} base rate {bb.mean():.4f} "
                     f"coin {bh.mean():.4f}; model vs base gain {m:+.5f} t {t:+.2f}; direction hit rate {hit:.3f} (n {n})")


def money_report(mon: pd.DataFrame, L: list):
    import realistic_engine as R
    L.append("\n--- MONEY TEST: Stage 5 monthly straddles, 1 Cr, near-close fills, FUT round; % of capital per year "
             "(excess over cash)")
    sessions = pd.DatetimeIndex(sorted(mon["date"].unique()))
    intr = R.interest_series(sessions)
    for lab, lo, hi in (("discovery 2013-2017", DISC0, HOLD0), ("holdout 2018+", HOLD0, pd.Timestamp("2100-01-01"))):
        x = mon[(mon["entry"] >= lo) & (mon["entry"] < hi)]
        mo = {}
        for nm, g in x.groupby("sizing"):
            s = (g["pnl"] - g["cost"]).groupby(g["date"]).sum()
            mo[nm] = s.resample("ME").sum()
        idx = mo["const"].index.union(mo["blend"].index).union(mo["naive"].index)
        yrs = (idx.max() - idx.min()).days / 365.25
        for nm in ("const", "blend", "naive"):
            s = mo[nm].reindex(idx).fillna(0)
            f = x[x["sizing"] == nm].drop_duplicates(["symbol", "pid"])["factor"]
            L.append(f"  {lab} {nm:6s}: {s.sum() / yrs:+.2%}/yr, Sharpe {s.mean() / s.std() * np.sqrt(12):+.2f}, worst month "
                     f"{s.min():+.2%}, mean factor {f.mean():.2f}, skipped cycles {(f == 0).mean():.0%}")
        for nm in ("blend", "naive"):
            d = (mo[nm].reindex(idx).fillna(0) - mo["const"].reindex(idx).fillna(0)).to_numpy()
            m, t, n = nw_t(d, 3)
            L.append(f"    {nm} - const: {m * 12:+.2%}/yr, NW t {t:+.2f} (n {n} months)")
    del intr


def lookahead_check(sym: str, closes: pd.Series, vix: pd.Series, imp: pd.DataFrame, full: pd.DataFrame, L: list):
    rng = np.random.default_rng(1)
    worst = 0.0
    cols = [c for c in full.columns if any(c.startswith(p) for p in ("har", "blend", "garch", "imp_sc", "w_"))]
    for _ in range(3):
        cut = int(rng.integers(len(closes) // 3, len(closes) - 30))
        tr = build(sym, closes.iloc[:cut + 1], vix, imp)
        a = tr[cols].to_numpy(float)
        b = full[cols].iloc[:cut + 1].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        worst = max(worst, float(np.nanmax(np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1e-12))),
                    float((np.isfinite(a) != np.isfinite(b)).sum()))
    L.append(f"  look-ahead (truncation) {sym}: max relative difference / mismatched availability {worst:.2e}")
    return worst


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    vix = pd.read_csv(os.path.join(UND, "india_vix.csv"), parse_dates=["date"]).set_index("date")["close"].astype(float)
    fc, prob, dirs, L = {}, {}, {}, [__doc__.strip().split("\n")[0], ""]
    L.append("CHECKS")
    for sym in SYMS:
        closes = load_closes(sym)
        imp = implied_table(sym, closes.index)
        print(f"  {sym}: implied table {len(imp)} sessions ({time.time() - t0:.0f}s)", flush=True)
        df = build(sym, closes, vix, imp)
        ps = pd.read_parquet(os.path.join(S4, "pstate.parquet"))
        ps = ps[ps["symbol"] == sym].set_index("date")["r"]
        j = df[["r"]].join(ps.rename("r4"), how="inner")
        L.append(f"  {sym}: returns vs Stage 4 returns corr {j['r'].corr(j['r4']):.6f}, "
                 f"max |diff| {np.abs(j['r'] - j['r4']).max():.2e} (n {len(j)})")
        lookahead_check(sym, closes, vix, imp, df, L)
        fc[sym] = df
        prob[sym] = prob_table(df)
        dirs[sym] = direction(df)
        print(f"  {sym}: forecasts done ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(fc.values()).to_parquet(os.path.join(OUT, "forecasts.parquet"))
    pd.concat([p.assign(symbol=s) for s, p in prob.items()]).to_parquet(os.path.join(OUT, "prob.parquet"))
    vol_report(fc, L)
    placebo(fc, L)
    prob_report(prob, dirs, L)
    mon = money(fc, prob)
    mon.to_parquet(os.path.join(OUT, "money.parquet"))
    money_report(mon, L)
    L.insert(1, f"run time {time.time() - t0:.0f}s")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
