"""Stage 4 data build: market-implied (Q) vs past-only physical (P) vs realised.

Per symbol and session t with a Stage 3 fit ("best" model: Bates + events where fitted,
else Bates), and two target expiries of that session's chain:
  short  the nearest expiry used in the fit (>= 2 trading days)
  month  the expiry closest to 30 calendar days (kept only if 20-45 days)
we record
  Q    expected quadratic variation to expiry (diffusive / jump / event parts), variance of
       ln(F_T/F_t), and the Q-probability u_Q = P^Q(F_T <= realised settlement) ("PIT")
  P    past-only GJR-GARCH-t forecast of sum r^2 over the sessions to settlement, raw and
       multiplied by a past-only calibration factor (sum RV / sum forecast over earlier rows
       of the same symbol whose settlement is before t); the naive trailing 22-session
       realised variance as an independent cross-check; PITs u_{v}{l} under Q's model
       moved to each P level (see attribution below)
  real realised settlement, ln(S_T/F_t), realised variance sum r^2 over (t, settlement],
       variance on threshold-jump days (3.5 / 4 / 5 GARCH sigma) in the window
Events: Q-implied variance of each event session's return (diffusive + jump part from the
fit on the session before, event part psi^2 from the last event-identified fit before the
session) vs the realised squared return and the P one-day forecast.
Stage 1 attribution: every Stage 1 entry with tdte >= 2 and expiry inside the fitted
maturity range is repriced under
  q_a       Q, the day's fitted model
  q_{v}{l}  Q's model moved to the P variance level l (c = calibrated GARCH, n = naive
            trailing RV) by variant v (d = diffusive only, p = proportional, j = jumps
            first; see at_level)
No lookahead: Q from session t's prices; P from returns up to t (GARCH parameters fitted on
returns before the month); realised quantities are only outcomes.
Outputs data/processed/stage4/{outcomes,events,attrib,pstate}.parquet
"""
from __future__ import annotations

import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import bates as bt  # noqa: E402
from src import calibrate as cb  # noqa: E402
from src import pmodel as pm  # noqa: E402
from src.contracts import SYMBOLS  # noqa: E402

S3 = os.path.join(PROJECT_ROOT, "data", "processed", "stage3", "fits")
OUT = os.path.join(PROJECT_ROOT, "data", "processed", "stage4")
CONTRACTS = os.path.join(PROJECT_ROOT, "data", "processed", "contracts")
STAGE1 = os.path.join(PROJECT_ROOT, "data", "processed", "stage1", "trades.parquet")
YAHOO = {"NIFTY": "nifty.csv", "BANKNIFTY": "banknifty.csv"}
H = 1e-4            # relative strike bump for the CDF
JUMP_CS = (3.5, 4.0, 5.0)
MIN_CAL_ROWS = 100  # earlier settled rows needed for the calibration factor
S_FLOOR = 0.05      # floor on the diffusive scale factor in (b)


# ---------------------------------------------------------------- helpers

def all_fits(symbol: str) -> pd.DataFrame:
    f = pd.concat([pd.read_parquet(x) for x in glob.glob(os.path.join(S3, f"{symbol}_*.parquet"))], ignore_index=True)
    return f[f["p_v0"].notna()]


def best_fits(f: pd.DataFrame) -> pd.DataFrame:
    f = f[f["model"].isin(["bates", "bates_ev"])].copy()
    f["rank"] = (f["model"] == "bates_ev").astype(int)
    return f.sort_values(["date", "rank"]).groupby("date").tail(1).set_index("date")


def params_of(row, n_events: int) -> bt.BatesParams:
    m = row["model"]
    x = np.array([row[f"p_{k}"] for k in cb.MODELS[m]], float)
    return cb.to_params(m, x, n_events)


def q_cdf(F, T, p, x) -> np.ndarray:
    """P^Q(ln(F_T/F) <= x) = d Put / dK at K = F e^x (DF = 1)."""
    x = np.atleast_1d(x)
    K = F * np.exp(x)
    Kb = np.concatenate([K * (1 - H), K * (1 + H)])
    put = bt.price_cos(F, Kb, T, p, 1.0, False)
    n = len(K)
    return np.clip((put[n:] - put[:n]) / (2 * H * K), 0.0, 1.0)


def qv_parts(T, p) -> tuple[float, float, float]:
    """Q-expected quadratic variation of ln F over (0, T]: diffusive, jump, event."""
    return bt._mean_variance(p, T), p.lam * T * (p.mu_j ** 2 + p.delta ** 2), p.ev


VARIANTS = ("d", "p", "j")


def at_level(p: bt.BatesParams, T: float, target: float, how: str) -> tuple[bt.BatesParams | None, bool]:
    """Q's model moved to expected QV = target (diffusive QV is linear in (v0, theta), jump
    QV in lam, event QV in ev):
      d  scale the diffusive part only (Q jumps and events kept; floored at S_FLOOR, flag)
      p  scale diffusive, jump intensity and event variance by the same factor
      j  scale jumps and events first (to zero if needed), then the diffusive part."""
    if not np.isfinite(target) or target <= 0:
        return None, False
    qd, qj, qe = qv_parts(T, p)
    sd = sj = 1.0
    flag = False
    if how == "d":
        sd = (target - qj - qe) / qd
        flag = sd < S_FLOOR
        sd = max(sd, S_FLOOR)
    elif how == "p":
        sd = sj = target / (qd + qj + qe)
    elif how == "j":
        if qj + qe > 0 and target >= qd:
            sj = (target - qd) / (qj + qe)
        else:
            sj, sd = 0.0, target / qd
    return bt.BatesParams(v0=p.v0 * sd, kappa=p.kappa, theta=p.theta * sd, sigma=p.sigma, rho=p.rho,
                          lam=p.lam * sj, mu_j=p.mu_j, delta=p.delta, ev=p.ev * sj), flag


class PState:
    """Past-only P quantities for one symbol."""

    def __init__(self, symbol: str, spot: pd.Series):
        y = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "underlying", YAHOO[symbol]),
                        parse_dates=["date"]).set_index("date")["close"]
        c = pm.close_series(y, spot)
        self.r = np.log(c).diff().dropna()
        self.params = pm.fit_monthly(self.r)
        self.state = pm.daily_state(self.r, self.params).set_index("date")
        self.jumps = {c: pm.jump_stats(self.r, self.state.reset_index(), c).set_index("date") for c in JUMP_CS}
        r2 = self.r ** 2
        self.cum_r2 = r2.cumsum()
        self.cum_jr2 = {c: r2.where(j["jump"].reindex(self.r.index).fillna(False).astype(bool), 0.0).cumsum()
                        for c, j in self.jumps.items()}
        self.cum_n = pd.Series(np.arange(1, len(self.r) + 1), index=self.r.index)
        self.naive_tab = pm.trailing_rv(self.r)

    def naive(self, t, n) -> float:
        return float(n * self.naive_tab.get(t, np.nan)) if n > 0 else np.nan

    def window(self, t, T_end):
        """Realised sum r^2, jump-day sum r^2 per threshold, and sessions over (t, T_end]."""
        if t not in self.cum_r2.index or T_end not in self.cum_r2.index:
            return np.nan, {c: np.nan for c in JUMP_CS}, 0
        return (float(self.cum_r2[T_end] - self.cum_r2[t]),
                {c: float(v[T_end] - v[t]) for c, v in self.cum_jr2.items()},
                int(self.cum_n[T_end] - self.cum_n[t]))

    def sessions(self, t, T_end) -> int:
        return self.window(t, T_end)[2]

    def forecast(self, t, n) -> float:
        if t not in self.state.index or n <= 0:
            return np.nan
        return pm.variance_forecast(self.state.loc[t], n)


def calibration(o: pd.DataFrame) -> pd.Series:
    """Per date t: sum RV / sum P forecast over rows settled strictly before t (past only)."""
    x = o.dropna(subset=["rv", "p_total"]).sort_values("settled")
    cs = pd.DataFrame(dict(settled=x["settled"].to_numpy(), rv=x["rv"].cumsum().to_numpy(),
                           p=x["p_total"].cumsum().to_numpy(), n=np.arange(1, len(x) + 1)))
    cs = cs.groupby("settled").last()
    dates = pd.DatetimeIndex(sorted(o["date"].unique()))
    pos = cs.index.searchsorted(dates, side="left") - 1      # last settlement < t
    out = np.full(len(dates), np.nan)
    ok = pos >= 0
    v = cs.iloc[pos[ok]]
    out[ok] = np.where(v["n"].to_numpy() >= MIN_CAL_ROWS, v["rv"].to_numpy() / v["p"].to_numpy(), np.nan)
    return pd.Series(out, index=dates)


# ---------------------------------------------------------------- per symbol

def build_symbol(symbol: str):
    t0 = time.time()
    fits_all = all_fits(symbol)
    fits = best_fits(fits_all)
    panel = cb.load_symbol(symbol)
    ev = cb.event_sessions(cb.load_events(), pd.DatetimeIndex(panel["date"].unique()))
    spot = pd.read_parquet(cb.PANEL, columns=["date", "spot"], filters=[("symbol", "=", symbol)]).groupby("date")["spot"].first()
    em = pd.read_parquet(os.path.join(CONTRACTS, "_expiry_map.parquet"))
    em = em[em["symbol"] == symbol].set_index("expiry")["settled"]
    P = PState(symbol, spot)
    by_day = dict(tuple(panel.groupby("date")))
    rows, chains = [], {}
    for d, fr in fits.iterrows():
        ch = cb.build_chain(symbol, d, by_day.get(d), ev) if d in by_day else None
        if ch is None:
            continue
        Ts = np.array([e.T for e in ch.expiries])
        targets = {"short": 0}
        j = int(np.argmin(np.abs(Ts - 30 / 365)))
        if 20 / 365 <= Ts[j] <= 45 / 365:
            targets["month"] = j
        for name, j in targets.items():
            e = ch.expiries[j]
            settled = em.get(e.expiry, pd.NaT)
            S_T = spot.get(settled, np.nan) if pd.notna(settled) else np.nan
            if not np.isfinite(S_T):
                continue
            p = params_of(fr, e.n_events)
            x = float(np.log(S_T / e.F))
            qd, qj, qe = qv_parts(e.T, p)
            rv, rjv, n_s = P.window(d, settled)
            rows.append(dict(symbol=symbol, date=d, target=name, model=fr["model"], expiry=e.expiry, settled=settled,
                             T=e.T, n_sessions=n_s, F=e.F, S_T=S_T, x=x, n_events=e.n_events,
                             k_lo=float(np.log(e.K.min() / e.F)), k_hi=float(np.log(e.K.max() / e.F)),
                             q_diff=qd, q_jump=qj, q_event=qe, q_var=bt.cumulants(e.T, p)[1],
                             q30=sum(qv_parts(30 / 365, params_of(fr, 0))),
                             u_q=float(q_cdf(e.F, e.T, p, x)[0]), rv=rv,
                             **{f"rv_jump_{c}": v for c, v in rjv.items()},
                             p_total=P.forecast(d, n_s), p_naive=P.naive(d, n_s), fit_rmse=fr["rmse_iv"], atm_iv=fr["atm_iv"],
                             _p=p))
    o = pd.DataFrame(rows)
    cal = calibration(o)
    o["cal"] = o["date"].map(cal)
    o["p_cal"] = o["p_total"] * o["cal"]
    for lvl, tgt in [("c", "p_cal"), ("n", "p_naive")]:
        for how in VARIANTS:
            us, fl = [], []
            for _, r in o.iterrows():
                q, f = at_level(r["_p"], r["T"], r[tgt], how)
                us.append(float(q_cdf(r["F"], r["T"], q, r["x"])[0]) if q is not None else np.nan)
                fl.append(f)
            o[f"u_{how}{lvl}"] = us
            if how == "d":
                o[f"floor_{lvl}"] = fl
    o = o.drop(columns="_p")
    print(f"{symbol}: {len(o)} outcome rows, {time.time() - t0:.0f}s", flush=True)
    return o, P, fits, fits_all, ev, cal


def events_table(symbol, P: PState, fits: pd.DataFrame, fits_all: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    ev_fits = fits_all[fits_all["model"] == "bates_ev"].set_index("date").sort_index()
    rows = []
    for _, e in ev.iterrows():
        sess = e["session"]
        if sess not in P.r.index:
            continue
        i = P.r.index.get_loc(sess)
        t = P.r.index[i - 1]
        if t not in fits.index:
            continue
        p = params_of(fits.loc[t], 0)
        dt = (sess - t).days / 365
        qd, qj, _ = qv_parts(dt, p)
        src = ev_fits[(ev_fits.index >= e["known_from"]) & (ev_fits.index < sess)]
        psi = float(src["p_psi"].iloc[-1]) if len(src) else np.nan
        st = P.state.loc[t] if t in P.state.index else None
        rows.append(dict(symbol=symbol, event=e["date"], session=sess, type=e["type"],
                         q_day_var=qd + qj, psi=psi, psi_from=src.index[-1] if len(src) else pd.NaT,
                         r=float(P.r.loc[sess]),
                         p_day_var=float(st["sigma2_next"] + st["mu"] ** 2) if st is not None else np.nan))
    return pd.DataFrame(rows)


def attribution(symbol, P: PState, fits: pd.DataFrame, ev: pd.DataFrame, cal: pd.Series) -> pd.DataFrame:
    tr = pd.read_parquet(STAGE1)
    tr = tr[(tr["symbol"] == symbol) & (tr["tdte"] >= 2)]
    out = []
    for (d, expiry, settled), g in tr.groupby(["date", "expiry", "settled"]):
        if d not in fits.index:
            continue
        fr = fits.loc[d]
        T, F = float(g["T"].iloc[0]), float(g["forward"].iloc[0])
        if not (fr["t_min"] - 1e-9 <= T <= fr["t_max"] + 1e-9):
            continue
        n_ev = int(((ev["known_from"] <= d) & (ev["session"] > d) & (ev["session"] <= expiry)).sum()) \
            if fr["model"] == "bates_ev" else 0
        p = params_of(fr, n_ev)
        n_s = P.sessions(d, settled)
        ptot = P.forecast(d, n_s)
        c = cal.get(d, np.nan)
        K = g["strike"].to_numpy(float)
        ic = (g["side"] == "CE").to_numpy()
        DF = float(g["df"].iloc[0])
        pnv = P.naive(d, n_s)
        res = dict(tid=g["tid"].to_numpy(), q_a=bt.price_cos(F, K, T, p, DF, ic), p_total=ptot, cal=c, p_naive=pnv,
                   q_qv=sum(qv_parts(T, p)), n_sessions=n_s)
        for lvl, tgt in [("c", ptot * c), ("n", pnv)]:
            for how in VARIANTS:
                q, _ = at_level(p, T, tgt, how)
                res[f"q_{how}{lvl}"] = bt.price_cos(F, K, T, q, DF, ic) if q is not None else np.full(len(K), np.nan)
        out.append(pd.DataFrame(res))
    a = pd.concat(out, ignore_index=True)
    return tr[["tid", "symbol", "date", "expiry", "final_expiry", "settled", "strike", "side", "tdte", "dbucket",
               "mark", "delta", "df", "payoff"]].merge(a, on="tid")


def main():
    os.makedirs(OUT, exist_ok=True)
    outs, evs, atts, pst = [], [], [], []
    for sym in SYMBOLS:
        o, P, fits, fits_all, ev, cal = build_symbol(sym)
        outs.append(o)
        evs.append(events_table(sym, P, fits, fits_all, ev))
        t0 = time.time()
        atts.append(attribution(sym, P, fits, ev, cal))
        print(f"{sym}: attribution {len(atts[-1])} trades, {time.time() - t0:.0f}s", flush=True)
        s = P.state[["r", "eps", "sigma2", "sigma2_next"]].join(P.jumps[4.0][["jump"]]).join(P.naive_tab.rename("naive_rv"))
        s["symbol"] = sym
        pst.append(s.reset_index())
        P.params.assign(symbol=sym).to_parquet(os.path.join(OUT, f"garch_params_{sym}.parquet"))
    pd.concat(outs, ignore_index=True).to_parquet(os.path.join(OUT, "outcomes.parquet"))
    pd.concat(evs, ignore_index=True).to_parquet(os.path.join(OUT, "events.parquet"))
    pd.concat(atts, ignore_index=True).to_parquet(os.path.join(OUT, "attrib.parquet"))
    pd.concat(pst, ignore_index=True).to_parquet(os.path.join(OUT, "pstate.parquet"))


if __name__ == "__main__":
    main()
