"""Seller's strike picker: can we choose WHICH options to sell (delta-hedged) better than
selling all of them, walk-forward, after costs?

Rules fixed before running (do not change after seeing results):
  Universe   Stage 4 attribution options (= Stage 1 entries: OTM, |delta| 0.02-0.50, traded
             >= 10 contracts, price >= 0.5, no carry forwards, inside that day's Bates fit),
             entered at the close, sold, delta-hedged daily to expiry (Stage 1 hedge, Stage 1
             costs incl. STT/fees/stamp/brokerage/half-spread; stress = Stage 1 stress costs).
  Risk unit  every option is sized so its stress loss is equal: L = worst loss of 1 short,
             delta-hedged unit on an instant index move of -15/-10/-5/+5/+10% with IV x1 or
             x1.5 (Stage 5 scenario, Black-76 at the option's own IV). Outcome
             r = seller net P&L per unit (hsell_net/100 * mark) / L  ("return per unit of
             crash risk").
  Books      a cycle = one (symbol, expiry) entry at horizon h. The book sells the chosen options
             with equal stress budgets, total = 25% of that symbol's capital (the Stage 5 engine's
             risk budget); cycle return on symbol capital = 0.25 * mean(r of chosen options);
             total capital = half per symbol -> monthly book return = mean over the two symbols
             of the summed returns of that symbol's cycles entered in the month.
    PRIMARY   h = 21 sessions, monthly expiries only (one cycle per symbol per month; this is
              the frozen engine's timing).
    SECONDARY h = 5 sessions, every expiry (weekly where listed; cycles do not overlap).
  Cycle needs >= 6 options; "chosen" = the top third (ceil(n/3)) by predicted seller r.
  Models     M0 sell all (baseline)            M1 ATM pair (nearest call and put, engine-like)
             M2 Bates: lowest S_P = ln(q_jc / mark) (Bates at P level; low = rich)
             M3 naive: lowest S_N = sqrt(p_naive / T) - iv
             M4 ridge regression (alpha 10, standardised features)
             M5 gradient boosting (sklearn HistGradientBoosting: depth 3, lr 0.05, 300 iters,
                min leaf 200, l2 1.0, seed 0)
  Features   (all known at the entry close) put flag, |delta|, tdte, standardised moneyness
             ln(K/F)/(iv sqrt T), iv, iv - ATM iv of the same expiry, S_P, S_N, ATM iv - naive
             vol, scheduled event before settlement (known by then), index return over the
             last 22 sessions, BANKNIFTY flag. Missing ATM iv -> median iv of the cross-section.
  Training   M4/M5 trained on ALL horizons (2-42 sessions, all expiries, both symbols) with
             target r winsorised at the 1st/99th percentile of the training set.
             Walk-forward: for test year Y, train on options SETTLED in [Y-5, Y) (rolling
             5 years = primary) or settled before Y (expanding = check); predict entries in Y.
             Test years 2013 .. 2026 (2008-2012 is the first training window).
  Primary test   PRIMARY book, rolling window: monthly difference (model book - M0 book),
             net costs, Newey-West (3 lags) one-sided p for > 0, Holm 5% over M2-M5.
  Verdict    a model "picks better" only if: Holm passes, AND the difference is > 0 under stress
             costs, AND > 0 in both halves (2013-2019, 2020-2026), AND > 0 on the one-session
             skip basis (sold at the next session's traded mark, hedged from then; so a stale or
             noisy close cannot create the result).
  Reported   book annual return / Sharpe / max drawdown (sum of monthly returns) / worst month
             for M0-M5, net and stress; vs M1; expanding window; SECONDARY book (its own Holm).
  Limits     sum-of-worst sizing ignores diversification between calls and puts (conservative);
             no margin model; Stage 1 lets the buyer side pick exercise vs sale (conservative for
             sellers).
Writes data/processed/strike_picker/ (dataset.parquet, preds.parquet, books.parquet, _report.txt)
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

from src import contracts as ct  # noqa: E402

S1 = os.path.join(PROJECT_ROOT, "data", "processed", "stage1", "trades.parquet")
S4 = os.path.join(PROJECT_ROOT, "data", "processed", "stage4")
EVENTS = os.path.join(PROJECT_ROOT, "config", "india_events.csv")
OUT = os.path.join(PROJECT_ROOT, "data", "processed", "strike_picker")

MOVES = np.array([-0.15, -0.10, -0.05, 0.05, 0.10])
IV_MULT = np.array([1.0, 1.5])
RISK = 0.25
MIN_UNIV = 6
ROLL_YEARS = 5
TEST_YEARS = range(2013, 2027)
HALF = pd.Period("2020-01", "M")
NW_LAGS = 3
FEATURES = ["is_put", "abs_delta", "tdte", "z", "iv", "iv_rel", "S_P", "S_N", "vrp", "event", "ret22", "is_bank"]
MODELS = ["M0", "M1", "M2", "M3", "M4", "M5"]
NAMES = {"M0": "sell all", "M1": "ATM pair", "M2": "Bates S_P", "M3": "naive S_N", "M4": "ridge", "M5": "boosting"}
LINES: list[str] = []


def out(s=""):
    print(s, flush=True)
    LINES.append(str(s))


# ---------------------------------------------------------------- dataset

def stress_unit(F, K, T, iv, df, delta, is_call):
    v0 = ct.b76_price(F, K, T, iv, df, is_call)
    worst = np.zeros(len(F))
    for m in MOVES:
        for vm in IV_MULT:
            v1 = ct.b76_price(F * (1 + m), K, T, iv * vm, df, is_call)
            worst = np.minimum(worst, -(v1 - v0) + delta * F * m)
    return -worst


def events_flag(d: pd.DataFrame) -> np.ndarray:
    ev = pd.read_csv(EVENTS, parse_dates=["date", "known_from"])
    f = np.zeros(len(d), bool)
    for e in ev.itertuples():
        f |= (d["date"] >= e.known_from).to_numpy() & (d["date"] < e.date).to_numpy() & (
            d["settled"] >= e.date).to_numpy()
    return f.astype(float)


def build_dataset() -> pd.DataFrame:
    import stage5_richcheap as rc
    t0 = time.time()
    a = rc.add_skip(rc.load())
    t = pd.read_parquet(S1, columns=["tid", "forward", "log_moneyness", "hbuy_gross", "hsell_net_stress"]
                        ).rename(columns={"hbuy_gross": "hbg"})
    a = a.drop(columns=[c for c in ("forward",) if c in a.columns]).merge(t, on="tid", how="left")
    a = a[a["hsell_net"].notna() & a["hsell_net_stress"].notna()].copy()
    is_call = (a["side"] == "CE").to_numpy()
    a["L"] = stress_unit(a["forward"].to_numpy(float), a["strike"].to_numpy(float), a["T"].to_numpy(float),
                         a["iv"].to_numpy(float), a["df"].to_numpy(float), a["delta"].to_numpy(float), is_call)
    a = a[a["L"] > 0].copy()
    unit = a["mark"] / 100 / a["L"]
    a["r"] = a["hsell_net"] * unit
    a["r_stress"] = a["hsell_net_stress"] * unit
    cost100 = -a["hbg"] - a["hsell_net"]
    a["r_skip"] = (-a["y_skip"] - cost100) / 100 * a["mark1"] / a["L"]
    a["is_put"] = (a["side"] == "PE").astype(float)
    a["abs_delta"] = a["delta"].abs()
    a["z"] = a["log_moneyness"] / (a["iv"] * np.sqrt(a["T"]))
    key = ["symbol", "date", "final_expiry"]

    def atm(g):
        x, y = g["log_moneyness"].to_numpy(), g["iv"].to_numpy()
        o = np.argsort(x)
        x, y = x[o], y[o]
        return float(np.interp(0.0, x, y)) if len(x) >= 2 and x[0] <= 0 <= x[-1] else np.nan

    at = a.groupby(key).apply(atm, include_groups=False).rename("atm_iv").reset_index()
    a = a.merge(at, on=key, how="left")
    a["atm_iv"] = a["atm_iv"].fillna(a.groupby(key)["iv"].transform("median"))
    a["iv_rel"] = a["iv"] - a["atm_iv"]
    a["vrp"] = a["atm_iv"] - np.sqrt(a["p_naive"] / a["T"])
    a["event"] = events_flag(a)
    ps = pd.read_parquet(os.path.join(S4, "pstate.parquet"))
    ps = ps.sort_values(["symbol", "date"])
    ps["ret22"] = ps.groupby("symbol")["r"].transform(lambda s: s.rolling(22, min_periods=15).sum())
    a = a.merge(ps[["symbol", "date", "ret22"]], on=["symbol", "date"], how="left")
    a["ret22"] = a["ret22"].fillna(0.0)
    a["is_bank"] = (a["symbol"] == "BANKNIFTY").astype(float)
    mon = a.groupby(["symbol", a["settled"].dt.to_period("M")])["settled"].transform("max")
    a["monthly"] = a["settled"] == mon
    a = a[np.isfinite(a[FEATURES]).all(axis=1) & np.isfinite(a["r"])].reset_index(drop=True)
    out(f"dataset: {len(a):,} options, {a['date'].min().date()}..{a['date'].max().date()}"
        f" ({time.time() - t0:.0f}s); skip outcome available {a['r_skip'].notna().mean():.0%}")
    return a


# ---------------------------------------------------------------- models

def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, Xt = train[FEATURES].to_numpy(float), test[FEATURES].to_numpy(float)
    lo, hi = np.quantile(train["r"], [0.01, 0.99])
    y = np.clip(train["r"].to_numpy(), lo, hi)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z, Zt = (X - mu) / sd, (Xt - mu) / sd
    A = Z.T @ Z + 10.0 * np.eye(Z.shape[1])
    b = np.linalg.solve(A, Z.T @ (y - y.mean()))
    gb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=200,
                                       l2_regularization=1.0, random_state=0).fit(X, y)
    return {"M4": y.mean() + Zt @ b, "M5": gb.predict(Xt), "coef": dict(zip(FEATURES, b))}


def walk_forward(a: pd.DataFrame) -> pd.DataFrame:
    preds, coefs = [], []
    for Y in TEST_YEARS:
        y0 = pd.Timestamp(f"{Y}-01-01")
        test = a[(a["date"] >= y0) & (a["date"] < pd.Timestamp(f"{Y + 1}-01-01"))]
        if test.empty:
            continue
        for win in ("roll", "expand"):
            lo = pd.Timestamp(f"{Y - ROLL_YEARS}-01-01") if win == "roll" else pd.Timestamp("1900-01-01")
            train = a[(a["settled"] >= lo) & (a["settled"] < y0)]
            p = fit_predict(train, test)
            preds.append(pd.DataFrame({"idx": test.index, "win": win, "M4": p["M4"], "M5": p["M5"]}))
            coefs.append(dict(year=Y, win=win, n_train=len(train), **p["coef"]))
        out(f"  trained {Y}: rolling n={coefs[-2]['n_train']:,}, expanding n={coefs[-1]['n_train']:,}")
    return pd.concat(preds, ignore_index=True), pd.DataFrame(coefs)


# ---------------------------------------------------------------- books

def books(a: pd.DataFrame, pr: pd.DataFrame, univ: pd.DataFrame, win: str) -> pd.DataFrame:
    p = pr[pr["win"] == win].set_index("idx")
    u = univ.join(p[["M4", "M5"]], how="inner")
    u["M2"], u["M3"] = -u["S_P"], -u["S_N"]
    rows = []
    for (sym, d, fe), g in u.groupby(["symbol", "date", "final_expiry"]):
        n = len(g)
        if n < MIN_UNIV:
            continue
        k = int(np.ceil(n / 3))
        row = dict(symbol=sym, date=d, settled=g["settled"].iloc[0], n=n)
        sel = {"M0": g.index}
        c, pu = g[g["side"] == "CE"], g[g["side"] == "PE"]
        if len(c) and len(pu):
            sel["M1"] = [c["log_moneyness"].abs().idxmin(), pu["log_moneyness"].abs().idxmin()]
        for m in ("M2", "M3", "M4", "M5"):
            sel[m] = g[m].nlargest(k).index
        for m, ix in sel.items():
            x = g.loc[ix]
            row[f"{m}"] = RISK * x["r"].mean()
            row[f"{m}_stress"] = RISK * x["r_stress"].mean()
            row[f"{m}_skip"] = RISK * x["r_skip"].mean() if x["r_skip"].notna().all() else np.nan
        rows.append(row)
    cy = pd.DataFrame(rows)
    cy["month"] = cy["date"].dt.to_period("M")
    cols = [c for c in cy.columns if c.startswith("M")]
    per_sym = cy.groupby(["symbol", "month"])[cols].sum(min_count=1).reset_index()
    return per_sym.groupby("month")[cols].mean()


def nw(x, lags=NW_LAGS):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12:
        return np.nan, np.nan, np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    t = x.mean() / np.sqrt(v / n)
    return x.mean(), t, 1 - norm.cdf(t)


def stats_line(s: pd.Series) -> str:
    s = s.dropna()
    cum = s.cumsum()
    dd = (cum.cummax() - cum).max()
    return (f"{12 * s.mean():+7.2%}/yr  Sharpe {s.mean() / s.std() * np.sqrt(12):+5.2f}  maxDD {dd:6.1%}"
            f"  worst month {s.min():+6.1%}")


def holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: (np.nan_to_num(p[k], nan=1.0)))
    m, adj, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * np.nan_to_num(p[k], nan=1.0)))
        adj[k] = run
    return adj


def report_book(bk: pd.DataFrame, title: str, verdict: bool):
    out("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    out(f"months {len(bk)} ({bk.index.min()}..{bk.index.max()})")
    for m in MODELS:
        if m not in bk:
            continue
        out(f"  {m} {NAMES[m]:10s} net {stats_line(bk[m])} | stress {stats_line(bk[m + '_stress'])}")
    out("  difference vs M0 (sell all), per month, book units:")
    res = {}
    for m in ("M1", "M2", "M3", "M4", "M5"):
        d = bk[m] - bk["M0"]
        mean, t, p = nw(d)
        ds = bk[m + "_stress"] - bk["M0_stress"]
        dk = bk[m + "_skip"] - bk["M0_skip"]
        h1, h2 = d[d.index < HALF].mean(), d[d.index >= HALF].mean()
        res[m] = dict(p=p, stress=ds.mean(), skip=dk.mean(), h1=h1, h2=h2)
        out(f"    {m} {NAMES[m]:10s} {12 * mean:+7.2%}/yr (t {t:+.2f}, p {p:.4f}) | stress {12 * ds.mean():+7.2%}/yr"
            f" | skip basis {12 * dk.mean():+7.2%}/yr ({dk.notna().sum()} mo) | halves {12 * h1:+.2%} / {12 * h2:+.2%}")
    out("  difference vs M1 (ATM pair):")
    for m in ("M2", "M3", "M4", "M5"):
        mean, t, p = nw(bk[m] - bk["M1"])
        out(f"    {m} {NAMES[m]:10s} {12 * mean:+7.2%}/yr (t {t:+.2f}, p {p:.4f})")
    if verdict:
        adj = holm({m: res[m]["p"] for m in ("M2", "M3", "M4", "M5")})
        out("  VERDICT (Holm over M2-M5; + stress, both halves, skip basis):")
        for m in ("M2", "M3", "M4", "M5"):
            r = res[m]
            ok = adj[m] < 0.05 and r["stress"] > 0 and r["h1"] > 0 and r["h2"] > 0 and r["skip"] > 0
            out(f"    {m} {NAMES[m]:10s} Holm p {adj[m]:.4f} -> {'PICKS BETTER' if ok else 'no'}")


def main():
    os.makedirs(OUT, exist_ok=True)
    ds_path = os.path.join(OUT, "dataset.parquet")
    if os.path.exists(ds_path):
        a = pd.read_parquet(ds_path)
        out(f"dataset (cached): {len(a):,} options")
    else:
        a = build_dataset()
        a.to_parquet(ds_path)
    out("walk-forward training:")
    pr, coefs = walk_forward(a)
    pr.to_parquet(os.path.join(OUT, "preds.parquet"))
    test = a[a["date"] >= pd.Timestamp(f"{TEST_YEARS[0]}-01-01")]
    prim = test[test["monthly"] & (test["tdte"] == 21)]
    sec = test[test["tdte"] == 5]
    out(f"PRIMARY universe: {len(prim):,} options; SECONDARY: {len(sec):,}")
    allb = []
    for win in ("roll", "expand"):
        for lab, u, ver in (("PRIMARY (monthly, 21 sessions)", prim, win == "roll"),
                            ("SECONDARY (every expiry, 5 sessions)", sec, win == "roll")):
            bk = books(a, pr, u, win)
            report_book(bk, f"{lab} - {'rolling 5y' if win == 'roll' else 'expanding'} window", ver)
            allb.append(bk.assign(win=win, book=lab.split()[0]))
    pd.concat(allb).to_parquet(os.path.join(OUT, "books.parquet"))
    out("\nRidge coefficients (standardised; + = higher seller return), rolling window, by year:")
    c = coefs[coefs["win"] == "roll"].set_index("year")[FEATURES]
    out(c.round(4).to_string())
    with open(os.path.join(OUT, "_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
