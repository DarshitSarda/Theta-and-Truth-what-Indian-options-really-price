"""SEBI Nov-2024 reform: did removing BANKNIFTY weekly options change how its options are priced?

Natural experiment. SEBI circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132 (1 Oct 2024): from 20 Nov
2024 one weekly-expiring index per exchange. NSE kept NIFTY weeklies; the last BANKNIFTY weekly
expired 13 Nov 2024. Treated = BANKNIFTY, comparison = NIFTY. Everything else in the circular
hit both on the same dates (lot sizes 20 Nov 2024, expiry-day extra margin 20 Nov 2024, upfront
premium and no calendar-spread benefit on expiry day 1 Feb 2025, intraday limits 1 Apr 2025),
so it cancels in the BANKNIFTY minus NIFTY gap - except the lot-size step, which was x3 for
NIFTY (25 -> 75) and x2 for BANKNIFTY (15 -> 30): a named confound, not removable.

Rules fixed before running (do not change after seeing results):
  Contracts  monthly expiries only (last settlement of each calendar month), which exist in both
             symbols before and after, so the measurement is identical in both regimes.
             Options = Stage 1 trades (OTM, |delta| 0.02-0.50, >= 10 contracts, price >= 0.5,
             no carry forwards), entered at the close h trading days before settlement.
  Outcomes   per symbol x monthly expiry (one number per expiry):
    P1 price of a month of volatility: hedged buyer gross return per Rs 100 (Stage 1 hbuy_gross),
       h = 21, |delta| 0.30-0.50, calls and puts averaged.
    P2 price of expiry-week options: same, h = 3, |delta| 0.15-0.50.
       (Hypothesis behind it: BANKNIFTY's short-dated demand now has only the monthly's last
       week to go to.)
    P3 crash pricing (skew): IV of the 10-delta put minus IV of the 10-delta call at h = 21,
       each interpolated linearly in |delta| within its side (needs bracketing points).
    P4 implied vs realised variance: ln(ATM IV^2 * T / sum of squared daily log returns from the
       entry close to the settlement close), ATM IV interpolated at log-moneyness 0, h = 21.
       (Calendar T vs trading-day RV shifts the level equally in both symbols; cancels.)
  Gap        G = BANKNIFTY - NIFTY, paired by settlement month.
  Windows    by settlement month: pre = Dec-2022 .. Sep-2024 (22 months; ends before the
             1-Oct-2024 announcement), post = Dec-2024 .. Sep-2026 (22 months; every entry is on
             or after 20-Nov-2024). Oct and Nov 2024 are the transition, excluded.
  Effect     delta = mean(G post) - mean(G pre). Two-sided; no direction is assumed.
  Inference  (a) OLS of G on a post dummy with Newey-West SE (3 lags), Holm across P1-P4;
             (b) placebo: the same 22/22-month comparison at every fake break month from
             Jan-2011 to Dec-2022 (all windows end before Oct-2024); placebo p = share of
             |placebo delta| >= |delta|. Overlapping placebos are not independent - (b) is a
             reality check on how much the gap drifts on its own, not an exact p-value.
  Verdict    "changed" only if Holm p < 0.05 AND placebo p < 0.10 AND delta has the same sign in
             both halves of the post window (settlements Dec-24..Jun-25 and Jul-25..Sep-26; the
             split also brackets SEBI's 3-Jul-2025 Jane Street interim order on BANKNIFTY
             expiry-day trading).
  Reported, not part of the verdict: each symbol's own pre/post change (which index moved);
  first stage (option premium turnover: BANKNIFTY vs NIFTY, and the share BANKNIFTY trades in
  contracts <= 5 sessions from expiry); Bates (model-based, its fit inputs changed for
  BANKNIFTY): ln(Q variance / P variance) and the jump share of Q variance for monthly targets
  15-25 sessions out; other horizons and wings.
  Disclosure: Stage 1's era table and Stage 5's era results already showed each symbol's own
  Nov-2024-onward averages. The BANKNIFTY-minus-NIFTY change tested here was never computed.
Writes data/processed/sebi2024/ (_report.txt, gaps.csv, placebo.csv)
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

S1 = os.path.join(PROJECT_ROOT, "data", "processed", "stage1", "trades.parquet")
S4 = os.path.join(PROJECT_ROOT, "data", "processed", "stage4")
PANEL = os.path.join(PROJECT_ROOT, "data", "processed", "contracts", "panel")
OUT = os.path.join(PROJECT_ROOT, "data", "processed", "sebi2024")

REFORM = pd.Timestamp("2024-11-20")
PRE = (pd.Period("2022-12", "M"), pd.Period("2024-09", "M"))
POST = (pd.Period("2024-12", "M"), pd.Period("2026-09", "M"))
POST_SPLIT = pd.Period("2025-07", "M")
N_WIN = 22
PLACEBO = (pd.Period("2011-01", "M"), pd.Period("2022-12", "M"))
NW_LAGS = 3
OUTCOMES = {"P1": "hedged buyer return, 21 sessions, |delta| .30-.50 (Rs/Rs100)",
            "P2": "hedged buyer return, 3 sessions, |delta| .15-.50 (Rs/Rs100)",
            "P3": "skew: 10d put IV - 10d call IV, 21 sessions (vol pts)",
            "P4": "ln(implied var / realised var), 21 sessions"}
LINES: list[str] = []


def out(s=""):
    print(s)
    LINES.append(str(s))


# ---------------------------------------------------------------- outcomes

def monthly_trades() -> pd.DataFrame:
    t = pd.read_parquet(S1)
    t = t[t["settled"].notna()].copy()
    mon = t.groupby(["symbol", t["settled"].dt.to_period("M")])["settled"].transform("max")
    t = t[t["settled"] == mon].copy()
    t["month"] = t["settled"].dt.to_period("M")
    t["ad"] = t["delta"].abs()
    return t


def interp_at(x, y, x0):
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    if len(x) < 2 or x0 < x[0] or x0 > x[-1]:
        return np.nan
    return float(np.interp(x0, x, y))


def per_expiry(t: pd.DataFrame, r: dict) -> pd.DataFrame:
    rows = []
    for (sym, m), g in t.groupby(["symbol", "month"]):
        row = dict(symbol=sym, month=m)
        a = g[(g["tdte"] == 21) & (g["ad"] > 0.30)]
        row["P1"] = a["hbuy_gross"].mean() if len(a) else np.nan
        b = g[(g["tdte"] == 3) & (g["ad"] > 0.15)]
        row["P2"] = b["hbuy_gross"].mean() if len(b) else np.nan
        c = g[g["tdte"] == 21]
        if len(c):
            pe, ce = c[c["side"] == "PE"], c[c["side"] == "CE"]
            row["P3"] = (interp_at(pe["ad"], pe["iv"], 0.10) - interp_at(ce["ad"], ce["iv"], 0.10)) * 100
            atm = interp_at(c["log_moneyness"], c["iv"], 0.0)
            d0, st = c["date"].iloc[0], c["settled"].iloc[0]
            rs = r[sym]
            rv = float((rs[(rs.index > d0) & (rs.index <= st)] ** 2).sum())
            row["P4"] = np.log(atm ** 2 * c["T"].iloc[0] / rv) if np.isfinite(atm) and rv > 0 else np.nan
            row["entry21"] = d0
        else:
            row["P3"] = row["P4"] = np.nan
        row["entry3"] = b["date"].min() if len(b) else pd.NaT
        rows.append(row)
    return pd.DataFrame(rows)


def gaps(pe: pd.DataFrame, cols) -> pd.DataFrame:
    w = pe.pivot(index="month", columns="symbol", values=cols)
    g = pd.DataFrame({c: w[(c, "BANKNIFTY")] - w[(c, "NIFTY")] for c in cols})
    return g.sort_index()


# ---------------------------------------------------------------- statistics

def nw_did(y: np.ndarray, post: np.ndarray, lags=NW_LAGS):
    X = np.column_stack([np.ones(len(y)), post.astype(float)])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    se = np.sqrt((XtXi @ S @ XtXi)[1, 1])
    from scipy import stats
    tstat = b[1] / se
    return b[1], se, 2 * stats.t.sf(abs(tstat), len(y) - 2)


def window(g: pd.Series, a, b):
    return g[(g.index >= a) & (g.index <= b)].dropna()


def did(g: pd.Series, pre, post):
    x0, x1 = window(g, *pre), window(g, *post)
    if len(x0) < 15 or len(x1) < 15:
        return None
    y = np.r_[x0.to_numpy(), x1.to_numpy()]
    d = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
    eff, se, p = nw_did(y, d)
    return dict(delta=eff, se=se, p=p, n_pre=len(x0), n_post=len(x1), pre=x0.mean(), post=x1.mean())


def placebo(g: pd.Series) -> pd.Series:
    res = {}
    for F in pd.period_range(*PLACEBO, freq="M"):
        r = did(g, (F - N_WIN, F - 1), (F, F + N_WIN - 1))
        if r:
            res[F] = r["delta"]
    return pd.Series(res, dtype=float)


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, adj, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


# ---------------------------------------------------------------- secondary

def first_stage():
    rows = []
    for sym in ("NIFTY", "BANKNIFTY"):
        for f in glob.glob(os.path.join(PANEL, f"symbol={sym}", "expiry_year=*", "*.parquet")):
            if int(f.split("expiry_year=")[1][:4]) < 2022:
                continue
            p = pd.read_parquet(f, columns=["date", "tdte", "contracts", "lot_size", "close"])
            p = p[p["date"] >= "2022-11-01"]
            p["prem"] = p["contracts"].fillna(0) * p["lot_size"] * p["close"].fillna(0)
            p["near"] = p["tdte"] <= 5
            a = p.groupby("date").agg(prem=("prem", "sum"))
            a["near"] = p[p["near"]].groupby("date")["prem"].sum()
            a["symbol"] = sym
            rows.append(a.reset_index())
    d = pd.concat(rows).groupby(["symbol", "date"]).sum().reset_index()
    d["month"] = d["date"].dt.to_period("M")
    m = d.groupby(["symbol", "month"])[["prem", "near"]].sum().reset_index()
    w = m.pivot(index="month", columns="symbol", values=["prem", "near"])
    fs = pd.DataFrame({"log_bank_over_nifty": np.log(w[("prem", "BANKNIFTY")] / w[("prem", "NIFTY")]),
                       "bank_near_share": w[("near", "BANKNIFTY")] / w[("prem", "BANKNIFTY")],
                       "nifty_near_share": w[("near", "NIFTY")] / w[("prem", "NIFTY")]})
    out("\nFIRST STAGE (did the reform change where BANKNIFTY options trade?) monthly premium turnover")
    for name, (a, b) in (("pre", PRE), ("post", POST)):
        x = fs[(fs.index >= a) & (fs.index <= b)]
        out(f"  {name:4s} BANKNIFTY/NIFTY premium turnover x{np.exp(x['log_bank_over_nifty'].mean()):.2f};"
            f" share traded <= 5 sessions from expiry: BANKNIFTY {x['bank_near_share'].mean():.0%},"
            f" NIFTY {x['nifty_near_share'].mean():.0%}")


def bates_secondary(mon_set):
    o = pd.read_parquet(os.path.join(S4, "outcomes.parquet"))
    o = o[(o["target"] == "month") & o["n_sessions"].between(15, 25)].copy()
    o["month"] = o["expiry"].dt.to_period("M")
    o = o[[(s, e) in mon_set for s, e in zip(o["symbol"], o["expiry"])]]
    o["prem"] = np.log(o["q_var"] / o["p_cal"])
    o["jshare"] = o["q_jump"] / o["q_var"]
    pe = o.groupby(["symbol", "month"])[["prem", "jshare"]].mean().reset_index()
    g = gaps(pe, ["prem", "jshare"])
    out("\nBATES (model-based; BANKNIFTY's fit inputs lost the weeklies, so read with care)")
    for c, lab in (("prem", "ln(Q var / P var), monthly 15-25 sessions out"), ("jshare", "jump share of Q variance")):
        r = did(g[c], PRE, POST)
        if r:
            out(f"  {lab:46s} gap pre {r['pre']:+.3f} post {r['post']:+.3f} change {r['delta']:+.3f}"
                f" (NW p {r['p']:.3f}, {r['n_pre']}/{r['n_post']} months)")


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    ps = pd.read_parquet(os.path.join(S4, "pstate.parquet"))
    r = {s: g.set_index("date")["r"].sort_index() for s, g in ps.groupby("symbol")}
    t = monthly_trades()
    pe = per_expiry(t, r)
    post_rows = pe[(pe["month"] >= POST[0]) & (pe["month"] <= POST[1])]
    early = post_rows[(post_rows["entry21"] < REFORM) | (post_rows["entry3"] < REFORM)]
    assert early.empty, f"post-window entries before the reform: {early}"
    cols = list(OUTCOMES)
    g = gaps(pe, cols)
    g.to_csv(os.path.join(OUT, "gaps.csv"))

    out("SEBI Nov-2024 reform: BANKNIFTY (lost weeklies) minus NIFTY (kept them), monthly contracts only")
    out(f"pre = settlements {PRE[0]}..{PRE[1]}, post = {POST[0]}..{POST[1]}; rules in the script header\n")
    res, pls = {}, {}
    for c in cols:
        res[c] = did(g[c], PRE, POST)
        pls[c] = placebo(g[c])
    adj = holm({c: res[c]["p"] for c in cols})
    pd.DataFrame(pls).to_csv(os.path.join(OUT, "placebo.csv"))
    out("PRIMARY")
    for c in cols:
        x, pl = res[c], pls[c]
        p_pl = float(np.mean(np.abs(pl) >= abs(x["delta"]))) if len(pl) else np.nan
        x1 = window(g[c], POST[0], POST_SPLIT - 1).mean() - x["pre"]
        x2 = window(g[c], POST_SPLIT, POST[1]).mean() - x["pre"]
        same = np.sign(x1) == np.sign(x2) == np.sign(x["delta"])
        ok = adj[c] < 0.05 and p_pl < 0.10 and same
        out(f"  {c} {OUTCOMES[c]}")
        out(f"     gap pre {x['pre']:+.3f}  post {x['post']:+.3f}  change {x['delta']:+.3f} (se {x['se']:.3f})"
            f"  NW p {x['p']:.4f}  Holm {adj[c]:.4f}  placebo p {p_pl:.2f} ({len(pl)} fake dates,"
            f" |placebo| 90th pct {np.quantile(np.abs(pl), 0.9):.3f})")
        out(f"     change by post half: Dec24-Jun25 {x1:+.3f}, Jul25-Sep26 {x2:+.3f}"
            f"  -> {'CHANGED' if ok else 'no reliable change'}")
        for sym in ("BANKNIFTY", "NIFTY"):
            s = pe[pe["symbol"] == sym].set_index("month")[c]
            a, b = window(s, *PRE).mean(), window(s, *POST).mean()
            out(f"       {sym:9s} own level pre {a:+.3f} post {b:+.3f} ({b - a:+.3f})")

    out("\nSECONDARY (descriptive; not part of the verdict)")
    first_stage()
    bates_secondary(set(zip(t["symbol"], t["settled"])))
    out("\nOther horizons / wings: hedged buyer return gap change (Rs/Rs100), NW p")
    for h in (2, 5, 10, 42):
        for lo, hi, lab in ((0.30, 0.50, "ATM-ish"), (0.02, 0.15, "wings")):
            for side in ("CE", "PE"):
                x = t[(t["tdte"] == h) & (t["ad"] > lo) & (t["ad"] <= hi) & (t["side"] == side)]
                pe2 = x.groupby(["symbol", "month"])["hbuy_gross"].mean().reset_index()
                if pe2["symbol"].nunique() < 2:
                    continue
                gg = gaps(pe2, ["hbuy_gross"])["hbuy_gross"]
                rr = did(gg, PRE, POST)
                if rr:
                    out(f"  h {h:2d} {lab:7s} {side}: pre {rr['pre']:+6.1f} post {rr['post']:+6.1f}"
                        f" change {rr['delta']:+6.1f} (p {rr['p']:.3f})")
    with open(os.path.join(OUT, "_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
