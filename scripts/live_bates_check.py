"""Does the live Bates view hold up on the live days themselves? (sanity check, low power)

Uses only the live snapshots, the live Bates fits (scripts/live_bates.py) and closes.
Unit of independent evidence = one settled expiry (overlapping entry days share outcomes).
  A  priced vs realised: for each fit day and fitted expiry that has settled, realised
     variance sum r^2 over (day, expiry] vs the Bates-priced variance (q_move^2) and the P
     model's (p_move^2); plus how often |ln(S_T / F)| fell inside the priced +/-1 sd move.
  B  rich/cheap: the Stage 5 core test on live data with its one-session skip - signal S_P on
     day t, option bought at the next session's traded mid, held to settlement, delta-hedged
     each live session at the signal-day IV (Black-76, live forwards; settlement = close).
     Within (day, expiry) cross-sections of >= 12 options, cell fixed effects (side x delta
     bucket): rank IC of S_P and of the naive S_N (trailing-RV vol minus IV), the partial
     rank slope of S_P beyond S_N, and the gross long-short (top vs bottom third of S_P
     residualised on S_N) per Rs 100 premium. Averaged per expiry, t across expiries.
Stage 5 holdout reference (bhavcopy, 2018-2026): partial slope +0.19, IC(S_P) +0.16,
long-short +3.8 per Rs 100. These live days are also inside that holdout, so this is a
check that the live pipeline sees the same thing - not new evidence.
Writes data/processed/live_bates/_check.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src import contracts as ct  # noqa: E402
from src import live  # noqa: E402
from live_bates import LiveP, OUT  # noqa: E402
from stage5_richcheap import demean, rk  # noqa: E402

MIN_N = 12
LINES: list[str] = []


def out(s=""):
    print(s)
    LINES.append(str(s))


def t_across(x):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return f"{x.mean():+.3f} (n {len(x)})"
    return f"{x.mean():+.3f} (t {x.mean() / (x.std(ddof=1) / np.sqrt(len(x))):+.2f}, {len(x)} expiries, {np.mean(x > 0):.0%} positive)"


def part_a(sym, info, y):
    ev = pd.read_csv(os.path.join(OUT, "expiry_view.csv"), parse_dates=["date", "expiry"])
    ev = ev[(ev["symbol"] == sym) & (ev["expiry"] <= info["date"].max()) & ev["expiry"].isin(y.index)]
    r2 = np.log(y).diff() ** 2
    cum = r2.cumsum()
    ev["rv"] = [float(cum[e] - cum[d]) for d, e in zip(ev["date"], ev["expiry"])]
    ev["x"] = np.log(y.reindex(ev["expiry"]).to_numpy() / ev["F"].to_numpy())
    ev["rq"] = ev["rv"] / ev["q_move"] ** 2
    ev["rp"] = ev["rv"] / ev["p_move"] ** 2
    ev["inside"] = np.abs(ev["x"]) <= ev["q_move"]
    g = ev.groupby("expiry").agg(rq=("rq", "mean"), rp=("rp", "mean"), inside=("inside", "mean"), n=("rq", "size"))
    out(f"  A priced vs realised: {len(ev)} (day, expiry) rows over {len(g)} settled expiries")
    out(f"    realised / Bates-priced variance: median expiry {g['rq'].median():.2f}, mean {g['rq'].mean():.2f},"
        f" expiries with realised < priced {np.mean(g['rq'] < 1):.0%}")
    out(f"    realised / P-model variance:      median expiry {g['rp'].median():.2f}, mean {g['rp'].mean():.2f},"
        f" expiries with realised < P {np.mean(g['rp'] < 1):.0%}")
    out(f"    final move inside the priced +/-1 sd: {ev['inside'].mean():.0%} of rows (a correct Q would give ~68%,"
        f" a premium-carrying one more)")


def outcomes(sym, lp, info, y):
    """Skip-day hedged buyer return for every option in the saved rich/cheap files."""
    fw = info[info["session_ok"] & (info["fwd_source"] != "none")][["date", "expiry", "forward"]]
    paths = {e: g.set_index("date")["forward"].sort_index() for e, g in fw.groupby("expiry")}
    sessions = sorted(lp["date"].unique())
    nxt = {d: sessions[i + 1] for i, d in enumerate(sessions[:-1])}
    mids = lp[lp["traded"]].set_index(["date", "expiry", "strike", "side"])["mid"]
    rates = ct.load_repo_rates(live.REPO_CSV)
    P = LiveP(sym)
    rows = []
    for f in sorted(os.listdir(os.path.join(OUT, "richcheap"))):
        if not f.startswith(sym + "_"):
            continue
        d = pd.Timestamp(f[len(sym) + 1:-4])
        if d not in nxt:
            continue
        rc = pd.read_csv(os.path.join(OUT, "richcheap", f), parse_dates=["expiry"])
        d1 = nxt[d]
        for e, g in rc.groupby("expiry"):
            if e > sessions[-1] or e not in y.index or d1 >= e or e not in paths:
                continue
            ST = float(y[e])
            path = paths[e]
            path = path[(path.index >= d1) & (path.index < e)]
            if path.empty or path.index[0] != d1:
                continue
            F = np.r_[path.to_numpy(float), ST]
            T = np.array([(e - k).days / 365 for k in path.index])
            DF = np.exp(-np.array([ct.rate_on(rates, k.date()) for k in path.index]) * T)
            n_s = live.trading_days_between(d, e)
            T0 = (e - d).days / 365
            sig_n = np.sqrt(P.naive(d, n_s) / T0) if T0 > 0 else np.nan
            for r in g.itertuples(index=False):
                m1 = mids.get((d1, e, r.strike, r.side))
                if m1 is None or not np.isfinite(m1) or m1 <= 0 or not np.isfinite(r.iv):
                    continue
                ic = r.side == "CE"
                dl, _, _, _ = ct.b76_greeks(F[:-1], r.strike, T, r.iv, DF, 0.0, ic)
                hedge = float(-np.sum(dl * np.diff(F)))
                pay = max(ST - r.strike, 0) if ic else max(r.strike - ST, 0)
                rows.append(dict(date=d, expiry=e, strike=r.strike, side=r.side, bucket=r.bucket, s_p=r.s_p,
                                 s_n=sig_n - r.iv, y=(pay - m1 + hedge) / m1 * 100, yu=(pay - m1) / m1 * 100))
    return pd.DataFrame(rows)


def part_b(sym, o):
    o["cell"] = o["side"] + o["bucket"].astype(str)
    cs = []
    for (d, e), g in o.groupby(["date", "expiry"]):
        if len(g) < MIN_N:
            continue
        c = g["cell"].to_numpy()
        ry, rP, rN = (demean(rk(g[k].to_numpy()), c) for k in ("y", "s_p", "s_n"))
        if ry @ ry == 0 or rP @ rP == 0:
            continue
        X = np.column_stack([rP, rN])
        b = np.linalg.lstsq(X, ry, rcond=None)[0] if np.linalg.matrix_rank(X) == 2 else [np.nan, np.nan]
        xP, xN = demean(g["s_p"].to_numpy(), c), demean(g["s_n"].to_numpy(), c)
        res = xP - (xP @ xN / (xN @ xN)) * xN if xN @ xN > 0 else xP
        q = pd.Series(res).rank(pct=True).to_numpy()
        yy = g["y"].to_numpy()
        cs.append(dict(date=d, expiry=e, n=len(g), ic_p=np.corrcoef(ry, rP)[0, 1],
                       ic_n=np.corrcoef(ry, rN)[0, 1] if rN @ rN > 0 else np.nan, b_p=b[0],
                       ls=yy[q > 2 / 3].mean() - yy[q <= 1 / 3].mean()))
    cs = pd.DataFrame(cs)
    if cs.empty:
        out("  B rich/cheap: no cross-sections with settled outcomes")
        return
    g = cs.groupby("expiry")[["ic_p", "ic_n", "b_p", "ls"]].mean()
    out(f"  B rich/cheap: {len(o):,} option outcomes, {len(cs)} day x expiry cross-sections, {len(g)} settled expiries")
    out(f"    IC naive S_N          {t_across(g['ic_n'])}")
    out(f"    IC Bates S_P          {t_across(g['ic_p'])}   (Stage 5 holdout +0.16)")
    out(f"    Bates beyond naive b  {t_across(g['b_p'])}   (Stage 5 holdout +0.19)")
    out(f"    long-short Rs/Rs100   {t_across(g['ls'])}   (Stage 5 holdout +3.8; costs ~6 per leg)")


def main():
    for sym in ("NIFTY", "BANKNIFTY"):
        lp, info = live.live_panel(sym)
        y = live.yahoo_close(sym)
        out("=" * 100 + f"\n{sym}: live days {lp['date'].min().date()} .. {lp['date'].max().date()}\n" + "=" * 100)
        part_a(sym, info, y)
        part_b(sym, outcomes(sym, lp, info, y))
    with open(os.path.join(OUT, "_check.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
