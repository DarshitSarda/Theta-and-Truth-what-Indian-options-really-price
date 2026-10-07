"""Reconciliation diagnostics: where does realistic 'exact' (fractional) differ from Stage 5?

Variants of run_cycle at 1 Cr, FUT hedge, rule exact, zero costs (gross P&L, % of capital per year):
  real        as run (next-day VWAP fills for entry and hedges)
  optclose    option entry filled at the next-day CLOSE instead of VWAP (hedges still VWAP)
  allclose    entry and hedges at next-day closes (pure one-day timing lag vs Stage 5)
Also: Stage 5 gross per cycle, and the per-cycle entry slippage close - VWAP as % of premium.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
import realistic_engine as R  # noqa: E402
from realistic_engine_selftest import ZeroCM  # noqa: E402


def main():
    pos = pd.read_parquet(os.path.join(R.D5, "positions.parquet"))
    pos = pos[pos["strategy"] == "eng_straddle_bs"].sort_values(["symbol", "entry"])
    d5 = pd.read_parquet(os.path.join(R.D5, "daily.parquet"))
    d5 = d5[d5["strategy"] == "eng_straddle_bs"]
    s5 = d5.groupby("pid")[["option", "hedge"]].sum().sum(axis=1)
    C, rows = 1e7, []
    for sym in ("NIFTY", "BANKNIFTY"):
        rd = R.RealData(sym)
        for p in pos[pos["symbol"] == sym].itertuples():
            c = rd.cycle(p)
            if c is None:
                continue
            c.units *= C
            out = dict(symbol=sym, pid=p.pid, entry=p.entry, stage5=s5.get(p.pid, np.nan))
            r = R.run_cycle(c, "exact", "FUT", [ZeroCM()], True)
            if r is None:
                continue
            out["real"] = r["pnl"].sum() / C
            f = r["fill"]
            out["fill"] = f
            out["slip_pct_prem"] = float((c.opt_close[f] - c.opt_vwap[f]).sum() / c.opt_vwap[f].sum())
            out["prem"] = r["prem"] / C
            ov, fc0 = c.opt_vwap.copy(), c.fut_close.copy()
            c.opt_vwap = np.where(np.isfinite(ov), c.opt_close, np.nan)
            out["optclose"] = R.run_cycle(c, "exact", "FUT", [ZeroCM()], True)["pnl"].sum() / C
            fv = c.fut_vwap.copy()
            c.fut_vwap = np.where(np.isfinite(fv), c.fut_close, np.nan)
            out["allclose"] = R.run_cycle(c, "exact", "FUT", [ZeroCM()], True)["pnl"].sum() / C
            # Stage 5 timing: a fill "at k" priced at close k-1 == trading at the decision close
            c.opt_vwap = np.vstack([c.opt_close[:1] * np.nan, c.opt_close[:-1]])
            c.fut_vwap = np.r_[np.nan, c.fut_close[:-1]]
            out["sameclose"] = R.run_cycle(c, "exact", "FUT", [ZeroCM()], True)["pnl"].sum() / C
            # same, but options only at the decision close; hedges one close late
            c.fut_vwap = np.where(np.isfinite(fv), c.fut_close, np.nan)
            out["hedgelag"] = R.run_cycle(c, "exact", "FUT", [ZeroCM()], True)["pnl"].sum() / C
            # Stage 5 timing and Stage 5's hedge instrument: the options-implied forward, not the futures close
            fwd = c.fut_close.copy()
            for k, d in enumerate(c.dates[:-1]):
                s = rd.mk.slice(d, p.final_expiry)
                if s is not None:
                    fwd[k] = s.F
            c.fut_close = fwd
            c.fut_vwap = np.r_[np.nan, fwd[:-1]]
            out["sameclose_fwd"] = R.run_cycle(c, "exact", "FUT", [ZeroCM()], True)["pnl"].sum() / C
            out["sameclose_fwd_last"] = R.run_cycle(c, "exact", "FUT", [ZeroCM()], True, True)["pnl"].sum() / C
            c.opt_vwap, c.fut_vwap, c.fut_close = ov, fv, fc0
            out["real_last"] = R.run_cycle(c, "exact", "FUT", [ZeroCM()], True, True)["pnl"].sum() / C
            rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R.OUT, "_diag_cycles.csv"), index=False)
    L = []
    for lab, lo, hi in (("2008-2017", "2008", "2018"), ("2018-2026", "2018", "2100")):
        x = df[(df["entry"] >= lo) & (df["entry"] < hi)]
        yrs = x.groupby("symbol")["entry"].agg(lambda e: (e.max() - e.min()).days / 365.25).max()
        L.append(f"{lab}: corr(stage5, sameclose) per cycle {x['stage5'].corr(x['sameclose']):.4f};"
                 f" sameclose {x['sameclose'].sum() / yrs:+.2%}, sameclose on forward {x['sameclose_fwd'].sum() / yrs:+.2%}"
                 f" (corr {x['stage5'].corr(x['sameclose_fwd']):.4f}), + hedge into settlement"
                 f" {x['sameclose_fwd_last'].sum() / yrs:+.2%} (corr {x['stage5'].corr(x['sameclose_fwd_last']):.4f});"
                 f" real + hedge into settlement {x['real_last'].sum() / yrs:+.2%}; hedgelag {x['hedgelag'].sum() / yrs:+.2%}")
        L.append(f"{lab}: gross %/yr  stage5 {x['stage5'].sum() / yrs:+.2%}  allclose {x['allclose'].sum() / yrs:+.2%}"
                 f"  optclose {x['optclose'].sum() / yrs:+.2%}  real {x['real'].sum() / yrs:+.2%}"
                 f" | premium sold {x['prem'].sum() / yrs:.0%}/yr; entry close-VWAP median"
                 f" {x['slip_pct_prem'].median():+.2%} of premium, mean {x['slip_pct_prem'].mean():+.2%};"
                 f" fill session counts {x['fill'].value_counts().to_dict()}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(R.OUT, "_diag.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
