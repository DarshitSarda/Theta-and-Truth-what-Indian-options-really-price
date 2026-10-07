"""EXPLORATORY follow-up to realistic_engine.py (decided 2026-10-07 AFTER seeing its results; user chose
"hedge near the close" + "allow hedges on the settlement day"). Not a pre-declared test.

Same simulator (realistic_engine.run_cycle), data, lots, costs, accounts, hedge rules. Changes:
  nearclose   decisions at the close of day t are filled at the close of day t (stands for trading
              ~15:15-15:25 at near-close prices); entry at the settlement close (as Stage 5);
              hedging continues through the session before settlement. Fill = close +/- the
              usual half-spread / 0.5 bp futures spread, plus extra slippage for the minutes
              before the close: S0 none, S1 futures 2 bp + options 0.5% of price, S2 5 bp + 1%.
  nextday+    the frozen next-day-VWAP rules, amended to allow a hedge order on the settlement
              session (filled at that day's VWAP; futures then settle at the closing index).
Selection fixed: best 2008-2017 excess Sharpe among implementable rules, or "don't trade" if no
rule has a positive one; judged on 2018-2026. Every rule is reported.
Writes data/processed/realistic/_nearclose.txt (+ _nearclose_summary.csv)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src import backtest as B  # noqa: E402
import realistic_engine as R  # noqa: E402

SLIPS = {"S0": (0.0, 0.0), "S1": (0.0002, 0.005), "S2": (0.0005, 0.01)}


class SlipCM:
    def __init__(self, base: R.CostModel, fut_bp: float, opt_frac: float):
        self.base, self.fut_bp, self.opt_frac = base, fut_bp, opt_frac

    def option(self, d, price, units):
        return self.base.option(d, price, units) + abs(units) * price * self.opt_frac

    def future(self, d, price, units):
        return self.base.future(d, price, units) + abs(units) * price * self.fut_bp

    def expiry(self, d, value, units, spot):
        return self.base.expiry(d, value, units, spot)


def nearclose(c: R.Cycle) -> R.Cycle:
    """A fill 'at session k' priced at close k-1 == trading at the decision close."""
    oc = c.opt_close.copy()
    oc_prev = np.vstack([np.full((1, 2), np.nan), oc[:-1]])
    return R.Cycle(c.dates, c.K, c.lot, c.fut_lot, c.units, oc, oc_prev, c.delta, c.fut_close,
                   np.r_[np.nan, c.fut_close[:-1]], c.DF)


def main():
    t0 = time.time()
    pos = pd.read_parquet(os.path.join(R.D5, "positions.parquet"))
    pos = pos[pos["strategy"] == "eng_straddle_bs"].sort_values(["symbol", "entry"])
    st = B.spread_table(os.path.join(R.D5, "_spread_table.parquet"))
    base = [R.CostModel(B.Costs(st), False), R.CostModel(B.Costs(st, stress=True), True)]
    variants = {f"nearclose {s}": (True, [SlipCM(b, *v) for b in base]) for s, v in SLIPS.items()}
    variants["nextday+ S0"] = (False, base)
    rows, cyc, sessions = [], [], None
    for sym in ("NIFTY", "BANKNIFTY"):
        rd = R.RealData(sym)
        sessions = rd.mk.sessions if sessions is None else sessions.union(rd.mk.sessions)
        built = []
        for p in pos[pos["symbol"] == sym].itertuples():
            c = rd.cycle(p)
            if c is not None:
                dN, fe = c.dates[-1], p.final_expiry
                c.fut_vwap[-1] = rd.fvw.get((dN, fe), np.nan)
                c.opt_vwap[-1] = [rd.ovw.get((dN, fe, c.K, sd), np.nan) for sd in ("CE", "PE")]
            built.append((p, c))
        for var, (nc, cms) in variants.items():
            for cap, C in R.CAPITALS.items():
                for instr in R.INSTR:
                    for rule in R.RULES:
                        if instr == "SYN" and rule == "none":
                            continue
                        for p, c in built:
                            r = None
                            if c is not None:
                                cc = nearclose(c) if nc else c
                                cc = R.Cycle(cc.dates, cc.K, cc.lot, cc.fut_lot, c.units * C, cc.opt_close,
                                             cc.opt_vwap, cc.delta, cc.fut_close, cc.fut_vwap, cc.DF)
                                r = R.run_cycle(cc, rule, instr, cms, rule == "exact", last_hedge=True)
                            cyc.append(dict(var=var, cap=cap, instr=instr, rule=rule, settled=p.settled,
                                            traded=r is not None, lots=r["lots"] if r else 0.0,
                                            trades=r["trades"].sum() if r else 0.0,
                                            margin_breach=float((r["margin"] > C).mean()) if r else np.nan))
                            if r is not None:
                                rows.append(pd.DataFrame({"date": c.dates, "var": var, "cap": cap, "instr": instr,
                                                          "rule": rule, "pnl": r["pnl"] / C,
                                                          "cost": r["cost"][0] / C, "cost_s": r["cost"][1] / C}))
            print(f"  {sym} {var} done ({time.time() - t0:.0f}s)", flush=True)
    daily = pd.concat(rows, ignore_index=True)
    cyc = pd.DataFrame(cyc)
    intr = R.interest_series(sessions)
    periods = {"2008-2017": ("2008-01-01", R.SELECT_END), "2018-2026": (R.SELECT_END, "2100-01-01")}
    res, L, allrows = {}, [__doc__.strip(), f"\nrun time {time.time() - t0:.0f}s"], []
    for var in variants:
        for lab, (lo, hi) in periods.items():
            s, ir = R.summarize(daily[daily["var"] == var], cyc[cyc["var"] == var], intr,
                                pd.Timestamp(lo), pd.Timestamp(hi))
            s.insert(0, "period", lab)
            s.insert(0, "var", var)
            res[(var, lab)] = s
            allrows.append(s)
    summ = pd.concat(allrows, ignore_index=True)
    summ.to_csv(os.path.join(R.OUT, "_nearclose_summary.csv"), index=False)
    cols = ["cap", "instr", "rule", "excess_yr", "stress_yr", "gross_yr", "cost_yr", "sharpe", "maxdd",
            "worst_month", "traded", "trades_per_cycle", "lots"]
    for var in variants:
        for lab in periods:
            s = res[(var, lab)][cols].copy()
            for c in ("excess_yr", "stress_yr", "gross_yr", "cost_yr", "maxdd", "worst_month", "traded"):
                s[c] = (s[c] * 100).round(2)
            s["cap"] = pd.Categorical(s["cap"], list(R.CAPITALS))
            L.append("\n" + "=" * 100 + f"\n{var} | {lab}  (% of capital per year; excess = without collateral interest)\n"
                     + "=" * 100)
            L.append(s.sort_values(["cap", "instr", "rule"]).round(2).to_string(index=False))
    L.append("\n" + "=" * 100 + "\nSELECTION (fixed: 'don't trade' unless the best 2008-17 Sharpe is > 0) -> 2018-2026\n"
             + "=" * 100)
    for var in variants:
        for cap in R.CAPITALS:
            sel = res[(var, "2008-2017")]
            cand = sel[(sel["cap"] == cap) & (sel["rule"] != "exact")]
            best = cand.loc[cand["sharpe"].idxmax()]
            jud = res[(var, "2018-2026")]
            j = jud[(jud["cap"] == cap) & (jud["instr"] == best["instr"]) & (jud["rule"] == best["rule"])].iloc[0]
            pick = f"{best['instr']} {best['rule']}" if best["sharpe"] > 0 else "DON'T TRADE"
            L.append(f"  {var:14s} {cap:4s}: best 2008-17 {best['instr']} {best['rule']} Sharpe {best['sharpe']:+.2f}"
                     f" excess {best['excess_yr']:+.2%} -> {pick}; that rule 2018-26 excess {j['excess_yr']:+.2%},"
                     f" stress {j['stress_yr']:+.2%}, Sharpe {j['sharpe']:+.2f}, maxDD {j['maxdd']:.1%},"
                     f" traded {j['traded']:.0%}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(R.OUT, "_nearclose.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
