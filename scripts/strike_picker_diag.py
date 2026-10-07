"""Descriptive diagnostics for strike_picker.py (not tests; nothing here changes the verdict).
  1  reconciliation: M1 (ATM pair) book vs the Stage 5 engine over the same years
  2  what each model chose in the PRIMARY rolling book (side x delta bucket mix)
  3  where the money is per unit of crash risk: mean seller r by side x delta bucket and horizon
Appends to data/processed/strike_picker/_report.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
import strike_picker as sp  # noqa: E402

D5 = os.path.join(PROJECT_ROOT, "data", "processed", "stage5")


def main():
    a = pd.read_parquet(os.path.join(sp.OUT, "dataset.parquet"))
    pr = pd.read_parquet(os.path.join(sp.OUT, "preds.parquet"))
    bk = pd.read_parquet(os.path.join(sp.OUT, "books.parquet"))
    L = ["\n" + "=" * 100, "DIAGNOSTICS (descriptive, after the verdict)", "=" * 100]

    b = bk[(bk["win"] == "roll") & (bk["book"] == "PRIMARY")]
    day = pd.read_parquet(os.path.join(D5, "daily.parquet"))
    e = day[day["strategy"] == "eng_straddle_bs"]
    L.append("1  reconciliation (book units, %/yr; sizing differs: sum of per-leg worst losses here):")
    for a0, a1 in (("2013-01-01", "2018-01-01"), ("2018-01-01", "2100-01-01")):
        m = b[(b.index >= pd.Period(a0[:7], "M")) & (b.index < pd.Period(a1[:7], "M"))]
        x = e[(e["date"] >= a0) & (e["date"] < a1)]
        yrs = (x["date"].max() - x["date"].min()).days / 365.25
        eng = (x["option"] + x["hedge"] - x["cost0"]).sum() / yrs
        L.append(f"   {a0[:4]}..{min(int(a1[:4]) - 1, 2026)}: picker M1 ATM pair {12 * m['M1'].mean():+.2%}/yr,"
                 f" M0 sell all {12 * m['M0'].mean():+.2%}/yr | Stage 5 engine {eng:+.2%}/yr")

    test = a[a["date"] >= "2013-01-01"]
    prim = test[test["monthly"] & (test["tdte"] == 21)]
    p = pr[pr["win"] == "roll"].set_index("idx")
    u = prim.join(p[["M4", "M5"]], how="inner")
    u["M2"], u["M3"] = -u["S_P"], -u["S_N"]
    u["cell"] = u["side"] + " " + u["dbucket"].astype(str)
    picks = {m: [] for m in ("M0", "M2", "M3", "M4", "M5")}
    for _, g in u.groupby(["symbol", "date", "final_expiry"]):
        if len(g) < sp.MIN_UNIV:
            continue
        k = int(np.ceil(len(g) / 3))
        picks["M0"].append(g["cell"])
        for m in ("M2", "M3", "M4", "M5"):
            picks[m].append(g.loc[g[m].nlargest(k).index, "cell"])
    mix = pd.DataFrame({m: pd.concat(v).value_counts(normalize=True) for m, v in picks.items()}).fillna(0)
    L.append("\n2  PRIMARY rolling: share of chosen options by side x |delta| bucket")
    L.append((mix.sort_index() * 100).round(0).astype(int).to_string())

    L.append("\n3  mean seller r per unit of crash risk (x 0.25 = book return per cycle), test years 2013+;"
             " per-expiry means averaged (one expiry = one observation)")
    t = test.copy()
    t["cell"] = t["side"] + " " + t["dbucket"].astype(str)
    g = t.groupby(["tdte", "cell", "symbol", "settled"])["r"].mean().groupby(["tdte", "cell"]).agg(["mean", "count"])
    tab = g["mean"].unstack("tdte")
    L.append((tab * 100).round(2).to_string())
    L.append("   (units: % of the option's stress loss earned per trade)")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(sp.OUT, "_report.txt"), "a", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
