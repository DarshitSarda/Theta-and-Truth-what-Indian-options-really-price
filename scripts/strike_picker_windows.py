"""Window-length sensitivity for strike_picker.py (asked by the user after the first run).

Rules fixed before running: same dataset, models, books and tests as strike_picker.py; only
the rolling training window changes: 1, 2, 3 and 5 years (5 = the original primary). Every
window is reported; none is chosen afterwards. "The window matters" only if a model beats
selling all (M0) under the original verdict rule for most windows, not one.
Also: the per-test-year gain of each model over M0 in the PRIMARY book, to show how much it
swings year to year.
Appends to data/processed/strike_picker/_windows.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
import strike_picker as sp  # noqa: E402

WINDOWS = [1, 2, 3, 5]
LINES: list[str] = []


def out(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def walk(a: pd.DataFrame, years: int) -> pd.DataFrame:
    preds = []
    for Y in sp.TEST_YEARS:
        y0 = pd.Timestamp(f"{Y}-01-01")
        test = a[(a["date"] >= y0) & (a["date"] < pd.Timestamp(f"{Y + 1}-01-01"))]
        train = a[(a["settled"] >= pd.Timestamp(f"{Y - years}-01-01")) & (a["settled"] < y0)]
        if test.empty or len(train) < 1000:
            continue
        assert train["settled"].max() < y0 <= test["date"].min()
        p = sp.fit_predict(train, test)
        preds.append(pd.DataFrame({"idx": test.index, "win": f"{years}y", "M4": p["M4"], "M5": p["M5"]}))
    return pd.concat(preds, ignore_index=True)


def main():
    a = pd.read_parquet(os.path.join(sp.OUT, "dataset.parquet"))
    test = a[a["date"] >= pd.Timestamp(f"{sp.TEST_YEARS[0]}-01-01")]
    univ = {"PRIMARY": test[test["monthly"] & (test["tdte"] == 21)], "SECONDARY": test[test["tdte"] == 5]}
    summary, yearly = [], {}
    for w in WINDOWS:
        pr = walk(a, w)
        for book, u in univ.items():
            bk = sp.books(a, pr, u, f"{w}y")
            res = {}
            for m in ("M4", "M5"):
                d = bk[m] - bk["M0"]
                mean, t, p = sp.nw(d)
                ds = (bk[m + "_stress"] - bk["M0_stress"]).mean()
                dk = (bk[m + "_skip"] - bk["M0_skip"]).mean()
                h1, h2 = d[d.index < sp.HALF].mean(), d[d.index >= sp.HALF].mean()
                res[m] = p
                summary.append(dict(window=f"{w}y", book=book, model=sp.NAMES[m], gain_yr=12 * mean, t=t, p=p,
                                    stress_yr=12 * ds, skip_yr=12 * dk, half1=12 * h1, half2=12 * h2,
                                    ret_yr=12 * bk[m].mean(),
                                    sharpe=bk[m].mean() / bk[m].std() * np.sqrt(12)))
                if book == "PRIMARY":
                    yearly[(f"{w}y", sp.NAMES[m])] = d.groupby(d.index.year).sum()
            adj = sp.holm({"M2": 1.0, "M3": 1.0, **res})
            for s in summary[-2:]:
                k = "M4" if s["model"] == "ridge" else "M5"
                s["holm_p"] = adj[k]
        out(f"  window {w}y done")
    s = pd.DataFrame(summary)
    out("\nGain over selling all (M0), %/yr, by training window (Holm over M2-M5 as in the main study)")
    for book in ("PRIMARY", "SECONDARY"):
        x = s[s["book"] == book].copy()
        for c in ("gain_yr", "stress_yr", "skip_yr", "half1", "half2", "ret_yr"):
            x[c] = (x[c] * 100).round(2)
        out(f"\n{book} ({'monthly, 21 sessions' if book == 'PRIMARY' else 'every expiry, 5 sessions'});"
            f" M0 sell all = reference")
        out(x[["window", "model", "ret_yr", "sharpe", "gain_yr", "t", "holm_p", "stress_yr", "skip_yr", "half1",
               "half2"]].round(3).to_string(index=False))
    y = pd.DataFrame(yearly)
    out("\nPRIMARY: gain over selling all by test year (% of capital), each window x model")
    out((y * 100).round(2).to_string())
    out(f"\nshare of test years with a positive gain: "
        + ", ".join(f"{k[0]} {k[1]} {np.mean(v > 0):.0%}" for k, v in yearly.items()))
    with open(os.path.join(sp.OUT, "_windows.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
