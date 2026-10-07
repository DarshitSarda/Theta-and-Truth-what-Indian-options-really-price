"""Stage 3 follow-up: continue every fit that stopped on its evaluation budget (status 0).

Each such fit is re-run from its own end point with a 3,000-evaluation budget, on the same
session's options only (no new information). Fit rows and per-option residuals are updated
in place; the originals are copied to data/processed/stage3/_backup_before_repolish/ first.
The hidden-strike and next-day columns are left as computed in the main run.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import calibrate as cb  # noqa: E402
from scripts.stage3_calibrate import at_bounds  # noqa: E402

BASE = os.path.join(PROJECT_ROOT, "data", "processed", "stage3")
BACKUP = os.path.join(BASE, "_backup_before_repolish")
BUDGET = 3000


def main():
    if not os.path.exists(BACKUP):
        for sub in ("fits", "resid"):
            shutil.copytree(os.path.join(BASE, sub), os.path.join(BACKUP, sub))
        print(f"backup written to {BACKUP}")
    panels = {}
    total = changed = 0
    for path in sorted(glob.glob(os.path.join(BASE, "fits", "*.parquet"))):
        f = pd.read_parquet(path)
        if f.empty or "status" not in f:
            continue
        todo = f.index[(f["status"] == 0) & f["p_v0"].notna()]
        if len(todo) == 0:
            continue
        rpath = os.path.join(BASE, "resid", os.path.basename(path))
        r = pd.read_parquet(rpath)
        for i in todo:
            row = f.loc[i]
            sym, d, m = row["symbol"], row["date"], row["model"]
            if sym not in panels:
                p = cb.load_symbol(sym)
                panels[sym] = (p, cb.event_sessions(cb.load_events(), pd.DatetimeIndex(p["date"].unique())))
            p, ev = panels[sym]
            ch = cb.build_chain(sym, d, p[p["date"] == d], ev)
            x0 = np.array([row[f"p_{k}"] for k in cb.MODELS[m]], float)
            res = cb._lsq(m, ch, x0, BUDGET)
            total += 1
            # restarting moves parameters sitting on a bound 1e-6 of the span inside it, so a
            # converged continuation may end a hair above the original objective
            if res.cost > row["cost"] * (1 + 1e-6) or (res.cost > row["cost"] and res.status <= 0):
                continue
            changed += 1
            err = cb.iv_errors(m, res.x, ch)
            rv = cb.residuals(m, res.x, ch)
            upd = dict(rmse_vega=float(np.sqrt(np.mean(rv ** 2))), rmse_iv=float(np.sqrt(np.nanmean(err ** 2))),
                       max_abs_iv=float(np.nanmax(np.abs(err))), iv_nan=int(np.isnan(err).sum()), cost=float(res.cost),
                       nfev=int(row["nfev"] + res.nfev), status=int(res.status), at_bounds=at_bounds(m, res.x),
                       **{f"p_{k}": float(v) for k, v in zip(cb.MODELS[m], res.x)})
            for k, v in upd.items():
                f.at[i, k] = v
            mask = ((r["date"] == d) & (r["model"] == m)).to_numpy()
            r.loc[mask, "err_iv"] = err
            r.loc[mask, "err_vega"] = rv
        f.to_parquet(path)
        r.to_parquet(rpath)
        print(f"{os.path.basename(path)}: {len(todo)} re-polished", flush=True)
    print(f"done: {total} fits continued, {changed} updated")


if __name__ == "__main__":
    main()
