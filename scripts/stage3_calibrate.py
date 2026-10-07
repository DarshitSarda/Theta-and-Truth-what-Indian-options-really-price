"""Stage 3: daily risk-neutral calibration, 2008-now, NIFTY and BANKNIFTY.

For every symbol and session: fit Heston, Merton, Bates, and (when identified) Bates with
scheduled-event jumps to that session's out-of-the-money options (src/calibrate.py).

Out-of-sample checks, both strictly forward/hidden:
  hidden strikes  every 5th session of a chunk: refit with every 4th option of each expiry
                  hidden, then score the hidden options
  next day        yesterday's parameters with only v0 re-estimated on today's options

Work is split in (symbol, year) chunks run in parallel; inside a chunk sessions run in
date order and each fit is warm-started from the previous session's fit (past only).
Outputs (resumable, one file per chunk):
  data/processed/stage3/fits/{SYMBOL}_{YEAR}.parquet    one row per session x model
  data/processed/stage3/resid/{SYMBOL}_{YEAR}.parquet   per-option IV errors of every fit

  python scripts/stage3_calibrate.py --pilot        (4 windows of 15 sessions per symbol)
  python scripts/stage3_calibrate.py                 (everything)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import calibrate as cb  # noqa: E402
from src.contracts import SYMBOLS  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "data", "processed", "stage3")
HOLDOUT_EVERY, HIDE_EVERY = 5, 4
PILOT_WINDOWS = [("2010-06-01", 15), ("2016-03-01", 15), ("2020-03-02", 15), ("2025-02-17", 15)]


def hide_masks(ch: cb.Chain) -> list[np.ndarray]:
    """True = hidden: every 4th option of each expiry in strike order (offset 1)."""
    return [(np.arange(len(e.K)) % HIDE_EVERY) == 1 for e in ch.expiries]


def at_bounds(model: str, x: np.ndarray) -> str:
    lb, ub = cb.bounds(model)
    span = ub - lb
    hit = [n for n, v, l, u, s in zip(cb.MODELS[model], x, lb, ub, span) if v <= l + 1e-4 * s or v >= u - 1e-4 * s]
    return ",".join(hit)


def run_chunk(symbol: str, sessions: list[pd.Timestamp], tag: str, panel: pd.DataFrame | None = None) -> None:
    fits_path = os.path.join(OUT, "fits", f"{symbol}_{tag}.parquet")
    if os.path.exists(fits_path):
        return
    t0 = time.time()
    p = panel if panel is not None else cb.load_symbol(symbol)
    ev = cb.event_sessions(cb.load_events(), pd.DatetimeIndex(p["date"].unique()))
    by_day = {d: g for d, g in p[p["date"].isin(sessions)].groupby("date")}
    rows, resid = [], []
    prev, prev_date = None, None
    for i, d in enumerate(sessions):
        day = by_day.get(d)
        ch = cb.build_chain(symbol, d, day, ev) if day is not None else None
        if ch is None:
            prev, prev_date = None, None
            continue
        fits = cb.fit_day(ch, prev)
        hold = {}
        if i % HOLDOUT_EVERY == 0:
            hm = hide_masks(ch)
            train = ch.subset([~m for m in hm])
            test = ch.subset(hm)
            if train.n >= cb.MIN_OPTIONS and test.n > 0:
                tf = cb.fit_day(train, prev)
                for m, f in tf.items():
                    if np.isfinite(f.x).all() and (m != "bates_ev" or cb.event_identified(train)):
                        r = cb.residuals(m, f.x, test)
                        hold[m] = float(np.sqrt(np.mean(r ** 2)))
        nxt = {}
        if prev is not None:
            for m, x in prev.items():
                if m == "bates_ev" and not cb.event_identified(ch):
                    continue
                if np.isfinite(x).all():
                    nxt[m] = cb.refit_v0(m, x, ch).rmse_vega
        for m, f in fits.items():
            if not np.isfinite(f.x).all():
                rows.append(dict(symbol=symbol, date=d, model=m, status=f.status, n_opt=ch.n))
                continue
            err = cb.iv_errors(m, f.x, ch)
            rv = cb.residuals(m, f.x, ch)
            rows.append(dict(
                symbol=symbol, date=d, model=m, n_opt=ch.n, n_exp=len(ch.expiries),
                t_min=ch.expiries[0].T, t_max=ch.expiries[-1].T,
                n_events=max(e.n_events for e in ch.expiries), atm_iv=np.sqrt(ch.atm_var()),
                rmse_vega=float(np.sqrt(np.mean(rv ** 2))), rmse_iv=float(np.sqrt(np.nanmean(err ** 2))),
                max_abs_iv=float(np.nanmax(np.abs(err))), iv_nan=int(np.isnan(err).sum()),
                cost=f.cost, nfev=f.nfev, status=f.status, secs=f.secs, start=f.start,
                at_bounds=at_bounds(m, f.x), hold_rmse=hold.get(m, np.nan), next_rmse=nxt.get(m, np.nan),
                prev_gap=(d - prev_date).days if prev_date is not None else np.nan,
                **{f"p_{k}": float(v) for k, v in zip(cb.MODELS[m], f.x)}))
            k = 0
            for e in ch.expiries:
                n = len(e.K)
                resid.append(pd.DataFrame(dict(symbol=symbol, date=d, model=m, expiry=e.expiry, T=e.T,
                                               k=np.log(e.K / e.F), is_call=e.is_call, iv_mkt=e.iv,
                                               err_iv=err[k:k + n], err_vega=rv[k:k + n])))
                k += n
        prev = {m: f.x for m, f in fits.items() if np.isfinite(f.x).all()}
        prev_date = d
    os.makedirs(os.path.join(OUT, "fits"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "resid"), exist_ok=True)
    if resid:
        pd.concat(resid, ignore_index=True).to_parquet(os.path.join(OUT, "resid", f"{symbol}_{tag}.parquet"))
    pd.DataFrame(rows).to_parquet(fits_path)
    print(f"{symbol} {tag}: {len(sessions)} sessions, {len(rows)} fits, {time.time() - t0:.0f}s", flush=True)


def _job(args):
    symbol, sessions, tag = args
    try:
        run_chunk(symbol, sessions, tag)
    except Exception as e:  # keep the pool alive; the chunk is retried on the next run
        import traceback
        print(f"FAILED {symbol} {tag}: {e!r}\n{traceback.format_exc()}", flush=True)


def jobs(pilot: bool) -> list[tuple]:
    out = []
    for sym in SYMBOLS:
        dates = pd.DatetimeIndex(cb.load_symbol(sym)["date"].unique()).sort_values()
        if pilot:
            for start, n in PILOT_WINDOWS:
                s = list(dates[dates >= start][:n])
                out.append((sym, s, f"pilot{start[:4]}"))
        else:
            for y in sorted(set(dates.year)):
                out.append((sym, list(dates[dates.year == y]), str(y)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    global OUT
    if a.pilot:
        OUT = os.path.join(OUT, "pilot")
    js = jobs(a.pilot)
    js.sort(key=lambda j: -len(j[1]) * (1 + (j[2] >= "2019")))     # long / heavy chunks first
    print(f"{len(js)} chunks, {sum(len(j[1]) for j in js)} sessions -> {OUT}", flush=True)
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.workers, initializer=_set_out, initargs=(OUT,)) as pool:
        for _ in pool.imap_unordered(_job, js):
            pass
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


def _set_out(out):
    global OUT
    OUT = out


if __name__ == "__main__":
    main()
