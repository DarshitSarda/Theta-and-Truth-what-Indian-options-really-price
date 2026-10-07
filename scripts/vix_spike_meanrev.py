"""Does a one-day India VIX spike precede positive NIFTY returns?

Motivation
----------
On 2026-09-15 NIFTY marked a local low and our option-chain features printed
sample extremes (highest iv_minus_trail20, largest 1-day IV jump, largest 1-day
VIX jump in 48 sessions). That is one episode, so it cannot support a signal.
The underlying hypothesis, however, needs no option data at all -- only India VIX
and NIFTY OHLC -- and we hold ~4.5k sessions of both back to 2008. So we test it
on the long history first. Only if it survives here is it worth asking whether
skew/curvature/wall features add anything on top.

Hypothesis
----------
H1: a large one-day *rise* in India VIX is followed by positive NIFTY returns
    over the next k sessions, beyond the unconditional drift.

Design (the parts that matter)
------------------------------
No lookahead:
  * The spike threshold is a percentile computed inside a TRAILING window only
    (default 250 sessions, strictly prior days). A full-sample percentile would
    leak the future into the signal definition.
  * Regime buckets use the trailing percentile of the VIX *level*, again prior
    days only.
  * Every signal input (VIX close, NIFTY open/high/low/close) is known at the
    close of day t. Nothing intraday-after-t and no same-day low is used.

Execution realism:
  * `cc` returns are close[t] -> close[t+k], which assumes we transact the close.
  * `oc` returns are open[t+1] -> close[t+k], which does not. Both are reported;
    `oc` is the conservative one.
  * No transaction costs. For multi-day index futures holds these are small but
    non-zero, so treat small edges accordingly.

Independence:
  * Episodes are DE-CLUSTERED per horizon: once a signal fires at t, later
    signals are suppressed until t+k has passed, so no two scored windows
    overlap. Vol spikes cluster heavily, so skipping this step would badly
    overstate the sample size.

Honest benchmarks:
  * NIFTY drifts upward, so a positive conditional mean proves nothing on its
    own. We compare against (a) the unconditional mean over the same span and
    (b) a REGIME-MATCHED PLACEBO: placebo dates drawn from the same trailing
    VIX-level decile as each episode. (b) is the important one -- it isolates
    the spike from the mere fact that spikes happen when vol is already high.

Multiple testing:
  * Primary pre-declared spec: threshold = 95th trailing percentile, horizons
    5 and 10 sessions, `cc` returns. Everything else is secondary/robustness and
    the total test count is printed so nothing is quietly cherry-picked.

Usage:  python scripts/vix_spike_meanrev.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UND = os.path.join(PROJECT_ROOT, "data", "raw", "underlying")

TRAIL_WINDOW = 250          # trailing sessions for percentile refs (~1 year)
HORIZONS = (1, 3, 5, 10, 20)
THRESHOLDS = (90, 95, 99)
PRIMARY_THRESHOLD = 95
PRIMARY_HORIZONS = (5, 10)
N_BOOT = 2000
RNG = np.random.default_rng(20260918)

_test_count = 0


# ---------------------------------------------------------------- data


def load_panel() -> pd.DataFrame:
    vix = (pd.read_csv(os.path.join(UND, "india_vix.csv"))[["date", "close"]]
           .rename(columns={"close": "vix"}))
    nif = pd.read_csv(os.path.join(UND, "nifty.csv"))[
        ["date", "open", "high", "low", "close"]]
    df = (nif.merge(vix, on="date", how="inner")
          .sort_values("date").reset_index(drop=True))
    df = df.dropna(subset=["open", "high", "low", "close", "vix"]).reset_index(drop=True)
    return df


def _trailing_pctile(s: pd.Series, window: int) -> pd.Series:
    """Percentile of today's value within the `window-1` STRICTLY PRIOR values.

    Using only prior days is what keeps the signal free of lookahead: the
    reference distribution is what an observer standing at close of day t
    would already have seen.
    """
    def f(a: np.ndarray) -> float:
        prior, today = a[:-1], a[-1]
        return float((prior < today).mean() * 100.0)

    return s.rolling(window, min_periods=window).apply(f, raw=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # % change is scale-free, so a "big move" is comparable at VIX 12 and VIX 60.
    d["vix_chg_pct"] = d["vix"].pct_change() * 100.0
    d["vix_chg_abs"] = d["vix"].diff()
    d["spike_pctile"] = _trailing_pctile(d["vix_chg_pct"], TRAIL_WINDOW)
    d["level_pctile"] = _trailing_pctile(d["vix"], TRAIL_WINDOW)
    # Where in the day's range did we close? 0 = at the low (capitulation).
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["close_in_range"] = (d["close"] - d["low"]) / rng
    d["nifty_ret"] = d["close"].pct_change() * 100.0

    for k in HORIZONS:
        # close[t] -> close[t+k]: assumes execution at the close of the signal day
        d[f"cc{k}"] = (d["close"].shift(-k) / d["close"] - 1.0) * 100.0
        # open[t+1] -> close[t+k]: conservative, no close execution assumed
        d[f"oc{k}"] = (d["close"].shift(-k) / d["open"].shift(-1) - 1.0) * 100.0
    return d


# ---------------------------------------------------------------- episodes


def declustered_mask(signal: pd.Series, horizon: int) -> np.ndarray:
    """Keep signals whose k-session forward windows do not overlap."""
    keep = np.zeros(len(signal), dtype=bool)
    blocked_until = -1
    for i, hit in enumerate(signal.to_numpy()):
        if hit and i > blocked_until:
            keep[i] = True
            blocked_until = i + horizon - 1
    return keep


def summarize(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "win": np.nan, "sd": np.nan}
    return {"n": len(x), "mean": x.mean(), "median": float(np.median(x)),
            "win": float((x > 0).mean() * 100.0),
            "sd": x.std(ddof=1) if len(x) > 1 else np.nan}


def boot_ci(x: np.ndarray, alpha: float = 0.05) -> tuple:
    """Percentile bootstrap CI for the mean, resampling episodes (which are
    already de-clustered, so i.i.d. resampling is defensible)."""
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    draws = RNG.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def regime_matched_placebo(d: pd.DataFrame, episode_idx: np.ndarray,
                           ret_col: str, signal: pd.Series) -> tuple:
    """p-value from placebo dates drawn from the same trailing VIX-level decile.

    This is the test that separates "a spike happened" from "vol was already
    high". Without it, any edge could just be the level effect.
    """
    valid = d[ret_col].notna() & d["level_pctile"].notna() & (~signal)
    pool_by_decile: dict[int, np.ndarray] = {}
    for dec in range(10):
        lo, hi = dec * 10, (dec + 1) * 10
        m = valid & (d["level_pctile"] >= lo) & (d["level_pctile"] < hi)
        pool_by_decile[dec] = d.index[m].to_numpy()

    deciles = np.clip((d.loc[episode_idx, "level_pctile"] // 10).astype(int), 0, 9)
    usable = [(i, dec) for i, dec in zip(episode_idx, deciles)
              if len(pool_by_decile[dec]) > 0 and not np.isnan(d.loc[i, ret_col])]
    if len(usable) < 3:
        return np.nan, np.nan

    actual = np.mean([d.loc[i, ret_col] for i, _ in usable])
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        picks = [RNG.choice(pool_by_decile[dec]) for _, dec in usable]
        means[b] = d.loc[picks, ret_col].mean()
    # one-sided: is the real conditional mean unusually HIGH vs regime-matched noise?
    p = float((means >= actual).mean())
    return actual, p


# ---------------------------------------------------------------- reporting


def run_block(d: pd.DataFrame, signal: pd.Series, label: str,
              horizons=HORIZONS, ret_kind: str = "cc", placebo: bool = True) -> None:
    global _test_count
    print(f"\n  {label}   (raw signal days: {int(signal.sum())})")
    print(f"    {'k':>3} {'n_ep':>5} {'mean%':>8} {'med%':>8} {'win%':>7} "
          f"{'boot 95% CI':>20} {'uncond%':>8} {'diff':>7} {'placebo p':>10}")
    for k in horizons:
        col = f"{ret_kind}{k}"
        keep = declustered_mask(signal, k)
        idx = d.index[keep]
        ep = d.loc[idx, col].to_numpy(float)
        st = summarize(ep)
        if st["n"] == 0:
            print(f"    {k:>3} {0:>5}   (no episodes)")
            continue
        lo, hi = boot_ci(ep)
        uncond = summarize(d[col].to_numpy(float))
        _test_count += 1
        if placebo:
            _, pval = regime_matched_placebo(d, idx, col, signal)
            pstr = f"{pval:.3f}" if not np.isnan(pval) else "n/a"
        else:
            pstr = "-"
        print(f"    {k:>3} {st['n']:>5} {st['mean']:>+8.2f} {st['median']:>+8.2f} "
              f"{st['win']:>7.1f} {f'[{lo:+.2f}, {hi:+.2f}]':>20} "
              f"{uncond['mean']:>+8.2f} {st['mean'] - uncond['mean']:>+7.2f} {pstr:>10}")


def main() -> None:
    d = build_features(load_panel())
    ok = d["spike_pctile"].notna()
    print("=" * 100)
    print("INDIA VIX SPIKE -> FORWARD NIFTY RETURNS")
    print("=" * 100)
    print(f"panel: {len(d)} sessions {d.date.iloc[0]} .. {d.date.iloc[-1]}")
    print(f"usable after {TRAIL_WINDOW}-session warm-up: {int(ok.sum())} "
          f"({d.loc[ok, 'date'].iloc[0]} onward)")
    print(f"VIX level: min {d.vix.min():.2f}  median {d.vix.median():.2f}  "
          f"max {d.vix.max():.2f}")
    print("\nNOTE: returns are % index moves, no transaction costs. 'cc' assumes")
    print("execution at the signal-day close; 'oc' enters at the next open.")

    d = d.loc[ok].reset_index(drop=True)

    print("\n" + "=" * 100)
    print(f"PRIMARY (pre-declared): spike >= p{PRIMARY_THRESHOLD} of trailing "
          f"{TRAIL_WINDOW}, horizons {PRIMARY_HORIZONS}, cc returns")
    print("=" * 100)
    sig = d["spike_pctile"] >= PRIMARY_THRESHOLD
    run_block(d, sig, f"VIX 1d %chg >= p{PRIMARY_THRESHOLD}", PRIMARY_HORIZONS, "cc")

    print("\n" + "=" * 100)
    print("SECONDARY: all thresholds x all horizons")
    print("=" * 100)
    for th in THRESHOLDS:
        run_block(d, d["spike_pctile"] >= th, f"spike >= p{th}", HORIZONS, "cc")

    print("\n" + "=" * 100)
    print("EXECUTION CHECK: same signal, entry at NEXT OPEN (oc returns)")
    print("=" * 100)
    run_block(d, d["spike_pctile"] >= PRIMARY_THRESHOLD,
              f"spike >= p{PRIMARY_THRESHOLD}, next-open entry", HORIZONS, "oc")

    print("\n" + "=" * 100)
    print("THE DECISIVE SPLIT: does it work when vol is ALREADY LOW?")
    print("  (our live sample sits in the bottom decile of the VIX distribution)")
    print("=" * 100)
    sig = d["spike_pctile"] >= PRIMARY_THRESHOLD
    for name, lo, hi in [("LOW  vol regime (level p0-33)", 0, 33),
                         ("MID  vol regime (level p33-66)", 33, 66),
                         ("HIGH vol regime (level p66-100)", 66, 101)]:
        sub = sig & (d["level_pctile"] >= lo) & (d["level_pctile"] < hi)
        run_block(d, sub, name, HORIZONS, "cc", placebo=False)

    print("\n" + "=" * 100)
    print("ROBUSTNESS")
    print("=" * 100)
    # a) conjunction with a weak close, the 2026-09-15 pattern
    sig_cap = (d["spike_pctile"] >= PRIMARY_THRESHOLD) & (d["close_in_range"] <= 0.25)
    run_block(d, sig_cap, "spike + closed in bottom 25% of day's range",
              HORIZONS, "cc")
    # b) require NIFTY actually fell (fear spike, not pre-event vol bid)
    sig_dn = (d["spike_pctile"] >= PRIMARY_THRESHOLD) & (d["nifty_ret"] < 0)
    run_block(d, sig_dn, "spike + NIFTY down that day", HORIZONS, "cc")
    # c) drop the first two years: India VIX was a new, thin product
    later = d[d["date"] >= "2010-01-01"].reset_index(drop=True)
    run_block(later, later["spike_pctile"] >= PRIMARY_THRESHOLD,
              "spike >= p95, 2010 onward only", HORIZONS, "cc")
    # d) absolute-points threshold instead of percentile
    sig_abs = d["vix_chg_abs"] >= 2.0
    run_block(d, sig_abs, "VIX rose >= 2.00 points (absolute)", HORIZONS, "cc")

    print("\n" + "=" * 100)
    print(f"TOTAL (horizon x spec) TESTS REPORTED: {_test_count}")
    print(f"Bonferroni-style threshold at family-wide 0.05: p < {0.05/_test_count:.5f}")
    print("Treat any single nominally-significant cell accordingly; horizons")
    print("within a spec overlap in information and are not independent tests.")
    print("=" * 100)

    # Where does the 2026-09-15 episode sit under this definition?
    print("\nThe episode that motivated this test:")
    recent = d[d["date"] >= "2026-09-09"][
        ["date", "vix", "vix_chg_pct", "spike_pctile", "level_pctile",
         "close_in_range", "cc5", "cc10"]]
    print(recent.to_string(index=False))


if __name__ == "__main__":
    main()
