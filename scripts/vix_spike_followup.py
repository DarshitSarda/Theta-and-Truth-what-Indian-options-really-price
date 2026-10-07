"""Follow-ups to vix_spike_meanrev.py: kill or confirm the two surviving cells.

The main run was null on its pre-declared spec. Two cells were left open:

  A. LOW vol regime (trailing level percentile 0-33) showed win rates of 78% at
     k=5/k=10. That is the ONLY regime relevant to our live sample (India VIX
     ~11-13, bottom decile), so it needs the regime-matched placebo that the
     main script skipped for the regime splits.

  B. "VIX rose >= 2.00 absolute points" produced the only small placebo
     p-values (0.002 at k=1, 0.022 at k=10). A real effect should be monotone
     in signal strength and reasonably stable across nearby thresholds. If
     +2.00 is a knife-edge, it is noise dressed up as a finding. We also check
     whether those episodes are concentrated in a couple of crisis years, in
     which case the "sample" is really two or three regimes, not 159 draws.

Also reported: the dose-response curve across percentile thresholds, since the
main run hinted the effect gets WORSE at p99 -- the opposite of what a genuine
mechanism produces.

Same no-lookahead rules as the main script (trailing-window percentiles only,
signal known at close of day t, de-clustered non-overlapping forward windows).

Usage:  python scripts/vix_spike_followup.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from vix_spike_meanrev import (HORIZONS, TRAIL_WINDOW, build_features,
                               declustered_mask, load_panel, summarize)

N_BOOT = 4000
RNG = np.random.default_rng(20260918)


def placebo_p(d: pd.DataFrame, episode_idx: np.ndarray, ret_col: str,
              exclude: pd.Series, n_buckets: int = 10) -> tuple:
    """Vectorised regime-matched placebo test.

    Placebo dates are drawn from the same trailing VIX-LEVEL bucket as each
    episode, so the null keeps the vol regime fixed and varies only "was there
    a spike". Anything that survives this is not just a level effect.
    """
    width = 100.0 / n_buckets
    valid = d[ret_col].notna() & d["level_pctile"].notna() & (~exclude)
    pools = {}
    for b in range(n_buckets):
        m = valid & (d["level_pctile"] >= b * width) & (d["level_pctile"] < (b + 1) * width)
        arr = d.loc[m, ret_col].to_numpy(float)
        if len(arr):
            pools[b] = arr

    buckets, obs = [], []
    for i in episode_idx:
        if np.isnan(d.at[i, ret_col]) or np.isnan(d.at[i, "level_pctile"]):
            continue
        b = int(min(d.at[i, "level_pctile"] // width, n_buckets - 1))
        if b in pools:
            buckets.append(b)
            obs.append(d.at[i, ret_col])
    if len(obs) < 3:
        return np.nan, np.nan, 0

    actual = float(np.mean(obs))
    # one column of placebo draws per episode, all B replications at once
    draws = np.column_stack([RNG.choice(pools[b], size=N_BOOT) for b in buckets])
    means = draws.mean(axis=1)
    return actual, float((means >= actual).mean()), len(obs)


def block(d: pd.DataFrame, signal: pd.Series, label: str,
          exclude: pd.Series | None = None, n_buckets: int = 10) -> None:
    if exclude is None:
        exclude = signal
    print(f"\n  {label}   (raw signal days: {int(signal.sum())})")
    print(f"    {'k':>3} {'n_ep':>5} {'mean%':>8} {'win%':>7} {'uncond%':>8} "
          f"{'diff':>7} {'placebo p':>10}")
    for k in HORIZONS:
        col = f"cc{k}"
        idx = d.index[declustered_mask(signal, k)]
        st = summarize(d.loc[idx, col].to_numpy(float))
        if st["n"] == 0:
            print(f"    {k:>3} {0:>5}   (no episodes)")
            continue
        unc = summarize(d[col].to_numpy(float))
        _, p, n = placebo_p(d, idx, col, exclude, n_buckets)
        pstr = f"{p:.3f}" if not np.isnan(p) else "n/a"
        print(f"    {k:>3} {st['n']:>5} {st['mean']:>+8.2f} {st['win']:>7.1f} "
              f"{unc['mean']:>+8.2f} {st['mean'] - unc['mean']:>+7.2f} {pstr:>10}")


def year_spread(d: pd.DataFrame, signal: pd.Series, label: str) -> None:
    yrs = pd.to_datetime(d.loc[signal, "date"]).dt.year.value_counts().sort_index()
    total = int(yrs.sum())
    top = yrs.sort_values(ascending=False)
    share = top.iloc[:3].sum() / total * 100.0 if total else np.nan
    print(f"\n  {label}: {total} signal days across {len(yrs)} calendar years")
    print("    " + "  ".join(f"{y}:{c}" for y, c in yrs.items()))
    print(f"    top-3 years hold {share:.0f}% of all signals "
          f"({', '.join(str(y) for y in top.index[:3])})")


def main() -> None:
    d = build_features(load_panel())
    d = d.loc[d["spike_pctile"].notna()].reset_index(drop=True)
    print("=" * 92)
    print("FOLLOW-UPS: LOW-VOL REGIME AND ABSOLUTE-THRESHOLD STABILITY")
    print(f"panel {len(d)} sessions {d.date.iloc[0]} .. {d.date.iloc[-1]}")
    print("=" * 92)

    spike95 = d["spike_pctile"] >= 95

    print("\n" + "=" * 92)
    print("A. LOW VOL REGIME, NOW WITH A REGIME-MATCHED PLACEBO")
    print("   placebos drawn from the same trailing level TERCILE (finer buckets")
    print("   would leave too few non-signal days inside the low band)")
    print("=" * 92)
    low = d["level_pctile"] < 33
    block(d, spike95 & low, "spike>=p95 AND trailing level p0-33",
          exclude=spike95, n_buckets=3)
    print("\n   For contrast, the same cell restricted to the bottom level decile")
    print("   (closest analogue to today's VIX ~11-13):")
    block(d, spike95 & (d["level_pctile"] < 10),
          "spike>=p95 AND trailing level p0-10", exclude=spike95, n_buckets=3)
    year_spread(d, spike95 & low, "low-vol spike episodes")

    print("\n" + "=" * 92)
    print("B. DOSE-RESPONSE: ABSOLUTE POINT THRESHOLDS")
    print("   a real effect should strengthen smoothly as the threshold rises")
    print("=" * 92)
    for thr in (0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
        block(d, d["vix_chg_abs"] >= thr, f"VIX rose >= {thr:.2f} points")
    year_spread(d, d["vix_chg_abs"] >= 2.0, "VIX +2.00pt episodes")

    print("\n" + "=" * 92)
    print("C. DOSE-RESPONSE: PERCENTILE THRESHOLDS (k=5 and k=10 only)")
    print("=" * 92)
    print(f"    {'threshold':>10} {'k':>3} {'n_ep':>5} {'mean%':>8} {'win%':>7} "
          f"{'diff vs uncond':>15} {'placebo p':>10}")
    for th in (80, 85, 90, 95, 97, 99):
        sig = d["spike_pctile"] >= th
        for k in (5, 10):
            col = f"cc{k}"
            idx = d.index[declustered_mask(sig, k)]
            st = summarize(d.loc[idx, col].to_numpy(float))
            unc = summarize(d[col].to_numpy(float))
            _, p, _ = placebo_p(d, idx, col, sig)
            pstr = f"{p:.3f}" if not np.isnan(p) else "n/a"
            print(f"    {'p' + str(th):>10} {k:>3} {st['n']:>5} {st['mean']:>+8.2f} "
                  f"{st['win']:>7.1f} {st['mean'] - unc['mean']:>+15.2f} {pstr:>10}")

    print("\n" + "=" * 92)
    print("D. SPLIT-HALF STABILITY of the low-vol cell (k=10)")
    print("   an effect present in only one half is not a signal")
    print("=" * 92)
    mid = d["date"].iloc[len(d) // 2]
    for name, sub in [(f"first half (..{mid})", d[d["date"] < mid]),
                      (f"second half ({mid}..)", d[d["date"] >= mid])]:
        sub = sub.reset_index(drop=True)
        sig = (sub["spike_pctile"] >= 95) & (sub["level_pctile"] < 33)
        idx = sub.index[declustered_mask(sig, 10)]
        st = summarize(sub.loc[idx, "cc10"].to_numpy(float))
        unc = summarize(sub["cc10"].to_numpy(float))
        if st["n"]:
            print(f"    {name:<28} n={st['n']:>3}  mean={st['mean']:+.2f}%  "
                  f"win={st['win']:.1f}%  uncond={unc['mean']:+.2f}%  "
                  f"diff={st['mean'] - unc['mean']:+.2f}")
        else:
            print(f"    {name:<28} no episodes")

    print("\n" + "=" * 92)
    print("E. IS THE SURVIVING SPEC EVEN REACHABLE IN TODAY'S REGIME?")
    print("   The absolute-points spec shows a monotone dose-response, but a")
    print("   +2pt jump off VIX 12 is a 16% move. If it only ever fires when VIX")
    print("   is already elevated, it cannot inform a VIX ~11-13 tape.")
    print("=" * 92)
    for thr in (1.5, 2.0, 2.5):
        sig = d["vix_chg_abs"] >= thr
        lv = d.loc[sig, "vix"]
        print(f"\n  VIX rose >= {thr:.2f}pt  ({int(sig.sum())} days)")
        print(f"    prevailing VIX level at signal: min {lv.min():.1f}  "
              f"p25 {lv.quantile(.25):.1f}  median {lv.median():.1f}  "
              f"p75 {lv.quantile(.75):.1f}  max {lv.max():.1f}")
        for cap in (15.0, 20.0):
            m = sig & (d["vix"] < cap)
            idx = d.index[declustered_mask(m, 10)]
            st = summarize(d.loc[idx, "cc10"].to_numpy(float))
            n_days = int(m.sum())
            share = n_days / max(int(sig.sum()), 1) * 100
            extra = (f"k=10 n_ep={st['n']} mean={st['mean']:+.2f}% win={st['win']:.0f}%"
                     if st["n"] else "no scorable episodes")
            print(f"    fired with VIX < {cap:.0f}: {n_days} days "
                  f"({share:.0f}% of spec)   {extra}")

    print("\n  What the 2026-09-15 session actually looked like under each spec:")
    row = d[d["date"] == "2026-09-15"]
    if len(row):
        r = row.iloc[0]
        print(f"    VIX {r.vix:.2f}, 1-day change {r.vix_chg_abs:+.2f}pt "
              f"({r.vix_chg_pct:+.1f}%), trailing spike pctile p{r.spike_pctile:.0f}, "
              f"level pctile p{r.level_pctile:.0f}")
        print(f"    percentile spec (>=p95):      "
              f"{'FIRES' if r.spike_pctile >= 95 else 'no fire'}  <- the null spec")
        for thr in (1.5, 2.0, 2.5):
            print(f"    absolute spec (>= {thr:.2f}pt):    "
                  f"{'FIRES' if r.vix_chg_abs >= thr else 'no fire'}"
                  f"{'  <- the spec with a gradient' if thr == 2.0 else ''}")
    print("=" * 92)


if __name__ == "__main__":
    main()
