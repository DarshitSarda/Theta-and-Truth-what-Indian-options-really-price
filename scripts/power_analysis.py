"""How much data does this pipeline need before any test can succeed?

Three tests have now come back null (VIX spike mean-reversion, VIX-NIFTY
lead-lag, VIX divergence). Before concluding the data is worthless, we should
check whether our tests were even CAPABLE of detecting a real effect. A null
result from an underpowered test is not evidence of absence -- it is no
evidence at all.

This script computes the MINIMUM DETECTABLE EFFECT (MDE): the smallest edge, in
% forward return, that a test could reliably find at a given sample size. If the
MDE is larger than any edge that plausibly exists, the test was decoration.

Two regimes are compared, and the gap between them is the main finding:

  1. RARE-SIGNAL DIRECTIONAL TESTS. A signal fires on ~5% of days and we ask
     whether forward returns beat the base rate. Only firing days count, and
     de-clustering removes overlapping windows, so usable observations are a
     small fraction of the calendar.

  2. ALWAYS-ON PREMIUM ESTIMATES (the VRP case). Every session contributes an
     observation because we are estimating a level (implied minus realised), not
     waiting for a rare trigger. Usable observations are sessions/horizon.

Regime 2 needs roughly 20x less calendar time than regime 1 for the same
precision. That is a statement about what this dataset can and cannot answer,
independent of anyone's market view.

Usage:  python scripts/power_analysis.py
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from vix_spike_meanrev import build_features, load_panel

SESSIONS_PER_YEAR = 250
HORIZONS = (1, 5, 10, 20)
FIRE_RATE = 0.05          # a "top 5% of trailing window" style signal
CLUSTER_PENALTY = 0.70    # empirical: real de-clustering loses ~30% vs the naive
                          # count, because signal days arrive in bursts
POWER_Z = stats.norm.ppf(0.80)

# sample sizes worth comparing, in sessions
SCENARIOS = [
    ("today's live sample", 48),
    ("1 year of collection", 250),
    ("3 years of collection", 750),
    ("5 years of collection", 1250),
    ("10 years of collection", 2500),
    ("bhavcopy backfill to 2008", 4500),
]


def mde(sigma: float, n: int, n_tests: int) -> float:
    """Smallest detectable difference in mean return, 80% power.

    n_tests > 1 applies a Bonferroni-corrected alpha, which is the honest
    setting when a panel of features is screened at once.
    """
    if n < 3:
        return np.nan
    alpha = 0.05 / n_tests
    z_a = stats.norm.ppf(1 - alpha / 2)
    return (z_a + POWER_Z) * sigma / np.sqrt(n)


def episodes_rare(sessions: int, k: int) -> int:
    """Independent (non-overlapping) episodes from a 5%-frequency signal."""
    naive = sessions * FIRE_RATE
    capped = min(naive, sessions / k)
    return int(max(capped * CLUSTER_PENALTY, 0))


def episodes_always_on(sessions: int, k: int) -> int:
    """Non-overlapping observations when every session is usable."""
    return int(sessions / k)


def main() -> None:
    d = build_features(load_panel())
    print("=" * 96)
    print("STATISTICAL POWER: WHAT SAMPLE SIZE DOES THIS PIPELINE ACTUALLY NEED?")
    print("=" * 96)

    sigma = {}
    for k in HORIZONS:
        sigma[k] = float(d[f"cc{k}"].dropna().std(ddof=1))
    print("\nNIFTY forward-return dispersion, 2008-2026 (the noise we fight):")
    for k in HORIZONS:
        base = float(d[f"cc{k}"].dropna().mean())
        print(f"  {k:>3}-session horizon: sd = {sigma[k]:5.2f}%   base-rate mean = {base:+.2f}%")

    print("\n" + "=" * 96)
    print("REGIME 1: RARE-SIGNAL DIRECTIONAL TEST (signal fires ~5% of days)")
    print("  MDE = smallest excess return over base rate detectable at 80% power")
    print("  'single' = one pre-declared test; 'screen' = 60 tests (20 features x 3 horizons)")
    print("=" * 96)
    print(f"  {'sample':<26}{'sessions':>9}{'k':>4}{'n_ep':>6}"
          f"{'MDE single':>12}{'MDE screen':>12}")
    for name, sessions in SCENARIOS:
        for k in (5, 10):
            n = episodes_rare(sessions, k)
            m1 = mde(sigma[k], n, 1)
            m60 = mde(sigma[k], n, 60)
            s1 = f"{m1:>10.2f}%" if not np.isnan(m1) else "     n/a  "
            s60 = f"{m60:>10.2f}%" if not np.isnan(m60) else "     n/a  "
            print(f"  {name:<26}{sessions:>9}{k:>4}{n:>6}{s1:>12}{s60:>12}")

    print("\n" + "=" * 96)
    print("REGIME 2: ALWAYS-ON ESTIMATE (the VRP case -- every session counts)")
    print("=" * 96)
    print(f"  {'sample':<26}{'sessions':>9}{'k':>4}{'n_obs':>6}"
          f"{'MDE single':>12}{'MDE screen':>12}")
    for name, sessions in SCENARIOS:
        for k in (5, 20):
            n = episodes_always_on(sessions, k)
            m1 = mde(sigma[k], n, 1)
            m60 = mde(sigma[k], n, 60)
            s1 = f"{m1:>10.2f}%" if not np.isnan(m1) else "     n/a  "
            s60 = f"{m60:>10.2f}%" if not np.isnan(m60) else "     n/a  "
            print(f"  {name:<26}{sessions:>9}{k:>4}{n:>6}{s1:>12}{s60:>12}")

    print("\n" + "=" * 96)
    print("HOW LONG WOULD WE HAVE TO WAIT? (daily collection only, no backfill)")
    print("  target: detect a plausibly-sized edge at 80% power")
    print("=" * 96)
    for k in (5, 10):
        for target in (0.30, 0.50, 1.00):
            # invert the MDE formula for the required number of episodes
            for label, n_tests in (("single test", 1), ("60-test screen", 60)):
                alpha = 0.05 / n_tests
                z_a = stats.norm.ppf(1 - alpha / 2)
                n_needed = ((z_a + POWER_Z) * sigma[k] / target) ** 2
                # invert episodes_rare to get calendar sessions
                sess = n_needed / (FIRE_RATE * CLUSTER_PENALTY)
                yrs = sess / SESSIONS_PER_YEAR
                print(f"  k={k:>2}  edge {target:.2f}%  {label:<15} "
                      f"needs {n_needed:>7.0f} episodes = {sess:>8.0f} sessions "
                      f"= {yrs:>6.1f} years of collection")
        print()

    print("=" * 96)
    print("WHAT 48 SESSIONS COULD HAVE FOUND")
    print("=" * 96)
    n_now = episodes_rare(48, 5)
    m_now = mde(sigma[5], max(n_now, 3), 39)
    print(f"  The earlier 13-feature x 3-horizon screen on 48 sessions had roughly")
    print(f"  {n_now} independent episodes per feature at k=5.")
    print(f"  Minimum detectable edge there: {m_now:.1f}% over 5 sessions.")
    print(f"  For scale, NIFTY's entire 5-session sd is {sigma[5]:.2f}% and its")
    print(f"  base-rate 5-session return is {float(d['cc5'].dropna().mean()):+.2f}%.")
    print("  An edge that large does not exist in a liquid index. That screen")
    print("  could not have detected a real signal, so its nulls carry no")
    print("  information about whether the option features work.")
    print("=" * 96)


if __name__ == "__main__":
    main()
