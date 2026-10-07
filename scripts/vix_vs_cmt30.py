"""Compare India VIX against our Layer-4 constant-maturity 30d ATM IV (NIFTY only).

This is NOT a level-agreement test. India VIX is a model-free variance-swap-style
integral over the whole OTM strip; cmt30 is an ATM number. The variance-swap rate
exceeds ATM IV by a skew/convexity margin, so a persistent positive spread is the
expected result, not a defect.

What actually validates the cmt30 construction is co-movement and a *stable*
spread. So we report:
  * Pearson/Spearman correlation in levels and in daily changes
  * spread mean/sd and an OLS trend, both raw and controlling for the vol level
  * the cmt30_in_range share, because a clamped cmt30 is not a 30-day number at
    all and any disagreement would then be extrapolation error, not construction
    error

The level control matters: the spread is mechanically wider when vol is low, so
any sample where vol trends will show a spurious time trend in the raw spread.
The verdict is taken from the level-controlled trend. `front_atm_iv` is the
control rather than `cmt30_iv`, which appears inside the spread and would induce
a mechanical negative coefficient.

The spread itself is a candidate feature (skew/convexity premium), not just a
diagnostic, so we also correlate it against smile shape at the *back* expiry.
Front-week rr25/bf25 is the wrong tenor to explain a 30-day strip integral and
gives misleading signs.

India VIX is computed on NIFTY options only -- BANKNIFTY is deliberately excluded.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIX_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "underlying", "india_vix.csv")
SIG_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "vol_surface", "signals")
SIG_DAILY = os.path.join(SIG_DIR, "_daily.csv")
SIG_ALL = os.path.join(SIG_DIR, "_signals_all.csv")


def load_joined() -> pd.DataFrame:
    vix = pd.read_csv(VIX_PATH)[["date", "close"]].rename(columns={"close": "india_vix"})
    sig = pd.read_csv(SIG_DAILY)
    sig = sig[sig["symbol"] == "NIFTY"][
        ["date", "cmt30_iv", "cmt30_in_range", "cmt7_iv", "cmt7_in_range",
         "front_atm_iv", "back_dte"]]

    # Longest-dated expiry per day: the tenor cmt30 actually reflects, and the
    # right place to read smile convexity for a 30-day strip comparison.
    per_exp = pd.read_csv(SIG_ALL)
    per_exp = per_exp[per_exp["symbol"] == "NIFTY"].sort_values(["date", "dte"])
    back = (per_exp.groupby("date").tail(1)[["date", "rr25", "bf25", "atm_curv"]]
            .rename(columns={"rr25": "back_rr25", "bf25": "back_bf25",
                             "atm_curv": "back_curv"}))

    j = (sig.merge(vix, on="date", how="inner")
            .merge(back, on="date", how="left")
            .sort_values("date").reset_index(drop=True))
    j["spread"] = j["india_vix"] - j["cmt30_iv"]
    return j


def _corr(a: pd.Series, b: pd.Series) -> tuple:
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan"), float("nan"), int(ok.sum())
    return (float(stats.pearsonr(a[ok], b[ok])[0]),
            float(stats.spearmanr(a[ok], b[ok])[0]),
            int(ok.sum()))


def report(j: pd.DataFrame) -> None:
    n = len(j)
    print(f"India VIX vs NIFTY cmt30 -- {n} matched sessions "
          f"({j['date'].iloc[0]} .. {j['date'].iloc[-1]})")

    in_range = int(j["cmt30_in_range"].sum())
    print(f"\ncmt30_in_range: {in_range}/{n}")
    if in_range < n:
        print("  WARNING: cmt30 is clamped on "
              f"{n - in_range} session(s). When clamped, linear-in-total-variance "
              "extrapolation past the last node collapses cmt30 to the longest "
              "listed expiry's ATM IV exactly, so it is not a 30-day quantity. "
              "Disagreement below is then extrapolation error, not construction "
              "error. Fix by listing an expiry beyond 30 DTE "
              "(config scraping.nifty_weekly_expiries).")

    print("\n-- levels --")
    print(f"  india_vix  mean {j['india_vix'].mean():6.2f}  sd {j['india_vix'].std():5.2f}")
    print(f"  cmt30_iv   mean {j['cmt30_iv'].mean():6.2f}  sd {j['cmt30_iv'].std():5.2f}")
    p, s, k = _corr(j["india_vix"], j["cmt30_iv"])
    print(f"  correlation (levels)   pearson {p:+.3f}  spearman {s:+.3f}  n={k}")

    d_vix = j["india_vix"].diff()
    d_cmt = j["cmt30_iv"].diff()
    p, s, k = _corr(d_vix, d_cmt)
    print(f"  correlation (changes)  pearson {p:+.3f}  spearman {s:+.3f}  n={k}")

    print("\n-- spread (india_vix - cmt30) --")
    sp = j["spread"]
    print(f"  mean {sp.mean():+.2f}  sd {sp.std():.2f}  "
          f"min {sp.min():+.2f}  max {sp.max():+.2f}")

    x = np.arange(n, dtype=float)
    lr = stats.linregress(x, sp.values)
    print(f"  raw OLS trend {lr.slope:+.4f} vol pts/session  "
          f"(t={lr.slope / lr.stderr:+.2f}, p={lr.pvalue:.3f})")

    t_ctrl, p_ctrl = _trend_controlling_for_level(j)
    print(f"  level-controlled trend (spread ~ 1 + front_atm_iv + t): "
          f"{p_ctrl:+.5f}/session (t={t_ctrl:+.2f})")
    if abs(t_ctrl) < 2.0:
        print("  verdict: STABLE -- no trend once the vol level is controlled for. "
              "The raw trend is a level effect: the spread widens as vol falls.")
    else:
        print("  verdict: DRIFTING -- trend survives the level control, investigate.")

    if n >= 10:
        roll = sp.rolling(10).mean().dropna()
        print(f"  rolling-10 mean of spread: {roll.min():+.2f} .. {roll.max():+.2f} "
              f"(range {roll.max() - roll.min():.2f})")

    print("\n-- spread vs smile shape (back expiry, tenor-matched) --")
    for feat in ("back_curv", "back_rr25", "back_bf25"):
        p, s, k = _corr(sp, j[feat])
        print(f"  spread vs {feat:<10} pearson {p:+.3f}  spearman {s:+.3f}  n={k}")
    p, _, _ = _corr(sp, j["front_atm_iv"])
    print(f"  spread vs {'front_atm_iv':<10} pearson {p:+.3f}   (level effect)")


def _trend_controlling_for_level(j: pd.DataFrame) -> tuple:
    """OLS t-stat and coefficient on time, controlling for the ATM vol level."""
    n = len(j)
    ok = j["spread"].notna() & j["front_atm_iv"].notna()
    y = j.loc[ok, "spread"].to_numpy(float)
    X = np.column_stack([np.ones(ok.sum()),
                         j.loc[ok, "front_atm_iv"].to_numpy(float),
                         np.arange(n, dtype=float)[ok.to_numpy()]])
    if len(y) <= X.shape[1]:
        return float("nan"), float("nan")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    se = np.sqrt((resid ** 2).sum() / dof * np.linalg.inv(X.T @ X)[2, 2])
    return float(beta[2] / se), float(beta[2])


def main() -> None:
    if not os.path.exists(VIX_PATH):
        sys.exit(f"missing {VIX_PATH} -- run scripts/fetch_underlying.py first")
    j = load_joined()
    if j.empty:
        sys.exit("no overlapping sessions between india_vix and signals/_daily.csv")
    report(j)


if __name__ == "__main__":
    main()
