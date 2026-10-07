"""NIFTY vs India VIX: is the negative correlation tradeable, and if not, what is?

Observation: when NIFTY rises, India VIX usually falls. True, strong, and known
(the "leverage effect", present in every equity index). The question is whether
anything can be DONE with it. Three distinct things get conflated:

  1. CONTEMPORANEOUS correlation. corr(nifty_ret[t], dvix[t]). Strong and
     negative. Not tradeable by itself: to know today's VIX move you need
     today's NIFTY move, which you only have once the day is over. Measuring it
     is still worth doing -- it sets the baseline the other two tests need.

  2. LEAD-LAG. Does dvix[t] say anything about nifty_ret[t+k]? THIS is the
     tradeable version. If the correlation is purely same-day, there is no
     directional trade in it.

  3. DIVERGENCE from the relationship. If NIFTY rallies 1% and VIX barely
     falls, someone is still paying for protection. We fit the expected VIX
     move from a TRAILING regression, take the residual, and test whether the
     residual leads returns. This is the only genuinely promising angle and the
     main reason for this script.

Plus two things the correlation implies even though they are not directional
signals: its ASYMMETRY (VIX rises harder on down days than it falls on up days,
which is what makes short vol a carry trade) and its STABILITY (a breakdown in
the usual correlation is itself a candidate warning sign).

No lookahead: every regression beta, residual scale, and percentile is computed
from STRICTLY PRIOR sessions and shifted before use, so any signal at date t is
computable at the close of t. Forward windows are de-clustered and judged
against the regime-matched placebo from vix_spike_followup.

Usage:  python scripts/vix_nifty_correlation.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from vix_spike_followup import placebo_p
from vix_spike_meanrev import (HORIZONS, TRAIL_WINDOW, build_features,
                               declustered_mask, load_panel, summarize)

BETA_WINDOW = 250     # trailing sessions for the dvix-on-return regression
CORR_WINDOW = 20      # trailing sessions for the rolling correlation


def add_divergence(d: pd.DataFrame) -> pd.DataFrame:
    """Residual of dvix[t] from a trailing regression on nifty_ret[t].

    beta/alpha come from the window ending at t-1 (shifted), so only the
    contemporaneous return and VIX change of day t are used at date t -- both
    known at the close.
    """
    x, y = d["nifty_ret"], d["vix_chg_abs"]
    cov = x.rolling(BETA_WINDOW).cov(y)
    var = x.rolling(BETA_WINDOW).var()
    beta = (cov / var).shift(1)
    alpha = (y.rolling(BETA_WINDOW).mean() - (cov / var) * x.rolling(BETA_WINDOW).mean()).shift(1)

    d["vix_beta"] = beta
    d["dvix_expected"] = alpha + beta * x
    d["dvix_resid"] = y - d["dvix_expected"]
    # scale the residual by its own trailing dispersion, prior days only
    mu = d["dvix_resid"].rolling(BETA_WINDOW).mean().shift(1)
    sd = d["dvix_resid"].rolling(BETA_WINDOW).std().shift(1)
    d["resid_z"] = (d["dvix_resid"] - mu) / sd
    d["roll_corr"] = x.rolling(CORR_WINDOW).corr(y)
    return d


def block(d: pd.DataFrame, signal: pd.Series, label: str,
          exclude: pd.Series | None = None, horizons=HORIZONS) -> None:
    if exclude is None:
        exclude = signal
    print(f"\n  {label}   (raw signal days: {int(signal.sum())})")
    print(f"    {'k':>3} {'n_ep':>5} {'mean%':>8} {'win%':>7} {'uncond%':>8} "
          f"{'diff':>7} {'placebo p':>10}")
    for k in horizons:
        col = f"cc{k}"
        idx = d.index[declustered_mask(signal, k)]
        st = summarize(d.loc[idx, col].to_numpy(float))
        if st["n"] == 0:
            print(f"    {k:>3} {0:>5}   (no episodes)")
            continue
        unc = summarize(d[col].to_numpy(float))
        _, p, _ = placebo_p(d, idx, col, exclude)
        pstr = f"{p:.3f}" if not np.isnan(p) else "n/a"
        print(f"    {k:>3} {st['n']:>5} {st['mean']:>+8.2f} {st['win']:>7.1f} "
              f"{unc['mean']:>+8.2f} {st['mean'] - unc['mean']:>+7.2f} {pstr:>10}")


def main() -> None:
    d = add_divergence(build_features(load_panel()))
    full = d.dropna(subset=["nifty_ret", "vix_chg_abs"])

    print("=" * 94)
    print("NIFTY vs INDIA VIX: CORRELATION, LEAD-LAG, AND DIVERGENCE")
    print(f"panel {len(full)} sessions {full.date.iloc[0]} .. {full.date.iloc[-1]}")
    print("=" * 94)

    # ---------------------------------------------------------------- A
    print("\n" + "=" * 94)
    print("A. THE CONTEMPORANEOUS RELATIONSHIP (same day -- NOT tradeable)")
    print("=" * 94)
    r_p = full["nifty_ret"].corr(full["vix_chg_abs"])
    r_s = full["nifty_ret"].corr(full["vix_chg_abs"], method="spearman")
    sl, ic, rv, pv, se = stats.linregress(full["nifty_ret"], full["vix_chg_abs"])
    print(f"  Pearson  corr(nifty_ret, dvix)  = {r_p:+.3f}")
    print(f"  Spearman corr(nifty_ret, dvix)  = {r_s:+.3f}")
    print(f"  OLS: dvix = {ic:+.3f} {sl:+.3f} * nifty_ret%     R^2 = {rv**2:.3f}")
    print(f"  -> a +1% NIFTY day comes with a {sl:+.2f}pt VIX move on average")
    up, dn = full[full.nifty_ret > 0], full[full.nifty_ret < 0]
    print(f"\n  Sign agreement (opposite directions): "
          f"{(np.sign(full.nifty_ret) != np.sign(full.vix_chg_abs)).mean()*100:.1f}% of days")
    print(f"    NIFTY up   ({len(up)} days): VIX fell on {(up.vix_chg_abs < 0).mean()*100:.1f}%")
    print(f"    NIFTY down ({len(dn)} days): VIX rose on {(dn.vix_chg_abs > 0).mean()*100:.1f}%")
    print("\n  Your observation is confirmed and it is very stable by period:")
    per = full.assign(yr=pd.to_datetime(full.date).dt.year // 3 * 3)
    for y, g in per.groupby("yr"):
        print(f"    {y}-{y+2}: corr {g.nifty_ret.corr(g.vix_chg_abs):+.3f}  ({len(g)} days)")
    print("\n  Why this alone cannot be traded: both terms are dated t. Knowing")
    print("  that VIX falls when NIFTY rises requires knowing NIFTY already rose.")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 94)
    print("B. ASYMMETRY (not a directional signal, but it is why short vol pays)")
    print("=" * 94)
    b_up = stats.linregress(up.nifty_ret, up.vix_chg_abs)
    b_dn = stats.linregress(dn.nifty_ret, dn.vix_chg_abs)
    print(f"  beta on UP   days: {b_up.slope:+.3f} pt per 1% move")
    print(f"  beta on DOWN days: {b_dn.slope:+.3f} pt per 1% move")
    print(f"  ratio (down/up): {abs(b_dn.slope / b_up.slope):.2f}x")
    for lo, hi, name in [(1.0, 99, "moves > +1%"), (-99, -1.0, "moves < -1%")]:
        m = full[(full.nifty_ret > lo) & (full.nifty_ret < hi)] if lo > 0 \
            else full[(full.nifty_ret > lo) & (full.nifty_ret < hi)]
        print(f"  {name:<14} (n={len(m):>4}): mean dvix {m.vix_chg_abs.mean():+.3f}pt, "
              f"median {m.vix_chg_abs.median():+.3f}pt")
    print("\n  VIX reacts harder to declines than to rallies of equal size. That")
    print("  convexity is priced, which is the structural source of the variance")
    print("  risk premium we already measure -- not a timing signal.")

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 94)
    print("C. LEAD-LAG: is ANY of the correlation predictive rather than same-day?")
    print("=" * 94)
    print("  corr(dvix[t], nifty_ret[t+k]).  k<0 means VIX lags the index.")
    print(f"  {'k':>4} {'corr':>9} {'p-value':>10}   interpretation")
    for k in (-3, -2, -1, 0, 1, 2, 3, 5):
        a = full["vix_chg_abs"]
        b = full["nifty_ret"].shift(-k)
        sub = pd.concat([a, b], axis=1).dropna()
        c, p = stats.pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
        tag = ("SAME DAY (mechanical)" if k == 0 else
               "VIX would PREDICT index" if k > 0 else "index predicts VIX")
        star = " *" if p < 0.05 else ""
        print(f"  {k:>4} {c:>+9.3f} {p:>10.4f}   {tag}{star}")
    print("\n  Also the reverse direction, corr(nifty_ret[t], dvix[t+k]):")
    for k in (1, 2, 3):
        sub = pd.concat([full["nifty_ret"], full["vix_chg_abs"].shift(-k)],
                        axis=1).dropna()
        c, p = stats.pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
        print(f"  k={k}: corr {c:+.3f} (p={p:.4f})")

    # ---------------------------------------------------------------- D
    d2 = d.dropna(subset=["resid_z", "cc1"]).reset_index(drop=True)
    print("\n" + "=" * 94)
    print("D. DIVERGENCE SIGNAL -- the promising version of your idea")
    print(f"   residual of dvix vs a trailing {BETA_WINDOW}-session regression on the")
    print("   same-day NIFTY return, standardised on trailing dispersion.")
    print("   resid_z > 0  =  VIX HIGHER than the index move justifies (sticky fear)")
    print("   resid_z < 0  =  VIX LOWER  than the index move justifies (complacency)")
    print("=" * 94)
    print(f"   usable sessions: {len(d2)}  {d2.date.iloc[0]} .. {d2.date.iloc[-1]}")
    for lab, sig in [
        ("resid_z >= +2  (VIX far too high for the move)", d2.resid_z >= 2),
        ("resid_z >= +1.5", d2.resid_z >= 1.5),
        ("resid_z <= -2  (VIX far too low  for the move)", d2.resid_z <= -2),
        ("resid_z <= -1.5", d2.resid_z <= -1.5),
    ]:
        block(d2, sig, lab)
    print("\n   Conjunction with direction (does 'sticky VIX on an UP day' warn?):")
    block(d2, (d2.resid_z >= 1.5) & (d2.nifty_ret > 0),
          "VIX sticky-high on a NIFTY UP day")
    block(d2, (d2.resid_z <= -1.5) & (d2.nifty_ret < 0),
          "VIX oddly calm on a NIFTY DOWN day")

    # ---------------------------------------------------------------- E
    d3 = d.dropna(subset=["roll_corr", "cc1"]).reset_index(drop=True)
    print("\n" + "=" * 94)
    print(f"E. CORRELATION BREAKDOWN: does a broken {CORR_WINDOW}d corr warn of trouble?")
    print("=" * 94)
    print(f"   rolling corr distribution: p10 {d3.roll_corr.quantile(.10):+.2f}  "
          f"median {d3.roll_corr.median():+.2f}  p90 {d3.roll_corr.quantile(.90):+.2f}")
    block(d3, d3.roll_corr > -0.2, "corr decayed to > -0.2 (relationship broken)")
    block(d3, d3.roll_corr > 0.0, "corr turned POSITIVE (VIX rising with index)")
    block(d3, d3.roll_corr < -0.85, "corr very tight < -0.85 (normal risk-on/off)")

    # ---------------------------------------------------------------- F
    print("\n" + "=" * 94)
    print("F. WHERE WE STAND NOW")
    print("=" * 94)
    tail = d.dropna(subset=["resid_z"]).tail(8)[
        ["date", "close", "nifty_ret", "vix", "vix_chg_abs", "dvix_expected",
         "dvix_resid", "resid_z", "roll_corr"]]
    print(tail.to_string(index=False,
                         float_format=lambda v: f"{v:>8.2f}"))
    print("=" * 94)


if __name__ == "__main__":
    main()
