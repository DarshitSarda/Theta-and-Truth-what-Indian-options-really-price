"""Do option-chain features, measured at the close, predict a NIFTY bounce?

Question
--------
When NIFTY is already in a pullback, do implied vol, skew, term structure,
PCR, OI walls or max pain -- as they stand at the close of day t -- tell us
anything about the return over the next 5-10 sessions that price alone does
not?

Data
----
Bhavcopy-derived features (data/processed/bhavcopy, 2008 onwards) plus
Yahoo NIFTY/BANKNIFTY OHLC and India VIX. Bhavcopy is published in the evening,
so a trade can only be placed at the NEXT open: every forward return here is
open[t+1] -> close[t+h]. No close execution is assumed.

Pre-declared design (fixed before looking at any result)
--------------------------------------------------------
Sample     "Stress" days: close at least 3% below its 20-session high. That is
           where a bounce call is actually made. Unconditional results are
           printed as secondary.
Outcomes   fwd5, fwd10 = open[t+1] -> close[t+5 / t+10], in %.
Features   Each raw feature is converted to its percentile within the 250
           STRICTLY PRIOR sessions (no lookahead, and a 2008 IV of 40 is
           compared with 2008, not with 2017). Changes of front-expiry
           features are only computed when the front expiry did not roll in
           between, so expiry rolls do not masquerade as signals.
Statistic  Spearman IC between feature percentile and forward return inside
           the sample, plus the partial IC after regressing the forward return
           on price-only state (drawdown depth, 5d return, trailing RV20).
           The partial IC is the one that matters: a feature that only restates
           "we already fell a lot" adds nothing.
P-values   Circular-shift placebo: the feature series is rotated against the
           price series by a random offset (>= 250 sessions), preserving both
           series' autocorrelation and clustering. p = share of |IC_placebo|
           >= |IC_observed|. Overlapping forward windows are therefore handled
           honestly, unlike a naive t-test.
Split      Discovery 2008-01-01 .. 2017-12-31, holdout 2018-01-01 .. end.
Pass rule  A feature/horizon PASSES only if
             (a) discovery partial IC has p < 0.05, AND
             (b) holdout partial IC has the SAME SIGN and p < 0.05 / K,
           where K = number of feature x horizon pairs (Bonferroni).
           BANKNIFTY is a replication check, not a second shot at passing.

Usage:  python scripts/bounce_event_study.py [--symbol NIFTY] [--root ...]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UND = os.path.join(PROJECT_ROOT, "data", "raw", "underlying")
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "data", "processed", "bhavcopy")

TRAIL = 250
MIN_TRAIL = 120
STRESS_DD = -3.0
HORIZONS = (5, 10)
BOUNCE_PCT = {5: 2.0, 10: 3.0}
SPLIT = "2018-01-01"
N_PLACEBO = 1000
MIN_SHIFT = 250
RNG = np.random.default_rng(20260923)

CONTROLS = ["dd20_p", "ret5_p", "rv20_p"]
FEATURES = {
    "front_atm_iv":    "front ATM IV level",
    "d5_cmt30_iv":     "5d change in 30-day constant-maturity IV",
    "iv_minus_rv20":   "front IV minus trailing RV20 (VRP proxy)",
    "rr25":            "25d risk reversal (neg = puts rich)",
    "d5_rr25":         "5d change in RR25 (same expiry)",
    "bf25":            "25d butterfly (smile curvature)",
    "term_slope":      "back - front expiry ATM IV (neg = inverted)",
    "pcr_all":         "PCR OI, all expiries",
    "d5_pcr_all":      "5d change in all-expiry PCR",
    "put_oi_chg1":     "1d % change in all-expiry put OI",
    "put_wall_dist":   "(spot - front put wall) / spot",
    "maxpain_dist":    "(front max pain - spot) / spot",
    "vix_d5":          "5d change in India VIX (benchmark)",
}


# ---------------------------------------------------------------- data

def trailing_pctile(s: pd.Series, window: int = TRAIL, min_obs: int = MIN_TRAIL) -> pd.Series:
    """Percentile of today's value among up to `window` strictly prior non-NaN values."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        prior = v[max(0, i - window):i]
        prior = prior[~np.isnan(prior)]
        if len(prior) >= min_obs:
            out[i] = (prior < v[i]).mean() * 100.0 + (prior == v[i]).mean() * 50.0
    return pd.Series(out, index=s.index)


def same_expiry_diff(val: pd.Series, expiry: pd.Series, k: int) -> pd.Series:
    d = val - val.shift(k)
    return d.where(expiry == expiry.shift(k))


def load_panel(symbol: str, root: str) -> pd.DataFrame:
    px = pd.read_csv(os.path.join(UND, f"{symbol.lower()}.csv"))[["date", "open", "high", "low", "close"]]
    vix = pd.read_csv(os.path.join(UND, "india_vix.csv"))[["date", "close"]].rename(columns={"close": "vix"})
    sig = pd.read_csv(os.path.join(root, "vol_surface", "signals", "_daily.csv"))
    vrp = pd.read_csv(os.path.join(root, "vol_surface", "vrp", "_daily.csv"))
    dm = pd.read_csv(os.path.join(root, "daily_metrics_all.csv"))

    sig = sig[sig["symbol"] == symbol][["date", "front_expiry", "front_atm_iv", "front_rr25",
                                        "front_bf25", "term_slope_raw", "cmt30_iv", "cmt30_in_range"]]
    # only trailing columns from VRP: rv_fwd* would be lookahead
    vrp = vrp[vrp["symbol"] == symbol][["date", "trailing_rv20_cc", "iv_minus_trail20"]]
    dm = dm[dm["symbol"] == symbol]
    agg = dm.groupby("date")[["total_call_oi", "total_put_oi"]].sum().reset_index()
    front = (dm[dm["is_front_expiry"].astype(str).str.lower() == "true"]
             .sort_values("dte").drop_duplicates("date")[["date", "put_wall", "max_pain_strike"]])

    df = (px.merge(vix, on="date", how="left").merge(sig, on="date", how="left")
            .merge(vrp, on="date", how="left").merge(agg, on="date", how="left")
            .merge(front, on="date", how="left"))
    df = df.dropna(subset=["open", "close"]).sort_values("date").reset_index(drop=True)
    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    d["dd20"] = (c / c.rolling(20, min_periods=20).max() - 1.0) * 100.0
    d["ret5"] = (c / c.shift(5) - 1.0) * 100.0
    d["rv20"] = np.log(c).diff().rolling(20, min_periods=20).std() * np.sqrt(252) * 100.0
    for h in HORIZONS:
        d[f"fwd{h}"] = (c.shift(-h) / d["open"].shift(-1) - 1.0) * 100.0

    # cmt7 needs an expiry inside 7 days, which the monthly-only era rarely has
    cmt30 = d["cmt30_iv"].where(d["cmt30_in_range"].astype(str).str.lower() == "true")
    pcr = d["total_put_oi"] / d["total_call_oi"].replace(0, np.nan)
    raw = pd.DataFrame({
        "front_atm_iv": d["front_atm_iv"],
        # weekly fronts roll almost every 5 sessions, so a same-expiry diff is mostly NaN
        "d5_cmt30_iv": cmt30 - cmt30.shift(5),
        "iv_minus_rv20": d["iv_minus_trail20"],
        "rr25": d["front_rr25"],
        "d5_rr25": same_expiry_diff(d["front_rr25"], d["front_expiry"], 5),
        "bf25": d["front_bf25"],
        "term_slope": d["term_slope_raw"],
        "pcr_all": pcr,
        "d5_pcr_all": pcr - pcr.shift(5),
        "put_oi_chg1": d["total_put_oi"].pct_change(fill_method=None) * 100.0,
        "put_wall_dist": (c - d["put_wall"]) / c * 100.0,
        "maxpain_dist": (d["max_pain_strike"] - c) / c * 100.0,
        "vix_d5": d["vix"] - d["vix"].shift(5),
    })
    for k in FEATURES:
        d[k] = raw[k]
        d[f"{k}_p"] = trailing_pctile(raw[k])
    for k in ("dd20", "ret5", "rv20"):
        d[f"{k}_p"] = trailing_pctile(d[k])
    d["stress"] = d["dd20"] <= STRESS_DD
    return d


# ---------------------------------------------------------------- stats

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 20:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def residualise(d: pd.DataFrame, y: str, mask: np.ndarray) -> np.ndarray:
    """y minus its OLS fit on price-only controls, fitted within `mask`."""
    out = np.full(len(d), np.nan)
    X = d[CONTROLS].to_numpy(float)
    Y = d[y].to_numpy(float)
    ok = mask & ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
    if ok.sum() < 30:
        return out
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    beta, *_ = np.linalg.lstsq(A, Y[ok], rcond=None)
    out[ok] = Y[ok] - A @ beta
    return out


def ic_with_placebo(f: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    """Observed IC in mask, and circular-shift placebo p-value (two-sided)."""
    ok = mask & ~np.isnan(f) & ~np.isnan(y)
    n = int(ok.sum())
    obs = _spearman(f[ok], y[ok])
    if np.isnan(obs):
        return np.nan, np.nan, n
    N = len(f)
    if N < 2 * MIN_SHIFT + 1:
        return obs, np.nan, n
    shifts = RNG.integers(MIN_SHIFT, N - MIN_SHIFT, N_PLACEBO)
    base = mask & ~np.isnan(y)
    null = np.empty(N_PLACEBO)
    for j, s in enumerate(shifts):
        fs = np.roll(f, s)
        m = base & ~np.isnan(fs)
        null[j] = _spearman(fs[m], y[m])
    null = null[~np.isnan(null)]
    p = float((np.abs(null) >= abs(obs)).mean()) if len(null) else np.nan
    return obs, p, n


def tercile_table(d: pd.DataFrame, feat: str, h: int, mask: np.ndarray) -> tuple[float, float, float, float]:
    """(mean fwd bottom tercile, top tercile, bounce hit-rate bottom, top)."""
    g = d.loc[mask, [f"{feat}_p", f"fwd{h}"]].dropna()
    if len(g) < 30:
        return (np.nan,) * 4
    lo, hi = g[g[f"{feat}_p"] <= 33.3], g[g[f"{feat}_p"] >= 66.7]
    b = BOUNCE_PCT[h]
    return (lo[f"fwd{h}"].mean(), hi[f"fwd{h}"].mean(),
            (lo[f"fwd{h}"] >= b).mean() * 100, (hi[f"fwd{h}"] >= b).mean() * 100)


def run_split(d: pd.DataFrame, cond: np.ndarray, label: str) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        resid = residualise(d, f"fwd{h}", cond)
        y = d[f"fwd{h}"].to_numpy(float)
        for feat in FEATURES:
            f = d[f"{feat}_p"].to_numpy(float)
            ic, p, n = ic_with_placebo(f, y, cond)
            pic, pp, _ = ic_with_placebo(f, resid, cond)
            lo, hi, blo, bhi = tercile_table(d, feat, h, cond)
            rows.append({"split": label, "h": h, "feature": feat, "n": n,
                         "ic": ic, "p": p, "partial_ic": pic, "partial_p": pp,
                         "fwd_lo_terc": lo, "fwd_hi_terc": hi,
                         "bounce%_lo": blo, "bounce%_hi": bhi})
    return pd.DataFrame(rows)


def price_baseline(d: pd.DataFrame, cond: np.ndarray, label: str) -> None:
    """How much do price-only controls explain? The bar option features must clear."""
    for h in HORIZONS:
        y = d[f"fwd{h}"].to_numpy(float)
        parts = []
        for c in CONTROLS:
            ic, p, n = ic_with_placebo(d[c].to_numpy(float), y, cond)
            parts.append(f"{c}: IC {ic:+.3f} (p={p:.3f})")
        base = d.loc[cond, f"fwd{h}"].dropna()
        print(f"  {label} fwd{h}: n={len(base)}, mean {base.mean():+.2f}%, "
              f"P(>= {BOUNCE_PCT[h]}%) {(base >= BOUNCE_PCT[h]).mean()*100:.1f}% | " + "; ".join(parts))


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "processed", "analysis"))
    args = ap.parse_args()

    d = build(load_panel(args.symbol, args.root))
    disc = (d["date"] < SPLIT).to_numpy()
    hold = ~disc
    stress = d["stress"].to_numpy()
    K = len(FEATURES) * len(HORIZONS)

    print(f"{args.symbol}: {len(d)} sessions {d['date'].iloc[0]} .. {d['date'].iloc[-1]}")
    print("feature coverage (share of stress days with a value):")
    cov = {k: d.loc[d["stress"], f"{k}_p"].notna().mean() * 100 for k in FEATURES}
    print("  " + ", ".join(f"{k} {v:.0f}%" for k, v in cov.items()))
    print(f"stress days: discovery {int((stress & disc).sum())}, holdout {int((stress & hold).sum())}")
    print(f"Bonferroni K = {K}; holdout threshold p < {0.05 / K:.4f}\n")

    print("PRICE-ONLY BASELINE (stress days)")
    price_baseline(d, stress & disc, "discovery")
    price_baseline(d, stress & hold, "holdout  ")

    res = pd.concat([run_split(d, stress & disc, "discovery"),
                     run_split(d, stress & hold, "holdout")], ignore_index=True)
    wide = res.pivot_table(index=["feature", "h"], columns="split",
                           values=["partial_ic", "partial_p", "ic"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide["PASS"] = ((wide["partial_p_discovery"] < 0.05)
                    & (np.sign(wide["partial_ic_holdout"]) == np.sign(wide["partial_ic_discovery"]))
                    & (wide["partial_p_holdout"] < 0.05 / K))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\nSTRESS DAYS - PRIMARY RESULT (partial IC = beyond price-only state)")
    show = wide[["feature", "h", "ic_discovery", "partial_ic_discovery", "partial_p_discovery",
                 "ic_holdout", "partial_ic_holdout", "partial_p_holdout", "PASS"]]
    print(show.round(3).to_string(index=False))

    print("\nTERCILES (stress days): mean fwd return % and bounce hit-rate %, low vs high feature percentile")
    terc = res[["split", "h", "feature", "n", "fwd_lo_terc", "fwd_hi_terc", "bounce%_lo", "bounce%_hi"]]
    print(terc.round(2).to_string(index=False))

    print("\nSECONDARY - ALL DAYS (unconditional), partial IC")
    alld = pd.concat([run_split(d, disc, "discovery"), run_split(d, hold, "holdout")])
    w2 = alld.pivot_table(index=["feature", "h"], columns="split",
                          values=["partial_ic", "partial_p"]).reset_index()
    w2.columns = ["_".join(c).strip("_") for c in w2.columns]
    print(w2.round(3).to_string(index=False))

    passed = wide[wide["PASS"]]
    print("\n" + "=" * 80)
    if passed.empty:
        print(f"VERDICT: no option feature passes the pre-declared rule for {args.symbol}.")
    else:
        print(f"VERDICT: {len(passed)} feature/horizon pair(s) pass for {args.symbol}:")
        for _, r in passed.iterrows():
            print(f"  {r['feature']} fwd{int(r['h'])}: partial IC disc {r['partial_ic_discovery']:+.3f}, "
                  f"holdout {r['partial_ic_holdout']:+.3f} (p={r['partial_p_holdout']:.4f})")
    print("=" * 80)

    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, f"bounce_study_{args.symbol.lower()}.csv"), index=False)
    d.to_csv(os.path.join(args.out, f"bounce_panel_{args.symbol.lower()}.csv"), index=False)


if __name__ == "__main__":
    main()
