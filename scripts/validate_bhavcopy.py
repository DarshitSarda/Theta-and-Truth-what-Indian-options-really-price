"""Phase A: can bhavcopy-derived features stand in for the live pipeline's?

The historical backfill is only worth anything if features rebuilt from
bhavcopy match the features the live pipeline computes on the same day. This
script compares the two on every session where we hold both.

Pass criteria are fixed here, BEFORE looking at any comparison, so the bar
cannot drift to fit the result:

  Positioning (OI is an end-of-day stock, so it should match almost exactly)
    * total CE/PE OI within 0.5% on >= 95% of expiry-days
    * max pain, put wall, call wall identical on >= 90% of expiry-days
    * PCR (OI) median |diff| < 0.01
    * composite_v0 correlation > 0.95

  Forward
    * median |F_bhav - F_live| < 10 bps (the two are measured at different
      moments -- a 30-min closing VWAP vs a post-close quote snapshot -- so
      some spot drift is expected)

  Vol (per expiry, DTE >= 3, both no-arb OK)
    * ATM IV    median |diff| < 0.50 vp, p90 < 1.00 vp
    * RR25      median |diff| < 0.30 vp
    * BF25      median |diff| < 0.30 vp
    * day-over-day change in front ATM IV: correlation > 0.90
      (an event study keys on moves, not levels)

Anything failing is reported with its DTE-bucket breakdown so the cause is
visible, rather than just a red flag.

Usage:  python scripts/validate_bhavcopy.py [--root data/processed/bhavcopy]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.chain_parser import load_chain_csv, max_pain  # noqa: E402

LIVE = os.path.join(PROJECT_ROOT, "data", "processed")
KEY = ["symbol", "date", "expiry"]

CRITERIA = {
    "oi_match_rate":        (">=", 0.95),
    "max_pain_match_rate":  (">=", 0.90),
    "put_wall_match_rate":  (">=", 0.90),
    "call_wall_match_rate": (">=", 0.90),
    "pcr_median_abs_diff":  ("<",  0.01),
    "composite_v0_corr":    (">",  0.95),
    "forward_median_bps":   ("<",  10.0),
    "atm_iv_median_abs":    ("<",  0.50),
    "atm_iv_p90_abs":       ("<",  1.00),
    "rr25_median_abs":      ("<",  0.30),
    "bf25_median_abs":      ("<",  0.30),
    "front_iv_change_corr": (">",  0.90),
}


def bucket(dte: pd.Series) -> pd.Series:
    return pd.cut(dte, [-1, 1, 4, 10, 20, 400], labels=["0-1", "2-4", "5-10", "11-20", "21+"])


def live_walls(dm_live: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in dm_live.iterrows():
        src = r["source_file"]
        if not isinstance(src, str):
            continue
        if not os.path.exists(src):
            # source_file is absolute from wherever the project lived when it was scraped
            parts = src.replace("\\", "/").split("/data/raw/", 1)
            src = os.path.join(PROJECT_ROOT, "data", "raw", parts[1]) if len(parts) == 2 else src
            if not os.path.exists(src):
                continue
        ch = load_chain_csv(src)
        piv = ch.pivot_table(index="STRIKE", columns="side", values="OI", aggfunc="sum").fillna(0)
        rows.append({"symbol": r["symbol"], "date": r["date"], "expiry": r["expiry"],
                     "max_pain_strike": max_pain(ch)["max_pain_strike"],
                     "put_wall": float(piv["PE"].idxmax()),
                     "call_wall": float(piv["CE"].idxmax())})
    return pd.DataFrame(rows)


def load_live_daily_metrics() -> pd.DataFrame:
    files = sorted(f for f in os.listdir(os.path.join(LIVE, "daily_metrics")) if f.endswith(".csv"))
    return pd.concat([pd.read_csv(os.path.join(LIVE, "daily_metrics", f)) for f in files],
                     ignore_index=True)


def verdict(name: str, value: float) -> str:
    op, thr = CRITERIA[name]
    ok = {"<": value < thr, ">": value > thr, ">=": value >= thr}[op] if np.isfinite(value) else False
    return f"{'PASS' if ok else 'FAIL'}  {name:<22} {value:>9.4f}  (need {op} {thr})", ok


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(LIVE, "bhavcopy"))
    args = ap.parse_args()
    root = args.root
    results = {}

    # ------------------------------------------------------------ positioning
    section("1. POSITIONING: OI, PCR, walls, max pain, composite_v0")
    live_dm = load_live_daily_metrics()
    bhav_dm = pd.read_csv(os.path.join(root, "daily_metrics_all.csv"))
    dates = sorted(set(live_dm["date"]) & set(bhav_dm["date"]))
    print(f"overlapping sessions: {len(dates)}  ({dates[0]} .. {dates[-1]})")

    lw = live_walls(live_dm[live_dm["date"].isin(dates)])
    live_pos = live_dm.merge(lw, on=KEY, how="left")
    m = live_pos.merge(bhav_dm, on=KEY, suffixes=("_live", "_bhav"))
    print(f"matched expiry-days: {len(m)}")

    ce_ratio = m["total_call_oi_bhav"] / m["total_call_oi_live"]
    pe_ratio = m["total_put_oi_bhav"] / m["total_put_oi_live"]
    oi_ok = ((ce_ratio - 1).abs() < 0.005) & ((pe_ratio - 1).abs() < 0.005)
    results["oi_match_rate"] = oi_ok.mean()
    for col in ("max_pain_strike", "put_wall", "call_wall"):
        key = col.replace("_strike", "") + "_match_rate"
        results[key] = (m[f"{col}_live"] == m[f"{col}_bhav"]).mean()
    results["pcr_median_abs_diff"] = (m["pcr_oi_live"] - m["pcr_oi_bhav"]).abs().median()
    results["composite_v0_corr"] = m["composite_v0_chg_oi_atm_band_live"].corr(
        m["composite_v0_chg_oi_atm_band_bhav"])

    print(f"CE OI ratio bhav/live: median {ce_ratio.median():.5f}  "
          f"[{ce_ratio.min():.4f}, {ce_ratio.max():.4f}]")
    print(f"PE OI ratio bhav/live: median {pe_ratio.median():.5f}  "
          f"[{pe_ratio.min():.4f}, {pe_ratio.max():.4f}]")
    mism = m[~oi_ok][KEY + ["dte_live"]].copy()
    mism["ce_ratio"], mism["pe_ratio"] = ce_ratio[~oi_ok], pe_ratio[~oi_ok]
    if len(mism):
        print(f"\nOI mismatches ({len(mism)}):")
        print(mism.head(15).to_string(index=False))
    walls_miss = m[(m["put_wall_live"] != m["put_wall_bhav"]) |
                   (m["call_wall_live"] != m["call_wall_bhav"]) |
                   (m["max_pain_strike_live"] != m["max_pain_strike_bhav"])]
    if len(walls_miss):
        print(f"\nwall / max-pain disagreements ({len(walls_miss)}):")
        print(walls_miss[KEY + ["dte_live", "max_pain_strike_live", "max_pain_strike_bhav",
                                "put_wall_live", "put_wall_bhav",
                                "call_wall_live", "call_wall_bhav"]].head(15).to_string(index=False))

    # ------------------------------------------------------------ forward
    section("2. FORWARD (parity-recovered) and spot")
    lh = pd.read_csv(os.path.join(LIVE, "vol_surface", "_health_all.csv"))
    bh = pd.read_csv(os.path.join(root, "vol_surface", "_health_all.csv"))
    bh = bh[bh["skipped"].isna()]
    f = lh.merge(bh, on=KEY, suffixes=("_live", "_bhav"))
    f["fwd_bps"] = (f["forward_bhav"] / f["forward_live"] - 1) * 1e4
    f["spot_bps"] = (f["spot_bhav"] / f["spot_live"] - 1) * 1e4
    f["fwd_minus_spot_bps"] = f["fwd_bps"] - f["spot_bps"]
    results["forward_median_bps"] = f["fwd_bps"].abs().median()
    print(f"matched expiry-days: {len(f)}")
    print(f"|F_bhav - F_live|:  median {f['fwd_bps'].abs().median():.1f} bps   "
          f"p90 {f['fwd_bps'].abs().quantile(.9):.1f} bps")
    print(f"spot used, bhav vs live: median {f['spot_bps'].abs().median():.1f} bps")
    print(f"forward gap net of spot gap: median |.| "
          f"{f['fwd_minus_spot_bps'].abs().median():.1f} bps")
    fut = f.dropna(subset=["fut_close"])
    if len(fut):
        fb = (fut["forward_bhav"] / fut["fut_close"] - 1) * 1e4
        fl = (fut["forward_live"] / fut["fut_close"] - 1) * 1e4
        print(f"vs futures close ({len(fut)} expiries with a future): "
              f"bhav median |.| {fb.abs().median():.1f} bps, live {fl.abs().median():.1f} bps")
    f["dte_bucket"] = bucket(f["dte_live"])
    print("\nby DTE bucket (median |F diff| bps):")
    print(f.groupby("dte_bucket", observed=True)["fwd_bps"]
          .agg(lambda s: s.abs().median()).round(1).to_string())
    print(f"\nskipped by bhav pipeline: "
          f"{pd.read_csv(os.path.join(root, 'vol_surface', '_health_all.csv'))['skipped'].notna().sum()}"
          f" expiry-days (includes dte<=1 by design)")

    # ------------------------------------------------------------ vol
    section("3. VOL SIGNALS per expiry (DTE >= 3, both smiles no-arb OK)")
    ls = pd.read_csv(os.path.join(LIVE, "vol_surface", "signals", "_signals_all.csv"))
    bs = pd.read_csv(os.path.join(root, "vol_surface", "signals", "_signals_all.csv"))
    v = ls.merge(bs, on=KEY, suffixes=("_live", "_bhav"))
    v = v[(v["dte_live"] >= 3) & v["no_arb_ok_live"] & v["no_arb_ok_bhav"]]
    print(f"matched expiry-days: {len(v)}")
    for col in ("atm_iv", "rr25", "bf25", "atm_skew"):
        v[f"{col}_diff"] = v[f"{col}_bhav"] - v[f"{col}_live"]
    results["atm_iv_median_abs"] = v["atm_iv_diff"].abs().median()
    results["atm_iv_p90_abs"] = v["atm_iv_diff"].abs().quantile(0.9)
    results["rr25_median_abs"] = v["rr25_diff"].abs().median()
    results["bf25_median_abs"] = v["bf25_diff"].abs().median()

    print(f"\n{'':<10}{'median diff':>13}{'median |d|':>12}{'p90 |d|':>10}{'corr lvl':>10}")
    for col in ("atm_iv", "rr25", "bf25", "atm_skew"):
        d = v[f"{col}_diff"]
        print(f"{col:<10}{d.median():>+13.3f}{d.abs().median():>12.3f}"
              f"{d.abs().quantile(.9):>10.3f}{v[col + '_live'].corr(v[col + '_bhav']):>10.3f}")

    v["dte_bucket"] = bucket(v["dte_live"])
    print("\nby DTE bucket (median |diff|, vol points):")
    tab = v.groupby("dte_bucket", observed=True).agg(
        n=("atm_iv_diff", "size"),
        atm_iv=("atm_iv_diff", lambda s: s.abs().median()),
        atm_iv_bias=("atm_iv_diff", "median"),
        rr25=("rr25_diff", lambda s: s.abs().median()),
        bf25=("bf25_diff", lambda s: s.abs().median()))
    print(tab.round(3).to_string())

    # front ATM IV, day-over-day changes -- what an event study keys on
    ld = pd.read_csv(os.path.join(LIVE, "vol_surface", "signals", "_daily.csv"))
    bd = pd.read_csv(os.path.join(root, "vol_surface", "signals", "_daily.csv"))
    cols = ["symbol", "date", "front_expiry", "front_atm_iv", "front_rr25", "cmt30_iv"]
    dd = ld[cols].merge(bd[cols], on=["symbol", "date"], suffixes=("_live", "_bhav"))
    same_front = dd["front_expiry_live"] == dd["front_expiry_bhav"]
    print(f"\ndaily rows matched: {len(dd)}; same front expiry chosen: {same_front.mean():.0%}")
    chg = []
    for sym, g in dd.sort_values("date").groupby("symbol"):
        g = g.copy()
        # a change is only meaningful when both sides kept the same front expiry
        same = (g["front_expiry_live"] == g["front_expiry_bhav"]) & \
               (g["front_expiry_live"] == g["front_expiry_live"].shift())
        g["d_live"] = g["front_atm_iv_live"].diff()
        g["d_bhav"] = g["front_atm_iv_bhav"].diff()
        chg.append(g[same])
    chg = pd.concat(chg).dropna(subset=["d_live", "d_bhav"])
    results["front_iv_change_corr"] = chg["d_live"].corr(chg["d_bhav"])
    print(f"front ATM IV daily change: corr {results['front_iv_change_corr']:.3f} "
          f"(n={len(chg)})")
    print(f"cmt30 IV level: median |diff| "
          f"{(dd['cmt30_iv_bhav'] - dd['cmt30_iv_live']).abs().median():.3f} vp, "
          f"corr {dd['cmt30_iv_live'].corr(dd['cmt30_iv_bhav']):.3f}")

    # ------------------------------------------------------------ verdict
    section("PHASE A VERDICT (criteria fixed before comparison)")
    oks = []
    for name in CRITERIA:
        line, ok = verdict(name, float(results.get(name, np.nan)))
        oks.append(ok)
        print("  " + line)
    print(f"\n  {sum(oks)}/{len(oks)} criteria passed")


if __name__ == "__main__":
    main()
