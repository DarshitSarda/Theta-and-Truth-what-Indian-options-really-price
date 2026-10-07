"""Validation gates for the Stage 0 contract history (pass rules fixed before running).

G1  integrity      every cached session present; no duplicate contract-days
G2  spot           NSE vs Yahoo spot p99 |diff| < 0.01% (UDiFF era); >= 99.5% of rows have spot
G3  forwards       carry-forward vs parity forward: median ATM IV effect < 0.25 vol pts (dte <= 90)
G3b local parity   panel's own parity forward vs vol-layer parity: median |diff| < 0.02%, p90 < 0.05%
G4  expiry         S_T equals NSE's expiry-day settle where NSE publishes it (>= 99%);
                   every expired contract has a payoff
G5  OI continuity  oi_shares[t] - chg_oi_shares[t] == oi_shares[t-1] on >= 99% of consecutive pairs
G6  parity         median |C - P - DF(F-K)| < 10 bp of spot on traded pairs (dte >= 7)
G7  vs pipeline    corr(panel IV, bhavcopy vol-layer IV) > 0.99 on the same OTM points
G8  India VIX      corr(30d ATM IV proxy, India VIX) > 0.95
G9  live chains    median |IV(live mid) - panel IV| < 0.5 vol pts near ATM (2026 overlap)
G10 rate           repo rate +/-2pp moves IV by < 0.25 vol pts at p99 (dte <= 90)
plus descriptive checks: liquidity, forward sources, famous days, example contracts.

Writes data/processed/contracts/_validation.txt
"""
from __future__ import annotations

import glob
import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import contracts as ct  # noqa: E402
from src.chain_parser import load_chain_csv  # noqa: E402

ROOT = os.path.join(PROJECT_ROOT, "data", "processed", "contracts")
PANEL = os.path.join(ROOT, "panel")
CACHE = os.path.join(PROJECT_ROOT, "data", "raw", "bhavcopy")
BC_VOL = os.path.join(PROJECT_ROOT, "data", "processed", "bhavcopy", "vol_surface")
RESULTS: list[tuple[str, bool, str]] = []

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def gate(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def sec(t: str) -> None:
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def load(sym: str, cols: list[str], filters=None) -> pd.DataFrame:
    f = [("symbol", "=", sym)] + (filters or [])
    df = pd.read_parquet(PANEL, columns=cols, filters=f)
    df["symbol"] = sym
    return df


def year_of(s) -> pd.Series:
    return pd.Series(pd.to_datetime(s)).dt.year.to_numpy()


def g1_integrity():
    sec("G1 integrity")
    sessions = set(pd.Timestamp(d) for d in ct.cached_sessions(CACHE))
    seen, dups, rows = set(), 0, {}
    for sym in ct.SYMBOLS:
        p = load(sym, ["date", "expiry", "strike", "side"])
        dups += int(p.duplicated(["date", "expiry", "strike", "side"]).sum())
        seen |= set(p["date"].unique())
        rows[sym] = p.groupby(year_of(p["date"])).size()
    print(pd.DataFrame(rows).fillna(0).astype(int).to_string())
    missing = sorted(sessions - set(pd.to_datetime(list(seen))))
    # independent calendar: every Yahoo NIFTY trading day must have a bhavcopy, unless
    # NSE's archive returned 404 for it (a .missing marker from the downloader)
    y = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "underlying", "nifty.csv"), usecols=["date"])
    ydays = set(pd.to_datetime(y["date"])[lambda s: s >= min(sessions)])
    no_file = sorted(ydays - sessions)
    marked = {pd.Timestamp(os.path.basename(m)[3:11]) for m in glob.glob(os.path.join(CACHE, "*", "fo_*.missing"))}
    unexplained = [d for d in no_file if d not in marked]
    print(f"Yahoo trading days without a bhavcopy: {[str(d.date()) for d in no_file]} "
          f"(NSE archive 404: {[str(d.date()) for d in no_file if d in marked]})")
    gate("G1 integrity", dups == 0 and not missing and not unexplained,
         f"{len(seen)} sessions with rows of {len(sessions)} cached; missing {len(missing)} "
         f"{[str(m.date()) for m in missing[:5]]}; duplicate contract-days {dups}; "
         f"Yahoo days not fetched {[str(d.date()) for d in unexplained]}; NSE-archive gaps {len(no_file) - len(unexplained)}")


def g2_spot():
    sec("G2 spot")
    yahoo = {s: pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "underlying", f"{s.lower()}.csv"),
                            usecols=["date", "close"]).set_index("date")["close"] for s in ct.SYMBOLS}
    diffs, cover = [], []
    for sym in ct.SYMBOLS:
        p = load(sym, ["date", "spot", "spot_source"]).drop_duplicates("date")
        cover.append(p["spot"].notna().mean())
        u = p[p["spot_source"] == "nse"].copy()
        u["y"] = u["date"].dt.strftime("%Y-%m-%d").map(yahoo[sym])
        diffs.append(((u["y"] / u["spot"] - 1) * 100).abs().dropna())
        print(f"{sym}: sessions {len(p)}, spot sources {p['spot_source'].value_counts().to_dict()}")
    d = pd.concat(diffs)
    p99 = d.quantile(0.99)
    gate("G2 spot", p99 < 0.01 and min(cover) >= 0.995,
         f"NSE vs Yahoo |diff| p99 {p99:.5f}% max {d.max():.4f}% (n={len(d)}); "
         f"sessions with spot {min(cover):.2%}")


def g3_forwards():
    sec("G3 forwards")
    cols = ["date", "expiry", "dte", "spot", "forward", "fwd_source", "contracts", "strike", "side",
            "mark", "df", "T", "iv", "log_moneyness"]
    effects = []
    for sym in ct.SYMBOLS:
        p = load(sym, cols)
        e = p.drop_duplicates(["date", "expiry"])
        w = p[p["contracts"] >= ct.MIN_TRADED]
        print(f"\n{sym} forward source share - by expiry-day / by traded contract-day:")
        a = e.groupby(year_of(e["date"]))["fwd_source"].value_counts(normalize=True).unstack().fillna(0)
        b = w.groupby(year_of(w["date"]))["fwd_source"].value_counts(normalize=True).unstack().fillna(0)
        print(pd.concat({"expiries": a, "traded rows": b}, axis=1).round(2).to_string())
        # carry accuracy: rebuild the day's carry without each parity expiry (leave-one-out)
        src = e[e["fwd_source"].isin(["parity", "futures"]) & (e["dte"] >= 7)].copy()
        src["c"] = np.log(src["forward"] / src["spot"]) / src["T"]
        src = src[src["date"].map(src.groupby("date").size()) >= 3]

        def loo(g):
            c = g["c"].to_numpy()
            return pd.Series([np.median(np.delete(c, i)) for i in range(len(c))], index=g.index)

        src["c_loo"] = src.groupby("date", group_keys=False)[["c"]].apply(loo)
        par = src[src["fwd_source"] == "parity"].copy()
        par["f_carry"] = par["spot"] * np.exp(par["c_loo"] * par["T"])
        par["ferr_pct"] = (par["f_carry"] / par["forward"] - 1) * 100
        par["bucket"] = pd.cut(par["dte"], [6, 14, 30, 60, 90, 2000])
        print(f"{sym} carry-forward error vs parity forward (%), leave-one-out:")
        print(par.groupby("bucket", observed=True)["ferr_pct"].describe()[["count", "50%"]]
              .join(par.groupby("bucket", observed=True)["ferr_pct"].apply(lambda s: s.abs().quantile(0.9)).rename("|p90|"))
              .round(3).to_string())
        # IV effect on near-ATM traded rows of parity expiries
        atm = p[(p["fwd_source"] == "parity") & (p["contracts"] >= ct.MIN_TRADED) & p["iv"].notna()
                & (p["log_moneyness"].abs() < 0.02) & (p["dte"].between(7, 90))]
        atm = atm.merge(par[["date", "expiry", "f_carry"]], on=["date", "expiry"], how="inner")
        iv2 = ct.b76_iv(atm["mark"], atm["f_carry"], atm["strike"], atm["T"], atm["df"], atm["side"] == "CE")
        eff_sym = np.abs(iv2 - atm["iv"].to_numpy()) * 100
        effects.append(eff_sym)
        eb = pd.DataFrame({"eff": eff_sym, "b": pd.cut(atm["dte"], [6, 14, 30, 60, 90])})
        print(f"{sym} near-ATM IV change (vol pts) if the carry forward were used, by dte:")
        print(eb.groupby("b", observed=True)["eff"].describe(percentiles=[0.5, 0.9])[["count", "50%", "90%"]]
              .round(3).to_string())
        cu = w[w["fwd_source"] == "carry"]
        cu = cu.assign(b=pd.cut(cu["dte"], [0, 7, 30, 90, 365, 4000]),
                       near=cu["log_moneyness"].abs() < 0.05)
        print(f"{sym} where carry forwards are actually used (traded rows): by dte, share within 5% of the forward")
        print(cu.groupby("b", observed=True).agg(rows=("near", "size"), near_atm=("near", "mean"),
                                                 contracts=("contracts", "sum")).round(3).to_string())
        print(f"  carry rows are {len(cu) / len(w):.1%} of traded rows and "
              f"{cu['contracts'].sum() / w['contracts'].sum():.2%} of traded contracts")
    # local parity (used where the vol layer has no forward) vs the vol-layer parity
    # forward, on a random sample of expiries that have both
    lp = []
    for sym in ct.SYMBOLS:
        p = load(sym, ["date", "expiry", "strike", "side", "close", "contracts", "spot", "df", "forward",
                       "fwd_source", "dte"], [("fwd_source", "=", "parity"), ("date", ">=", pd.Timestamp("2015-01-01"))])
        keys = p[["date", "expiry"]].drop_duplicates().sample(1500, random_state=0)
        g = p.merge(keys, on=["date", "expiry"]).groupby(["date", "expiry"])
        for (_, _), sub in g:
            f = ct._local_parity(sub, sub["spot"].iat[0], sub["df"].iat[0])
            lp.append((sym, sub["dte"].iat[0], (f / sub["forward"].iat[0] - 1) * 100))
    lp = pd.DataFrame(lp, columns=["symbol", "dte", "err_pct"])
    ok_lp = lp["err_pct"].notna()
    print(f"\nlocal parity vs vol-layer parity forward (%): available on {ok_lp.mean():.1%} of sampled expiries")
    print(lp[ok_lp].assign(b=pd.cut(lp["dte"], [1, 7, 30, 90, 4000])).groupby(["symbol", "b"], observed=True)["err_pct"]
          .agg(n="size", median="median", p90_abs=lambda s: s.abs().quantile(0.9)).round(4).to_string())
    lp_med = lp.loc[ok_lp, "err_pct"].abs().median()
    lp_p90 = lp.loc[ok_lp, "err_pct"].abs().quantile(0.9)
    gate("G3b local parity", lp_med < 0.02 and lp_p90 < 0.05,
         f"|local - vol-layer parity forward| median {lp_med:.4f}% p90 {lp_p90:.4f}% (n={ok_lp.sum():,})")

    eff = np.concatenate(effects)
    eff = eff[np.isfinite(eff)]
    gate("G3 forwards", np.median(eff) < 0.25,
         f"near-ATM IV change if carry forward replaced parity forward: median {np.median(eff):.3f} "
         f"p90 {np.quantile(eff, 0.9):.3f} vol pts (n={len(eff)})")


def g4_expiry(summary: pd.DataFrame):
    sec("G4 expiry settlement")
    rows = []
    for sym in ct.SYMBOLS:
        p = load(sym, ["date", "expiry", "dte", "strike", "side", "settle", "close", "contracts", "spot",
                       "intrinsic", "source_format"], [("dte", "=", 0)])
        rows.append(p)
    e = pd.concat(rows)
    pub = e[e["settle"] > 0]
    match = (pub["settle"] - pub["spot"]).abs() < 0.011
    print(f"expiry-day rows {len(e):,}; with NSE settle > 0: {len(pub):,} "
          f"(first {pub['date'].min().date() if len(pub) else None})")
    itm = e[(e["contracts"] >= ct.MIN_TRADED) & (e["intrinsic"] > 0)]
    itm = itm.assign(diff=itm["close"] - itm["intrinsic"])
    print("traded in-the-money options on expiry day: close - intrinsic (index points), by year")
    print(itm.groupby([itm["symbol"], year_of(itm["date"])])["diff"].median().unstack(0).round(2).to_string())
    em = pd.read_parquet(os.path.join(ROOT, "_expiry_map.parquet"))
    print("\nexpiry labels by how they settled:")
    print(em.groupby(["symbol", "event"]).size().unstack(0).fillna(0).astype(int).to_string())
    odd = em[~em["event"].isin(["normal", "live"])]
    print("\nlabels not settled on their own date (match = share of open positions carried to the successor):")
    print(odd[["symbol", "expiry", "event", "last_seen", "successor", "match", "n_open", "final_expiry", "settled"]]
          .to_string(index=False))
    n_unexpl = int((em["event"] == "unexplained").sum())
    exp = summary[summary["expiry"] <= pd.Timestamp(ct.cached_sessions(CACHE)[-1])]
    no_pay = int(exp["payoff"].isna().sum())
    gate("G4 expiry", match.mean() >= 0.99 and no_pay == 0 and n_unexpl == 0,
         f"S_T == NSE settle on {match.mean():.2%} of {len(pub):,} published rows; "
         f"expired contracts without payoff: {no_pay} of {len(exp):,}; unexplained expiry labels {n_unexpl}")
    if no_pay:
        print(exp[exp["payoff"].isna()].groupby(["symbol", "expiry"]).size().head(20).to_string())


def g5_oi():
    sec("G5 OI continuity")
    sess = np.array(ct.cached_sessions(CACHE), dtype="datetime64[D]")
    nxt = {pd.Timestamp(a): pd.Timestamp(b) for a, b in zip(sess[:-1], sess[1:])}
    out = []
    for sym in ct.SYMBOLS:
        p = load(sym, ["date", "expiry", "strike", "side", "oi_shares", "chg_oi", "lot_size"])
        p = p.sort_values(["expiry", "strike", "side", "date"])
        p["chg_sh"] = p["chg_oi"] * p["lot_size"]
        prev = p.groupby(["expiry", "strike", "side"], observed=True).shift()
        consecutive = prev["date"].map(nxt) == p["date"]
        gap = prev["date"].notna() & ~consecutive
        ok = (p["oi_shares"] - p["chg_sh"] - prev["oi_shares"]).abs() < 0.5
        r = pd.DataFrame({"year": year_of(p["date"]), "ok": ok, "cons": consecutive, "gap": gap})
        by = r[r["cons"]].groupby("year")["ok"].mean()
        out.append((sym, r.loc[r["cons"], "ok"].mean(), int(r["gap"].sum()), int(r["cons"].sum()), by))
        per_day = r[r["cons"]].assign(date=p["date"]).groupby("date")["ok"].mean()
        bad = per_day[per_day < 0.98]
        print(f"{sym}: sessions where < 98% of contracts reconcile with the previous session "
              f"(a missing session in between, or an NSE reporting change): {len(bad)}")
        if len(bad):
            print("  " + ", ".join(f"{d.date()} {v:.1%}" for d, v in bad.sort_values().head(25).items()))
        sw = p[(p["date"] == "2024-07-08") & consecutive]
        print(f"{sym}: format switch 2024-07-05 -> 2024-07-08 continuity {ok[sw.index].mean():.2%} (n={len(sw)})")
    print(pd.DataFrame({s: by for s, _, _, _, by in out}).round(4).to_string())
    worst = min(m for _, m, _, _, _ in out)
    gate("G5 OI continuity", worst >= 0.99,
         "; ".join(f"{s}: {m:.2%} of {n:,} consecutive pairs, {g:,} contracts skip a session"
                   for s, m, g, n, _ in out))


def g6_parity():
    sec("G6 put-call parity on our forward + repo DF")
    res = []
    for sym in ct.SYMBOLS:
        p = load(sym, ["date", "expiry", "strike", "side", "mark", "contracts", "forward", "df", "spot", "dte",
                       "log_moneyness"], [("dte", ">=", 7)])
        p = p[p["contracts"] >= ct.MIN_TRADED]
        c = p[p["side"] == "CE"].set_index(["date", "expiry", "strike"])
        q = p[p["side"] == "PE"].set_index(["date", "expiry", "strike"])
        j = c.join(q[["mark"]], rsuffix="_pe", how="inner")
        j["res_bp"] = (j["mark"] - j["mark_pe"] - j["df"] * (j["forward"] - j.index.get_level_values("strike"))) / j["spot"] * 1e4
        j["region"] = pd.cut(j["log_moneyness"], [-9, -0.03, 0.03, 9], labels=["K<F (ITM call)", "near F", "K>F (ITM put)"])
        print(f"\n{sym}: median residual (bp of spot) by region and year")
        print(j.groupby([year_of(j.index.get_level_values("date")), "region"], observed=True)["res_bp"].median()
              .unstack().round(1).to_string())
        res.append(j["res_bp"].abs())
    r = pd.concat(res)
    gate("G6 parity", r.median() < 10, f"median |residual| {r.median():.2f} bp, p90 {r.quantile(0.9):.1f} bp (n={len(r):,})")


def g7_pipeline():
    sec("G7 panel IV vs bhavcopy vol-layer IV (same OTM points)")
    files = sorted(glob.glob(os.path.join(BC_VOL, "nifty", "*.parquet")) + glob.glob(os.path.join(BC_VOL, "banknifty", "*.parquet")))
    pick = files[:: max(1, len(files) // 400)]
    v = pd.concat([pd.read_parquet(f, columns=["symbol", "date", "expiry", "strike", "otm_side", "our_iv", "liquid"])
                   for f in pick])
    v = v[v["liquid"] & v["our_iv"].notna()]
    v["date"] = pd.to_datetime(v["date"])
    v["expiry"] = pd.to_datetime(v["expiry"], format="%d-%b-%Y")
    out = []
    for sym in ct.SYMBOLS:
        p = load(sym, ["date", "expiry", "strike", "side", "iv", "df", "dte"],
                 [("date", "in", list(v.loc[v["symbol"] == sym, "date"].unique()))])
        m = v[v["symbol"] == sym].merge(p, left_on=["date", "expiry", "strike", "otm_side"],
                                        right_on=["date", "expiry", "strike", "side"], how="inner")
        out.append(m.dropna(subset=["iv"]))
    m = pd.concat(out)
    m["our_iv"] = m["our_iv"] / 100.0  # vol layer stores percent
    m["diff"] = (m["iv"] - m["our_iv"]) * 100
    m["bucket"] = pd.cut(m["dte"], [1, 7, 30, 60, 90])
    print(m.groupby("bucket", observed=True)["diff"].describe()[["count", "mean", "50%", "std"]].round(3).to_string())
    corr = m["iv"].corr(m["our_iv"])
    gate("G7 vs pipeline", corr > 0.99,
         f"corr {corr:.4f}; median (panel - pipeline) {m['diff'].median():.3f} vol pts (n={len(m):,}); "
         "a negative offset is expected: the pipeline divides by the biased parity DF")


def atm30(sym: str) -> pd.Series:
    p = load(sym, ["date", "expiry", "strike", "side", "iv", "contracts", "dte", "log_moneyness"],
             [("dte", ">=", 15), ("dte", "<=", 45)])
    p = p[(p["contracts"] >= ct.MIN_TRADED) & p["iv"].notna()]
    piv = p.pivot_table(index=["date", "expiry", "strike"], columns="side", values="iv")
    lm = p.groupby(["date", "expiry", "strike"])["log_moneyness"].first()
    piv = piv.join(lm).dropna(subset=["CE", "PE"])
    piv["atm"] = piv[["CE", "PE"]].mean(axis=1)
    piv["absm"] = piv["log_moneyness"].abs()
    piv = piv.reset_index()
    piv["dte"] = (piv["expiry"] - piv["date"]).dt.days
    best = piv.sort_values("absm").groupby(["date", "expiry"]).first().reset_index()
    best = best[best["absm"] < 0.02]
    # linear interpolation to 30 days between the nearest expiries on each side
    out = {}
    for d, g in best.groupby("date"):
        g = g.sort_values("dte")
        lo, hi = g[g["dte"] <= 30].tail(1), g[g["dte"] >= 30].head(1)
        if len(lo) and len(hi):
            t0, t1 = lo["dte"].iat[0], hi["dte"].iat[0]
            v0, v1 = lo["atm"].iat[0] ** 2 * t0, hi["atm"].iat[0] ** 2 * t1
            w = 0 if t1 == t0 else (30 - t0) / (t1 - t0)
            out[d] = np.sqrt((v0 + w * (v1 - v0)) / 30)
        else:
            out[d] = g.iloc[(g["dte"] - 30).abs().argmin()]["atm"]
    return pd.Series(out) * 100


def g8_vix():
    sec("G8 30-day ATM IV (panel) vs India VIX")
    vix = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "underlying", "india_vix.csv"),
                      usecols=["date", "close"])
    vix = vix.set_index(pd.to_datetime(vix["date"]))["close"]
    a = atm30("NIFTY").rename("atm30").to_frame().join(vix.rename("vix"), how="inner")
    a = a[a["vix"] > 0]
    by = a.groupby(a.index.year).apply(lambda g: pd.Series({"n": len(g), "corr": g["atm30"].corr(g["vix"]),
                                                             "median_spread(atm-vix)": (g["atm30"] - g["vix"]).median()}))
    print(by.round(3).to_string())
    corr = a["atm30"].corr(a["vix"])
    gate("G8 India VIX", corr > 0.95, f"corr {corr:.4f} over {len(a)} days; "
         f"median spread {(a['atm30'] - a['vix']).median():.2f} vol pts (VIX includes OTM wings, so ATM sits below)")
    return a


def g9_live():
    sec("G9 live chains (bid/ask) vs bhavcopy panel, 2026 overlap")
    rows = []
    for sym in ct.SYMBOLS:
        files = sorted(f for f in glob.glob(os.path.join(PROJECT_ROOT, "data", "raw", sym.lower(), "option_chain", "2026-*.csv"))
                       if re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-[A-Z][a-z]{2}-\d{4}\.csv", os.path.basename(f)))
        dates = sorted({os.path.basename(f)[:10] for f in files})
        p = load(sym, ["date", "expiry", "strike", "side", "mark", "iv", "forward", "df", "T", "contracts",
                       "log_moneyness", "dte"], [("date", "in", [pd.Timestamp(d) for d in dates])])
        for f in files:
            d, exp = os.path.basename(f)[:10], os.path.basename(f)[11:-4]
            try:
                ch = load_chain_csv(f)
            except Exception:
                continue
            ch = ch[(ch["BID"] > 0) & (ch["ASK"] > 0)].copy()
            ch["mid"] = (ch["BID"] + ch["ASK"]) / 2
            ch["spread_pct"] = (ch["ASK"] - ch["BID"]) / ch["mid"]
            ch["date"] = pd.Timestamp(d)
            ch["expiry"] = pd.to_datetime(exp, format="%d-%b-%Y")
            m = ch.merge(p, left_on=["date", "expiry", "STRIKE", "side"],
                         right_on=["date", "expiry", "strike", "side"], how="inner")
            m["symbol"] = sym
            rows.append(m)
    m = pd.concat(rows, ignore_index=True)
    m = m[(m["contracts"] >= ct.MIN_TRADED) & m["iv"].notna() & (m["spread_pct"] < 0.10)]
    m["iv_live"] = ct.b76_iv(m["mid"], m["forward"], m["strike"], m["T"], m["df"], m["side"] == "CE")
    m["d_iv"] = (m["iv_live"] - m["iv"]) * 100
    m["d_px_pct"] = (m["mid"] / m["mark"] - 1) * 100
    near = m[(m["log_moneyness"].abs() < 0.03) & m["dte"].between(2, 40)]
    print(f"overlap rows {len(m):,}; sessions {m['date'].nunique()} ({m['date'].min().date()}..{m['date'].max().date()})")
    print("near-ATM, 2-40 dte: live-mid IV minus panel IV (vol pts), and price diff (%)")
    print(near.groupby("symbol")[["d_iv", "d_px_pct"]].describe().T.round(3).to_string())
    med = near["d_iv"].abs().median()
    gate("G9 live chains", med < 0.5, f"median |IV diff| {med:.3f} vol pts, median signed {near['d_iv'].median():.3f} "
         f"(n={len(near):,}); live mid is a ~21:00+ IST snapshot, panel close is the 15:00-15:30 VWAP")


def g10_rate():
    sec("G10 repo-rate sensitivity")
    p = pd.concat([load(s, ["mark", "forward", "strike", "T", "df", "rate", "side", "iv", "dte", "contracts",
                            "log_moneyness"], [("dte", ">=", 2)]) for s in ct.SYMBOLS])
    p = p[(p["contracts"] >= ct.MIN_TRADED) & p["iv"].notna()].sample(200000, random_state=0)
    otm = np.where(p["side"] == "CE", p["log_moneyness"] >= 0, p["log_moneyness"] <= 0)
    for bump in (0.005, 0.02):
        e = np.maximum(*[np.abs(ct.b76_iv(p["mark"], p["forward"], p["strike"], p["T"],
                                          np.exp(-(p["rate"] + s * bump) * p["T"]), p["side"] == "CE")
                                - p["iv"].to_numpy()) * 100 for s in (-1, 1)])
        t = pd.DataFrame({"eff": e, "otm": np.where(otm, "OTM", "ITM"), "b": pd.cut(p["dte"], [1, 7, 30, 90, 365, 3000])})
        print(f"+/-{bump * 100:.1f}pp: abs IV change (vol pts) p99 by side of the forward and dte")
        print(t.groupby(["otm", "b"], observed=True)["eff"].quantile(0.99).unstack(0).round(4).to_string())
    out = {}
    for bump in (-0.02, 0.02):
        df2 = np.exp(-(p["rate"] + bump) * p["T"])
        iv2 = ct.b76_iv(p["mark"], p["forward"], p["strike"], p["T"], df2, p["side"] == "CE")
        out[bump] = np.abs(iv2 - p["iv"].to_numpy()) * 100
    eff = np.maximum(out[-0.02], out[0.02])
    b = pd.cut(p["dte"], [1, 7, 30, 90, 365, 3000])
    t = pd.DataFrame({"eff": eff, "b": b}).groupby("b", observed=True)["eff"].describe(percentiles=[0.5, 0.99])
    print("abs IV change (vol pts) from a +/-2pp rate error:\n" + t[["count", "50%", "99%"]].round(4).to_string())
    short = eff[(p["dte"] <= 90).to_numpy()]
    short = short[np.isfinite(short)]
    gate("G10 rate", np.quantile(short, 0.99) < 0.25, f"dte<=90: median {np.median(short):.4f}, p99 {np.quantile(short, 0.99):.4f} vol pts")


def descriptive(summary: pd.DataFrame, a30: pd.DataFrame):
    sec("Liquidity and IV coverage")
    for sym in ct.SYMBOLS:
        p = load(sym, ["date", "quality", "iv_status"])
        q = p.groupby(year_of(p["date"]))["quality"].value_counts(normalize=True).unstack().round(3)
        s = p.groupby(year_of(p["date"]))["iv_status"].value_counts(normalize=True).unstack().round(3)
        print(f"\n{sym}\n" + pd.concat({"quality": q, "iv_status": s}, axis=1).fillna(0).to_string())

    sec("Contracts")
    s = summary.copy()
    s["era"] = np.where(s["expiry"].dt.year < 2020, "2008-2019", "2020-2026")
    s["life"] = (s["last_date"] - s["first_date"]).dt.days
    t = s.groupby(["symbol", "era"]).agg(contracts=("strike", "size"),
                                         never_traded=("n_traded", lambda x: (x == 0).mean()),
                                         median_life_days=("life", "median"),
                                         median_traded_days=("n_traded", "median"),
                                         median_first_trade_dte=("first_trade_dte", "median"))
    print(t.round(2).to_string())

    sec("Famous days")
    for d in ["2020-03-23", "2024-06-04", "2024-06-05"]:
        ts = pd.Timestamp(d)
        p = load("NIFTY", ["date", "expiry", "strike", "side", "oi", "contracts", "dte", "spot", "iv",
                           "log_moneyness"], [("date", "=", ts)])
        front = p[p["dte"] > 0]["expiry"].min()
        f = p[p["expiry"] == front]
        near = f[(f["log_moneyness"].abs() < 0.01) & f["iv"].notna()]
        print(f"{d}: NIFTY spot {p['spot'].iat[0]:.0f}, contracts traded {p['contracts'].sum():,.0f}, "
              f"front {front.date()} near-ATM IV {near['iv'].median()*100:.1f}, "
              f"30d ATM {a30['atm30'].get(ts, np.nan):.1f} vs VIX {a30['vix'].get(ts, np.nan):.1f}; "
              f"top put OI {f[f.side=='PE'].nlargest(1,'oi')[['strike','oi']].values.tolist()}, "
              f"top call OI {f[f.side=='CE'].nlargest(1,'oi')[['strike','oi']].values.tolist()}")

    sec("Example contracts")
    for sym, exp, k, side in [("NIFTY", "2026-06-30", 23000, "CE"), ("NIFTY", "2020-03-26", 8000, "PE")]:
        e = pd.Timestamp(exp)
        p = load(sym, ["date", "expiry", "strike", "side", "dte", "spot", "mark", "quality", "oi", "contracts",
                       "iv", "delta", "fwd_source"], [("expiry_year", "=", e.year)])
        c = p[(p["expiry"] == e) & (p["strike"] == k) & (p["side"] == side)].sort_values("date")
        r = s[(s["symbol"] == sym) & (s["expiry"] == e) & (s["strike"] == k) & (s["side"] == side)]
        print(f"\n{sym} {exp} {k} {side}: {len(c)} rows; summary: "
              + r[["first_date", "first_trade_date", "first_trade_price", "n_traded", "spot_at_expiry", "payoff"]]
              .to_string(index=False, header=False))
        show = c[c["mark"].notna()]
        show = pd.concat([show.iloc[:: max(1, len(show) // 12)], show.tail(2)]).drop_duplicates()
        print(show[["date", "dte", "spot", "mark", "iv", "delta", "oi", "contracts", "fwd_source"]].round(3).to_string(index=False))


def main():
    summary = pd.read_parquet(os.path.join(ROOT, "_contracts.parquet"))
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        g1_integrity()
        g2_spot()
        g3_forwards()
        g4_expiry(summary)
        g5_oi()
        g6_parity()
        g7_pipeline()
        a30 = g8_vix()
        g9_live()
        g10_rate()
        descriptive(summary, a30)
        sec("SUMMARY")
        for name, ok, detail in RESULTS:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    with open(os.path.join(ROOT, "_validation.txt"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


if __name__ == "__main__":
    main()
