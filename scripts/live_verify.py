"""Live snapshots (src/live.py) vs the bhavcopy panel on the sessions both cover.

The bhavcopy is NSE's official end-of-day file (closes, settles, volumes); the live chain is
a snapshot of the option-chain page after the close (bid/ask/LTP). If the live panel is
built right, on the same session and contract:
  spot     snapshot underlying value = NSE index close
  forward  live parity forward ~ panel forward (parity/futures), a few bp
  price    live mid ~ bhavcopy close, within about a half-spread
  IV       live IV ~ panel IV, a fraction of a vol point near the money
Writes data/processed/live/_verify.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import live  # noqa: E402
from src.backtest import PANEL  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "data", "processed", "live")
LINES: list[str] = []


def out(s=""):
    print(s)
    LINES.append(str(s))


def q(x, ps=(0.5, 0.9, 0.99)):
    x = pd.Series(x).dropna()
    return " ".join(f"p{int(p * 100)} {x.quantile(p):.3g}" for p in ps) + f" (n {len(x)})"


def main():
    os.makedirs(OUT, exist_ok=True)
    for sym in ("NIFTY", "BANKNIFTY"):
        lp, info = live.live_panel(sym)
        days = sorted(lp["date"].unique())
        cols = ["date", "expiry", "strike", "side", "dte", "close", "settle", "contracts", "mark", "spot", "forward",
                "fwd_source", "iv", "delta", "lot_size"]
        bp = pd.read_parquet(PANEL, columns=cols, filters=[("symbol", "=", sym), ("date", ">=", days[0])])
        both = sorted(set(days) & set(bp["date"].unique()))
        out("=" * 100 + f"\n{sym}: live sessions {len(days)} ({days[0].date()} .. {days[-1].date()}), "
            f"overlap with bhavcopy {len(both)}\n" + "=" * 100)
        rel = live.expiry_relabels(live.chain_files(sym))
        out(f"expiry relabels detected: {rel or 'none'}")
        out(f"forward source per file: {info['fwd_source'].value_counts().to_dict()}")
        stale = info[info["scraped_at"].notna()].copy()
        stale["scr"] = pd.to_datetime(stale["scraped_at"]).dt.tz_localize(None)
        early = stale[stale["scr"] < stale["date"] + pd.Timedelta(hours=21)]
        out(f"files scraped before 21:00 IST on their session: {len(early)} of {len(stale)}"
            + (f" (sessions {sorted(set(early['date'].dt.date.astype(str)))[:8]})" if len(early) else ""))

        sp = info.groupby("date")["spot"].median()
        bs = bp.groupby("date")["spot"].first()
        d = ((sp.reindex(both) / bs.reindex(both) - 1) * 1e4)
        out(f"\nspot: snapshot vs NSE close, bp: {q(d.abs())}; worst {d.abs().idxmax().date()} {d.abs().max():.1f}bp")
        off = d[d.abs() > live.SNAPSHOT_TOL_BP]
        out(f"sessions whose snapshot is not at the close (> {live.SNAPSHOT_TOL_BP}bp from NSE close): "
            + (", ".join(f"{k.date()} {v:+.0f}bp" for k, v in off.items()) or "none"))
        y = live.yahoo_close(sym)
        dy = ((y.reindex(both) / bs.reindex(both) - 1) * 1e4)
        out(f"spot: Yahoo close vs NSE close, bp: {q(dy.abs())}")

        fi = info[info["fwd_source"] != "none"][["date", "expiry", "forward", "fwd_source", "dte"]]
        bf = bp[bp["fwd_source"].isin(["parity", "futures", "expiry_day"])].drop_duplicates(["date", "expiry"])
        m = fi.merge(bf[["date", "expiry", "forward", "fwd_source"]], on=["date", "expiry"], suffixes=("", "_b"))
        m["bp"] = (m["forward"] / m["forward_b"] - 1) * 1e4
        out(f"\nforward: live vs panel, bp (|diff|): {q(m['bp'].abs())}; mean signed {m['bp'].mean():+.2f}bp")
        for src, g in m.groupby("fwd_source"):
            out(f"   live {src:13s}: {q(g['bp'].abs())}")
        w = m.reindex(m["bp"].abs().sort_values().index[-3:])
        out("   worst: " + "; ".join(f"{r.date.date()} {r.expiry.date()} dte {r.dte} {r.bp:+.1f}bp ({r.fwd_source}/{r.fwd_source_b})"
                                     for r in w.itertuples()))

        a = lp[lp["date"].isin(both)]
        c = a.merge(bp, on=["date", "expiry", "strike", "side"], suffixes=("", "_b"))
        tb = c["contracts"] >= live.MIN_TRADED
        out(f"\ntraded flags: bhavcopy-traded contracts also live-traded {c.loc[tb, 'traded'].mean():.1%}; "
            f"live-traded also bhavcopy-traded {tb[c['traded']].mean():.1%}")
        t = c[c["traded"] & tb & (c["mark_b"] > 0)].copy()
        t["rel"] = t["mid"] / t["mark_b"] - 1
        t["half"] = (t["ask"] - t["bid"]) / 2
        t["in_half"] = (t["mid"] - t["mark_b"]).abs() / t["half"].where(t["half"] > 0)
        t["ltp_rel"] = t["ltp"] / t["mark_b"] - 1
        t["absd"] = t["delta_b"].abs()
        out(f"price: live mid vs bhavcopy close, contracts traded in both: {len(t):,}")
        g = t[t["dte"] <= 1]
        out(f"   dte 0-1 (expiry / day before): |mid/close-1| {q(g['rel'].abs())}")
        t = t[(t["dte"] >= 2) & ~t["date"].isin(off.index)]
        out("   dte >= 2, snapshot at the close:")
        for lab, mm in (("|delta| 0.30-0.50", t["absd"].between(0.3, 0.5)), ("|delta| 0.10-0.30", t["absd"].between(0.1, 0.3)),
                        ("|delta| 0.02-0.10", t["absd"].between(0.02, 0.1)), ("ITM (|delta| > 0.5)", t["absd"] > 0.5)):
            g = t[mm]
            out(f"   {lab:20s}: |mid/close-1| {q(g['rel'].abs())}; |mid-close| in half-spreads {q(g['in_half'])};"
                f" |LTP/close-1| p50 {g['ltp_rel'].abs().median():.3g}")
        out(f"   mean signed mid/close-1 (OTM, |delta| 0.02-0.5): {t.loc[t['absd'].between(0.02, 0.5), 'rel'].mean():+.4f}")
        iv = t[t["absd"].between(0.02, 0.5) & t["iv"].notna() & t["iv_b"].notna() & (t["dte"] >= 2)]
        otm = np.where(iv["side"] == "CE", iv["strike"] >= iv["forward_b"], iv["strike"] <= iv["forward_b"])
        iv = iv[otm]
        dv = (iv["iv"] - iv["iv_b"]) * 100
        out(f"IV: live - panel, vol points, OTM |delta| 0.02-0.5, dte >= 2: |diff| {q(dv.abs())}; mean signed {dv.mean():+.3f}")
        near = iv["absd"].between(0.3, 0.5)
        out(f"   near the money (|delta| 0.3-0.5): |diff| {q(dv[near].abs())}; mean signed {dv[near].mean():+.3f}")
        lots = bp.groupby("date")["lot_size"].max()
        out(f"lot size (bhavcopy) on the last overlap day: {lots.iloc[-1]:g}")
        out()
    with open(os.path.join(OUT, "_verify.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
