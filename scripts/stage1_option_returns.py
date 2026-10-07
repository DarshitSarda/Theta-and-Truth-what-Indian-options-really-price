"""Stage 1: model-free returns of NIFTY / BANKNIFTY options, 2008 -> 2026.

Two questions, answered without any pricing model:
  A. Buy-and-hold: buy an option at the close, hold it to expiry. What did buyers
     and sellers earn per Rs 100 of premium, after costs?
  B. Delta-hedged: same, but hedge the index exposure every day with the forward
     (futures) at a delta from the entry IV. What remains is the price of
     volatility itself: negative for buyers = implied vol was too high.

Rules fixed before running (do not change after seeing results):
  Entries   every expiry, at the close 1, 2, 3, 5, 10, 21, 42 trading days before it;
            out-of-the-money options only (call K >= F, put K <= F), |delta| 0.02-0.50,
            traded >= 10 contracts that day, price >= Rs 0.5, valid IV, forward not
            from the carry fallback (Stage 0 decision).
  Cells     symbol x side x |delta| bucket (0.02-0.05, 0.05-0.15, 0.15-0.30, 0.30-0.50)
            x horizon. Trades in a cell are averaged per settlement date first, so
            one expiry = one observation (trades within an expiry are not independent).
  Periods   discovery = settlements before 2018-01-01; holdout = 2018 onwards.
            Eras for the structure question: 2008-13, 2014-19, 2020 to 19-Nov-2024,
            20-Nov-2024 onwards (SEBI: one weekly per exchange, bigger lots).
  Costs     half the bid-ask spread (from the 2026 live chains, by price), exchange fee
            + GST, stamp duty, Rs 20/order brokerage + GST, STT by era (sale of option
            0.017% -> 0.05% (Jun-2016) -> 0.0625% (Apr-2023) -> 0.1% (Oct-2024) -> 0.15% (Apr-2026); exercise
            0.125% of the settlement value before Jun-2016, of intrinsic after, 0.15% from
            Apr-2026; NSE fee 0.03553% from Apr-2026; futures STT rise in Apr-2026 adds 1.5 bp
            per unit of hedge turnover - corrected 2026-10-07, earlier results kept). A buyer
            holding an ITM option may instead sell it at the expiry-day close if that
            is better. Futures hedge: 2 bp of notional per unit of delta traded.
            Stress costs: spreads x3 before 2020, x2 after.
  Verdict   a cell "holds up" if: significant in discovery after Benjamini-Hochberg
            (q = 0.05, two-sided, all cells of that test), same sign and p < 0.05 in
            the holdout, and same sign of the holdout mean under stress costs.
  Structure pre-2020 vs post-2020 mean per cell, Welch t-test, BH across cells.
  Engine    simulated Black-Scholes markets priced at the true vol: the hedged
            P&L must average ~0 (|mean| < 1% of premium) and be far less noisy
            than unhedged.

Usage: python -u scripts/stage1_option_returns.py
Writes data/processed/stage1/ (trades.parquet, cells_*.csv, _report.txt)
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
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import contracts as ct  # noqa: E402
from src.chain_parser import load_chain_csv  # noqa: E402

PANEL = os.path.join(PROJECT_ROOT, "data", "processed", "contracts", "panel")
CONTRACTS = os.path.join(PROJECT_ROOT, "data", "processed", "contracts")
OUT = os.path.join(PROJECT_ROOT, "data", "processed", "stage1")

HORIZONS = [1, 2, 3, 5, 10, 21, 42]
DELTA_BINS = [0.02, 0.05, 0.15, 0.30, 0.50]
DELTA_LABELS = ["0.02-0.05", "0.05-0.15", "0.15-0.30", "0.30-0.50"]
MIN_PRICE = 0.5
HOLDOUT = pd.Timestamp("2018-01-01")
POST2020 = pd.Timestamp("2020-01-01")
ERAS = [("2008-13", "2008-01-01"), ("2014-19", "2014-01-01"), ("2020-Nov24", "2020-01-01"),
        ("Nov24-now", "2024-11-20")]
FUT_COST = 0.0002
FIN26 = "2026-04-01"        # Finance Act 2026 STT rates (NSE/FATAX/73524)
FUT_STT_STEP = 0.00015      # futures STT 0.02% -> 0.05% on sales ~ half of hedge turnover
Q_FDR = 0.05

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


# ---------------------------------------------------------------- costs

def stt_sell_rate(d: pd.Series) -> np.ndarray:
    return np.select([d < "2016-06-01", d < "2023-04-01", d < "2024-10-01", d < FIN26],
                     [0.00017, 0.0005, 0.000625, 0.001], 0.0015)


def exchange_rate(d: pd.Series) -> np.ndarray:
    return np.select([d < "2024-10-01", d < FIN26], [0.00053, 0.000350], 0.0003553) * 1.18


def spread_model() -> pd.DataFrame:
    """Median half-spread (Rs) by option price, from live 2026 chains (OTM, traded)."""
    rows = []
    pat = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-[A-Z][a-z]{2}-\d{4}\.csv")
    for sym in ct.SYMBOLS:
        for f in glob.glob(os.path.join(PROJECT_ROOT, "data", "raw", sym.lower(), "option_chain", "2026-*.csv")):
            if not pat.fullmatch(os.path.basename(f)):
                continue
            try:
                ch = load_chain_csv(f)
            except Exception:
                continue
            ch = ch[(ch["BID"] > 0) & (ch["ASK"] > ch["BID"]) & (ch["VOLUME"] > 0)]
            rows.append(pd.DataFrame({"mid": (ch["BID"] + ch["ASK"]) / 2, "half": (ch["ASK"] - ch["BID"]) / 2}))
    s = pd.concat(rows)
    s = s[s["mid"] >= MIN_PRICE]
    edges = [0.5, 2, 5, 10, 20, 50, 100, 200, 500, 1e9]
    s["b"] = pd.cut(s["mid"], edges)
    t = s.groupby("b", observed=True).agg(n=("half", "size"), mid=("mid", "median"), half=("half", "median"))
    t["half_pct"] = t["half"] / t["mid"]
    return t.reset_index()


def half_spread(price: np.ndarray, model: pd.DataFrame, mult: np.ndarray) -> np.ndarray:
    """Interpolate the median half-spread as a share of price; at least half a tick."""
    pct = np.interp(np.log(price), np.log(model["mid"].to_numpy()), model["half_pct"].to_numpy())
    return np.maximum(pct * price, 0.025) * mult


# ---------------------------------------------------------------- data

def load_entries() -> pd.DataFrame:
    cols = ["date", "expiry", "strike", "side", "tdte", "dte", "mark", "close", "contracts", "spot", "forward",
            "fwd_source", "iv", "delta", "log_moneyness", "lot_size", "rate", "T", "df"]
    out = []
    for sym in ct.SYMBOLS:
        p = pd.read_parquet(PANEL, columns=cols, filters=[("symbol", "=", sym), ("tdte", "in", HORIZONS),
                                                          ("contracts", ">=", ct.MIN_TRADED)])
        p["symbol"] = sym
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    otm = np.where(p["side"] == "CE", p["log_moneyness"] >= 0, p["log_moneyness"] <= 0)
    keep = (otm & (p["mark"] >= MIN_PRICE) & p["iv"].notna() & p["fwd_source"].isin(["parity", "futures", "parity_local"])
            & p["delta"].abs().between(DELTA_BINS[0], DELTA_BINS[-1]) & (p["dte"] > 0))
    p = p[keep].copy()
    p["dbucket"] = pd.cut(p["delta"].abs(), DELTA_BINS, labels=DELTA_LABELS, include_lowest=True)
    return p


def attach_outcome(e: pd.DataFrame) -> pd.DataFrame:
    em = pd.read_parquet(os.path.join(CONTRACTS, "_expiry_map.parquet"))[["symbol", "expiry", "final_expiry", "settled"]]
    e = e.merge(em, on=["symbol", "expiry"], how="left")
    summ = pd.read_parquet(os.path.join(CONTRACTS, "_contracts.parquet"),
                           columns=["symbol", "expiry", "strike", "side", "spot_at_expiry", "payoff"])
    e = e.merge(summ.rename(columns={"expiry": "final_expiry"}), on=["symbol", "final_expiry", "strike", "side"], how="left")
    e = e[e["payoff"].notna() & e["settled"].notna()].copy()
    # the option's own close on the settlement session (to sell instead of exercising)
    last = []
    for sym in ct.SYMBOLS:
        q = pd.read_parquet(PANEL, columns=["date", "expiry", "strike", "side", "mark"],
                            filters=[("symbol", "=", sym), ("dte", "<=", 1)])
        q["symbol"] = sym
        last.append(q)
    q = pd.concat(last).merge(em, on=["symbol", "expiry"])
    q = q[q["date"] == q["settled"]][["symbol", "final_expiry", "strike", "side", "mark"]]
    q = q.rename(columns={"mark": "exit_close"}).drop_duplicates(["symbol", "final_expiry", "strike", "side"])
    return e.merge(q, on=["symbol", "final_expiry", "strike", "side"], how="left")


def forward_paths() -> pd.DataFrame:
    """Forward of each final expiry on every session up to settlement (settlement = S_T)."""
    em = pd.read_parquet(os.path.join(CONTRACTS, "_expiry_map.parquet"))[["symbol", "expiry", "final_expiry"]]
    out = []
    for sym in ct.SYMBOLS:
        p = pd.read_parquet(PANEL, columns=["date", "expiry", "forward", "fwd_source", "spot"],
                            filters=[("symbol", "=", sym), ("dte", "<=", 80)])
        p = p.drop_duplicates(["date", "expiry"])
        p["symbol"] = sym
        out.append(p)
    p = pd.concat(out).merge(em, on=["symbol", "expiry"])
    p["rank"] = p["fwd_source"].map({"expiry_day": 0, "parity": 1, "futures": 2, "parity_local": 3, "carry": 4}).fillna(9)
    p = p.sort_values("rank").drop_duplicates(["symbol", "final_expiry", "date"])
    p.loc[p["fwd_source"] == "expiry_day", "forward"] = p["spot"]
    return p[["symbol", "final_expiry", "date", "forward"]].sort_values(["symbol", "final_expiry", "date"])


# ---------------------------------------------------------------- delta hedge

def hedge_pnl(tr: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """Daily delta hedge with the forward, delta from the entry IV.

    gain = payoff - P0*exp(rT) - sum_k delta_k (F_{k+1} - F_k); the last F is S_T.
    Returns per trade: hedge P&L (Rs, per unit) and futures turnover in delta-units x F.
    """
    parts = []
    for sym in tr["symbol"].unique():
        a = tr.loc[tr["symbol"] == sym, ["tid", "final_expiry", "date", "settled"]]
        b = paths.loc[paths["symbol"] == sym, ["final_expiry", "date", "forward"]].rename(columns={"date": "d"})
        m = a.merge(b, on="final_expiry")
        parts.append(m[(m["d"] >= m["date"]) & (m["d"] <= m["settled"])][["tid", "d", "forward", "settled"]])
    m = pd.concat(parts, ignore_index=True)
    info = tr.set_index("tid")[["iv", "strike", "side", "rate", "spot_at_expiry"]]
    m = m.join(info, on="tid").sort_values(["tid", "d"])
    m.loc[m["d"] == m["settled"], "forward"] = m["spot_at_expiry"]
    m["forward"] = m.groupby("tid")["forward"].ffill()
    T = (m["settled"] - m["d"]).dt.days.to_numpy() / 365.0
    F = m["forward"].to_numpy(float)
    K, sig = m["strike"].to_numpy(float), m["iv"].to_numpy(float)
    DF = np.exp(-m["rate"].to_numpy(float) * T)
    is_call = (m["side"] == "CE").to_numpy()
    with np.errstate(all="ignore"):
        dlt, _, _, _ = ct.b76_greeks(F, K, np.maximum(T, 1e-9), sig, DF, 0.0, is_call)
    dlt = np.where(T > 0, dlt, 0.0)
    m["delta_h"] = dlt
    m["dF"] = m.groupby("tid")["forward"].shift(-1) - m["forward"]
    m["hedge"] = -m["delta_h"] * m["dF"].fillna(0.0)
    m["turn"] = (m["delta_h"] - m.groupby("tid")["delta_h"].shift().fillna(0.0)).abs() * m["forward"]
    m["turn_26"] = np.where(m["d"] >= FIN26, m["turn"], 0.0)
    g = m.groupby("tid").agg(hedge=("hedge", "sum"), turnover=("turn", "sum"), turnover_26=("turn_26", "sum"),
                             n_steps=("d", "size"))
    return g


def engine_check(n=20000, sigma=0.15, days=21, seed=0):
    """Black-Scholes world priced at the true vol: hedged P&L should be ~0."""
    rng = np.random.default_rng(seed)
    T0 = days / 365.0
    dt = 1 / 365.0
    out = {}
    for side, K_mult in (("CE", 1.03), ("PE", 0.97)):
        F0, K = 100.0, 100.0 * K_mult
        is_call = np.full(n, side == "CE")
        z = rng.standard_normal((n, days))
        F = F0 * np.exp(np.cumsum(-0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * z, axis=1))
        F = np.hstack([np.full((n, 1), F0), F])
        P0 = ct.b76_price(F0, K, T0, sigma, 1.0, side == "CE")
        hedge = np.zeros(n)
        for k in range(days):
            T = T0 - k * dt
            d, _, _, _ = ct.b76_greeks(F[:, k], K, T, sigma, 1.0, 0.0, is_call)
            hedge -= d * (F[:, k + 1] - F[:, k])
        pay = np.maximum(F[:, -1] - K, 0) if side == "CE" else np.maximum(K - F[:, -1], 0)
        un = (pay - P0) / P0 * 100
        hd = (pay - P0 + hedge) / P0 * 100
        out[side] = (un.mean(), un.std(), hd.mean(), hd.std(), hd.std() / np.sqrt(n))
    return out


# ---------------------------------------------------------------- statistics

def cell_stats(t: pd.DataFrame, col: str, keys: list[str]) -> pd.DataFrame:
    """Per settlement date mean first, then mean / t / p across settlements."""
    per = t.groupby(keys + ["settled"], observed=True)[col].mean().reset_index()

    def f(x):
        v = x[col].to_numpy()
        n = len(v)
        m = v.mean()
        se = v.std(ddof=1) / np.sqrt(n) if n > 2 else np.nan
        tt = m / se if se and se > 0 else np.nan
        p = 2 * stats.t.sf(abs(tt), n - 1) if np.isfinite(tt) else np.nan
        return pd.Series({"n_exp": n, "mean": m, "t": tt, "p": p, "win_share": (v > 0).mean(),
                          "worst": v.min(), "median": np.median(v)})

    return per.groupby(keys, observed=True).apply(f).reset_index()


def bh(p: pd.Series, q=Q_FDR) -> pd.Series:
    p = p.fillna(1.0)
    n = len(p)
    order = np.argsort(p.to_numpy())
    ranked = p.to_numpy()[order]
    thresh = q * np.arange(1, n + 1) / n
    passed = ranked <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    out[order[:k]] = True
    return pd.Series(out, index=p.index)


def verdict(t: pd.DataFrame, col: str, stress_col: str) -> pd.DataFrame:
    keys = ["symbol", "side", "dbucket", "tdte"]
    disc = cell_stats(t[t["settled"] < HOLDOUT], col, keys)
    hold = cell_stats(t[t["settled"] >= HOLDOUT], col, keys)
    hs = cell_stats(t[t["settled"] >= HOLDOUT], stress_col, keys)[keys + ["mean"]].rename(columns={"mean": "hold_stress"})
    disc["bh"] = bh(disc["p"])
    v = disc.merge(hold, on=keys, how="outer", suffixes=("_disc", "_hold")).merge(hs, on=keys, how="left")
    v["holds_up"] = (v["bh"].fillna(False) & (np.sign(v["mean_disc"]) == np.sign(v["mean_hold"]))
                     & (v["p_hold"] < 0.05) & (np.sign(v["hold_stress"]) == np.sign(v["mean_hold"])))
    return v


def era_table(t: pd.DataFrame, col: str) -> pd.DataFrame:
    keys = ["symbol", "side", "dbucket", "tdte"]
    t = t.copy()
    t["era"] = ERAS[0][0]
    for name, start in ERAS[1:]:
        t.loc[t["settled"] >= start, "era"] = name
    s = cell_stats(t, col, keys + ["era"])
    return s.pivot_table(index=keys, columns="era", values="mean", observed=True)[[e for e, _ in ERAS]]


def structure_test(t: pd.DataFrame, col: str) -> pd.DataFrame:
    keys = ["symbol", "side", "dbucket", "tdte"]
    per = t.groupby(keys + ["settled"], observed=True)[col].mean().reset_index()
    per["post"] = per["settled"] >= POST2020
    rows = []
    for k, g in per.groupby(keys, observed=True):
        a, b = g.loc[~g["post"], col], g.loc[g["post"], col]
        if len(a) > 5 and len(b) > 5:
            tt, p = stats.ttest_ind(b, a, equal_var=False)
            rows.append(dict(zip(keys, k), pre=a.mean(), post=b.mean(), diff=b.mean() - a.mean(), p=p,
                             n_pre=len(a), n_post=len(b)))
    s = pd.DataFrame(rows)
    s["bh"] = bh(s["p"])
    return s


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        print("=" * 90 + "\nENGINE CHECK (simulated Black-Scholes market, priced at true vol, 21 days, daily hedge)")
        ec = engine_check()
        for side, (um, us, hm, hs_, hse) in ec.items():
            print(f"  {side}: unhedged mean {um:+.2f} (sd {us:.0f}) | hedged mean {hm:+.2f} +/- {hse:.2f} (sd {hs_:.1f}) per Rs100")
        ok = all(abs(v[2]) < 1.0 and v[3] < v[1] / 3 for v in ec.values())
        print(f"  [{'PASS' if ok else 'FAIL'}] hedged mean within 1% of premium and >3x less noisy")

        sm = spread_model()
        print("\nSPREAD MODEL (live chains 2026, median half-spread by price)")
        print(sm.round(3).to_string(index=False))

        e = attach_outcome(load_entries())
        e = e.reset_index(drop=True)
        e["tid"] = np.arange(len(e))
        print(f"\nTrades: {len(e):,} ({e['settled'].min().date()} .. {e['settled'].max().date()}); "
              f"settlements {e['settled'].nunique():,}")

        paths = forward_paths()
        h = hedge_pnl(e, paths)
        e = e.join(h, on="tid")

        d = e["date"]
        P0 = e["mark"].to_numpy(float)
        lot = e["lot_size"].to_numpy(float)
        brk = 20 * 1.18 / lot
        exch = exchange_rate(d)
        stt_s = stt_sell_rate(d)
        stamp = np.where(d >= "2020-07-01", 0.00003, 0.0)
        pre = (d < POST2020).to_numpy()
        ST = e["spot_at_expiry"].to_numpy(float)
        pay = e["payoff"].to_numpy(float)
        ex_rate = np.where(e["settled"] < FIN26, 0.00125, 0.0015)
        stt_ex = np.where(pay > 0, ex_rate * np.where(e["settled"] < "2016-06-01", ST, pay), 0.0)
        xc = e["exit_close"].to_numpy(float)
        stt_s_exit = stt_sell_rate(e["settled"])
        exch_exit = exchange_rate(e["settled"])
        carry = P0 * (np.exp(e["rate"].to_numpy(float) * e["T"].to_numpy(float)) - 1)

        for tag, mult in (("", 1.0), ("_stress", np.where(pre, 3.0, 2.0))):
            hs_in = half_spread(P0, sm, mult)
            buy_in = P0 + hs_in + P0 * (exch + stamp) + brk
            sell_in = P0 - hs_in - P0 * (exch + stt_s) - brk
            sell_close = np.where(np.isfinite(xc), xc - half_spread(np.maximum(xc, 0.05), sm, mult)
                                  - xc * (exch_exit + stt_s_exit) - brk, -np.inf)
            buy_out = np.maximum(pay - stt_ex, sell_close)
            buy_out = np.where(pay > 0, buy_out, 0.0)
            hedge_cost = e["turnover"].to_numpy(float) * FUT_COST + e["turnover_26"].to_numpy(float) * FUT_STT_STEP
            e["buy_net" + tag] = (buy_out - buy_in - carry) / P0 * 100
            e["sell_net" + tag] = (sell_in - pay) / P0 * 100
            e["hbuy_net" + tag] = (buy_out - buy_in - carry + e["hedge"] - hedge_cost) / P0 * 100
            e["hsell_net" + tag] = (sell_in - pay - e["hedge"] - hedge_cost) / P0 * 100
        e["buy_gross"] = (pay - P0) / P0 * 100
        e["hbuy_gross"] = (pay - P0 - carry + e["hedge"]) / P0 * 100

        e.drop(columns=["close"]).to_parquet(os.path.join(OUT, "trades.parquet"), index=False)

        print("\nSANITY")
        print(f"  share of trades ending in the money: {(pay > 0).mean():.1%}; "
              f"hedge steps per trade median {e['n_steps'].median():.0f}")
        print(f"  cost per Rs100 premium (round trip, median): buyer "
              f"{(e['buy_gross'] - e['buy_net']).median():.1f}, by |delta|:")
        print((e.assign(c=e["buy_gross"] - e["buy_net"]).groupby("dbucket", observed=True)["c"].median()).round(1).to_string())

        pd.options.display.float_format = "{:,.1f}".format
        for title, col in (("A. BUY-AND-HOLD", "buy"), ("B. DELTA-HEDGED", "hbuy")):
            print("\n" + "=" * 90 + f"\n{title}: buyer P&L per Rs100 premium, net of costs (seller ~ minus this, minus own costs)")
            v = verdict(e, f"{col}_net", f"{col}_net_stress")
            sv = verdict(e, f"{col.replace('buy', 'sell')}_net", f"{col.replace('buy', 'sell')}_net_stress")
            v.to_csv(os.path.join(OUT, f"cells_{col}.csv"), index=False)
            sv.to_csv(os.path.join(OUT, f"cells_{col.replace('buy', 'sell')}.csv"), index=False)
            et = era_table(e, f"{col}_net")
            print("\nBy era (buyer, net):")
            print(et.to_string())
            hu = v[v["holds_up"]]
            hus = sv[sv["holds_up"]]
            print(f"\nCells that hold up (discovery BH + holdout + stress): buyer side {len(hu)} of {len(v)}, "
                  f"seller side {len(hus)} of {len(sv)}")
            show = ["symbol", "side", "dbucket", "tdte", "n_exp_disc", "mean_disc", "n_exp_hold", "mean_hold",
                    "hold_stress", "win_share_hold", "worst_hold"]
            if len(hus):
                print("Seller side, holds up:")
                print(hus[show].to_string(index=False))
            if len(hu):
                print("Buyer side, holds up:")
                print(hu[show].to_string(index=False))
            st = structure_test(e, f"{col}_net")
            st.to_csv(os.path.join(OUT, f"structure_{col}.csv"), index=False)
            print(f"\nStructure pre-2020 vs post-2020 (buyer net): {int(st['bh'].sum())} of {len(st)} cells differ (BH q=0.05)")
            if st["bh"].any():
                print(st[st["bh"]].sort_values("p").head(20).to_string(index=False))

        print("\n" + "=" * 90 + "\nPOOLED VIEW (all horizons, buyer net per Rs100, mean of settlement means)")
        e["era"] = ERAS[0][0]
        for name, start in ERAS[1:]:
            e.loc[e["settled"] >= start, "era"] = name
        for col in ("buy_net", "hbuy_net"):
            per = e.groupby(["symbol", "side", "dbucket", "era", "settled"], observed=True)[col].mean().reset_index()
            print(f"\n{col}:")
            print(per.pivot_table(index=["symbol", "side", "dbucket"], columns="era", values=col, aggfunc="mean",
                                  observed=True)[[x for x, _ in ERAS]].to_string())
        print("\nWorst settlements for sellers of 0.02-0.15 delta puts (all horizons pooled, seller net per Rs100):")
        w = e[(e["side"] == "PE") & e["dbucket"].isin(DELTA_LABELS[:2])]
        print(w.groupby(["symbol", "settled"])["sell_net"].mean().nsmallest(8).to_string())

    with open(os.path.join(OUT, "_report.txt"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


if __name__ == "__main__":
    main()
