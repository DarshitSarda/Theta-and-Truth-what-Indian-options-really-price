"""Realistic execution of the frozen engine: whole lots, next-day VWAP fills, real Zerodha/NSE
costs, futures vs synthetic-futures hedging, hedge bands, account sizes, collateral interest.

Rules fixed before running (agreed with the user 2026-10-07; do not change after seeing results):
  Strategy   the frozen Stage 5 engine exactly (stage5 positions, strategy eng_straddle_bs): at each
             monthly settlement close sell the ATM straddle of the next monthly expiry, sized so the
             Stage 5 crash scenario costs 25% of the symbol's capital (half the book per symbol),
             held to settlement. Only execution changes.
  Accounts   total capital C = Rs 10 lakh, 25 lakh, 50 lakh, 1 crore, 5 crore. Option lots =
             floor(Stage 5 units x C / lot), the contract's own lot size; 0 lots -> cycle not traded.
  Timing     every decision uses the close of day t; the order fills at the VWAP of day t+1.
             Entry: decided at the settlement close, filled next session; if either leg has no
             usable VWAP, retried for up to 3 sessions, else the cycle is skipped. Hedge orders with
             no usable VWAP are not filled (re-decided at the next close). No hedge orders are
             placed for the settlement session; everything settles at the closing index value.
             Usable VWAP: inside the day's range (build_vwap.py), >= 10 contracts, precision <= 2%.
  Marks      daily MTM at the close: Stage 5 marks (traded close, else smile model); futures at
             their close; settlement at the closing index.
  Hedges     target = -(option delta at the close, Black-76 at the smile IV) in index units.
    instruments  FUT  same-expiry monthly index future
                 SYN  synthetic future at the straddle's strike (buy call + sell put), i.e. the
                      straddle legs' quantities are adjusted; target / DF synthetic units
    rules    exact   fractional options and hedge, daily (benchmark, not implementable)
             round   daily, hedge = target rounded to whole lots
             band1 / band2   trade (to the rounded target) only if |target - hedge| > 1 / 2 lots
             every2 / every5 rebalance to the rounded target every 2nd / 5th decision day
             none    no hedge
  Costs (per executed order; GST 18% on brokerage + exchange + SEBI)
    options  half-spread (Stage 5 table by price; stress x3 before 2020, x2 after) + exchange fee
             on premium (0.053% to Sep-2024, 0.03503% to Mar-2026, 0.03553% after) + SEBI Rs 10/cr +
             STT on sales (Stage 1 eras, 0.15% from Apr-2026) + stamp 0.003% on buys (from Jul-2020)
             + Rs 20 brokerage.
    futures  0.5 bp of notional (stress x2) + exchange 0.0019% (to Sep-2024), 0.00173% (to
             Mar-2026), 0.00183% (after) + SEBI + STT on sales 0.017% (to May-2013), 0.01% (to
             Mar-2023), 0.0125% (to Sep-2024), 0.02% (to Mar-2026), 0.05% (after) + stamp 0.002%
             on buys (from Jul-2020) + Rs 20 brokerage.
    expiry   net long ITM legs (possible with SYN): the cheaper of exercise STT (0.125%, 0.15% from
             Apr-2026; on the index value before Jun-2016, else on intrinsic) and selling at the
             settlement close at intrinsic (half-spread + fees + brokerage), as in Stage 5.
  Note       "round" is the +/-0.5-lot band (with whole-lot hedges, |target - hedge| > 0.5 lot is
             the same as the rounded target differing from the hedge).
  Interest   the whole capital sits in cash-equivalents earning RBI repo - 0.5 pp (daily accrual);
             Sharpe and the verdict use the excess return (strategy P&L without interest).
  Margin     approx. 12% x F x (short option units + |futures units|) - premium received;
             reported as the share of days it exceeds C (no trading change).
  Selection  per account size: among the implementable rules (FUT/SYN x round/band1/band2/every2/
             every5, plus none) the highest excess Sharpe over 2008-2017 (normal costs) is chosen;
             it is then judged on 2018-2026. Every rule is reported.
  Verdict    "viable at C" if the chosen rule's 2018-2026 excess return is > 0 under normal AND stress
             costs and at least 80% of cycles were tradeable.
  Checks     synthetic market (scripts/realistic_engine_selftest.py) must pass first; reconciliation
             of rule "exact" with the Stage 5 engine.
Writes data/processed/realistic/ (daily.parquet, cycles.parquet, _report.txt)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src import backtest as B  # noqa: E402
from src import contracts as ct  # noqa: E402

D5 = os.path.join(PROJECT_ROOT, "data", "processed", "stage5")
VW = os.path.join(PROJECT_ROOT, "data", "processed", "vwap", "vwap.parquet")
OUT = os.path.join(PROJECT_ROOT, "data", "processed", "realistic")
CAPITALS = {"10L": 1e6, "25L": 2.5e6, "50L": 5e6, "1Cr": 1e7, "5Cr": 5e7}
RULES = ["exact", "round", "band1", "band2", "every2", "every5", "none"]
INSTR = ["FUT", "SYN"]
MIN_CONTRACTS, MAX_PREC = 10, 0.02
ENTRY_TRIES = 3
GST = 1.18
SEBI = 1e-6
FUT_SPREAD = 0.00005
MARGIN = 0.12
SELECT_END = pd.Timestamp("2018-01-01")


# ---------------------------------------------------------------- costs

def _ts(d):
    return pd.Timestamp(d)


def opt_fee_rate(d, buy: bool) -> float:
    """Exchange + SEBI (with GST) + STT (sell) / stamp (buy), as a fraction of premium."""
    d = _ts(d)
    exch = 0.00053 if d < _ts("2024-10-01") else 0.0003503 if d < B.FIN26 else 0.0003553
    tax = B.stamp_rate(d) if buy else B.stt_sell_rate(d)
    return (exch + SEBI) * GST + tax


def fut_fee_rate(d, buy: bool) -> float:
    d = _ts(d)
    exch = 0.000019 if d < _ts("2024-10-01") else 0.0000173 if d < B.FIN26 else 0.0000183
    if buy:
        tax = 0.00002 if d >= _ts("2020-07-01") else 0.0
    else:
        tax = 0.00017 if d < _ts("2013-06-01") else 0.0001 if d < _ts("2023-04-01") else \
            0.000125 if d < _ts("2024-10-01") else 0.0002 if d < B.FIN26 else 0.0005
    return (exch + SEBI) * GST + tax


BROKERAGE = 20 * GST


@dataclass
class CostModel:
    costs: B.Costs                      # half-spread table (normal or stress)
    stress: bool

    def option(self, d, price, units) -> float:
        if units == 0:
            return 0.0
        hs = float(self.costs.half_spread(np.array([price]), d)[0])
        return abs(units) * (hs + price * opt_fee_rate(d, units > 0)) + BROKERAGE

    def future(self, d, price, units) -> float:
        if units == 0:
            return 0.0
        return abs(units) * price * (FUT_SPREAD * (2.0 if self.stress else 1.0) + fut_fee_rate(d, units > 0)) \
            + BROKERAGE

    def expiry(self, d, value, units, spot) -> float:
        """Long ITM option at settlement: exercise STT, or selling it at the close if cheaper."""
        if units <= 0 or value <= 0:
            return 0.0
        base = spot if _ts(d) < _ts("2016-06-01") else value
        return min(units * B.stt_exercise_rate(d) * base, self.option(d, value, -units))


# ---------------------------------------------------------------- core (array-based, testable)

@dataclass
class Cycle:
    """One straddle cycle on arrays indexed by session k = 0..N (0 = decision day, N = settlement)."""
    dates: list
    K: float
    lot: float                          # option lot (also synthetic lot)
    fut_lot: float
    units: float                        # straddle size in index units (> 0), before rounding
    opt_close: np.ndarray               # (N+1, 2) call, put marks at the close; row N = intrinsic
    opt_vwap: np.ndarray                # (N+1, 2) usable VWAP or nan
    delta: np.ndarray                   # (N+1, 2) per-unit Black-76 deltas at the close (row N unused)
    fut_close: np.ndarray               # (N+1,) row N = closing index
    fut_vwap: np.ndarray                # (N+1,) usable VWAP or nan
    DF: np.ndarray                      # (N+1,)


def run_cycle(c: Cycle, rule: str, instr: str, cms: list, fractional: bool) -> dict | None:
    """Simulate one cycle. Returns per-day arrays (length N+1): pnl (gross), cost per cost model,
    trades, margin; plus summary. None if the cycle cannot be entered (0 lots or no fill)."""
    N = len(c.dates) - 1
    n = c.units if fractional else np.floor(c.units / c.lot) * c.lot
    if n <= 0:
        return None
    fill = next((k for k in range(1, min(1 + ENTRY_TRIES, N)) if np.isfinite(c.opt_vwap[k]).all()), None)
    if fill is None:
        return None
    q = np.zeros(2)                      # option units held (call, put); short < 0
    h = 0.0                              # hedge units (futures, or synthetic = +call/-put)
    pend_h = 0.0                         # hedge order decided at the previous close
    pnl = np.zeros(N + 1)
    cost = np.zeros((len(cms), N + 1))
    trades = np.zeros(N + 1)
    margin = np.zeros(N + 1)
    hpos, qpos = np.zeros(N + 1), np.zeros((N + 1, 2))
    hlot = c.fut_lot if instr == "FUT" else c.lot
    d = c.dates
    for k in range(fill, N + 1):
        # 1. holdings from the previous close, marked to today's close
        if k > fill:
            pnl[k] += float(q @ (c.opt_close[k] - c.opt_close[k - 1]))
            if instr == "FUT":
                pnl[k] += h * (c.fut_close[k] - c.fut_close[k - 1])
        # 2. today's fills at VWAP, marked to today's close
        if k == fill:
            px, dq = c.opt_vwap[k], np.array([-n, -n])
            pnl[k] += float(dq @ (c.opt_close[k] - px))
            for i, cm in enumerate(cms):
                cost[i, k] += sum(cm.option(d[k], px[j], dq[j]) for j in range(2))
            prem = float(-(dq @ px))
            q += dq
            trades[k] += 2
        if pend_h != 0:
            if instr == "FUT" and np.isfinite(c.fut_vwap[k]):
                px = c.fut_vwap[k]
                pnl[k] += pend_h * (c.fut_close[k] - px)
                for i, cm in enumerate(cms):
                    cost[i, k] += cm.future(d[k], px, pend_h)
                h += pend_h
                trades[k] += 1
            elif instr == "SYN" and np.isfinite(c.opt_vwap[k]).all():
                px, dq = c.opt_vwap[k], np.array([pend_h, -pend_h])
                pnl[k] += float(dq @ (c.opt_close[k] - px))
                for i, cm in enumerate(cms):
                    cost[i, k] += sum(cm.option(d[k], px[j], dq[j]) for j in range(2))
                q += dq
                h += pend_h
                trades[k] += 2
            pend_h = 0.0
        if k == N:
            for i, cm in enumerate(cms):
                cost[i, k] += sum(cm.expiry(d[k], c.opt_close[k, j], q[j], c.fut_close[k]) for j in range(2))
        hpos[k], qpos[k] = h, q
        # 3. margin
        shorts = float(np.abs(np.minimum(q, 0)).sum())
        margin[k] = MARGIN * c.fut_close[k] * (shorts + (abs(h) if instr == "FUT" else 0.0)) - prem
        # 4. decision at today's close, filled at tomorrow's VWAP (none for the settlement session)
        if rule == "none" or k >= N - 1:
            continue
        straddle_delta = float(-n * (c.delta[k, 0] + c.delta[k, 1]))
        target = -straddle_delta if instr == "FUT" else -straddle_delta / c.DF[k]
        cur = h
        if rule == "exact":
            pend_h = target - cur
            continue
        rt = np.round(target / hlot) * hlot
        gap = abs(target - cur)
        if rule == "round":
            go = rt != cur
        elif rule.startswith("band"):
            go = gap > float(rule[4:]) * hlot and rt != cur
        else:
            every = int(rule[5:])
            go = ((k - fill) % every == 0) and rt != cur
        pend_h = (rt - cur) if go else 0.0
    return dict(pnl=pnl, cost=cost, trades=trades, margin=margin, lots=n / c.lot, fill=fill, prem=prem,
                hpos=hpos, qpos=qpos)


# ---------------------------------------------------------------- real data

class RealData:
    def __init__(self, sym: str):
        t0 = time.time()
        self.sym = sym
        self.mk = B.Market(sym)
        em = pd.read_parquet(os.path.join(B.CONTRACTS, "_expiry_map.parquet"))
        fo = em[em["symbol"] == sym].set_index("expiry")["final_expiry"].to_dict()
        v = pd.read_parquet(VW, filters=[("symbol", "=", sym)])
        v["final_expiry"] = v["expiry"].map(fo).fillna(v["expiry"])
        good = v["usable"] & (v["contracts"] >= MIN_CONTRACTS) & (v["prec"] <= MAX_PREC * v["vwap"].abs())
        o = v[(v["instrument"] == "OPT")]
        self.ovw = o[good.loc[o.index]].set_index(["date", "final_expiry", "strike", "side"])["vwap"].to_dict()
        self.olot = o.set_index(["date", "final_expiry", "strike", "side"])["lot_size"].to_dict()
        f = v[v["instrument"] == "FUT"]
        self.fvw = f[good.loc[f.index]].set_index(["date", "final_expiry"])["vwap"].to_dict()
        self.fcl = f.set_index(["date", "final_expiry"])["close"].to_dict()
        self.flot = f.set_index(["date", "final_expiry"])["lot_size"].to_dict()
        print(f"  {sym}: market + VWAP loaded ({time.time() - t0:.0f}s)", flush=True)

    def cycle(self, pos) -> Cycle | None:
        mk, fe = self.mk, pos.final_expiry
        K = float(pos.strikes.split(",")[0])
        days = list(mk.sessions[(mk.sessions >= pos.entry) & (mk.sessions <= pos.settled)])
        N = len(days) - 1
        if N < 3:
            return None
        oc, ov, dl = np.full((N + 1, 2), np.nan), np.full((N + 1, 2), np.nan), np.zeros((N + 1, 2))
        fc, fv, DF = np.full(N + 1, np.nan), np.full(N + 1, np.nan), np.ones(N + 1)
        sides = np.array(["CE", "PE"])
        last_s = None
        for k, d in enumerate(days):
            if k == N:
                oc[k] = mk.settle_value(fe, [K, K], sides)
                fc[k] = float(mk.spot[d])
                continue
            s = mk.slice(d, fe) or last_s
            if s is None:
                return None
            last_s = s
            oc[k], _ = mk.price(s, [K, K], sides)
            dl[k] = mk.delta_bs(s, [K, K], sides)
            DF[k] = s.DF
            fc[k] = self.fcl.get((d, fe), s.F)
            fv[k] = self.fvw.get((d, fe), np.nan)
            ov[k] = [self.ovw.get((d, fe, K, sd), np.nan) for sd in sides]
        lot = self.olot.get((days[1], fe, K, "CE")) or self.olot.get((days[1], fe, K, "PE")) or float(mk.lot.get(days[1]))
        flot = self.flot.get((days[1], fe), lot)
        return Cycle(days, K, float(lot), float(flot), float(pos.qty), oc, ov, dl, fc, fv, DF)


# ---------------------------------------------------------------- run + report

def interest_series(sessions: pd.DatetimeIndex) -> pd.Series:
    rates = ct.load_repo_rates(os.path.join(PROJECT_ROOT, "config", "india_repo_rate.csv"))
    gaps = np.r_[1, np.diff(sessions.values).astype("timedelta64[D]").astype(int)]
    r = np.array([ct.rate_on(rates, d.date()) for d in sessions]) - 0.005
    return pd.Series(r * gaps / 365.0, index=sessions)


def simulate_all():
    pos = pd.read_parquet(os.path.join(D5, "positions.parquet"))
    pos = pos[pos["strategy"] == "eng_straddle_bs"].sort_values(["symbol", "entry"])
    st = B.spread_table(os.path.join(D5, "_spread_table.parquet"))
    cms = [CostModel(B.Costs(st), False), CostModel(B.Costs(st, stress=True), True)]
    rows, cyc = [], []
    sessions = None
    for sym in ("NIFTY", "BANKNIFTY"):
        rd = RealData(sym)
        sessions = rd.mk.sessions if sessions is None else sessions.union(rd.mk.sessions)
        built = [(p, rd.cycle(p)) for p in pos[pos["symbol"] == sym].itertuples()]
        print(f"  {sym}: {sum(c is not None for _, c in built)}/{len(built)} cycles with data", flush=True)
        for cap_name, C in CAPITALS.items():
            for instr in INSTR:
                for rule in RULES:
                    if instr == "SYN" and rule in ("none",):
                        continue
                    for p, c in built:
                        r = None if c is None else run_cycle(
                            Cycle(c.dates, c.K, c.lot, c.fut_lot, c.units * C, c.opt_close, c.opt_vwap, c.delta,
                                  c.fut_close, c.fut_vwap, c.DF), rule, instr, cms, rule == "exact")
                        cyc.append(dict(symbol=sym, cap=cap_name, instr=instr, rule=rule, pid=p.pid, entry=p.entry,
                                        settled=p.settled, traded=r is not None,
                                        lots=r["lots"] if r else 0.0, trades=r["trades"].sum() if r else 0.0,
                                        prem=r["prem"] if r else 0.0,
                                        margin_breach=float((r["margin"] > C).mean()) if r else np.nan))
                        if r is None:
                            continue
                        rows.append(pd.DataFrame({"date": c.dates, "symbol": sym, "cap": cap_name, "instr": instr,
                                                  "rule": rule, "pid": p.pid, "pnl": r["pnl"] / C,
                                                  "cost": r["cost"][0] / C, "cost_s": r["cost"][1] / C}))
        print(f"  {sym}: simulated", flush=True)
    daily = pd.concat(rows, ignore_index=True)
    return daily, pd.DataFrame(cyc), interest_series(sessions)


def summarize(daily, cyc, intr, lo, hi):
    d = daily[(daily["date"] >= lo) & (daily["date"] < hi)]
    g = d.groupby(["cap", "instr", "rule", "date"])[["pnl", "cost", "cost_s"]].sum().reset_index()
    out = []
    days = intr[(intr.index >= lo) & (intr.index < hi)]
    yrs = (days.index.max() - days.index.min()).days / 365.25
    for (cap, instr, rule), x in g.groupby(["cap", "instr", "rule"]):
        s = x.set_index("date").reindex(days.index).fillna(0.0)
        ex = s["pnl"] - s["cost"]
        exs = s["pnl"] - s["cost_s"]
        cum = ex.cumsum()
        cy = cyc[(cyc["cap"] == cap) & (cyc["instr"] == instr) & (cyc["rule"] == rule) &
                 (cyc["settled"] >= lo) & (cyc["settled"] < hi)]
        out.append(dict(cap=cap, instr=instr, rule=rule, excess_yr=ex.sum() / yrs, stress_yr=exs.sum() / yrs,
                        with_interest_yr=(ex.sum() + days.sum()) / yrs, gross_yr=s["pnl"].sum() / yrs,
                        cost_yr=s["cost"].sum() / yrs, sharpe=ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
                        maxdd=(cum.cummax() - cum).max(), worst_month=ex.resample("ME").sum().min(),
                        traded=cy["traded"].mean(), trades_per_cycle=cy.loc[cy["traded"], "trades"].mean(),
                        lots=cy.loc[cy["traded"], "lots"].mean(), margin_breach=cy["margin_breach"].mean()))
    return pd.DataFrame(out), days.sum() / yrs


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    daily, cyc, intr = simulate_all()
    daily.to_parquet(os.path.join(OUT, "daily.parquet"), index=False)
    cyc.to_parquet(os.path.join(OUT, "cycles.parquet"), index=False)
    L = [f"Realistic execution of the frozen engine ({time.time() - t0:.0f}s); rules in the script header"]
    periods = {"2008-2017 (selection)": ("2008-01-01", SELECT_END), "2018-2026 (judged)": (SELECT_END, "2100-01-01"),
               "full 2008-2026": ("2008-01-01", "2100-01-01")}
    res = {}
    for lab, (lo, hi) in periods.items():
        s, ir = summarize(daily, cyc, intr, pd.Timestamp(lo), pd.Timestamp(hi))
        res[lab] = s
        L.append("\n" + "=" * 110 + f"\n{lab}  (collateral interest alone: {ir:+.2%}/yr)\n" + "=" * 110)
        show = s.copy()
        for c in ("excess_yr", "stress_yr", "with_interest_yr", "gross_yr", "cost_yr", "maxdd", "worst_month",
                  "traded", "margin_breach"):
            show[c] = (show[c] * 100).round(2)
        show["cap"] = pd.Categorical(show["cap"], list(CAPITALS))
        show = show.sort_values(["cap", "instr", "rule"])
        L.append("% of capital per year unless noted; excess = strategy P&L without collateral interest")
        L.append(show.round(2).to_string(index=False))
    L.append("\n" + "=" * 110 + "\nSELECTION (best excess Sharpe 2008-2017 among implementable rules) and VERDICT\n" + "=" * 110)
    sel, jud = res["2008-2017 (selection)"], res["2018-2026 (judged)"]
    for cap in CAPITALS:
        cand = sel[(sel["cap"] == cap) & (sel["rule"] != "exact")]
        if cand["sharpe"].notna().sum() == 0:
            L.append(f"  {cap}: nothing tradeable in 2008-2017")
            continue
        best = cand.loc[cand["sharpe"].idxmax()]
        j = jud[(jud["cap"] == cap) & (jud["instr"] == best["instr"]) & (jud["rule"] == best["rule"])].iloc[0]
        ex = jud[(jud["cap"] == cap) & (jud["instr"] == "FUT") & (jud["rule"] == "exact")].iloc[0]
        ok = j["excess_yr"] > 0 and j["stress_yr"] > 0 and j["traded"] >= 0.8
        L.append(f"  {cap}: chosen {best['instr']} {best['rule']} (2008-17 Sharpe {best['sharpe']:+.2f}) -> 2018-26 excess"
                 f" {j['excess_yr']:+.2%}/yr, stress {j['stress_yr']:+.2%}/yr, with interest {j['with_interest_yr']:+.2%}/yr,"
                 f" Sharpe {j['sharpe']:+.2f}, maxDD {j['maxdd']:.1%}, cycles traded {j['traded']:.0%}"
                 f" | exact benchmark {ex['excess_yr']:+.2%}/yr -> {'VIABLE' if ok else 'not viable'}")
    d5 = pd.read_parquet(os.path.join(D5, "daily.parquet"))
    e = d5[d5["strategy"] == "eng_straddle_bs"]
    L.append("\nRECONCILIATION with the Stage 5 engine (close fills, 2 bp futures, fractional), excess %/yr:")
    for lab, (lo, hi) in periods.items():
        x = e[(e["date"] >= lo) & (e["date"] < hi)]
        yrs = (x["date"].max() - x["date"].min()).days / 365.25
        ex = res[lab]
        r1 = ex[(ex["cap"] == "1Cr") & (ex["instr"] == "FUT") & (ex["rule"] == "exact")].iloc[0]
        L.append(f"  {lab}: Stage 5 {(x['option'] + x['hedge'] - x['cost0']).sum() / yrs:+.2%} gross"
                 f" {(x['option'] + x['hedge']).sum() / yrs:+.2%} | realistic exact (1 Cr) {r1['excess_yr']:+.2%}"
                 f" gross {r1['gross_yr']:+.2%}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
