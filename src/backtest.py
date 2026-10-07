"""Stage 5 portfolio simulator: positions in NIFTY / BANKNIFTY options marked to market
every session, optionally delta-hedged at the close with the expiry's forward (futures).

Prices: the session's traded mark when the contract traded (>= MIN_TRADED contracts),
otherwise Black-76 at the IV interpolated (linear in log-moneyness, flat beyond the
quoted range) from that session's traded out-of-the-money IVs of the same expiry; the
previous session's smile when none traded. Settlement session: intrinsic value at the
settlement price (= closing spot, Stage 4 check A2). Contracts are identified by
(final_expiry, strike, side), so NSE's relabelled expiries stay one contract.

Costs follow Stage 1: half-spread by price (live 2026 chains), exchange fee + GST,
stamp duty (buys), STT on sales by era and on exercise of long ITM options (a long may
instead sell at the settlement-session close if better), Rs 20 + GST brokerage per order
spread over one lot, futures 2 bp of notional traded. Stress: option spreads x3 before
2020 and x2 after, futures cost x2.
Interest on premium and margin is ignored (P&L is in excess of cash).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src import contracts as ct  # noqa: E402

PANEL = os.path.join(PROJECT_ROOT, "data", "processed", "contracts", "panel")
CONTRACTS = os.path.join(PROJECT_ROOT, "data", "processed", "contracts")
FWD_OK = ("parity", "futures", "parity_local")
MAX_DTE = 60
FUT_COST = 0.0002
FIN26 = pd.Timestamp("2026-04-01")   # Finance Act 2026 STT rates (NSE/FATAX/73524)
FUT_STT_STEP = 0.00015               # futures STT 0.02% -> 0.05% on sales ~ half of hedge turnover
POST2020 = pd.Timestamp("2020-01-01")


# ---------------------------------------------------------------- costs (Stage 1 rules)

def stt_sell_rate(d) -> float:
    d = pd.Timestamp(d)
    return 0.00017 if d < pd.Timestamp("2016-06-01") else 0.0005 if d < pd.Timestamp("2023-04-01") \
        else 0.000625 if d < pd.Timestamp("2024-10-01") else 0.001 if d < FIN26 else 0.0015


def stt_exercise_rate(d) -> float:
    return 0.00125 if pd.Timestamp(d) < FIN26 else 0.0015


def exchange_rate(d) -> float:
    d = pd.Timestamp(d)
    return (0.00053 if d < pd.Timestamp("2024-10-01") else 0.00035 if d < FIN26 else 0.0003553) * 1.18


def stamp_rate(d) -> float:
    return 0.00003 if pd.Timestamp(d) >= pd.Timestamp("2020-07-01") else 0.0


class Costs:
    def __init__(self, spread_table: pd.DataFrame, stress: bool = False):
        self.lm = np.log(spread_table["mid"].to_numpy(float))
        self.pct = spread_table["half_pct"].to_numpy(float)
        self.stress = stress

    def spread_mult(self, d) -> float:
        if not self.stress:
            return 1.0
        return 3.0 if pd.Timestamp(d) < POST2020 else 2.0

    def half_spread(self, price, d) -> np.ndarray:
        price = np.maximum(np.asarray(price, float), 0.05)
        return np.maximum(np.interp(np.log(price), self.lm, self.pct) * price, 0.025) * self.spread_mult(d)

    def trade(self, price, qty, d, lot) -> np.ndarray:
        """Cost (Rs, >= 0) of trading qty units (+ buy, - sell) at price on session d."""
        price, qty = np.asarray(price, float), np.asarray(qty, float)
        buy = qty > 0
        fee = price * (exchange_rate(d) + np.where(buy, stamp_rate(d), stt_sell_rate(d)))
        brk = 20 * 1.18 / lot
        return np.abs(qty) * (self.half_spread(price, d) + fee + brk)

    def fut(self, notional_traded, d=None) -> float:
        step = FUT_STT_STEP if d is not None and pd.Timestamp(d) >= FIN26 else 0.0
        return float(notional_traded) * (FUT_COST * (2.0 if self.stress else 1.0) + step)


def spread_table(cache: str) -> pd.DataFrame:
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    import stage1_option_returns as s1
    t = s1.spread_model()[["mid", "half_pct"]]
    t.to_parquet(cache)
    return t


# ---------------------------------------------------------------- market data

@dataclass
class Slice:
    """One (session, final expiry): forward, time, discount, smile and traded marks."""
    F: float
    T: float
    DF: float
    k: np.ndarray            # log(K / F) of the smile points (sorted)
    iv: np.ndarray
    marks: dict = field(default_factory=dict)   # (strike, side) -> mark
    stale: bool = False


class Market:
    def __init__(self, symbol: str):
        self.symbol = symbol
        em = pd.read_parquet(os.path.join(CONTRACTS, "_expiry_map.parquet"))
        em = em[em["symbol"] == symbol]
        self.final_of = em.set_index("expiry")["final_expiry"].to_dict()
        fe = em.drop_duplicates("final_expiry").set_index("final_expiry")
        self.settled = fe["settled"].to_dict()
        flt = [("symbol", "=", symbol), ("dte", "<=", MAX_DTE)]
        a = pd.read_parquet(PANEL, columns=["date", "expiry", "forward", "fwd_source", "spot", "rate", "lot_size"],
                            filters=flt)
        a = a.drop_duplicates(["date", "expiry", "fwd_source"])
        a["final_expiry"] = a["expiry"].map(self.final_of)
        a = a[a["final_expiry"].notna()]
        day = a.groupby("date").agg(spot=("spot", "first"), rate=("rate", "first"), lot=("lot_size", "max"))
        self.sessions = pd.DatetimeIndex(sorted(day.index))
        self.spot = day["spot"]
        self.rate = day["rate"].ffill()
        self.lot = day["lot"].ffill()
        rank = a["fwd_source"].map({"expiry_day": 0, "parity": 1, "futures": 2, "parity_local": 3})
        fw = a[rank.notna()].assign(rank=rank).sort_values("rank").drop_duplicates(["date", "final_expiry"])
        self.fwd = fw.set_index(["date", "final_expiry"])["forward"].to_dict()
        cols = ["date", "expiry", "strike", "side", "tdte", "mark", "contracts", "fwd_source", "iv", "log_moneyness"]
        tr = pd.read_parquet(PANEL, columns=cols, filters=flt + [("contracts", ">=", ct.MIN_TRADED)])
        tr["final_expiry"] = tr["expiry"].map(self.final_of)
        tr = tr[tr["final_expiry"].notna() & tr["mark"].notna() & (tr["mark"] > 0)]
        self.traded = tr
        self._build_slices(tr)

    def _build_slices(self, tr: pd.DataFrame):
        otm = np.where(tr["side"] == "CE", tr["log_moneyness"] >= 0, tr["log_moneyness"] <= 0)
        sm = tr[otm & tr["iv"].notna() & tr["fwd_source"].isin(FWD_OK) & (tr["iv"] > 0.01) & (tr["iv"] < 3)]
        smiles = {key: (g["log_moneyness"].to_numpy(float), g["iv"].to_numpy(float))
                  for key, g in sm.sort_values("log_moneyness").groupby(["date", "final_expiry"])}
        marks = {key: dict(zip(zip(g["strike"].to_numpy(float), g["side"].to_numpy()), g["mark"].to_numpy(float)))
                 for key, g in tr.groupby(["date", "final_expiry"])}
        self.smiles, self.marks = smiles, marks

    def slice(self, d, fe) -> Slice | None:
        d, fe = pd.Timestamp(d), pd.Timestamp(fe)
        st = self.settled.get(fe)
        F = self.fwd.get((d, fe))
        if F is None or st is None or d > st:
            return None
        T = max((st - d).days, 0) / 365.0
        DF = float(np.exp(-self.rate.get(d, 0.06) * T))
        sm = self.smiles.get((d, fe))
        stale = False
        if sm is None:
            i = self.sessions.searchsorted(d) - 1
            while i >= 0 and sm is None and (d - self.sessions[i]).days <= 10:
                sm = self.smiles.get((self.sessions[i], fe))
                i -= 1
            stale = True
        if sm is None:
            return None
        return Slice(F, T, DF, sm[0], sm[1], self.marks.get((d, fe), {}), stale)

    @staticmethod
    def iv_at(s: Slice, K) -> np.ndarray:
        k = np.log(np.asarray(K, float) / s.F)
        if len(s.k) == 1:
            return np.full(k.shape, s.iv[0])
        return np.interp(k, s.k, s.iv)

    def price(self, s: Slice, K, side) -> tuple[np.ndarray, np.ndarray]:
        """(value, traded flag) for strikes K / sides on a slice; model value if untraded."""
        K = np.asarray(K, float)
        side = np.asarray(side)
        is_call = side == "CE"
        mk = np.array([s.marks.get((k, sd), np.nan) for k, sd in zip(K, side)])
        if s.T <= 0:
            return np.where(is_call, np.maximum(s.F - K, 0), np.maximum(K - s.F, 0)), np.isfinite(mk)
        model = ct.b76_price(s.F, K, s.T, self.iv_at(s, K), s.DF, is_call)
        return np.where(np.isfinite(mk), mk, model), np.isfinite(mk)

    def delta_bs(self, s: Slice, K, side) -> np.ndarray:
        K = np.asarray(K, float)
        is_call = np.asarray(side) == "CE"
        if s.T <= 0:
            return np.zeros(len(K))
        d, _, _, _ = ct.b76_greeks(s.F, K, s.T, self.iv_at(s, K), s.DF, 0.0, is_call)
        return d

    def settle_value(self, fe, K, side) -> np.ndarray:
        S = float(self.spot[self.settled[pd.Timestamp(fe)]])
        K = np.asarray(K, float)
        return np.where(np.asarray(side) == "CE", np.maximum(S - K, 0), np.maximum(K - S, 0))

    def next_session(self, d, n=1):
        i = self.sessions.searchsorted(pd.Timestamp(d))
        j = i + n
        return self.sessions[j] if 0 <= j < len(self.sessions) else None


# ---------------------------------------------------------------- positions

@dataclass
class Position:
    strategy: str
    symbol: str
    entry: pd.Timestamp
    exit: pd.Timestamp              # settlement session = hold to expiry
    final_expiry: pd.Timestamp
    strikes: np.ndarray
    sides: np.ndarray
    qty: np.ndarray                 # units of the index (+ long, - short), already sized
    hedge: str | None = "bs"        # None / "bs" / "mv"
    info: dict = field(default_factory=dict)


def _expiry_cost(mk: Market, costs: Costs, d, fe, pos: Position, vals, lot) -> float:
    """Exercise STT on long ITM legs, or selling them at the settlement close if cheaper."""
    long_itm = (pos.qty > 0) & (vals > 0)
    if not long_itm.any():
        return 0.0
    ex = stt_exercise_rate(d) * np.where(d < pd.Timestamp("2016-06-01"), float(mk.spot[d]), vals)
    close_mk = np.array([mk.marks.get((d, fe), {}).get((k, sd), np.nan) for k, sd in zip(pos.strikes, pos.sides)])
    sell = close_mk - costs.half_spread(np.nan_to_num(close_mk, nan=0.05), d) \
        - close_mk * (exchange_rate(d) + stt_sell_rate(d)) - 20 * 1.18 / lot
    loss = np.where(np.isfinite(sell), np.minimum(ex, vals - sell), ex)
    return float(np.sum(np.where(long_itm, pos.qty * np.maximum(loss, 0), 0.0)))


def simulate(pos: Position, mk: Market, costs: list[Costs], mv_delta=None) -> pd.DataFrame:
    """Daily rows of one position: option and hedge P&L, cost under each cost model
    (cost0, cost1, ...), net delta after the hedge, short notional, stale-smile flag.
    Empty if the position cannot be entered (a leg untraded on the entry session).

    mv_delta(date, final_expiry, strikes, sides, slice) -> per-unit deltas or None."""
    fe = pos.final_expiry
    st = mk.settled[fe]
    days = mk.sessions[(mk.sessions >= pos.entry) & (mk.sessions <= pos.exit)]
    lot = float(mk.lot.get(pos.entry, 50))
    nc = len(costs)
    rows = []
    prev_val, h_prev, F_prev = None, 0.0, None
    for d in days:
        at_settle = d == st
        s = None if at_settle else mk.slice(d, fe)
        c = np.zeros(nc)
        if d == pos.entry:
            if s is None:
                return pd.DataFrame()
            vals, traded = mk.price(s, pos.strikes, pos.sides)
            if not traded.all():
                return pd.DataFrame()
            c += [float(k.trade(vals, pos.qty, d, lot).sum()) for k in costs]
        elif at_settle:
            vals = mk.settle_value(fe, pos.strikes, pos.sides)
            c += [_expiry_cost(mk, k, d, fe, pos, vals, lot) for k in costs]
        elif s is None:
            rows.append(dict(date=d, option=0.0, hedge=0.0, **{f"cost{i}": 0.0 for i in range(nc)},
                             delta_net=np.nan, short_notional=np.nan, stale=True))
            continue
        else:
            vals, _ = mk.price(s, pos.strikes, pos.sides)
            if d == pos.exit:
                c += [float(k.trade(vals, -pos.qty, d, lot).sum()) for k in costs]
        F = float(mk.spot[d]) if at_settle else s.F
        value = float(np.sum(pos.qty * vals))
        opt = 0.0 if prev_val is None else value - prev_val
        hedge_pnl = h_prev * (F - F_prev) if F_prev is not None else 0.0
        opt_delta = 0.0
        if s is not None and d != pos.exit:
            dl = mk.delta_bs(s, pos.strikes, pos.sides)
            if pos.hedge == "mv" and mv_delta is not None:
                m = mv_delta(d, fe, pos.strikes, pos.sides, s)
                dl = dl if m is None else m
            opt_delta = float(np.sum(pos.qty * dl))
        h = -opt_delta if pos.hedge else 0.0
        if pos.hedge and not at_settle:
            c += [k.fut(abs(h - h_prev) * F, d) for k in costs]
        rows.append(dict(date=d, option=opt, hedge=hedge_pnl, **{f"cost{i}": c[i] for i in range(nc)},
                         delta_net=opt_delta + h, short_notional=float(np.abs(np.minimum(pos.qty, 0)).sum() * F),
                         stale=bool(s.stale) if s is not None else False))
        prev_val, h_prev, F_prev = value, h, F
    out = pd.DataFrame(rows)
    out["strategy"], out["symbol"], out["entry"] = pos.strategy, pos.symbol, pos.entry
    return out


# ---------------------------------------------------------------- sizing

MOVES = np.array([-0.15, -0.10, -0.05, 0.05, 0.10])
IV_MULT = np.array([1.0, 1.5])


def stress_loss(mk: Market, s: Slice, strikes, sides, qty, hedged: bool) -> float:
    """Worst loss (Rs, > 0) of the position on an instant index move with IV x1 or x1.5
    (time unchanged), hedge (if any) at today's Black-76 delta."""
    strikes, qty = np.asarray(strikes, float), np.asarray(qty, float)
    is_call = np.asarray(sides) == "CE"
    v0, _ = mk.price(s, strikes, sides)
    iv = mk.iv_at(s, strikes)
    h = -float(np.sum(qty * mk.delta_bs(s, strikes, sides))) if hedged else 0.0
    worst = 0.0
    for m in MOVES:
        F1 = s.F * (1 + m)
        for vm in IV_MULT:
            v1 = ct.b76_price(F1, strikes, max(s.T, 1e-6), iv * vm, s.DF, is_call)
            pnl = float(np.sum(qty * (v1 - v0))) + h * (F1 - s.F)
            worst = min(worst, pnl)
    return -worst
