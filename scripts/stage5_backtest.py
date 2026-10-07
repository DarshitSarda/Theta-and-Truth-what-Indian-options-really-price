"""Stage 5 build: can the Stage 4 mispricings be captured after costs, without one crash
wiping the book out? Simulates every position; scripts/stage5_report.py evaluates.

Rules fixed before running (do not change after seeing results):
  Book      capital 1, half per symbol (NIFTY, BANKNIFTY). Every position is sized on its
            entry session from that session's data only; P&L is a fraction of book capital
            (not compounded). Daily mark-to-market (src/backtest.py), Stage 1 costs, stress
            costs (option spreads x3 before 2020, x2 after; futures x2).
  Engine    at the close of each monthly expiry's settlement session, sell the next monthly
            expiry and hold it to settlement, delta-hedged at every close with that
            expiry's forward. Variants: ATM straddle (strike nearest the forward with both
            sides traded) and 25-delta strangle (traded OTM call / put with Black-76 delta
            nearest 0.25). Size: hedged stress loss (instant -15..+10% move, IV x1 / x1.5)
            = 25% of the symbol's capital. The variant with the higher discovery net
            Sharpe (book, settlements before 2018) is the engine; the other is reported.
            Unhedged copies (same size) are references only.
  O1        crash put spread: 5 sessions before each monthly settlement sell the traded put
            with |delta| nearest 0.04 (0.02-0.07), buy the highest traded put strike
            <= 0.97 x the short strike; hold to settlement, unhedged. Max loss (width - credit)
            = 3% of the symbol's capital. Skipped if either leg did not trade.
  O2        Budget: 3 sessions before each Budget session (known_from <= entry), sell an
            ATM straddle in the nearest expiry settling >= 2 sessions after the Budget
            session, delta-hedged, closed at the Budget session's close. Size: stress loss =
            10% of the symbol's capital. Elections excluded (Stage 4: under-priced).
  G1        size the engine by a variance-premium signal at entry, m = clip(R / median of R
            over discovery entries of that symbol, 0, 2):
              Bates  R = E^Q[QV to settlement] (Stage 3 fit) / calibrated GARCH forecast
              simple R = ATM IV^2 T / (trailing 22-session mean r^2 x sessions to settlement)
            compared on the entries where both exist.
  H1        hedge the engine with the Bates minimum-variance delta (dC/dF + rho sigma /F
            dC/dv0, the session's Stage 3 parameters) instead of Black-76 at the smile IV.
  Weekly    exploratory only (no decision): the same straddle engine on consecutive weekly
            expiries (<= 9 days apart), holdout years only.
Outputs data/processed/stage5/{positions,daily}.parquet
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src import backtest as B  # noqa: E402
from src import bates as bt  # noqa: E402
from src import calibrate as cb  # noqa: E402
from src import pmodel as pm  # noqa: E402
import stage4_build as s4  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "data", "processed", "stage5")
S4 = os.path.join(PROJECT_ROOT, "data", "processed", "stage4", "outcomes.parquet")
HOLDOUT = pd.Timestamp("2018-01-01")
C_SYM = 0.5
ENGINE_RISK, O1_RISK, O2_RISK = 0.25, 0.03, 0.10
O1_DELTA, O1_WIDTH, O1_TDTE = 0.04, 0.97, 5
O2_LEAD = 3
H = 1e-3


# ---------------------------------------------------------------- leg selection

def atm_straddle(s: B.Slice):
    both = [k for (k, sd) in s.marks if sd == "CE" and (k, "PE") in s.marks]
    if not both:
        return None
    K = min(both, key=lambda k: abs(np.log(k / s.F)))
    return np.array([K, K]), np.array(["CE", "PE"])


def strangle(mk: B.Market, s: B.Slice, target=0.25):
    legs = []
    for side, cond in (("CE", lambda k: k >= s.F), ("PE", lambda k: k <= s.F)):
        ks = np.array([k for (k, sd) in s.marks if sd == side and cond(k)])
        if len(ks) == 0:
            return None
        d = np.abs(mk.delta_bs(s, ks, np.array([side] * len(ks))))
        legs.append(ks[np.argmin(np.abs(d - target))])
    return np.array(legs), np.array(["CE", "PE"])


def put_spread(mk: B.Market, s: B.Slice):
    ks = np.array(sorted(k for (k, sd) in s.marks if sd == "PE" and k <= s.F))
    if len(ks) < 2:
        return None
    d = np.abs(mk.delta_bs(s, ks, np.array(["PE"] * len(ks))))
    ok = (d >= 0.02) & (d <= 0.07)
    if not ok.any():
        return None
    Ks = ks[ok][np.argmin(np.abs(d[ok] - O1_DELTA))]
    wings = ks[ks <= O1_WIDTH * Ks]
    if len(wings) == 0:
        return None
    return np.array([Ks, wings.max()]), np.array(["PE", "PE"])


# ---------------------------------------------------------------- Bates helpers

class BatesView:
    def __init__(self, symbol: str, mk: B.Market, events: pd.DataFrame):
        self.fits = s4.best_fits(s4.all_fits(symbol))
        self.ev = events
        self.mk = mk

    def params(self, d, fe):
        if d not in self.fits.index:
            return None
        fr = self.fits.loc[d]
        st = self.mk.settled[fe]
        n_ev = int(((self.ev["known_from"] <= d) & (self.ev["session"] > d) & (self.ev["session"] <= st)).sum()) \
            if fr["model"] == "bates_ev" else 0
        return s4.params_of(fr, n_ev)

    def mv_delta(self, d, fe, K, sides, s: B.Slice):
        p = self.params(d, fe)
        if p is None or s.T <= 1 / 365:
            return None
        ic = np.asarray(sides) == "CE"
        K = np.asarray(K, float)
        up = bt.price_cos(s.F * (1 + H), K, s.T, p, s.DF, ic)
        dn = bt.price_cos(s.F * (1 - H), K, s.T, p, s.DF, ic)
        vu = bt.price_cos(s.F, K, s.T, dataclasses.replace(p, v0=p.v0 * (1 + H)), s.DF, ic)
        vd = bt.price_cos(s.F, K, s.T, dataclasses.replace(p, v0=p.v0 * (1 - H)), s.DF, ic)
        cF = (up - dn) / (2 * H * s.F)
        cv = (vu - vd) / (2 * H * p.v0)
        out = cF + cv * p.rho * p.sigma / s.F
        return out if np.all(np.isfinite(out)) else None

    def expected_qv(self, d, fe, T):
        p = self.params(d, fe)
        return np.nan if p is None else float(sum(s4.qv_parts(T, p)))


# ---------------------------------------------------------------- strategies

def monthly_expiries(mk: B.Market) -> list[pd.Timestamp]:
    fe = pd.Series(sorted(mk.settled))
    return sorted(fe.groupby([fe.dt.year, fe.dt.month]).max().tolist())


def build_symbol(sym: str, costs):
    t0 = time.time()
    mk = B.Market(sym)
    events = cb.event_sessions(cb.load_events(), mk.sessions)
    last = mk.sessions[-1]
    mk.settled = {f: s for f, s in mk.settled.items() if s <= last}
    bv = BatesView(sym, mk, events)
    P = s4.PState(sym, mk.spot)
    o4 = pd.read_parquet(S4)
    cal = s4.calibration(o4[o4["symbol"] == sym])
    naive = pm.trailing_rv(P.r)
    ev = events[events["type"] == "budget"]
    positions, daily = [], []
    pid = [0]

    def run(pos: B.Position, mv=None, **info):
        r = B.simulate(pos, mk, costs, mv)
        if len(r) == 0:
            return False
        pid[0] += 1
        r["pid"] = pid[0]
        daily.append(r)
        positions.append(dict(pid=pid[0], strategy=pos.strategy, symbol=sym, entry=pos.entry, exit=pos.exit,
                              final_expiry=pos.final_expiry, settled=mk.settled[pos.final_expiry],
                              strikes=",".join(f"{k:g}" for k in pos.strikes), sides=",".join(pos.sides),
                              qty=float(abs(pos.qty[0])), hedge=pos.hedge or "none", **info))
        return True

    mon = monthly_expiries(mk)
    for prev, fe in zip(mon[:-1], mon[1:]):
        entry = mk.settled[prev]
        if entry not in mk.sessions or mk.settled[fe] <= entry:
            continue
        s = mk.slice(entry, fe)
        if s is None or s.stale:
            continue
        n_s = P.sessions(entry, mk.settled[fe])
        eqv = bv.expected_qv(entry, fe, s.T)
        p_cal = P.forecast(entry, n_s) * cal.get(entry, np.nan) if n_s > 0 else np.nan
        nv = float(naive.get(entry, np.nan)) * n_s
        iv_atm = float(B.Market.iv_at(s, np.array([s.F]))[0])
        sig = dict(n_sessions=n_s, eqv=eqv, p_cal=p_cal, p_naive=nv, iv_atm=iv_atm, T=s.T,
                   r_bates=eqv / p_cal if p_cal and p_cal > 0 else np.nan,
                   r_simple=iv_atm ** 2 * s.T / nv if nv and nv > 0 else np.nan)
        for name, legs in (("straddle", atm_straddle(s)), ("strangle", strangle(mk, s))):
            if legs is None:
                continue
            K, sd = legs
            unit = np.array([-1.0, -1.0])
            sl = B.stress_loss(mk, s, K, sd, unit, hedged=True)
            q = unit * ENGINE_RISK * C_SYM / sl
            prem = float(np.sum(mk.price(s, K, sd)[0]))
            info = dict(sig, stress_unit=sl, premium_unit=prem, F=s.F)
            for hedge in ("bs", "mv", None):
                run(B.Position(f"eng_{name}_{hedge or 'none'}", sym, entry, mk.settled[fe], fe, K, sd, q, hedge),
                    bv.mv_delta if hedge == "mv" else None, **info)
        # O1: crash put spread, O1_TDTE sessions before settlement
        i = mk.sessions.searchsorted(mk.settled[fe]) - O1_TDTE
        if i > 0:
            e1 = mk.sessions[i]
            s1 = mk.slice(e1, fe)
            legs = put_spread(mk, s1) if s1 is not None and not s1.stale else None
            if legs is not None:
                K, sd = legs
                px = mk.price(s1, K, sd)[0]
                ml = (K[0] - K[1]) - (px[0] - px[1])
                if ml > 0:
                    q = np.array([-1.0, 1.0]) * O1_RISK * C_SYM / ml
                    run(B.Position("o1_putspread", sym, e1, mk.settled[fe], fe, K, sd, q, None),
                        credit_unit=float(px[0] - px[1]), maxloss_unit=ml, F=s1.F)

    # O2: Budget straddle
    for _, e in ev.iterrows():
        sess = e["session"]
        if sess not in mk.sessions:
            continue
        j = mk.sessions.get_loc(sess)
        if j < O2_LEAD:
            continue
        entry = mk.sessions[j - O2_LEAD]
        if e["known_from"] > entry or j + 2 >= len(mk.sessions):
            continue
        cands = sorted(f for f, st in mk.settled.items() if st >= mk.sessions[j + 2] and f >= entry)
        for fe in cands[:3]:
            s = mk.slice(entry, fe)
            if s is None or s.stale:
                continue
            legs = atm_straddle(s)
            if legs is None:
                continue
            K, sd = legs
            unit = np.array([-1.0, -1.0])
            sl = B.stress_loss(mk, s, K, sd, unit, hedged=True)
            q = unit * O2_RISK * C_SYM / sl
            if run(B.Position("o2_budget", sym, entry, sess, fe, K, sd, q, "bs"), event=e["date"],
                   stress_unit=sl, premium_unit=float(np.sum(mk.price(s, K, sd)[0])), F=s.F):
                break

    # exploratory weekly straddle (holdout only)
    allx = sorted(mk.settled)
    for prev, fe in zip(allx[:-1], allx[1:]):
        entry = mk.settled[prev]
        if entry < HOLDOUT or (fe - prev).days > 9 or mk.settled[fe] <= entry:
            continue
        s = mk.slice(entry, fe)
        if s is None or s.stale:
            continue
        legs = atm_straddle(s)
        if legs is None:
            continue
        K, sd = legs
        unit = np.array([-1.0, -1.0])
        sl = B.stress_loss(mk, s, K, sd, unit, hedged=True)
        run(B.Position("wk_straddle_bs", sym, entry, mk.settled[fe], fe, K, sd, unit * ENGINE_RISK * C_SYM / sl, "bs"),
            stress_unit=sl, premium_unit=float(np.sum(mk.price(s, K, sd)[0])), F=s.F)

    print(f"{sym}: {len(positions)} positions, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(positions), pd.concat(daily, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    st = B.spread_table(os.path.join(OUT, "_spread_table.parquet"))
    costs = [B.Costs(st), B.Costs(st, stress=True)]
    pos, day = [], []
    for sym in ("NIFTY", "BANKNIFTY"):
        p, d = build_symbol(sym, costs)
        if len(pos):
            off = pos[-1]["pid"].max()
            p["pid"] += off
            d["pid"] += off
        pos.append(p)
        day.append(d)
    pd.concat(pos, ignore_index=True).to_parquet(os.path.join(OUT, "positions.parquet"))
    pd.concat(day, ignore_index=True).to_parquet(os.path.join(OUT, "daily.parquet"))


if __name__ == "__main__":
    main()
