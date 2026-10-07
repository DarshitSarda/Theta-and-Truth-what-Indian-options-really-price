"""Paper trading of the Stage 5 engine on the live option-chain snapshots.

Rules: exactly the Stage 5 engine (scripts/stage5_backtest.py), frozen 2026-10-07:
  at the close of each monthly expiry's settlement session (monthly = last listed expiry of
  the calendar month), sell the ATM straddle of the next monthly expiry (strike nearest the
  forward with both sides traded), hold to settlement, delta-hedge at every close with the
  expiry's forward (Black-76 delta at the smile IV). Size: hedged stress loss (instant
  -15..+10%, IV x1 / x1.5) = 25% of the symbol's capital; book capital 1, half per symbol.
  Same simulator (src/backtest.simulate), same Stage 1 cost model and stress costs. Extra
  column cost_q: the entry trade at the snapshot's actual bid (sell) instead of the
  modelled half-spread, plus the same fees.
  A cycle whose entry session has no usable snapshot is skipped (flagged), as in Stage 5.
Phases: "replay" = entries before FREEZE (data that existed when the rules were fixed; used
  to verify against the Stage 5 backtest), "forward" = entries on or after FREEZE: the only
  out-of-sample record.
Outputs data/processed/paper/: positions.csv, daily.csv, cycles.csv, _report.txt,
  journal.csv (append-only: one row per position and new session, with a hash of the rules
  and code; a changed hash on an already journaled session is reported, never overwritten).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src import backtest as B  # noqa: E402
from src import live  # noqa: E402
from stage5_backtest import C_SYM, ENGINE_RISK, atm_straddle, monthly_expiries  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "data", "processed", "paper")
D5 = os.path.join(PROJECT_ROOT, "data", "processed", "stage5")
FREEZE = pd.Timestamp("2026-10-07")
STRATEGY = "eng_straddle_bs"
ILLUSTRATIVE_CAPITAL = 1e7          # Rs 1 crore, to show lots
RULES = dict(strategy=STRATEGY, c_sym=C_SYM, engine_risk=ENGINE_RISK, freeze=str(FREEZE.date()),
             moves=B.MOVES.tolist(), iv_mult=B.IV_MULT.tolist(), fut_cost=B.FUT_COST, min_traded=live.MIN_TRADED)
LINES: list[str] = []


def out(s=""):
    print(s)
    LINES.append(str(s))


def code_hash() -> str:
    h = hashlib.sha256(json.dumps(RULES, sort_keys=True).encode())
    for f in ("src/backtest.py", "src/live.py", "scripts/paper_engine.py", "scripts/stage5_backtest.py"):
        h.update(open(os.path.join(PROJECT_ROOT, f), "rb").read())
    return h.hexdigest()[:12]


def last_lot(symbol: str) -> float:
    p = pd.read_parquet(B.PANEL, columns=["date", "lot_size"], filters=[("symbol", "=", symbol),
                                                                       ("date", ">=", pd.Timestamp("2026-01-01"))])
    return float(p.sort_values("date")["lot_size"].iloc[-1])


def quoted_entry_cost(mk: live.LiveMarket, panel: pd.DataFrame, pos: B.Position, lot: float) -> float:
    d, fe = pos.entry, pos.final_expiry
    q = panel[(panel["date"] == d) & (panel["final_expiry"] == fe)].set_index(["strike", "side"])
    c = 0.0
    for K, sd, n in zip(pos.strikes, pos.sides, pos.qty):
        r = q.loc[(K, sd)]
        px = r["bid"] if n < 0 else r["ask"]
        fee = r["mid"] * (B.exchange_rate(d) + (B.stt_sell_rate(d) if n < 0 else B.stamp_rate(d)))
        c += abs(n) * (abs(r["mid"] - px) + fee + 20 * 1.18 / lot)
    return float(c)


def run_symbol(sym: str, costs):
    panel, info = live.live_panel(sym)
    lot = last_lot(sym)
    mk = live.LiveMarket(sym, panel, info, lot)
    last = mk.sessions[-1]
    mon = monthly_expiries(mk)
    positions, daily, flags = [], [], []
    for prev, fe in zip(mon[:-1], mon[1:]):
        entry = mk.settled[prev]
        if entry < mk.sessions[0] or entry > last or fe <= entry:
            continue
        s = mk.slice(entry, fe)
        if s is None or s.stale:
            flags.append(f"{sym}: the {fe.date()} chain was not scraped (or unusable) on its entry session {entry.date()}"
                         " -> cycle skipped")
            continue
        legs = atm_straddle(s)
        if legs is None:
            flags.append(f"{sym}: no traded ATM pair on {entry.date()} for {fe.date()} -> cycle skipped")
            continue
        K, sd = legs
        unit = np.array([-1.0, -1.0])
        sl = B.stress_loss(mk, s, K, sd, unit, hedged=True)
        qty = unit * ENGINE_RISK * C_SYM / sl
        pos = B.Position(STRATEGY, sym, entry, mk.settled[fe], fe, K, sd, qty, "bs")
        r = B.simulate(pos, mk, costs)
        if len(r) == 0:
            flags.append(f"{sym}: entry legs not traded on {entry.date()} -> cycle skipped")
            continue
        cq = quoted_entry_cost(mk, panel, pos, lot)
        c0_entry = float(costs[0].trade(mk.price(s, K, sd)[0], qty, entry, lot).sum())
        r["cost_q"] = r["cost0"]
        r.loc[r["date"] == entry, "cost_q"] += cq - c0_entry
        pid = f"{sym}_{fe.date()}"
        r["pid"] = pid
        done = mk.settled[fe] <= last
        prem = float(np.sum(mk.price(s, K, sd)[0]))
        positions.append(dict(pid=pid, symbol=sym, phase="replay" if entry < FREEZE else "forward", entry=entry,
                              final_expiry=fe, strike=float(K[0]), F_entry=s.F, iv_entry=float(mk.iv_at(s, K[:1])[0]),
                              premium_per_unit=prem, qty_units=float(abs(qty[0])), stress_unit=sl,
                              premium_book=prem * abs(qty[0]), lot=lot,
                              lots_per_crore=float(abs(qty[0]) * ILLUSTRATIVE_CAPITAL / lot),
                              status="settled" if done else "open", through=min(mk.settled[fe], last)))
        daily.append(r)
        missing = r[r["stale"]]
        for d in missing["date"]:
            flags.append(f"{sym} {pid}: no usable snapshot on {d.date()} (P&L carried to the next session, hedge not re-set)")
    for d in mk.dropped:
        flags.append(f"{sym}: snapshot of {d.date()} was not taken at the close -> session dropped")
    nxt = [m for m in mon if mk.settled[m] > last]
    if nxt:
        flags.append(f"{sym}: next entry session {nxt[0].date()} (settlement of the {nxt[0]:%b} monthly) - scrape it after the close")
    return pd.DataFrame(positions), (pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()), flags, mk


def verify_against_stage5(pos: pd.DataFrame, day: pd.DataFrame):
    out("\nREPLAY vs STAGE 5 BACKTEST (same rules; bhavcopy closes there, live mids here)")
    p5 = pd.read_parquet(os.path.join(D5, "positions.parquet"))
    p5 = p5[p5["strategy"] == STRATEGY]
    d5 = pd.read_parquet(os.path.join(D5, "daily.parquet"))
    d5 = d5[d5["strategy"] == STRATEGY]
    for _, p in pos.iterrows():
        m = p5[(p5["symbol"] == p.symbol) & (p5["final_expiry"] == p.final_expiry)]
        if m.empty:
            out(f"  {p.pid}: not in Stage 5 (its data ended {d5['date'].max().date()})")
            continue
        m = m.iloc[0]
        a = day[day["pid"] == p.pid].set_index("date")
        b = d5[d5["pid"] == m.pid].set_index("date")
        both = a.index.intersection(b.index)
        ga, gb = (a.option + a.hedge).loc[both], (b.option + b.hedge).loc[both]
        out(f"  {p.pid}: entry {p.entry.date()} vs {m.entry.date()}; strike {p.strike:g} vs {m.strikes}; "
            f"qty {p.qty_units:.3e} vs {m.qty:.3e} ({p.qty_units / m.qty - 1:+.1%})")
        out(f"     common days {len(both)}: cumulative gross live {ga.sum():+.5f} vs backtest {gb.sum():+.5f} (book units);"
            f" daily corr {ga.corr(gb):.3f}; option {a.option.loc[both].sum():+.5f} vs {b.option.loc[both].sum():+.5f};"
            f" hedge {a.hedge.loc[both].sum():+.5f} vs {b.hedge.loc[both].sum():+.5f}; entry cost {a.cost0.iloc[0]:.5f} vs {b.cost0.iloc[0]:.5f}")


def journal(pos: pd.DataFrame, day: pd.DataFrame, h: str):
    path = os.path.join(OUT, "journal.csv")
    old = pd.read_csv(path, parse_dates=["session"]) if os.path.exists(path) else pd.DataFrame()
    rows = []
    for _, p in pos.iterrows():
        a = day[day["pid"] == p.pid].sort_values("date")
        cum = (a.option + a.hedge - a.cost0).cumsum()
        for d, v in zip(a["date"], cum):
            rows.append(dict(run_at=datetime.now().isoformat(timespec="seconds"), session=d, pid=p.pid, phase=p.phase,
                             cum_net=float(v), code_hash=h))
    new = pd.DataFrame(rows)
    if len(old):
        key = ["session", "pid"]
        mm = new.merge(old.drop_duplicates(key, keep="first"), on=key, how="left", suffixes=("", "_old"))
        changed = mm[mm["cum_net_old"].notna() & ((mm["cum_net"] - mm["cum_net_old"]).abs() > 1e-9)]
        if len(changed):
            out(f"\nJOURNAL WARNING: {len(changed)} journaled (session, position) values differ from what was recorded"
                f" earlier (code hash then {sorted(set(changed['code_hash_old']))}, now {h}). Earlier rows are kept.")
            out(changed[["session", "pid", "cum_net_old", "cum_net"]].tail(6).to_string(index=False))
        new = mm[mm["cum_net_old"].isna()][new.columns]
    if len(new):
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
    out(f"\njournal: {len(new)} new rows appended (code hash {h})")


def main():
    os.makedirs(OUT, exist_ok=True)
    st = B.spread_table(os.path.join(D5, "_spread_table.parquet"))
    costs = [B.Costs(st), B.Costs(st, stress=True)]
    P, D, F = [], [], []
    for sym in ("NIFTY", "BANKNIFTY"):
        p, d, f, mk = run_symbol(sym, costs)
        P.append(p)
        D.append(d)
        F += f
    pos = pd.concat(P, ignore_index=True)
    day = pd.concat(D, ignore_index=True)
    day["net"] = day.option + day.hedge - day.cost0
    day["net_quoted"] = day.option + day.hedge - day.cost_q
    day["net_stress"] = day.option + day.hedge - day.cost1
    pos.to_csv(os.path.join(OUT, "positions.csv"), index=False)
    day.to_csv(os.path.join(OUT, "daily.csv"), index=False)
    cyc = day.groupby("pid").agg(gross=("option", "sum"), hedge=("hedge", "sum"), cost=("cost0", "sum"),
                                 cost_quoted=("cost_q", "sum"), cost_stress=("cost1", "sum"), days=("date", "size"),
                                 last=("date", "max"), stale_days=("stale", "sum"))
    cyc["gross"] = cyc["gross"] + cyc.pop("hedge")
    cyc["net"] = cyc["gross"] - cyc["cost"]
    cyc["net_quoted"] = cyc["gross"] - cyc["cost_quoted"]
    cyc["net_stress"] = cyc["gross"] - cyc["cost_stress"]
    cyc = pos.set_index("pid")[["symbol", "phase", "status", "entry", "final_expiry", "strike", "iv_entry",
                                "premium_book", "lots_per_crore"]].join(cyc)
    cyc.to_csv(os.path.join(OUT, "cycles.csv"))
    h = code_hash()
    out("=" * 100 + f"\nPAPER ENGINE (Stage 5 monthly ATM straddle, daily hedge) - run {datetime.now():%Y-%m-%d %H:%M},"
        f" data through {day['date'].max().date()}, code hash {h}\n" + "=" * 100)
    out("P&L in % of book capital (capital 1 = both symbols); 'lots/crore' = lots per Rs 1 crore of book capital")
    show = cyc.copy()
    for c in ("gross", "cost", "net", "net_quoted", "net_stress", "premium_book"):
        show[c] = (show[c] * 100).round(2)
    show["iv_entry"] = (show["iv_entry"] * 100).round(1)
    show["lots_per_crore"] = show["lots_per_crore"].round(1)
    out(show[["phase", "status", "entry", "final_expiry", "strike", "iv_entry", "premium_book", "lots_per_crore",
              "gross", "cost", "net", "net_quoted", "net_stress", "stale_days"]].to_string())
    for ph in ("replay", "forward"):
        x = cyc[cyc["phase"] == ph]
        out(f"\n{ph}: {len(x)} cycles ({(x['status'] == 'settled').sum()} settled); net {x['net'].sum() * 100:+.2f}% of book,"
            f" quoted-spread net {x['net_quoted'].sum() * 100:+.2f}%, stress {x['net_stress'].sum() * 100:+.2f}%")
    out("\nFLAGS")
    for f in F:
        out("  " + f)
    verify_against_stage5(pos[pos["phase"] == "replay"], day)
    journal(pos, day, h)
    with open(os.path.join(OUT, "_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
