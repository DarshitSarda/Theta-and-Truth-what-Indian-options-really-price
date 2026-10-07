"""Stage 5 report: evaluates the positions built by scripts/stage5_backtest.py under the
rules fixed there.

Book P&L is a fraction of capital per session (positions are sized on capital 1, half per
symbol), summed over positions; months with no position count as 0. Inference on monthly
sums with a Newey-West (lag 2) standard error; Sharpe from daily P&L x sqrt(252).
Decisions (holdout = P&L dates from 2018-01-01; Holm across D1-D5 at 0.05):
  D1 engine      holdout monthly mean > 0; also stress-cost mean > 0 and max drawdown < 40%
  D2 O1, D3 O2   overlay's own holdout monthly mean > 0; stress mean > 0; adding it does not
                 lower the book's holdout Sharpe
  D4 G1          Bates-sized engine minus simple-sized engine > 0 and minus constant > 0
                 (monthly P&L difference; the larger p counts)
  D5 H1          Bates minimum-variance hedge lowers the engine's squared daily P&L
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(PROJECT_ROOT, "data", "processed", "stage5")
HOLDOUT = pd.Timestamp("2018-01-01")
ERAS = [("2008-13", "2008-01-01", "2014-01-01"), ("2014-19", "2014-01-01", "2020-01-01"),
        ("2020-Nov24", "2020-01-01", "2024-11-20"), ("Nov24-now", "2024-11-20", "2100-01-01")]
LAG = 2
MARGIN_RATE, MARGIN_CAP = 0.12, 0.80
LINES: list[str] = []


def out(s=""):
    print(s)
    LINES.append(str(s))


def hac_t(x: pd.Series) -> tuple[float, float, float]:
    """Mean, se (Newey-West lag LAG), one-sided p (mean > 0) of a monthly series."""
    v = np.asarray(x, float)
    n = len(v)
    if n < 6:
        return np.nan, np.nan, np.nan
    m = v.mean()
    e = v - m
    s = e @ e
    for l in range(1, LAG + 1):
        s += 2 * (1 - l / (LAG + 1)) * (e[l:] @ e[:-l])
    se = np.sqrt(max(s, 0) / n ** 2)
    return m, se, float(stats.norm.sf(m / se)) if se > 0 else np.nan


def metrics(day: pd.Series, cal: pd.DatetimeIndex) -> dict:
    d = day.reindex(cal, fill_value=0.0)
    if len(d) == 0:
        return {}
    mon = d.resample("ME").sum()
    m, se, p = hac_t(mon)
    cum = d.cumsum()
    dd = (cum - cum.cummax()).min()
    sd = d.std() * np.sqrt(252)
    return dict(ann_ret=d.mean() * 252, ann_vol=sd, sharpe=d.mean() * 252 / sd if sd > 0 else np.nan,
                t=m / se if se else np.nan, p=p, max_dd=dd, worst_month=mon.min(), worst_day=d.min(),
                pos_months=(mon > 0).mean(), months=len(mon))


def load():
    pos = pd.read_parquet(os.path.join(D, "positions.parquet"))
    day = pd.read_parquet(os.path.join(D, "daily.parquet"))
    day["gross"] = day.option + day.hedge
    day["net"] = day.gross - day.cost0
    day["stress"] = day.gross - day.cost1
    return pos, day


def book(day: pd.DataFrame, strategies, col="net", weights: pd.Series | None = None) -> pd.Series:
    x = day[day.strategy.isin(strategies)]
    v = x[col] if weights is None else x[col] * x.pid.map(weights).fillna(0.0)
    return v.groupby(x.date).sum()


def split(cal):
    return cal[cal < HOLDOUT], cal[cal >= HOLDOUT]


def table(rows: dict, cal) -> pd.DataFrame:
    disc, hold = split(cal)
    r = []
    for name, s in rows.items():
        for per, c in (("disc", disc), ("hold", hold)):
            r.append(dict(book=name, per=per, **metrics(s, c)))
    return pd.DataFrame(r)


def main():
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
    pos, day = load()
    cal = pd.DatetimeIndex(sorted(day.date.unique()))
    cal = cal[cal >= pd.Timestamp("2008-01-01")]
    out(f"Stage 5 report. positions {len(pos)}, position-days {len(day)}, sessions {len(cal)} "
        f"({cal.min().date()} .. {cal.max().date()})")
    out("Positions by strategy x symbol:")
    out(pos.groupby(["strategy", "symbol"]).size().unstack().to_string())
    out(f"Stale-smile position-days (valued from an earlier session's smile): {day.stale.mean():.2%}")

    # ------------------------------------------------ engine choice (discovery only)
    out("\n" + "=" * 100 + "\nENGINE: monthly short straddle / strangle, daily delta hedge (discovery picks)\n" + "=" * 100)
    eng = {f"{n}_{h}": book(day, [f"eng_{n}_{h}"]) for n in ("straddle", "strangle") for h in ("bs", "mv", "none")}
    eng.update({f"{n}_bs_stress": book(day, [f"eng_{n}_bs"], "stress") for n in ("straddle", "strangle")})
    eng.update({f"{n}_bs_gross": book(day, [f"eng_{n}_bs"], "gross") for n in ("straddle", "strangle")})
    t = table(eng, cal)
    out(t.round(4).to_string(index=False))
    disc = t[(t.per == "disc") & t.book.isin(["straddle_bs", "strangle_bs"])]
    choice = disc.loc[disc.sharpe.idxmax(), "book"].split("_")[0]
    out(f"\nChosen engine (higher discovery net Sharpe): {choice}")
    E = f"eng_{choice}_bs"

    # per symbol / era
    out("\nEngine by symbol and era (net):")
    rows = []
    for sym in ("NIFTY", "BANKNIFTY"):
        s = book(day[day.symbol == sym], [E])
        for name, a, b in ERAS:
            c = cal[(cal >= a) & (cal < b)]
            rows.append(dict(symbol=sym, era=name, **metrics(s, c)))
    s = book(day, [E])
    for name, a, b in ERAS:
        c = cal[(cal >= a) & (cal < b)]
        rows.append(dict(symbol="BOOK", era=name, **metrics(s, c)))
    out(pd.DataFrame(rows).round(4).to_string(index=False))
    out("\nEngine calendar-year net return (book):")
    out(s.reindex(cal, fill_value=0).groupby(cal.year).sum().round(4).to_string())
    out("\nWorst 10 sessions for the engine (book, net) and worst 6 months:")
    out(s.nsmallest(10).round(4).to_string())
    out(s.reindex(cal, fill_value=0).resample("ME").sum().nsmallest(6).round(4).to_string())
    cost_share = day[day.strategy == E][["gross", "cost0", "cost1"]].sum()
    out(f"\nEngine gross P&L {cost_share.gross:.4f}, costs {cost_share.cost0:.4f} (stress {cost_share.cost1:.4f}) "
        f"-> costs take {cost_share.cost0 / cost_share.gross:.0%} of gross (stress {cost_share.cost1 / cost_share.gross:.0%})")
    pe = pos[pos.strategy == E]
    ent = pe.qty * 2 * pe.F
    out(f"Short option notional at entry / symbol capital: median {(ent / 0.5).median():.2f}, max {(ent / 0.5).max():.2f}"
        f" (margin check in ROBUSTNESS)")
    out(f"Engine premium collected per entry / capital: median {(pe.premium_unit * pe.qty).median():.4f}")

    # ------------------------------------------------ overlays
    out("\n" + "=" * 100 + "\nOVERLAYS\n" + "=" * 100)
    ov = {"o1_putspread": book(day, ["o1_putspread"]), "o1_stress": book(day, ["o1_putspread"], "stress"),
          "o2_budget": book(day, ["o2_budget"]), "o2_stress": book(day, ["o2_budget"], "stress"),
          "engine": book(day, [E]), "engine+o1": book(day, [E, "o1_putspread"]),
          "engine+o2": book(day, [E, "o2_budget"]), "engine+o1+o2": book(day, [E, "o1_putspread", "o2_budget"])}
    to = table(ov, cal)
    out(to.round(4).to_string(index=False))
    o1 = pos[pos.strategy == "o1_putspread"]
    out(f"\nO1 entries {len(o1)} (monthly cycles available ~{len(pe)}); median credit / max loss "
        f"{(o1.credit_unit / o1.maxloss_unit).median():.3f}")
    o1p = day[day.strategy == "o1_putspread"].groupby("pid").net.sum()
    out(f"O1 per-trade net (fraction of capital): mean {o1p.mean():.5f}, win share {(o1p > 0).mean():.2f}, "
        f"worst {o1p.min():.4f} on {pos.set_index('pid').loc[o1p.idxmin(), 'settled'].date()}")
    o2p = day[day.strategy == "o2_budget"].groupby("pid").net.sum()
    o2i = pos.set_index("pid").loc[o2p.index]
    out("O2 per Budget (net, fraction of capital):")
    out(pd.DataFrame(dict(symbol=o2i.symbol, event=o2i.event.dt.date, net=o2p.round(5),
                          gross=day[day.strategy == "o2_budget"].groupby("pid").gross.sum().round(5))).to_string(index=False))

    # ------------------------------------------------ G1 sizing signals
    out("\n" + "=" * 100 + "\nG1: size the engine by a variance-premium signal\n" + "=" * 100)
    pe = pos[pos.strategy == E].copy()
    pe = pe[np.isfinite(pe.r_bates) & np.isfinite(pe.r_simple) & (pe.r_bates > 0) & (pe.r_simple > 0)]
    for c in ("r_bates", "r_simple"):
        med = pe[pe.entry < HOLDOUT].groupby("symbol")[c].median()
        pe["m_" + c[2:]] = (pe[c] / pe.symbol.map(med)).clip(0, 2)
        out(f"  {c}: discovery median by symbol {med.round(3).to_dict()}; corr with realised next-cycle net "
            f"P&L per unit qty shown below")
    pn = day[day.strategy == E].groupby("pid").net.sum()
    pe["pnl"] = pe.pid.map(pn)
    out(f"  corr(r_bates, r_simple) {pe.r_bates.corr(pe.r_simple):.2f}; Spearman corr with cycle P&L: "
        f"bates {pe.r_bates.corr(pe.pnl, method='spearman'):.3f}, simple {pe.r_simple.corr(pe.pnl, method='spearman'):.3f}")
    ids = pe.pid
    w_const = pd.Series(1.0, index=ids)
    w_b = pe.set_index("pid").m_bates
    w_s = pe.set_index("pid").m_simple
    g = {"const": book(day, [E], weights=w_const), "bates": book(day, [E], weights=w_b),
         "simple": book(day, [E], weights=w_s)}
    g["bates-simple"] = g["bates"].sub(g["simple"], fill_value=0)
    g["bates-const"] = g["bates"].sub(g["const"], fill_value=0)
    g["simple-const"] = g["simple"].sub(g["const"], fill_value=0)
    tg = table(g, cal[cal >= pe.entry.min()])
    out(tg.round(4).to_string(index=False))

    # ------------------------------------------------ H1 hedge
    out("\n" + "=" * 100 + "\nH1: Bates minimum-variance delta vs Black-76 smile delta (engine)\n" + "=" * 100)
    bs = book(day, [E]).reindex(cal, fill_value=0)
    mv = book(day, [f"eng_{choice}_mv"]).reindex(cal, fill_value=0)
    sq = (bs ** 2 - mv ** 2)
    rows = []
    for per, c in zip(("disc", "hold"), split(cal)):
        m, se, p = hac_t(sq.reindex(c).resample("ME").sum())
        rows.append(dict(per=per, sd_bs=bs.reindex(c).std() * np.sqrt(252), sd_mv=mv.reindex(c).std() * np.sqrt(252),
                         ret_bs=bs.reindex(c).mean() * 252, ret_mv=mv.reindex(c).mean() * 252, sq_diff_t=m / se, p=p))
    th = pd.DataFrame(rows)
    out(th.round(5).to_string(index=False))

    # ------------------------------------------------ decisions
    out("\n" + "=" * 100 + "\nDECISIONS (holdout, Holm across D1-D5)\n" + "=" * 100)

    def get(tab, name, col):
        r = tab[(tab.book == name) & (tab.per == "hold")]
        return float(r[col].iloc[0]) if len(r) else np.nan

    dec = []
    e_st = f"{choice}_bs_stress"
    dec.append(dict(id="D1 engine", p=get(t, f"{choice}_bs", "p"),
                    extra=bool(get(t, e_st, "ann_ret") > 0 and get(t, f"{choice}_bs", "max_dd") > -0.40),
                    note=f"stress ann_ret {get(t, e_st, 'ann_ret'):.4f}, max_dd {get(t, f'{choice}_bs', 'max_dd'):.3f}"))
    for did, ovn, st, comb in (("D2 O1 put spread", "o1_putspread", "o1_stress", "engine+o1"),
                               ("D3 O2 budget", "o2_budget", "o2_stress", "engine+o2")):
        dec.append(dict(id=did, p=get(to, ovn, "p"),
                        extra=bool(get(to, st, "ann_ret") > 0 and get(to, comb, "sharpe") >= get(to, "engine", "sharpe")),
                        note=f"stress ann_ret {get(to, st, 'ann_ret'):.4f}, Sharpe engine {get(to, 'engine', 'sharpe'):.3f}"
                             f" -> with {get(to, comb, 'sharpe'):.3f}"))
    dec.append(dict(id="D4 G1 Bates sizing", p=max(get(tg, "bates-simple", "p"), get(tg, "bates-const", "p")),
                    extra=True, note=f"p vs simple {get(tg, 'bates-simple', 'p'):.3f}, vs const {get(tg, 'bates-const', 'p'):.3f}"))
    hp = float(th[th.per == "hold"].p.iloc[0])
    dec.append(dict(id="D5 H1 Bates hedge", p=hp, extra=True,
                    note=f"holdout sd bs {th[th.per == 'hold'].sd_bs.iloc[0]:.4f} vs mv {th[th.per == 'hold'].sd_mv.iloc[0]:.4f}"))
    dd = pd.DataFrame(dec)
    order = dd.p.fillna(1).sort_values().index
    alpha, holm, stop = 0.05, {}, False
    for i, ix in enumerate(order):
        thr = alpha / (len(dd) - i)
        ok = (not stop) and dd.loc[ix, "p"] <= thr
        stop = stop or not ok
        holm[ix] = ok
    dd["holm_pass"] = pd.Series(holm)
    dd["survives"] = dd.holm_pass & dd.extra
    out(dd.to_string(index=False))

    # ------------------------------------------------ robustness (added after the first run)
    out("\n" + "=" * 100 + "\nROBUSTNESS (added after the first run; no decision): margin-feasible engine\n" + "=" * 100)
    pe = pos[pos.strategy == E].copy()
    pe["notional"] = pe.qty * 2 * pe.F
    pe["margin"] = MARGIN_RATE * pe.notional
    pe["w"] = np.minimum(1.0, MARGIN_CAP * 0.5 / pe.margin)
    out(f"  Estimated margin = {MARGIN_RATE:.0%} of short notional; cap = {MARGIN_CAP:.0%} of the symbol's capital.")
    out(f"  Entries needing a cut: {(pe.w < 1).mean():.1%}; median weight when cut {pe.w[pe.w < 1].median():.2f}")
    out("  Entry margin / symbol capital by year (median, max):")
    out((pe.margin / 0.5).groupby(pe.entry.dt.year).agg(["median", "max"]).round(2).T.to_string())
    w = pe.set_index("pid").w
    rb = {"engine_capped": book(day, [E], weights=w), "engine_capped_stress": book(day, [E], "stress", weights=w)}
    out(table(rb, cal).round(4).to_string(index=False))

    # ------------------------------------------------ exploratory
    out("\n" + "=" * 100 + "\nEXPLORATORY (no decision): weekly straddle engine, holdout only; crash windows\n" + "=" * 100)
    wk = {"weekly_straddle": book(day, ["wk_straddle_bs"]), "weekly_stress": book(day, ["wk_straddle_bs"], "stress")}
    out(table(wk, cal).query("per == 'hold'").round(4).to_string(index=False))
    for a, b, lab in (("2020-02-15", "2020-04-30", "COVID crash"), ("2024-05-25", "2024-06-15", "2024 election"),
                      ("2008-09-01", "2008-12-31", "2008 crisis"), ("2016-11-07", "2016-11-30", "demonetisation")):
        c = cal[(cal >= a) & (cal <= b)]
        parts = {k: float(v.reindex(c, fill_value=0).sum()) for k, v in
                 (("engine", book(day, [E])), ("o1", book(day, ["o1_putspread"])), ("unhedged", book(day, [f"eng_{choice}_none"])))}
        out(f"  {lab:15s} {a}..{b}: " + ", ".join(f"{k} {v:+.4f}" for k, v in parts.items()))

    with open(os.path.join(D, "_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
