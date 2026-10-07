"""Stage 5 audit of the simulator (src/backtest.py) and the engine's risk.

A1  Synthetic Black-Scholes market (no costs): a short ATM straddle hedged daily at the
    implied vol must earn ~0 when realised vol = implied, and about the theoretical
    variance-swap-like amount when realised < implied; hedging must cut the noise.
A2  Stage 1 cross-check: Stage 1 trades re-run through the simulator as single long
    options. Option P&L must equal payoff - mark, entry costs must equal Stage 1's,
    hedge P&L (smile-IV deltas here, entry-IV deltas in Stage 1) must agree closely.
A3  Moving-block bootstrap (3-month blocks) of the engine's monthly net P&L: drawdown
    and losing-year odds over 10 years.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import backtest as B  # noqa: E402
from src import contracts as ct  # noqa: E402

D5 = os.path.join(PROJECT_ROOT, "data", "processed", "stage5")
LINES: list[str] = []


def out(s=""):
    print(s)
    LINES.append(str(s))


class ZeroCosts:
    def trade(self, price, qty, d, lot):
        return np.zeros(len(np.atleast_1d(qty)))

    def fut(self, x):
        return 0.0

    def half_spread(self, price, d):
        return np.zeros(len(np.atleast_1d(price)))


class FakeMarket(B.Market):
    def __init__(self, F_path: np.ndarray, sessions: pd.DatetimeIndex, sigma: float):
        fe = sessions[-1]
        self.symbol = "FAKE"
        self.sessions = sessions
        self.settled = {fe: fe}
        self.spot = pd.Series(F_path, index=sessions)
        self.rate = pd.Series(0.0, index=sessions)
        self.lot = pd.Series(50.0, index=sessions)
        self.fwd = {(d, fe): float(f) for d, f in zip(sessions, F_path)}
        self.smiles = {(d, fe): (np.array([-3.0, 3.0]), np.array([sigma, sigma])) for d in sessions}
        T = np.array([(fe - d).days / 365.0 for d in sessions])
        self.marks = {}
        for d, f, t in zip(sessions[:-1], F_path[:-1], T[:-1]):
            c = float(ct.b76_price(f, 100.0, t, sigma, 1.0, True))
            p = float(ct.b76_price(f, 100.0, t, sigma, 1.0, False))
            self.marks[(d, fe)] = {(100.0, "CE"): c, (100.0, "PE"): p}


def a1(n_paths=3000, seed=7):
    out("=" * 90 + "\nA1 synthetic Black-Scholes market, short ATM straddle, 21 sessions, no costs\n" + "=" * 90)
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2031-01-06", periods=22)
    dt = np.diff(sessions).astype("timedelta64[D]").astype(float) / 365.0
    T0 = (sessions[-1] - sessions[0]).days / 365.0
    sig_i = 0.20
    zc = [ZeroCosts()]
    for sig_r in (0.20, 0.16, 0.24):
        hed, unh = [], []
        for _ in range(n_paths):
            z = rng.standard_normal(len(dt))
            F = 100 * np.exp(np.concatenate([[0], np.cumsum(-0.5 * sig_r ** 2 * dt + sig_r * np.sqrt(dt) * z)]))
            mk = FakeMarket(F, sessions, sig_i)
            for hedge, acc in (("bs", hed), (None, unh)):
                pos = B.Position("t", "FAKE", sessions[0], sessions[-1], sessions[-1], np.array([100.0, 100.0]),
                                 np.array(["CE", "PE"]), np.array([-1.0, -1.0]), hedge)
                r = B.simulate(pos, mk, zc)
                acc.append((r.option + r.hedge).sum())
        hed, unh = np.array(hed), np.array(unh)
        prem = 2 * ct.b76_price(100.0, 100.0, T0, sig_i, 1.0, True)
        theory = 2 * (ct.b76_price(100.0, 100.0, T0, sig_i, 1.0, True) - ct.b76_price(100.0, 100.0, T0, sig_r, 1.0, True))
        out(f"  realised {sig_r:.2f} vs implied {sig_i:.2f}: hedged mean {hed.mean():+.3f} +/- {hed.std() / np.sqrt(len(hed)):.3f}"
            f" (theory ~{theory:+.3f}; premium {prem:.2f}); hedged sd {hed.std():.2f} vs unhedged sd {unh.std():.2f};"
            f" unhedged mean {unh.mean():+.3f}")
    out("  PASS if the 0.20 row is ~0 within 3 s.e., the others are near theory, and hedged sd << unhedged sd.")


def a2(n=600, seed=11):
    out("\n" + "=" * 90 + "\nA2 Stage 1 trades re-run through the simulator (single long option, hedged daily)\n" + "=" * 90)
    tr = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "processed", "stage1", "trades.parquet"))
    tr = tr[tr.tdte.isin([5, 10, 21])]
    st = B.spread_table(os.path.join(D5, "_spread_table.parquet"))
    costs = [B.Costs(st)]
    rows = []
    for sym in ("NIFTY", "BANKNIFTY"):
        mk = B.Market(sym)
        s = tr[tr.symbol == sym].sample(n // 2, random_state=seed)
        for _, t in s.iterrows():
            fe = pd.Timestamp(t.final_expiry)
            if fe not in mk.settled:
                continue
            pos = B.Position("x", sym, t.date, mk.settled[fe], fe, np.array([t.strike]), np.array([t.side]),
                             np.array([1.0]), "bs")
            r = B.simulate(pos, mk, costs)
            if len(r) == 0:
                rows.append(dict(tid=t.tid, ok=False))
                continue
            pos.hedge = None
            u = B.simulate(pos, mk, costs)
            sl = mk.slice(t.date, fe)
            m100 = 100 / t.mark
            rows.append(dict(tid=t.tid, ok=True, symbol=sym, opt=r.option.sum(), s1_opt=t.payoff - t.mark,
                             hedge=r.hedge.sum(), s1_hedge=t.hedge, mark=t.mark,
                             opt_cost=u.cost0.sum() * m100, s1_opt_cost=t.buy_gross - t.buy_net,
                             hnet=(r.option + r.hedge - r.cost0).sum() * m100, s1_hnet=t.hbuy_net,
                             mark_sim=float(mk.price(sl, np.array([t.strike]), np.array([t.side]))[0][0])))
    x = pd.DataFrame(rows)
    out(f"  sampled {len(x)}, simulated {x.ok.sum()} (others: entry leg not traded under the >= 10 contracts rule)")
    x = x[x.ok]
    out(f"  entry mark identical: {np.isclose(x.mark, x.mark_sim).mean():.1%}")
    out(f"  option P&L = payoff - mark: max |diff| {np.abs(x.opt - x.s1_opt).max():.2e}")
    x.to_parquet(os.path.join(D5, "_audit_a2.parquet"))
    dc = x.opt_cost - x.s1_opt_cost
    out(f"  option costs per Rs100 (entry + expiry): sim mean {x.opt_cost.mean():.3f} vs Stage 1 {x.s1_opt_cost.mean():.3f};"
        f" |diff| median {dc.abs().median():.2e}, max {dc.abs().max():.3f}")
    out(f"  hedged net buyer return per Rs100: sim mean {x.hnet.mean():+.2f} vs Stage 1 {x.s1_hnet.mean():+.2f};"
        f" corr {x.hnet.corr(x.s1_hnet):.4f}")
    d = (x.hedge - x.s1_hedge) / x.mark * 100
    out(f"  hedge P&L corr {x.hedge.corr(x.s1_hedge):.4f}; mean per Rs100 premium: sim {(x.hedge / x.mark * 100).mean():+.2f}"
        f" vs Stage 1 {(x.s1_hedge / x.mark * 100).mean():+.2f}; median |diff| {d.abs().median():.2f} per Rs100")
    g = ((x.opt + x.hedge) / x.mark * 100).mean(), ((x.s1_opt + x.s1_hedge) / x.mark * 100).mean()
    out(f"  hedged gross buyer return per Rs100: sim {g[0]:+.2f} vs Stage 1 {g[1]:+.2f} (differ only by the hedge IV choice)")


def a3(n_boot=5000, block=3, years=10, seed=3):
    out("\n" + "=" * 90 + "\nA3 moving-block bootstrap of the engine's monthly net P&L (3-month blocks)\n" + "=" * 90)
    day = pd.read_parquet(os.path.join(D5, "daily.parquet"))
    e = day[day.strategy == "eng_straddle_bs"]
    s = (e.option + e.hedge - e.cost0).groupby(e.date).sum()
    ss = (e.option + e.hedge - e.cost1).groupby(e.date).sum()
    rng = np.random.default_rng(seed)
    for lab, ser in (("net", s), ("stress", ss)):
        for per, m in (("full", ser), ("hold", ser[ser.index >= "2018-01-01"])):
            mon = m.resample("ME").sum().to_numpy()
            nb = years * 12 // block
            starts = rng.integers(0, len(mon) - block + 1, size=(n_boot, nb))
            paths = mon[starts[:, :, None] + np.arange(block)].reshape(n_boot, -1)
            cum = paths.cumsum(1)
            dd = (cum - np.maximum.accumulate(cum, 1)).min(1)
            yrs = paths.reshape(n_boot, years, 12).sum(2)
            out(f"  {lab:6s} {per:4s}: 10-yr total median {np.median(cum[:, -1]):+.2f}, P(total < 0) {np.mean(cum[:, -1] < 0):.1%};"
                f" max drawdown median {np.median(dd):.2f}, 5% worst {np.quantile(dd, 0.05):.2f};"
                f" losing years {np.mean(yrs < 0):.0%}")


def main():
    a1()
    a2()
    a3()
    with open(os.path.join(D5, "_audit.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
