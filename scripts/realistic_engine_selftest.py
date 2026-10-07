"""Synthetic-market self-test of realistic_engine.run_cycle (must pass before the real run).

Market: driftless GBM forward, r = 0 (DF = 1), 21 sessions, Black-76 marks at a constant implied
vol; each session's VWAP = prices at the half-session point of the path (so fills differ from
closes, like real VWAP fills). Settlement = intrinsic at the last close.
  S1 no edge (implied = realised = 20%), zero costs, exact hedge: mean P&L ~ 0
  S2 implied 20% > realised 16%, zero costs, exact hedge: mean P&L ~ straddle(20%) - straddle(16%)
  S3 whole lots (3.4 lots wanted -> 3 sold), rule round: unbiased vs exact, noisier; option and
     hedge positions always whole lots
  S4 wider bands trade less (round >= band1 >= band2 >= none)
  S5 zero costs: synthetic-future hedge P&L == futures hedge P&L (parity holds exactly here)
  S6 cost formula = hand calculation for one option sale and one futures buy
  S7 entry retry: no usable entry VWAP for 3 sessions -> cycle skipped
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src import backtest as B  # noqa: E402
from src import contracts as ct  # noqa: E402
import realistic_engine as R  # noqa: E402

N, F0, K = 21, 100.0, 100.0
DAYS = list(pd.bdate_range("2025-01-06", periods=N + 1))


class ZeroCM:
    def option(self, d, price, units):
        return 0.0

    def future(self, d, price, units):
        return 0.0

    def expiry(self, d, value, units, spot):
        return 0.0


def make_cycle(rng, s_imp, s_real, units=1.0, lot=1.0):
    dt = 1 / 252
    z = rng.standard_normal(2 * N)
    path = F0 * np.exp(np.cumsum(-0.5 * s_real ** 2 * dt / 2 + s_real * np.sqrt(dt / 2) * z))
    path = np.r_[F0, path]                      # index 2k = close k, 2k-1 = mid-session k
    Fc = path[0::2]
    Fm = np.r_[np.nan, path[1::2]]
    T = (N - np.arange(N + 1)) / 252
    Tm = T + 0.5 / 252
    oc, ov, dl = np.zeros((N + 1, 2)), np.full((N + 1, 2), np.nan), np.zeros((N + 1, 2))
    for j, call in enumerate((True, False)):
        oc[:N, j] = ct.b76_price(Fc[:N], K, T[:N], s_imp, 1.0, call)
        oc[N, j] = max(Fc[N] - K, 0) if call else max(K - Fc[N], 0)
        ov[1:, j] = ct.b76_price(Fm[1:], K, Tm[1:], s_imp, 1.0, call)
        dl[:N, j] = ct.b76_greeks(Fc[:N], K, T[:N], s_imp, 1.0, 0.0, call)[0]
    fv = Fm.copy()
    return R.Cycle(DAYS, K, lot, lot, units, oc, ov, dl, Fc, fv, np.ones(N + 1))


def batch(n, s_imp, s_real, rule, instr, units=1.0, lot=1.0, frac=True, seed=0, cms=None):
    rng = np.random.default_rng(seed)
    cms = cms or [ZeroCM()]
    res = []
    for _ in range(n):
        c = make_cycle(rng, s_imp, s_real, units, lot)
        r = R.run_cycle(c, rule, instr, cms, frac)
        res.append(r)
    return res


def main():
    L, ok_all = [], True

    def check(name, ok, msg):
        nonlocal ok_all
        ok_all &= bool(ok)
        L.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")

    n = 4000
    prem0 = 2 * ct.b76_price(F0, K, N / 252, 0.20, 1.0, True)
    r1 = batch(n, 0.20, 0.20, "exact", "FUT")
    p1 = np.array([x["pnl"].sum() for x in r1])
    m, se = p1.mean(), p1.std() / np.sqrt(n)
    check("S1 no edge", abs(m) < 3 * se + 0.01 * prem0, f"mean {m:+.4f} +/- {se:.4f} (premium {prem0:.3f})")
    sd_exact = p1.std()

    r2 = batch(n, 0.20, 0.16, "exact", "FUT", seed=1)
    p2 = np.array([x["pnl"].sum() for x in r2])
    theory = 2 * (ct.b76_price(F0, K, N / 252, 0.20, 1.0, True) - ct.b76_price(F0, K, N / 252, 0.16, 1.0, True))
    m2, se2 = p2.mean(), p2.std() / np.sqrt(n)
    check("S2 premium capture", abs(m2 - theory) < 3 * se2 + 0.1 * theory,
          f"mean {m2:+.4f} +/- {se2:.4f} vs theory {theory:+.4f}")

    r3 = batch(n, 0.20, 0.20, "round", "FUT", units=3.4, lot=1.0, frac=False, seed=0)
    p3 = np.array([x["pnl"].sum() for x in r3]) / 3.0
    whole = all(np.allclose(x["hpos"], np.round(x["hpos"])) and np.allclose(x["qpos"], np.round(x["qpos"]))
                for x in r3)
    sold = {round(x["lots"], 6) for x in r3}
    check("S3 whole lots", whole and sold == {3.0} and abs(p3.mean()) < 3 * p3.std() / np.sqrt(n) + 0.01 * prem0
          and p3.std() > sd_exact,
          f"lots sold {sold}; per-unit mean {p3.mean():+.4f}, sd {p3.std():.4f} vs exact sd {sd_exact:.4f}")

    tr = {}
    for rule in ("round", "band1", "band2", "none"):
        rr = batch(500, 0.20, 0.20, rule, "FUT", units=5.0, lot=1.0, frac=False, seed=2)
        tr[rule] = np.mean([x["trades"].sum() for x in rr])
    check("S4 bands trade less", tr["round"] >= tr["band1"] >= tr["band2"] >= tr["none"] == 2,
          ", ".join(f"{k} {v:.1f}" for k, v in tr.items()))

    a = batch(300, 0.20, 0.18, "exact", "FUT", seed=3)
    b = batch(300, 0.20, 0.18, "exact", "SYN", seed=3)
    diff = max(abs(x["pnl"].sum() - y["pnl"].sum()) for x, y in zip(a, b))
    check("S5 synthetic == futures (zero cost)", diff < 1e-8, f"max |difference| {diff:.2e}")

    st = B.spread_table(os.path.join(R.D5, "_spread_table.parquet"))
    cm = R.CostModel(B.Costs(st), False)
    d = pd.Timestamp("2026-05-04")
    got = cm.option(d, 150.0, -65)
    hs = float(B.Costs(st).half_spread(np.array([150.0]), d)[0])
    hand = 65 * (hs + 150 * ((0.0003553 + 1e-6) * 1.18 + 0.0015)) + 20 * 1.18
    gotf = cm.future(d, 23000.0, 65)
    handf = 65 * 23000 * (0.00005 + (0.0000183 + 1e-6) * 1.18 + 0.00002) + 20 * 1.18
    check("S6 cost formulas", abs(got - hand) < 1e-9 and abs(gotf - handf) < 1e-9,
          f"option sale {got:.2f} = {hand:.2f}; futures buy {gotf:.2f} = {handf:.2f}")

    rng = np.random.default_rng(9)
    c = make_cycle(rng, 0.2, 0.2)
    c.opt_vwap[1:1 + R.ENTRY_TRIES] = np.nan
    r = R.run_cycle(c, "round", "FUT", [ZeroCM()], False)
    c2 = make_cycle(np.random.default_rng(9), 0.2, 0.2)
    c2.opt_vwap[1:3] = np.nan
    r2_ = R.run_cycle(c2, "round", "FUT", [ZeroCM()], False)
    check("S7 entry retry", r is None and r2_ is not None and r2_["fill"] == 3,
          f"3 missing -> {'skipped' if r is None else 'entered'}; 2 missing -> filled on session {r2_['fill']}")

    e_new = cm.expiry(d, 50.0, 65, 23000.0)
    e_old = cm.expiry(pd.Timestamp("2012-05-31"), 50.0, 65, 5000.0)
    sale_old = cm.option(pd.Timestamp("2012-05-31"), 50.0, -65)
    check("S8 expiry cost", abs(e_new - 65 * 0.0015 * 50) < 1e-9 and abs(e_old - min(65 * 0.00125 * 5000, sale_old)) < 1e-9,
          f"2026: {e_new:.2f} (exercise STT on intrinsic); 2012: {e_old:.2f} = min(exercise on index"
          f" {65 * 0.00125 * 5000:.2f}, sell {sale_old:.2f})")

    worst = 0.0
    for t in range(200):
        rng = np.random.default_rng(100 + t)
        c = make_cycle(rng, 0.2, 0.17, units=4.6, lot=1.0)
        instr = "SYN" if t % 2 else "FUT"
        base = R.run_cycle(c, "band1", instr, [cm], False)
        k = int(rng.integers(base["fill"], N - 1))
        c2 = make_cycle(np.random.default_rng(100 + t), 0.2, 0.17, units=4.6, lot=1.0)
        for arr in (c2.opt_close, c2.opt_vwap, c2.delta):
            arr[k + 1:] *= rng.uniform(0.5, 1.5, arr[k + 1:].shape)
        c2.fut_close[k + 1:] *= rng.uniform(0.9, 1.1, N - k)
        c2.fut_vwap[k + 1:] *= rng.uniform(0.9, 1.1, N - k)
        alt = R.run_cycle(c2, "band1", instr, [cm], False)
        worst = max(worst, np.abs(base["pnl"][:k + 1] - alt["pnl"][:k + 1]).max(),
                    np.abs(base["cost"][:, :k + 1] - alt["cost"][:, :k + 1]).max(),
                    np.abs(base["hpos"][:k + 2] - alt["hpos"][:k + 2]).max(),
                    np.abs(base["qpos"][:k + 2] - alt["qpos"][:k + 2]).max())
    check("S9 no look-ahead", worst < 1e-9,
          f"scrambling all data after day k changes nothing up to day k (positions to k+1): max diff {worst:.1e}")

    L.append("ALL PASS" if ok_all else "SOME CHECKS FAILED")
    txt = "\n".join(L)
    print(txt)
    os.makedirs(R.OUT, exist_ok=True)
    with open(os.path.join(R.OUT, "_selftest.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
