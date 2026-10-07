"""Synthetic self-test of forecast_ledger learners (must pass before the real run).

Market: GJR-GARCH(1,1) with t(7) shocks, 4,500 sessions, true conditional variance known, so the true
h-step variance forecast is known exactly.
  L1 target() = brute-force sums
  L2 garch_h() = brute-force expected-variance recursion (symmetric shocks: E[I(eps<0)] = 1/2)
  L3 past_stats() = brute-force mean over rows settled by the origin
  L4 "implied" = truth x 1.3 x lognormal noise (sd 0.25): blend beats it (NW t > 2) and calibrate()
     removes its premium (calibrated implied / realised within 5% over the 2nd half; checked against
     realised, since one finite path's realised level differs from its expectation)
  L5 "implied" = truth exactly: blend does NOT beat it (t < 2), puts the largest weight on it (> 0.45)
  L6 truncation: har / calibrate / blend outputs at t unchanged when data after t is removed
  L7 p_inside(a, a^2) = 0.6827 for small a (the market's own +/-1 sigma probability)
"""
from __future__ import annotations

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
import forecast_ledger as FL  # noqa: E402

N = 4500
OM, AL, GA, BE, NU = 2e-6, 0.03, 0.10, 0.88, 7


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(NU, N) / np.sqrt(NU / (NU - 2))
    s2 = np.empty(N + 1)
    s2[0] = OM / (1 - AL - GA / 2 - BE)
    r = np.empty(N)
    for t in range(N):
        r[t] = np.sqrt(s2[t]) * z[t]
        s2[t + 1] = OM + (AL + GA * (r[t] < 0)) * r[t] ** 2 + BE * s2[t]
    return r, s2


def true_h(s2, h):
    """E[sum r^2 over t+1..t+h | info at t]; s2[t+1] is known at t."""
    phi = AL + GA / 2 + BE
    VL = OM / (1 - phi)
    s_next = s2[1:N + 1]
    return h * VL + (1 - phi ** h) / (1 - phi) * (s_next - VL)


def main():
    L, ok_all = [], True

    def check(name, ok, msg):
        nonlocal ok_all
        ok_all &= bool(ok)
        L.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")

    r, s2 = simulate()
    h = 5
    y = FL.target(r, h)
    brute = np.array([np.sum(r[t + 1:t + 1 + h] ** 2) if t + h < N else np.nan for t in range(N)])
    check("L1 target", np.allclose(y, brute, equal_nan=True), "matches brute-force sums")

    import pandas as pd
    months = pd.DatetimeIndex(["2020-01-01"] * N)
    prm = pd.DataFrame({"month": [pd.Timestamp("2020-01-01")], "mu": [0.0], "omega": [OM * 1e4], "alpha": [AL],
                        "gamma": [GA], "beta": [BE]})
    g = FL.garch_h(s2[1:N + 1], months, prm, h)
    check("L2 garch_h", np.allclose(g, true_h(s2, h), rtol=1e-10), "matches the expected-variance recursion")

    rng = np.random.default_rng(3)
    t_idx = np.sort(rng.integers(0, 3000, 600))
    s_idx = t_idx + rng.integers(1, 30, 600)
    vals = rng.normal(size=600)
    m, _ = FL.past_stats(t_idx, s_idx, vals, window=None, min_n=10)
    bf = np.array([vals[s_idx <= t].mean() if (s_idx <= t).sum() >= 10 else np.nan for t in t_idx])
    check("L3 past_stats", np.allclose(m, bf, equal_nan=True), "matches brute force (rows settled by the origin)")

    truth = true_h(s2, h)
    noisy = truth * 1.3 * np.exp(rng.normal(0, 0.25, N))
    harf = FL.har(r, h, 756)
    gcal = FL.calibrate(g, y, h)
    b, w = FL.blend([noisy, gcal, harf], y, h, 250)
    sc = FL.calibrate(noisy, y, h)
    ok = np.isfinite(b) & np.isfinite(y) & np.isfinite(sc)
    mm, t, n = FL.nw_t(FL.qlike(y[ok], noisy[ok]) - FL.qlike(y[ok], b[ok]), h)
    late = ok & (np.arange(N) > N // 2)
    ratio = sc[late].sum() / y[late].sum()
    check("L4 beats biased noisy implied", t > 2 and abs(ratio - 1) < 0.05,
          f"QLIKE gain {mm:+.4f} t {t:+.2f}; calibrated implied / realised (2nd half) {ratio:.3f} "
          f"(raw implied {noisy[late].sum() / y[late].sum():.3f})")

    b2, w2 = FL.blend([truth, gcal, harf], y, h, 250)
    ok2 = np.isfinite(b2) & np.isfinite(y)
    mm2, t2, _ = FL.nw_t(FL.qlike(y[ok2], truth[ok2]) - FL.qlike(y[ok2], b2[ok2]), h)
    wm = np.nanmean(w2, axis=0)
    check("L5 no false win vs a perfect forecast", t2 < 2 and wm[0] >= wm.max() - 1e-12 and wm[0] > 0.45,
          f"gain {mm2:+.5f} t {t2:+.2f}; mean weights truth/garch/har {wm[0]:.2f}/{wm[1]:.2f}/{wm[2]:.2f}")

    worst = 0.0
    for cut in (1500, 2600, 4000):
        rr = r[:cut + 1]
        yy = FL.target(rr, h)
        hf = FL.har(rr, h, 756)
        gc = FL.calibrate(g[:cut + 1], yy, h)
        bb, _ = FL.blend([noisy[:cut + 1], gc, hf], yy, h, 250)
        for a_, b_ in ((hf, harf), (gc, gcal), (bb, b)):
            fa, fb = np.isfinite(a_), np.isfinite(b_[:cut + 1])
            worst = max(worst, float((fa != fb).sum()),
                        float(np.max(np.abs(a_[fa] - b_[:cut + 1][fa]) / np.abs(b_[:cut + 1][fa]))) if fa.any() else 0.0)
    check("L6 no look-ahead", worst < 1e-9, f"max relative difference after truncation {worst:.1e}")

    p = FL.p_inside(0.01, 0.01 ** 2)
    check("L7 market +/-1 sigma probability", abs(p - 0.6827) < 1e-3, f"{p:.4f}")

    L.append("ALL PASS" if ok_all else "SOME CHECKS FAILED")
    txt = "\n".join(L)
    print(txt)
    os.makedirs(FL.OUT, exist_ok=True)
    with open(os.path.join(FL.OUT, "_selftest.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
