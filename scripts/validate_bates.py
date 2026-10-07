"""Stage 2 validation of src/bates.py (pass rules fixed before running).

Prices are for F = 100, so "abs err" is also in % of the forward.

V1  CF identities      phi(0)=1, phi(-i)=1 (forward is a martingale), phi(-u)=conj(phi(u)):
                       max error < 1e-12 over 2,000 random parameter sets / u
V2  cumulants          analytic mean/variance of ln(F_T/F0) vs numerical derivatives of
                       log phi at 0: relative diff < 1e-4 (moderate parameters)
V3  Black limit        no vol-of-vol, no jumps, v0 = theta: COS and Lewis vs Black-76,
                       max abs err < 1e-10. General formula at tiny vol-of-vol: with rho = 0
                       gap < 1e-10 at 1e-6 (second order); with rho != 0 the true price moves
                       linearly, so the gap must shrink 10x from 1e-6 to 1e-7 (ratio 9-11).
                       (Revised after run 1: the original "< 1e-7 at 1e-6" rule was wrong
                       physics, not a loosening - see _validation.txt history in handoff.)
V4  Merton limit       no vol-of-vol, v0 = theta, jumps on: vs Merton series < 1e-10
V5  Heston vs QuantLib 300 random cases, T 30d-3y, strikes +/-3 sd: max abs err < 1e-6
V6  Bates vs QuantLib  same, with jumps: max abs err < 1e-6
V7  COS vs Lewis       1,500 random cases, T 1 day-5 years, strikes +/-8 sd, wide
                       parameters: max abs err < 1e-8, and relative err < 1e-6 where
                       price > 1e-4
V8  Monte Carlo        Andersen QE variance + exact Poisson jumps, 4 parameter sets x
                       3 strikes: |COS - MC| / SE < 3 for all 12. If a case fails, it is
                       rerun at a quarter of the time step (discretisation, not pricer)
                       and must then pass.
V9  no-arbitrage       parity between puts (expanded under Q) and calls (expanded under the
                       share measure, an independent series) < 1e-9; price within
                       [intrinsic, DF*F] (calls) / [intrinsic, DF*K] (puts); calls
                       non-increasing and convex in K; prices non-decreasing in T and v0
                       (tolerance 1e-10)
V10 extremes           T = 6 hours, 1 day, 2 days and extreme parameters (vol-of-vol 3,
                       kappa 20, lam 10, delta 0.4): finite, >= -1e-12, |COS - Lewis| < 1e-8
V11 speed              10 expiries x 101 strikes in < 1 s (median of 3 runs)
V12 smile              rho = 0, no jumps: IV symmetric in log-moneyness (|diff| < 1e-6 vol);
                       rho = -0.7 with negative jumps: IV(0.95F) > IV(F) > IV(1.05F)

V13 event variance     scheduled-event jumps (extra log-variance ev) vs Black-76 at total
                       variance vT + ev: max abs err < 1e-10

Writes data/processed/stage2/_validation.txt
"""
from __future__ import annotations

import io
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import QuantLib as ql  # noqa: E402

from src import bates as bt  # noqa: E402
from src.contracts import b76_iv  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "data", "processed", "stage2")
F0 = 100.0
RESULTS: list[tuple[str, bool, str]] = []
rng = np.random.default_rng(20261005)


def gate(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def sec(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def rand_params(n, jumps=True, wide=False):
    out = []
    for _ in range(n):
        out.append(bt.BatesParams(
            v0=rng.uniform(0.003, 0.30 if wide else 0.15),
            kappa=rng.uniform(0.2, 15.0 if wide else 8.0),
            theta=rng.uniform(0.003, 0.30 if wide else 0.15),
            sigma=rng.uniform(0.05, 2.5 if wide else 1.2),
            rho=rng.uniform(-0.95, 0.5),
            lam=rng.uniform(0.0, 5.0 if wide else 2.0) if jumps else 0.0,
            mu_j=rng.uniform(-0.3, 0.1) if jumps else 0.0,
            delta=rng.uniform(0.01, 0.35 if wide else 0.2) if jumps else 0.0))
    return out


def strikes(T, p, n_sd, n=9):
    _, c2 = bt.cumulants(T, p)
    sd = np.sqrt(c2)
    return F0 * np.exp(np.linspace(-n_sd, n_sd, n) * sd)


# ---------------------------------------------------------------- QuantLib


def ql_price(p: bt.BatesParams, days: int, K: float, is_call: bool) -> float:
    today = ql.Date(5, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    r = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    q = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    s0 = ql.QuoteHandle(ql.SimpleQuote(F0))
    if p.lam == 0.0:
        proc = ql.HestonProcess(r, q, s0, p.v0, p.kappa, p.theta, p.sigma, p.rho)
        eng = ql.AnalyticHestonEngine(ql.HestonModel(proc), 1e-13, 1000000)
    else:
        proc = ql.BatesProcess(r, q, s0, p.v0, p.kappa, p.theta, p.sigma, p.rho, p.lam, p.mu_j, p.delta)
        eng = ql.BatesEngine(ql.BatesModel(proc), 1e-13, 1000000)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K),
                           ql.EuropeanExercise(today + days))
    opt.setPricingEngine(eng)
    return opt.NPV()


# ---------------------------------------------------------------- Monte Carlo


def mc_price(p: bt.BatesParams, T: float, K: np.ndarray, n_paths=400_000, steps=None, seed=1):
    """Andersen (2008) QE for v, central-discretised log-forward, exact Poisson jumps."""
    r = np.random.default_rng(seed)
    steps = steps or max(20, int(np.ceil(T * 200)))
    dt = T / steps
    k, th, s, rho = p.kappa, p.theta, p.sigma, p.rho
    ekd = np.exp(-k * dt)
    K0 = -rho * k * th * dt / s
    K1 = 0.5 * dt * (k * rho / s - 0.5) - rho / s
    K2 = 0.5 * dt * (k * rho / s - 0.5) + rho / s
    K3 = K4 = 0.5 * dt * (1 - rho * rho)
    v = np.full(n_paths, p.v0)
    X = np.zeros(n_paths)
    for _ in range(steps):
        m = th + (v - th) * ekd
        s2 = v * s * s * ekd * (1 - ekd) / k + th * s * s * (1 - ekd) ** 2 / (2 * k)
        psi = s2 / (m * m)
        vn = np.empty_like(v)
        lo = psi <= 1.5
        z = r.standard_normal(n_paths)
        u = r.random(n_paths)
        b2 = 2 / psi[lo] - 1 + np.sqrt(2 / psi[lo]) * np.sqrt(2 / psi[lo] - 1)
        a = m[lo] / (1 + b2)
        vn[lo] = a * (np.sqrt(b2) + z[lo]) ** 2
        hi = ~lo
        pp = (psi[hi] - 1) / (psi[hi] + 1)
        beta = (1 - pp) / m[hi]
        uh = u[hi]
        vn[hi] = np.where(uh <= pp, 0.0, np.log((1 - pp) / np.maximum(1 - uh, 1e-300)) / beta)
        zx = r.standard_normal(n_paths)
        X += K0 + K1 * v + K2 * vn + np.sqrt(np.maximum(K3 * v + K4 * vn, 0.0)) * zx
        if p.lam > 0:
            nj = r.poisson(p.lam * dt, n_paths)
            X += nj * p.mu_j + np.sqrt(nj) * p.delta * r.standard_normal(n_paths) - p.lam * p.kbar * dt
        v = vn
    FT = F0 * np.exp(X)
    out = []
    for k_ in K:
        pay = np.maximum(k_ - FT, 0.0) if k_ < F0 else np.maximum(FT - k_, 0.0)
        out.append((pay.mean(), pay.std(ddof=1) / np.sqrt(n_paths)))
    return out, steps


# ---------------------------------------------------------------- gates


def v1_cf():
    sec("V1 characteristic-function identities")
    e0 = em = ec = 0.0
    for p in rand_params(2000, wide=True):
        T = rng.choice([1 / 365, 7 / 365, 0.25, 1.0, 5.0])
        e0 = max(e0, abs(bt.cf(0.0, T, p) - 1))
        if p.kappa > p.rho * p.sigma:
            em = max(em, abs(bt.cf(-1j, T, p) - 1))
        u = rng.uniform(0.1, 200)
        ec = max(ec, abs(bt.cf(-u, T, p) - np.conj(bt.cf(u, T, p))))
    gate("V1 CF identities", max(e0, em, ec) < 1e-12,
         f"|phi(0)-1| {e0:.1e}, |phi(-i)-1| {em:.1e}, |phi(-u)-conj phi(u)| {ec:.1e}")


def v2_cumulants():
    sec("V2 cumulants vs numerical derivatives of log phi")
    worst = 0.0
    for p in rand_params(300):
        T = rng.choice([7 / 365, 0.25, 1.0, 3.0])
        c1, c2 = bt.cumulants(T, p)
        h = 1e-3 / np.sqrt(c2)
        lp, lm = bt.log_cf(h, T, p), bt.log_cf(-h, T, p)
        n1 = np.imag(lp - lm) / (2 * h)
        n2 = -np.real(lp + lm) / (h * h)
        worst = max(worst, abs(n1 - c1) / np.sqrt(c2), abs(n2 - c2) / c2)
    gate("V2 cumulants", worst < 1e-4, f"max relative diff {worst:.2e}")


def v3_black():
    sec("V3 Black-76 limit")
    ec = el = e0 = 0.0
    ratios = []
    for _ in range(200):
        v = rng.uniform(0.003, 0.3)
        T = rng.choice([1 / 365, 7 / 365, 30 / 365, 1.0, 5.0])
        rho = rng.uniform(-0.9, 0.5)
        mk = lambda s, r=rho: bt.BatesParams(v0=v, kappa=p0.kappa, theta=v, sigma=s, rho=r)
        p0 = bt.BatesParams(v0=v, kappa=rng.uniform(0.2, 8), theta=v, sigma=0.0, rho=rho)
        K = strikes(T, p0, 6)
        bl = bt.black76(F0, K, T, np.sqrt(v), 1.0, K >= F0)
        ec = max(ec, np.abs(bt.price_cos(F0, K, T, p0, 1.0, K >= F0) - bl).max())
        j = rng.integers(len(K))
        el = max(el, abs(bt.price_lewis(F0, K[j], T, p0, 1.0, K[j] >= F0) - bl[j]))
        # rho != 0: the true price moves linearly in sigma, so the gap must shrink 10x per decade
        g6 = np.abs(bt.price_cos(F0, K, T, mk(1e-6), 1.0, K >= F0) - bl).max()
        g7 = np.abs(bt.price_cos(F0, K, T, mk(1e-7), 1.0, K >= F0) - bl).max()
        if g7 > 1e-11:
            ratios.append(g6 / g7)
        # rho = 0: the first-order term vanishes, the gap is second order
        e0 = max(e0, np.abs(bt.price_cos(F0, K, T, mk(1e-6, 0.0), 1.0, K >= F0) - bl).max())
    ratios = np.array(ratios)
    gate("V3 Black limit", max(ec, el) < 1e-10 and e0 < 1e-10 and np.all((ratios > 9) & (ratios < 11)),
         f"exact: COS {ec:.1e}, Lewis {el:.1e}; vol-of-vol 1e-6 with rho=0 {e0:.1e}; "
         f"rho!=0 gap ratio 1e-6/1e-7 in [{ratios.min():.2f}, {ratios.max():.2f}] (n={len(ratios)})")


def v4_merton():
    sec("V4 Merton limit")
    err = 0.0
    for _ in range(200):
        v = rng.uniform(0.003, 0.2)
        T = rng.choice([1 / 365, 7 / 365, 30 / 365, 1.0, 3.0])
        p = bt.BatesParams(v0=v, kappa=1.0, theta=v, sigma=0.0, rho=0.0, lam=rng.uniform(0.05, 5),
                           mu_j=rng.uniform(-0.3, 0.1), delta=rng.uniform(0.01, 0.3))
        K = strikes(T, p, 6)
        m = bt.merton(F0, K, T, np.sqrt(v), p.lam, p.mu_j, p.delta, 1.0, K >= F0)
        err = max(err, np.abs(bt.price_cos(F0, K, T, p, 1.0, K >= F0) - m).max())
    gate("V4 Merton limit", err < 1e-10, f"max abs err {err:.1e}")


def v5_v6_quantlib():
    for name, jumps in (("V5 Heston vs QuantLib", False), ("V6 Bates vs QuantLib", True)):
        sec(name)
        errs = []
        for p in rand_params(60, jumps=jumps):
            days = int(rng.choice([30, 90, 365, 1095]))
            T = days / 365
            K = strikes(T, p, 3, n=5)
            ours = bt.price_cos(F0, K, T, p, 1.0, K >= F0)
            for k_, o in zip(K, ours):
                errs.append(abs(o - ql_price(p, days, k_, k_ >= F0)))
        errs = np.array(errs)
        gate(name, errs.max() < 1e-6, f"max abs err {errs.max():.1e}, median {np.median(errs):.1e} (n={len(errs)})")


def v7_cos_lewis():
    sec("V7 COS vs Lewis reference")
    ae, re_, n = 0.0, 0.0, 0
    worst = None
    t0 = time.time()
    for p in rand_params(300, wide=True):
        T = float(rng.choice([1 / 365, 2 / 365, 7 / 365, 30 / 365, 0.25, 1.0, 5.0]))
        K = strikes(T, p, 8, n=5)
        cos = bt.price_cos(F0, K, T, p, 1.0, K >= F0)
        for k_, c in zip(K, cos):
            lw = bt.price_lewis(F0, k_, T, p, 1.0, k_ >= F0)
            a = abs(c - lw)
            if a > ae:
                ae, worst = a, (p, T, k_, c, lw)
            if lw > 1e-4:
                re_ = max(re_, a / lw)
            n += 1
    print(f"  {n} prices in {time.time() - t0:.0f}s; worst case: {worst}")
    gate("V7 COS vs Lewis", ae < 1e-8 and re_ < 1e-6, f"max abs err {ae:.1e}, max rel err (price>1e-4) {re_:.1e} (n={n})")


def v8_mc():
    sec("V8 Monte Carlo")
    cases = [
        ("NIFTY-like 30d", bt.BatesParams(0.02, 3.0, 0.025, 0.5, -0.7, 0.5, -0.08, 0.06), 30 / 365),
        ("crash-heavy 2d", bt.BatesParams(0.04, 5.0, 0.03, 0.8, -0.6, 3.0, -0.10, 0.08), 2 / 365),
        ("long, wild 2y", bt.BatesParams(0.06, 1.0, 0.08, 1.2, -0.85, 0.3, -0.20, 0.15), 2.0),
        ("Heston 90d", bt.BatesParams(0.03, 2.0, 0.04, 0.6, -0.5), 90 / 365),
    ]
    zs, ok = [], True
    for name, p, T in cases:
        K = strikes(T, p, 1.5, n=3)
        cos = bt.price_cos(F0, K, T, p, 1.0, K >= F0)
        mc, steps = mc_price(p, T, K)
        for k_, c, (m, se) in zip(K, cos, mc):
            z = (m - c) / se
            line = f"  {name:16s} K={k_:7.2f} COS {c:9.5f}  MC {m:9.5f} +/- {se:.5f}  z={z:+.2f}  ({steps} steps)"
            if abs(z) >= 3:
                (m2, se2), = mc_price(p, T, [k_], steps=steps * 4, seed=2)[0]
                z = (m2 - c) / se2
                line += f" -> quarter step: MC {m2:.5f} z={z:+.2f}"
            print(line)
            zs.append(z)
            ok &= abs(z) < 3
    gate("V8 Monte Carlo", ok, f"max |z| {np.max(np.abs(zs)):.2f}, mean |z| {np.mean(np.abs(zs)):.2f} (n={len(zs)})")


def v9_arbitrage():
    sec("V9 no-arbitrage")
    par = bnd = mono = conv = tmono = vmono = 0.0
    for p in rand_params(200, wide=True):
        T = float(rng.choice([1 / 365, 7 / 365, 30 / 365, 1.0]))
        DF = float(np.exp(-0.06 * T))
        K = F0 * np.exp(np.linspace(-6, 6, 61) * np.sqrt(bt.cumulants(T, p)[1]))
        P, C = bt.price_cos_sides(F0, K, T, p, DF)
        par = max(par, np.abs(C - P - DF * (F0 - K)).max())
        Cp = bt.price_cos(F0, K, T, p, DF, True)
        bnd = max(bnd, (DF * np.maximum(F0 - K, 0) - Cp).max(), (Cp - DF * F0).max(),
                  (DF * np.maximum(K - F0, 0) - P).max(), (P - DF * K).max())
        mono = max(mono, np.diff(Cp).max())
        conv = max(conv, -(np.diff(np.diff(Cp) / np.diff(K))).min())   # slopes non-decreasing
        C2 = bt.price_cos(F0, K, T * 1.5, p, 1.0, True)
        tmono = max(tmono, (bt.price_cos(F0, K, T, p, 1.0, True) - C2).max())
        pv = bt.BatesParams(p.v0 * 1.2, p.kappa, p.theta, p.sigma, p.rho, p.lam, p.mu_j, p.delta)
        vmono = max(vmono, (bt.price_cos(F0, K, T, p, 1.0, True) - bt.price_cos(F0, K, T, pv, 1.0, True)).max())
    ok = par < 1e-9 and max(bnd, mono, conv, tmono, vmono) < 1e-10
    gate("V9 no-arbitrage", ok, f"parity {par:.1e}; worst violation: bounds {bnd:.1e}, monotone K {mono:.1e}, "
         f"convex K {conv:.1e}, monotone T {tmono:.1e}, monotone v0 {vmono:.1e}")


def v10_extremes():
    sec("V10 short maturities and extreme parameters")
    worst, neg, finite = 0.0, 0.0, True
    ext = [bt.BatesParams(0.3, 20.0, 0.3, 3.0, -0.95, 10.0, -0.3, 0.4),
           bt.BatesParams(0.005, 0.2, 0.005, 3.0, 0.5, 10.0, 0.1, 0.4),
           bt.BatesParams(0.01, 20.0, 0.2, 0.05, -0.95, 0.0, 0.0, 0.0),
           bt.BatesParams(0.15, 8.0, 0.02, 2.0, -0.9, 0.1, -0.25, 0.05)]
    for p in ext + rand_params(30, wide=True):
        for T in (0.25 / 365, 1 / 365, 2 / 365):
            K = strikes(T, p, 8, n=7)
            c = bt.price_cos(F0, K, T, p, 1.0, K >= F0)
            finite &= bool(np.isfinite(c).all())
            neg = min(neg, c.min())
            for k_, x in zip(K, c):
                worst = max(worst, abs(x - bt.price_lewis(F0, k_, T, p, 1.0, k_ >= F0)))
    gate("V10 extremes", finite and neg >= -1e-12 and worst < 1e-8,
         f"all finite {finite}, most negative {neg:.1e}, max |COS - Lewis| {worst:.1e}")


def v11_speed():
    sec("V11 speed")
    p = bt.BatesParams(0.02, 3.0, 0.025, 0.5, -0.7, 0.5, -0.08, 0.06)
    Ts = [d / 365 for d in (1, 2, 5, 7, 14, 21, 30, 60, 90, 180)]
    K = F0 * np.linspace(0.8, 1.2, 101)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for T in Ts:
            bt.price_cos(F0, K, T, p, 1.0, K >= F0)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    gate("V11 speed", med < 1.0, f"1,010 prices in {med * 1000:.0f} ms (median of 3)")


def v12_smile():
    sec("V12 smile shape")
    T = 30 / 365
    p0 = bt.BatesParams(0.02, 3.0, 0.025, 0.5, 0.0)
    k = np.linspace(0.02, 0.15, 8)
    Kc, Kp = F0 * np.exp(k), F0 * np.exp(-k)
    ivc = b76_iv(bt.price_cos(F0, Kc, T, p0, 1.0, True), np.full(8, F0), Kc, np.full(8, T), np.ones(8), np.ones(8, bool))
    ivp = b76_iv(bt.price_cos(F0, Kp, T, p0, 1.0, False), np.full(8, F0), Kp, np.full(8, T), np.ones(8), np.zeros(8, bool))
    sym = np.nanmax(np.abs(ivc - ivp))
    p1 = bt.BatesParams(0.02, 3.0, 0.025, 0.5, -0.7, 0.5, -0.08, 0.06)
    K3 = np.array([95.0, 100.0, 105.0])
    iv3 = b76_iv(bt.price_cos(F0, K3, T, p1, 1.0, K3 >= F0), np.full(3, F0), K3, np.full(3, T), np.ones(3), K3 >= F0)
    print(f"  rho=0: IV(k) vs IV(-k) {np.round(ivc * 100, 4)} / {np.round(ivp * 100, 4)}")
    print(f"  rho=-0.7, jumps: IV at 95/100/105 = {np.round(iv3 * 100, 3)}")
    gate("V12 smile", sym < 1e-6 and iv3[0] > iv3[1] > iv3[2],
         f"rho=0 asymmetry {sym:.1e} vol; skewed case {iv3[0] * 100:.2f} > {iv3[1] * 100:.2f} > {iv3[2] * 100:.2f}")


def v13_event():
    sec("V13 scheduled-event variance")
    err = 0.0
    for _ in range(200):
        v, T, ev = rng.uniform(0.003, 0.2), float(rng.choice([2 / 365, 7 / 365, 30 / 365, 0.25])), rng.uniform(1e-5, 0.01)
        p = bt.BatesParams(v0=v, kappa=1.0, theta=v, sigma=0.0, rho=0.0, ev=ev)
        K = strikes(T, p, 6)
        bl = bt.black76(F0, K, T, np.sqrt(v + ev / T), 1.0, K >= F0)
        err = max(err, np.abs(bt.price_cos(F0, K, T, p, 1.0, K >= F0) - bl).max())
    gate("V13 event variance", err < 1e-10, f"vs Black at total variance vT + ev: max abs err {err:.1e}")


def main():
    os.makedirs(OUT, exist_ok=True)
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        for f in (v1_cf, v2_cumulants, v3_black, v4_merton, v5_v6_quantlib, v7_cos_lewis, v8_mc,
                  v9_arbitrage, v10_extremes, v11_speed, v12_smile, v13_event):
            f()
        sec("SUMMARY")
        for name, ok, detail in RESULTS:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    with open(os.path.join(OUT, "_validation.txt"), "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())


if __name__ == "__main__":
    main()
