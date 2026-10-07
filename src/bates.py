"""Bates (1996) option pricing: Heston stochastic variance + Merton lognormal jumps.

Everything is on the forward of the option's own expiry, so rates and dividends
only enter through the discount factor DF (same convention as src/contracts.py):

    dF/F = sqrt(v) dW1 - lam*kbar dt + (e^J - 1) dN,    J ~ N(mu_j, delta^2)
    dv   = kappa (theta - v) dt + sigma sqrt(v) dW2,    d<W1, W2> = rho dt
    kbar = exp(mu_j + delta^2/2) - 1                    (jump compensator)

    price = DF * E[(F_T - K)^+]   (call),   DF * E[(K - F_T)^+]   (put)

T is in years (calendar days / 365 in this project). Parameters are passed as a
BatesParams; sigma = 0 is handled exactly (deterministic variance), lam = 0 is
Heston, sigma = 0 and v0 = theta is Merton, both zero is Black-76.

Two independent pricers share only the characteristic function:
  price_cos    Fang-Oosterlee COS expansion, vectorised over strikes - production.
  price_lewis  Lewis (2001) single integral by adaptive quadrature - slow reference.
The characteristic function uses the "little Heston trap" form (Albrecher et al.
2007), which stays on the principal branch of the complex log for any maturity.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import integrate
from scipy.special import gammaln, ndtr

COS_L = 10.0         # minimum half-width of the range in standard deviations
COS_N_MIN, COS_N_MAX = 256, 2 ** 16
COS_TAIL = 1e-15     # probability left outside the range on each side (Chernoff bound)
COS_CF_EPS = 1e-15   # |phi| below which the series is truncated


@dataclass(frozen=True)
class BatesParams:
    v0: float          # initial variance
    kappa: float       # mean-reversion speed of variance
    theta: float       # long-run variance
    sigma: float       # vol of variance
    rho: float         # correlation of index and variance shocks
    lam: float = 0.0   # jump intensity (jumps per year)
    mu_j: float = 0.0  # mean log jump size
    delta: float = 0.0  # std of log jump size
    ev: float = 0.0     # extra log-variance from scheduled events before expiry (each a
                        # compensated jump N(-psi^2/2, psi^2), so ev = sum psi^2)

    @property
    def kbar(self) -> float:
        return float(np.expm1(self.mu_j + 0.5 * self.delta ** 2))


# ---------------------------------------------------------------- characteristic function


def _mean_variance(p: BatesParams, T: float) -> float:
    """E[integral_0^T v dt] = theta T + (v0 - theta)(1 - e^{-kappa T})/kappa."""
    if p.kappa * T < 1e-10:
        return p.v0 * T
    return p.theta * T + (p.v0 - p.theta) * (-np.expm1(-p.kappa * T)) / p.kappa


def _clog1p(z):
    """Accurate complex log(1 + z) for small |z| (numpy's complex log1p is not)."""
    x, y = np.real(z), np.imag(z)
    return 0.5 * np.log1p(2.0 * x + x * x + y * y) + 1j * np.arctan2(y, 1.0 + x)


def log_cf(u, T: float, p: BatesParams):
    """log E[exp(i u X)], X = ln(F_T / F_0). u may be complex (array or scalar)."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    q = u * u + iu                       # u^2 + i u
    if p.sigma < 1e-12:
        diff = -0.5 * q * _mean_variance(p, T)
    else:
        s2 = p.sigma * p.sigma
        beta = p.kappa - p.rho * p.sigma * iu
        d = np.sqrt(beta * beta + s2 * q)
        # q = 0 (u = 0 or u = -i) gives A = B = 0 exactly; with kappa < rho*sigma the
        # principal d makes beta + d = 0 there, so substitute a dummy to avoid 0/0
        zero = q == 0
        bpd = np.where(zero, 1.0, beta + d)
        bmd_s2 = -q / bpd                    # (beta - d)/sigma^2 without cancellation
        g = bmd_s2 * s2 / bpd
        one_m_e = -np.expm1(-d * T)
        one_m_ge = 1.0 - g * (1.0 - one_m_e)
        # log((1 - g e)/(1 - g)) = log1p(g (1 - e)/(1 - g))
        A = p.kappa * p.theta * (bmd_s2 * T - 2.0 / s2 * _clog1p(g * one_m_e / (1.0 - g)))
        B = bmd_s2 * one_m_e / one_m_ge
        diff = A + B * p.v0
    jump = 0.0
    if p.lam > 0.0:
        jump = p.lam * T * (np.exp(iu * p.mu_j - 0.5 * p.delta ** 2 * u * u) - 1.0 - iu * p.kbar)
    if p.ev > 0.0:
        jump = jump - 0.5 * p.ev * q
    return diff + jump


def cf(u, T: float, p: BatesParams):
    return np.exp(log_cf(u, T, p))


def cumulants(T: float, p: BatesParams) -> tuple[float, float]:
    """Mean and variance of X = ln(F_T/F_0).

    Diffusion part: X = -I/2 + M with I = int v dt, M = int sqrt(v) dW1, so
    Var X = E[I] + Var(I)/4 - Cov(I, M); Var(I) and Cov(I, M) are the CIR moments
    (derived symbolically, checked against derivatives of log_cf). Jumps add
    lam T (mu_j^2 + delta^2).
    """
    w = _mean_variance(p, T)
    c1 = -0.5 * w + p.lam * T * (p.mu_j - p.kbar)
    k, th, s, r, v0 = p.kappa, p.theta, p.sigma, p.rho, p.v0
    if s < 1e-12 or k * T < 1e-8:
        c2 = w
    else:
        kT = k * T
        e1, e2 = np.exp(-kT), np.exp(-2 * kT)
        var_i = s * s / k ** 3 * (kT * th * (1 + 2 * e1) - 2 * kT * v0 * e1 - 2.5 * th
                                  + 2 * th * e1 + 0.5 * th * e2 + v0 * (1 - e2))
        cov_im = r * s / k ** 2 * (kT * (th + (th - v0) * e1) + (v0 - 2 * th) * (1 - e1))
        c2 = max(w + 0.25 * var_i - cov_im, 0.25 * w)
    c2 += p.lam * T * (p.mu_j ** 2 + p.delta ** 2) + p.ev
    c1 -= 0.5 * p.ev
    return float(c1), float(c2)


# ---------------------------------------------------------------- COS (production)


def _chi_psi(uk, a, c, d):
    """Fang-Oosterlee chi_k(c, d) and psi_k(c, d) on the interval [a, b]."""
    cd, cc = uk * (d - a), uk * (c - a)
    chi = (np.cos(cd) * np.exp(d) - np.cos(cc) * np.exp(c)
           + uk * (np.sin(cd) * np.exp(d) - np.sin(cc) * np.exp(c))) / (1.0 + uk * uk)
    psi = np.empty_like(uk)
    psi[0] = d - c
    psi[1:] = (np.sin(cd[1:]) - np.sin(cc[1:])) / uk[1:]
    return chi, psi


def explosion_time(omega, p: BatesParams):
    """Time T* beyond which E[(F_T/F_0)^omega] is infinite (Andersen & Piterbarg 2007,
    Prop. 3.1). Lognormal jumps have all moments, so only the Heston part matters.
    Vectorised over omega."""
    w = np.atleast_1d(np.asarray(omega, float))
    out = np.full(w.shape, np.inf)
    if p.sigma < 1e-12:
        return out if np.ndim(omega) else float(out[0])
    k = p.rho * p.sigma * w - p.kappa
    D = k * k - p.sigma ** 2 * w * (w - 1.0)
    live = (w < 0.0) | (w > 1.0)
    pos = live & (D >= 0.0) & (k > 0.0)
    sD = np.sqrt(np.abs(D))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(pos, np.log((k + sD) / (k - sD)) / sD, out)
        out = np.where(live & (D < 0.0), 2.0 / sD * np.arctan2(sD, k), out)
    return out if np.ndim(omega) else float(out[0])


_OMEGA_STEPS = np.geomspace(0.02, 3e4, 49)    # large orders give tight bounds at short T
_OMEGAS = np.concatenate([-_OMEGA_STEPS, [0.25, 0.5, 0.75, 1.0], 1.0 + _OMEGA_STEPS])


def tail_ranges(T: float, p: BatesParams, tol: float = None):
    """Ranges holding all but <= tol probability on each side of the variable expanded by
    COS: X = ln(F_T/F_0) under Q, and Z = -X under the share measure (weight F_T/F_0).
    Returns ((lo_Q, hi_Q), (lo_S, hi_S)).

    Chernoff bounds from the moments M(w) = E[(F_T/F_0)^w] = phi(-i w), using only w with
    T < 0.9 T*(w) so every moment used is finite. Under Q: P(X<a) <= M(w) e^{-w a} (w<0),
    P(X>b) <= M(w) e^{-w b} (w>0). Under the share measure the weight shifts w by one.
    """
    lt = np.log(COS_TAIL if tol is None else tol)
    ws = _OMEGAS[T < 0.9 * explosion_time(_OMEGAS, p)]
    with np.errstate(over="ignore", invalid="ignore"):
        logm = np.real(log_cf(-1j * ws, T, p))
    ok = np.isfinite(logm)
    ws, logm = ws[ok], logm[ok]

    def left(shift):            # a <= (lt - logM)/(shift - w), over w < shift
        m = ws < shift
        return float(np.max((lt - logm[m]) / (shift - ws[m]))) if m.any() else np.nan

    def right(shift):           # b >= (logM - lt)/(w - shift), over w > shift
        m = ws > shift
        return float(np.min((logm[m] - lt) / (ws[m] - shift))) if m.any() else np.nan

    q = (left(0.0), right(0.0))
    s = (-right(1.0), -left(1.0))
    return q, s


@lru_cache(maxsize=256)
def _tail_ranges_cached(T: float, p: BatesParams):
    return tail_ranges(T, p)


def tail_range(T: float, p: BatesParams, share: bool, tol: float = None) -> tuple[float, float]:
    r = tail_ranges(T, p, tol) if tol is not None else _tail_ranges_cached(T, p)
    return r[1 if share else 0]


_CUTOFF_STEPS = 2.0 ** (np.arange(64) / 4.0)


def _cf_cutoff(fn, sd: float) -> float:
    """Smallest frequency beyond which |phi| stays below COS_CF_EPS on a grid with four
    points per doubling (|phi| is not monotone when jumps are present)."""
    g = _CUTOFF_STEPS / sd
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        small = np.abs(fn(g)) < COS_CF_EPS
    bad = np.nonzero(~small)[0]
    if len(bad) == 0:
        return float(g[0])
    return float(g[min(bad[-1] + 1, len(g) - 1)])


def _cos_unit_put(x: np.ndarray, T: float, p: BatesParams, share: bool, N: int | None):
    """E[(1 - e^{x + Z})^+] by COS, Z = ln(F_T/F_0) under Q, or -ln(F_T/F_0) under the
    share measure. Both are bounded payoffs, so truncation error stays below the tail mass."""
    c1, c2 = cumulants(T, p)
    sd = np.sqrt(c2)
    lo, hi = tail_range(T, p, share)
    m = -c1 if share else c1
    lo, hi = np.nanmin([lo, m - COS_L * sd]), np.nanmax([hi, m + COS_L * sd])
    a, b = x.min() + lo, x.max() + hi
    if a >= 0.0:
        return np.zeros_like(x)
    fn = (lambda u: cf(-u - 1j, T, p)) if share else (lambda u: cf(u, T, p))
    if N is None:
        n = _cf_cutoff(fn, sd) * (b - a) / np.pi
        N = int(min(max(2 ** int(np.ceil(np.log2(max(n, 1.0)))), COS_N_MIN), COS_N_MAX))
    uk = np.arange(N) * np.pi / (b - a)
    phi = fn(uk)
    chi, psi = _chi_psi(uk, a, a, min(0.0, b))
    U = 2.0 / (b - a) * (psi - chi)
    U[0] *= 0.5
    rot = np.exp(1j * np.outer(x - a, uk))          # (strikes, N)
    return np.real((rot * phi) @ U)


def price_cos_sides(F: float, K, T: float, p: BatesParams, DF: float = 1.0,
                    N: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(put, call) for every strike, each from its own independent expansion:
    puts under Q, calls under the share measure. Used for the parity test."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    put = DF * K * _cos_unit_put(np.log(F / K), T, p, False, N)
    call = DF * F * _cos_unit_put(np.log(K / F), T, p, True, N)
    return put, call


def price_cos(F: float, K, T: float, p: BatesParams, DF: float = 1.0, is_call=True,
              N: int | None = None, puts_only: bool = False) -> np.ndarray:
    """Bates prices for many strikes of one expiry (COS method).

    The out-of-the-money option of each strike is expanded directly (puts under Q,
    calls under the share measure); the in-the-money one follows from parity, which
    only adds the exact intrinsic DF*(F - K).
    puts_only=True expands every strike under Q (one series, half the cost); calls then
    come from parity, which is accurate only for moderately out-of-the-money calls
    (checked for the calibration universe in stage3 gates).
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), K.shape)
    if T <= 0.0:
        intr = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
        return DF * intr
    if puts_only:
        put = DF * K * _cos_unit_put(np.log(F / K), T, p, False, N)
        return np.where(is_call, put + DF * (F - K), put)
    lo = K < F
    otm = np.empty_like(K)
    if lo.any():
        otm[lo] = DF * K[lo] * _cos_unit_put(np.log(F / K[lo]), T, p, False, N)
    if (~lo).any():
        otm[~lo] = DF * F * _cos_unit_put(np.log(K[~lo] / F), T, p, True, N)
    fwd = DF * (F - K)
    put = np.where(lo, otm, otm - fwd)
    call = np.where(lo, otm + fwd, otm)
    return np.where(is_call, call, put)


# ---------------------------------------------------------------- Lewis (reference)


def _lewis_upper(T: float, p: BatesParams, tol: float = 1e-16) -> float:
    U = 10.0
    while U < 1e8:
        if abs(cf(U - 0.5j, T, p)) / (U * U) < tol:
            return U
        U *= 2.0
    return U


def price_lewis(F: float, K: float, T: float, p: BatesParams, DF: float = 1.0,
                is_call: bool = True) -> float:
    """Lewis (2001): C = DF [F - sqrt(FK)/pi * int_0^inf Re(e^{iuk} phi(u - i/2)) / (u^2 + 1/4) du],
    k = ln(F/K). Adaptive quadrature over geometric sub-intervals; slow, for testing."""
    if T <= 0.0:
        return DF * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    k = np.log(F / K)

    def f(u):
        return float(np.real(np.exp(1j * u * k) * cf(u - 0.5j, T, p))) / (u * u + 0.25)

    U = _lewis_upper(T, p)
    edges = np.concatenate([[0.0], np.geomspace(1e-3, U, 60)])
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        val, _ = integrate.quad(f, lo, hi, epsabs=1e-15, epsrel=1e-13, limit=400)
        total += val
    call = DF * (F - np.sqrt(F * K) / np.pi * total)
    return call if is_call else call - DF * (F - K)


# ---------------------------------------------------------------- closed forms (for tests)


def black76(F, K, T, sig, DF=1.0, is_call=True):
    F, K = np.asarray(F, float), np.asarray(K, float)
    v = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    call = DF * (F * ndtr(d1) - K * ndtr(d2))
    return np.where(is_call, call, call - DF * (F - K))


def merton(F, K, T, sig, lam, mu_j, delta, DF=1.0, is_call=True, n_max=200):
    """Merton (1976) series: Poisson mixture of Black prices on the forward."""
    kbar = np.expm1(mu_j + 0.5 * delta ** 2)
    out = 0.0
    for n in range(n_max + 1):
        w = np.exp(-lam * T + n * np.log(lam * T) - gammaln(n + 1)) if lam * T > 0 else float(n == 0)
        if w < 1e-300 and n > lam * T:
            break
        Fn = F * np.exp(-lam * kbar * T) * (1.0 + kbar) ** n
        sn = np.sqrt(sig * sig + n * delta * delta / T)
        out = out + w * black76(Fn, K, T, sn, DF, is_call)
    return out
