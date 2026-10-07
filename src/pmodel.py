"""Past-only ("physical measure") model of daily index returns, for Stage 4.

GJR-GARCH(1,1) with Student-t shocks on daily close-to-close log returns (percent),
refitted at the start of every month on all returns strictly before that month; within the
month the variance recursion runs forward with those fixed parameters, so the forecast made
at the close of day t uses returns up to and including t only.

Jumps: day t is a jump day if |r_t - mu| > JUMP_C * sigma_t, where sigma_t is the GARCH
forecast for day t made at t-1. Jump intensity / mean / sd are expanding-window statistics
of jump days up to and including t (crude with daily data: a jump day's return also contains
that day's diffusive move). Used as a diagnostic of realised tail moves only: the Student-t
shocks already produce about as many > 4 sigma days as observed, so these are not added on
top of the GARCH variance.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

JUMP_C = 4.0
FIRST_FIT = pd.Timestamp("2009-01-01")
MIN_JUMPS = 3


def close_series(yahoo: pd.Series, panel_spot: pd.Series) -> pd.Series:
    """Daily closes: panel spot (NSE-filled) where available, Yahoo only before the panel starts."""
    p = panel_spot.dropna()
    y = yahoo[yahoo.index < p.index.min()].dropna()
    return pd.concat([y, p]).sort_index()


def _recursion(eps: np.ndarray, omega, alpha, gamma, beta, s0) -> np.ndarray:
    """sig2[i] = variance forecast for day i made at i-1 (sig2[0] = s0)."""
    n = len(eps)
    s = np.empty(n + 1)
    s[0] = s0
    for i in range(n):
        s[i + 1] = omega + (alpha + gamma * (eps[i] < 0)) * eps[i] ** 2 + beta * s[i]
    return s


def fit_monthly(r: pd.Series) -> pd.DataFrame:
    """Per month: GJR parameters fitted on returns before the month (percent units)."""
    months = pd.date_range(FIRST_FIT, r.index.max() + pd.offsets.MonthBegin(1), freq="MS")
    return pd.DataFrame([fit_month(r, m) for m in months])


def fit_month(r: pd.Series, m: pd.Timestamp) -> dict:
    """GJR parameters for month m from the returns before m (percent units)."""
    x = r[r.index < m] * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    pr = res.params
    return dict(month=m, mu=pr["mu"], omega=pr["omega"], alpha=pr["alpha[1]"], gamma=pr["gamma[1]"],
                beta=pr["beta[1]"], nu=pr["nu"], n_obs=len(x), converged=res.convergence_flag == 0)


def daily_state(r: pd.Series, params: pd.DataFrame) -> pd.DataFrame:
    """For each day t >= FIRST_FIT: sigma2_t (forecast for t made at t-1), sigma2_next (for t+1
    made at t), the month's parameters, the jump flag, and expanding jump statistics.
    Units: decimal returns, variance per session."""
    out = []
    x = r * 100
    for _, pm in params.iterrows():
        m0, m1 = pm["month"], pm["month"] + pd.offsets.MonthBegin(1)
        hist = x[x.index < m1]
        eps = (hist - pm["mu"]).to_numpy()
        s0 = float(np.var(eps[:250]))
        s = _recursion(eps, pm["omega"], pm["alpha"], pm["gamma"], pm["beta"], s0)
        idx = np.nonzero(np.asarray(hist.index >= m0))[0]
        if len(idx) == 0:
            continue
        out.append(pd.DataFrame(dict(date=hist.index[idx], r=hist.to_numpy()[idx] / 100, eps=eps[idx] / 100,
                                     sigma2=s[idx] / 1e4, sigma2_next=s[idx + 1] / 1e4,
                                     mu=pm["mu"] / 100, omega=pm["omega"] / 1e4, alpha=pm["alpha"],
                                     gamma=pm["gamma"], beta=pm["beta"])))
    d = pd.concat(out, ignore_index=True)
    return d


def jump_stats(r: pd.Series, state: pd.DataFrame, c: float = JUMP_C) -> pd.DataFrame:
    """Expanding (from the first return) jump intensity per year, mean and sd of jump-day log
    returns, using the month-specific GARCH sigma for detection. Days before FIRST_FIT are
    classified with the first month's parameters (fitted on those same early days - the only
    in-sample element, affecting the jump history available in early 2009)."""
    first = state.iloc[0]
    x = r * 100
    pre = x[x.index < FIRST_FIT]
    eps_pre = (pre - first["mu"] * 100).to_numpy()
    s_pre = _recursion(eps_pre, first["omega"] * 1e4, first["alpha"], first["gamma"], first["beta"],
                       float(np.var(eps_pre[:250])))[:-1]
    flag_pre = np.abs(eps_pre) > c * np.sqrt(s_pre)
    j = pd.concat([pd.DataFrame(dict(date=pre.index, r=pre.to_numpy() / 100, jump=flag_pre)),
                   pd.DataFrame(dict(date=state["date"], r=state["r"],
                                     jump=np.abs(state["eps"]) > c * np.sqrt(state["sigma2"])))], ignore_index=True)
    years = (j["date"] - j["date"].iloc[0]).dt.days / 365.0
    cnt = j["jump"].cumsum()
    jr = j["r"].where(j["jump"])
    s1 = jr.fillna(0).cumsum()
    s2 = (jr ** 2).fillna(0).cumsum()
    mean = s1 / cnt.where(cnt > 0)
    var = (s2 - cnt * mean ** 2) / (cnt - 1).where(cnt > 1)
    out = pd.DataFrame(dict(date=j["date"], r=j["r"], jump=j["jump"], n_jumps=cnt,
                            lam_p=cnt / years.where(years > 0.5), mu_p=mean, delta_p=np.sqrt(var.clip(lower=0))))
    out.loc[out["n_jumps"] < MIN_JUMPS, ["lam_p", "mu_p", "delta_p"]] = np.nan
    return out


NAIVE_W = 22


def trailing_rv(r: pd.Series) -> pd.Series:
    """Mean r^2 over the last NAIVE_W sessions up to and including t: the naive forecast of
    the per-session variance. Chosen over raw/expanding HAR as the cross-check because it had
    the smallest forecast bias before 2018 (expanding-window fits inherit the 2008 level)."""
    return (r ** 2).rolling(NAIVE_W).mean()


def variance_forecast(st: pd.Series, n: int) -> float:
    """E[sum_{k=1..n} r_{t+k}^2 | t] (decimal^2) from the day-t state row: GJR persistence with
    symmetric shocks is alpha + beta + gamma/2."""
    phi = st["alpha"] + st["beta"] + 0.5 * st["gamma"]
    lr = st["omega"] / (1 - phi) if phi < 1 else np.nan
    if not np.isfinite(lr) or n <= 0:
        return np.nan
    tot = n * lr + (st["sigma2_next"] - lr) * (1 - phi ** n) / (1 - phi)
    return float(tot + n * st["mu"] ** 2)
