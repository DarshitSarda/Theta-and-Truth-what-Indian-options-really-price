"""Daily risk-neutral calibration of Heston / Merton / Bates / Bates + scheduled events.

One "chain" = one symbol on one date: the out-of-the-money options that pass the Stage 1
filters, grouped by expiry, each expiry with its own forward, discount factor and T.

Objective: robust least squares on vega-scaled price errors, r = (model - market) / vega,
i.e. implied-vol errors in vol points to first order (soft-L1 loss, scale 1 vol point, so
a few stale quotes cannot drag the fit). Exact IV errors are reported after the fit.

No lookahead: a fit on date t sees only date t's prices, starts from earlier fits, and
uses scheduled events only once their date was public (config/india_events.csv).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src import bates as bt
from src.contracts import b76_iv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(PROJECT_ROOT, "data", "processed", "contracts", "panel")
EVENTS_CSV = os.path.join(PROJECT_ROOT, "config", "india_events.csv")

# ---------------------------------------------------------------- data selection (fixed)
MIN_TRADED = 10
MIN_PRICE = 0.5
DELTA_RANGE = (0.02, 0.50)
FWD_OK = ("parity", "futures", "parity_local")
MIN_TDTE, MAX_DTE = 2, 100
MAX_EXPIRIES, MAX_PER_EXPIRY, MIN_PER_EXPIRY = 6, 15, 4
MIN_OPTIONS = 10

# ---------------------------------------------------------------- models
MODELS = {
    "heston": ["v0", "kappa", "theta", "sigma", "rho"],
    "merton": ["v0", "lam", "mu_j", "delta"],
    "bates": ["v0", "kappa", "theta", "sigma", "rho", "lam", "mu_j", "delta"],
    "bates_ev": ["v0", "kappa", "theta", "sigma", "rho", "lam", "mu_j", "delta", "psi"],
}
BOUNDS = {"v0": (1e-4, 1.5), "kappa": (0.05, 25.0), "theta": (1e-4, 1.5), "sigma": (0.02, 6.0),
          "rho": (-0.99, 0.95), "lam": (0.0, 12.0), "mu_j": (-0.6, 0.4), "delta": (0.005, 0.6),
          "psi": (0.0, 0.12)}


@dataclass
class Expiry:
    expiry: pd.Timestamp
    F: float
    DF: float
    T: float
    K: np.ndarray
    is_call: np.ndarray
    mkt: np.ndarray
    vega: np.ndarray      # Rs per vol point
    iv: np.ndarray        # decimal
    n_events: int = 0


@dataclass
class Chain:
    symbol: str
    date: pd.Timestamp
    expiries: list[Expiry] = field(default_factory=list)

    @property
    def n(self) -> int:
        return sum(len(e.K) for e in self.expiries)

    def atm_var(self) -> float:
        e = self.expiries[0]
        i = np.argmin(np.abs(np.log(e.K / e.F)))
        return float(e.iv[i] ** 2)

    def subset(self, keep: list[np.ndarray]) -> "Chain":
        out = []
        for e, m in zip(self.expiries, keep):
            if m.sum() == 0:
                continue
            out.append(Expiry(e.expiry, e.F, e.DF, e.T, e.K[m], e.is_call[m], e.mkt[m], e.vega[m], e.iv[m], e.n_events))
        return Chain(self.symbol, self.date, out)


def load_symbol(symbol: str) -> pd.DataFrame:
    cols = ["date", "expiry", "strike", "side", "dte", "tdte", "mark", "contracts", "forward", "fwd_source",
            "df", "T", "log_moneyness", "iv", "delta", "vega"]
    p = pd.read_parquet(PANEL, columns=cols, filters=[("symbol", "=", symbol), ("contracts", ">=", MIN_TRADED),
                                                      ("dte", "<=", MAX_DTE), ("tdte", ">=", MIN_TDTE)])
    return select_options(p)


def select_options(p: pd.DataFrame) -> pd.DataFrame:
    """The fixed Stage 3 option filter (traded, OTM, priced, usable forward, delta range)."""
    p = p[(p["contracts"] >= MIN_TRADED) & (p["dte"] <= MAX_DTE) & (p["tdte"] >= MIN_TDTE)]
    otm = np.where(p["side"] == "CE", p["log_moneyness"] >= 0, p["log_moneyness"] <= 0)
    keep = (otm & (p["mark"] >= MIN_PRICE) & p["iv"].notna() & p["vega"].gt(0) & p["fwd_source"].isin(FWD_OK)
            & p["delta"].abs().between(*DELTA_RANGE))
    return p[keep].sort_values(["date", "expiry", "strike"]).reset_index(drop=True)


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_CSV, parse_dates=["date", "known_from"])
    return ev


def event_sessions(events: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Map each event to the first session on or after its date (when the market reacts)."""
    s = np.sort(sessions.unique())
    idx = np.searchsorted(s, events["date"].to_numpy())
    ok = idx < len(s)
    out = events[ok].copy()
    out["session"] = s[idx[ok]]
    return out


def build_chain(symbol: str, date: pd.Timestamp, day: pd.DataFrame, ev: pd.DataFrame | None) -> Chain | None:
    """Thin the day's options to <= MAX_PER_EXPIRY per expiry (evenly in log-moneyness,
    deterministic) and keep the nearest MAX_EXPIRIES usable expiries."""
    out = []
    for expiry, g in day.groupby("expiry", sort=True):
        if len(g) < MIN_PER_EXPIRY:
            continue
        if g["forward"].nunique() > 1 or g["df"].nunique() > 1:
            g = g[g["forward"] == g["forward"].iloc[0]]
        g = g.sort_values("log_moneyness")
        if len(g) > MAX_PER_EXPIRY:
            g = g.iloc[np.unique(np.round(np.linspace(0, len(g) - 1, MAX_PER_EXPIRY)).astype(int))]
        n_ev = 0
        if ev is not None and len(ev):
            n_ev = int(((ev["known_from"] <= date) & (ev["session"] > date) & (ev["session"] <= expiry)).sum())
        out.append(Expiry(expiry, float(g["forward"].iloc[0]), float(g["df"].iloc[0]), float(g["T"].iloc[0]),
                          g["strike"].to_numpy(float), (g["side"] == "CE").to_numpy(), g["mark"].to_numpy(float),
                          g["vega"].to_numpy(float), g["iv"].to_numpy(float), n_ev))
        if len(out) == MAX_EXPIRIES:
            break
    ch = Chain(symbol, date, out)
    return ch if ch.n >= MIN_OPTIONS and out else None


def event_identified(ch: Chain) -> bool:
    """The event term is only identified if some expiries end before the event and some after."""
    n = [e.n_events for e in ch.expiries]
    return max(n) > 0 and min(n) == 0


# ---------------------------------------------------------------- pricing / residuals

def to_params(model: str, x: np.ndarray, n_events: int = 0) -> bt.BatesParams:
    d = dict(zip(MODELS[model], x))
    if model == "merton":
        return bt.BatesParams(v0=d["v0"], kappa=1.0, theta=d["v0"], sigma=0.0, rho=0.0,
                              lam=d["lam"], mu_j=d["mu_j"], delta=d["delta"])
    p = dict(v0=d["v0"], kappa=d["kappa"], theta=d["theta"], sigma=d["sigma"], rho=d["rho"])
    if model in ("bates", "bates_ev"):
        p.update(lam=d["lam"], mu_j=d["mu_j"], delta=d["delta"])
    if model == "bates_ev":
        p["ev"] = n_events * d["psi"] ** 2
    return bt.BatesParams(**p)


PUTS_ONLY = True     # one Q-measure series per expiry; verified against the two-sided pricer (gate C0)


def model_prices(model: str, x: np.ndarray, ch: Chain, puts_only: bool | None = None) -> list[np.ndarray]:
    po = PUTS_ONLY if puts_only is None else puts_only
    return [bt.price_cos(e.F, e.K, e.T, to_params(model, x, e.n_events), e.DF, e.is_call, puts_only=po)
            for e in ch.expiries]


def residuals(model: str, x: np.ndarray, ch: Chain) -> np.ndarray:
    pr = model_prices(model, x, ch)
    r = np.concatenate([(m - e.mkt) / e.vega for m, e in zip(pr, ch.expiries)])
    return np.where(np.isfinite(r), r, 1e3)


def iv_errors(model: str, x: np.ndarray, ch: Chain) -> np.ndarray:
    """Exact model IV minus market IV, in vol points (NaN where the model price has no IV)."""
    out = []
    for m, e in zip(model_prices(model, x, ch), ch.expiries):
        n = len(e.K)
        iv = b76_iv(m, np.full(n, e.F), e.K, np.full(n, e.T), np.full(n, e.DF), e.is_call)
        out.append((iv - e.iv) * 100)
    return np.concatenate(out)


# ---------------------------------------------------------------- fitting

def bounds(model: str):
    lb, ub = zip(*(BOUNDS[k] for k in MODELS[model]))
    return np.array(lb), np.array(ub)


def clip(model: str, x) -> np.ndarray:
    lb, ub = bounds(model)
    span = ub - lb
    return np.clip(np.asarray(x, float), lb + 1e-6 * span, ub - 1e-6 * span)


def generic_starts(model: str, v: float) -> list[np.ndarray]:
    s = {"heston": [[v, 2.0, v, 0.8, -0.6], [v, 6.0, 1.3 * v, 1.5, -0.8]],
         "merton": [[0.8 * v, 0.5, -0.10, 0.10], [0.6 * v, 3.0, -0.05, 0.05]],
         "bates": [[0.8 * v, 3.0, v, 0.6, -0.6, 0.3, -0.10, 0.08]],
         "bates_ev": []}
    return [clip(model, x) for x in s[model]]


@dataclass
class Fit:
    model: str
    x: np.ndarray
    cost: float
    rmse_vega: float
    nfev: int
    status: int
    secs: float
    start: int


SCOUT_NFEV, POLISH_NFEV, TOL = 12, 300, 1e-9


def _lsq(model, ch, x0, max_nfev):
    lb, ub = bounds(model)
    return least_squares(lambda x: residuals(model, x, ch), clip(model, x0), bounds=(lb, ub), method="trf",
                         loss="soft_l1", f_scale=1.0, x_scale="jac", ftol=TOL, xtol=TOL, gtol=TOL,
                         max_nfev=max_nfev)


def fit_one(model: str, ch: Chain, starts: list[np.ndarray]) -> Fit:
    """Short scouting runs from every start, then the best one is run to convergence."""
    lb, _ = bounds(model)
    t0 = time.time()
    nfev, scouts = 0, []
    for i, x0 in enumerate(starts):
        try:
            r = _lsq(model, ch, x0, SCOUT_NFEV if len(starts) > 1 else POLISH_NFEV)
        except Exception:
            continue
        nfev += r.nfev
        scouts.append((r.cost, i, r))
    if not scouts:
        return Fit(model, np.full(len(lb), np.nan), np.inf, np.nan, nfev, -9, time.time() - t0, -1)
    _, best_i, best = min(scouts, key=lambda s: s[0])
    if len(starts) > 1 and best.status == 0:          # scout hit its budget: polish
        r = _lsq(model, ch, best.x, POLISH_NFEV)
        nfev += r.nfev
        if r.cost <= best.cost:
            best = r
    rm = float(np.sqrt(np.mean(best.fun ** 2)))
    return Fit(model, best.x, float(best.cost), rm, nfev, int(best.status), time.time() - t0, best_i)


def fit_day(ch: Chain, prev: dict[str, np.ndarray] | None) -> dict[str, Fit]:
    """All four models on one chain. Starts: previous fit (past information only), generic
    starts (only when there is no previous fit, for Bates), and nesting starts from today's
    simpler fits so Bates is never worse than either of its sub-models."""
    v = ch.atm_var()
    prev = prev or {}
    fits = {}
    for m in ("heston", "merton"):
        st = ([prev[m]] if m in prev else []) + generic_starts(m, v)
        fits[m] = fit_one(m, ch, st)
    h, me = fits["heston"].x, fits["merton"].x
    st = [prev["bates"]] if "bates" in prev else generic_starts("bates", v)
    if np.isfinite(h).all():
        st.append(np.r_[h, 0.02, -0.05, 0.05])
    if np.isfinite(me).all():
        st.append(np.r_[me[0], 2.0, me[0], 0.05, -0.3, me[1:]])
    fits["bates"] = fit_one("bates", ch, st)
    if event_identified(ch):
        b = fits["bates"].x
        st = ([prev["bates_ev"]] if "bates_ev" in prev else [])
        if np.isfinite(b).all():
            st += [np.r_[b, 0.01], np.r_[b, 0.03]]
        fits["bates_ev"] = fit_one("bates_ev", ch, st)
    return fits


def refit_v0(model: str, x: np.ndarray, ch: Chain) -> Fit:
    """Next-day test: keep yesterday's structural parameters, re-estimate only v0."""
    lb, ub = bounds(model)
    t0 = time.time()

    def res(z):
        xx = x.copy()
        xx[0] = z[0]
        return residuals(model, xx, ch)

    r = least_squares(res, [np.clip(x[0], lb[0] * 1.01, ub[0] * 0.99)], bounds=([lb[0]], [ub[0]]), method="trf",
                      loss="soft_l1", f_scale=1.0, xtol=1e-10, ftol=1e-10, max_nfev=100)
    xx = x.copy()
    xx[0] = r.x[0]
    return Fit(model, xx, float(r.cost), float(np.sqrt(np.mean(r.fun ** 2))), r.nfev, int(r.status), time.time() - t0, 0)
