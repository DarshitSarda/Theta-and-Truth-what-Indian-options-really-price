"""Stage 0: per-contract daily history of NIFTY/BANKNIFTY options from the bhavcopy.

One row per (date, symbol, expiry, strike, side): NSE's end-of-day record plus
what is needed to study a single contract through its life - spot, forward,
discount factor, days to expiry in calendar and trading days, a usable mark
price, Black-76 implied vol and greeks.

Pricing inputs and why:
  * Spot. UDiFF files carry NSE's index close; legacy files do not, so the Yahoo
    close is used (identical to NSE's figure wherever both exist). Yahoo lacks some
    NSE sessions (1 Jan, Muhurat and Saturday/Budget sessions); for those, NSE's
    ind_close_all file (data/raw/underlying_supplement/nse_index_close.csv), then on
    an expiry day the expiring future's final settle.
  * Forward. Put-call-parity forward from the bhavcopy vol layer when its fit is
    good and agrees with the futures close within 0.5%; else the futures close for
    that expiry (if futures traded); else a local parity forward from the traded
    pairs nearest spot (the vol layer skips dte = 1, and weeklies have no futures);
    else spot carried at the day's index carry.
  * Discount factor. exp(-r*T) with r from config/india_repo_rate.csv. The parity
    slope is NOT used: in-the-money options trade below fair value, which flattens
    the slope and implies 10-30% rates. The forward (intercept) is unaffected.
  * Time. T = calendar days / 365, matching the rest of the project. Trading days
    to expiry are stored alongside for later trading-time work.
  * Mark. The close only when the contract traded that day. NSE carries an
    untraded contract's last close forward unchanged, and its settle price on such
    days is NSE's own theoretical value, so neither is a market price.
  * Expiry. Panel rows carry the expiry NSE published that day (what the market
    priced, so no lookahead). NSE moves expiries for holidays: it relabels live
    contracts (e.g. 27-Feb-2014 -> 26-Feb-2014 on 30-Dec-2013), settles a day
    early under the old label (24-Dec-2008 for 25-Dec), or a day late (28-Nov-2008
    after the 26/11 closure). `final_expiries` maps each label to the session the
    series actually settled; the contract summary is keyed on that.
"""
from __future__ import annotations

import glob
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .bhavcopy import expiring_future_settle, fmt_expiry
from .trading_calendar import is_trading_day

SYMBOLS = ("NIFTY", "BANKNIFTY")
MIN_TRADED = 10
PARITY_MIN_R2 = 0.99
PARITY_FUT_TOL = 0.005
_SQRT_2PI = np.sqrt(2.0 * np.pi)

PANEL_COLS = [
    "date", "symbol", "expiry", "strike", "side", "dte", "tdte", "tdte_projected",
    "open", "high", "low", "close", "settle", "last", "contracts", "trades",
    "oi", "chg_oi", "oi_shares", "lot_size", "quality", "mark",
    "spot", "spot_source", "forward", "fwd_source", "rate", "df", "T",
    "log_moneyness", "intrinsic", "iv", "iv_status", "delta", "gamma", "vega", "theta",
    "source_format",
]


# ---------------------------------------------------------------- Black-76 (vectorised)


def _pdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def b76_price(F, K, T, sig, DF, is_call):
    v = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    call = DF * (F * ndtr(d1) - K * ndtr(d2))
    put = DF * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def b76_iv(price, F, K, T, DF, is_call, lo=1e-4, hi=5.0, tol=1e-10, max_iter=100):
    """Implied vol by safeguarded Newton-bisection on arrays.

    NaN where the price breaks the no-arbitrage bounds (at/below discounted
    intrinsic, at/above the forward or strike cap) or cannot be bracketed.
    """
    price, F, K, T, DF = (np.asarray(a, dtype=float) for a in (price, F, K, T, DF))
    is_call = np.asarray(is_call, dtype=bool)
    out = np.full(price.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        intrinsic = DF * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
        upper = DF * np.where(is_call, F, K)
        ok = (np.isfinite(price) & np.isfinite(F) & np.isfinite(DF) & (price > intrinsic + 1e-10)
              & (price < upper - 1e-10) & (T > 0) & (F > 0) & (K > 0))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    p, f_, k, t, df, c = price[idx], F[idx], K[idx], T[idx], DF[idx], is_call[idx]
    a = np.full(idx.size, lo)
    b = np.full(idx.size, hi)
    bracketed = (b76_price(f_, k, t, a, df, c) <= p) & (b76_price(f_, k, t, b, df, c) >= p)
    x = np.clip(np.sqrt(2.0 * np.pi / t) * p / (df * f_), lo, hi)
    sqrt_t = np.sqrt(t)
    for _ in range(max_iter):
        fx = b76_price(f_, k, t, x, df, c) - p
        a = np.where(fx < 0, x, a)
        b = np.where(fx >= 0, x, b)
        v = x * sqrt_t
        d1 = (np.log(f_ / k) + 0.5 * v * v) / v
        vega = df * f_ * _pdf(d1) * sqrt_t
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            xn = x - fx / vega
        bad = ~np.isfinite(xn) | (xn <= a) | (xn >= b)
        xn = np.where(bad, 0.5 * (a + b), xn)
        step = np.abs(xn - x)
        x = xn
        if np.all(step < tol):
            break
    x = np.where(bracketed, x, np.nan)
    out[idx] = x
    return out


def b76_greeks(F, K, T, sig, DF, r, is_call):
    """Forward delta, gamma (per point of forward), vega (per vol point),
    theta (per calendar day, forward held fixed)."""
    sqrt_t = np.sqrt(T)
    v = sig * sqrt_t
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    nd1 = _pdf(d1)
    price = b76_price(F, K, T, sig, DF, is_call)
    delta = np.where(is_call, DF * ndtr(d1), -DF * ndtr(-d1))
    gamma = DF * nd1 / (F * v)
    vega = DF * F * nd1 * sqrt_t / 100.0
    theta = (r * price - DF * F * nd1 * sig / (2.0 * sqrt_t)) / 365.0
    return delta, gamma, vega, theta


# ---------------------------------------------------------------- inputs


def load_repo_rates(path: str) -> pd.Series:
    df = pd.read_csv(path, usecols=["effective_date", "repo_rate_pct"])
    s = pd.Series(df["repo_rate_pct"].to_numpy(float) / 100.0,
                  index=pd.to_datetime(df["effective_date"])).sort_index()
    return s


def rate_on(rates: pd.Series, d: date) -> float:
    return float(rates.loc[:pd.Timestamp(d)].iloc[-1])


def cached_sessions(cache_dir: str) -> list[date]:
    """Trading sessions = days NSE published a bhavcopy (files present in the cache)."""
    out = []
    for p in glob.glob(os.path.join(cache_dir, "*", "fo_*.parquet")):
        s = os.path.basename(p)[3:11]
        out.append(date(int(s[:4]), int(s[4:6]), int(s[6:8])))
    return sorted(out)


class SessionClock:
    """Trading days between dates: known sessions, then projected NSE trading days."""

    def __init__(self, sessions: list[date], horizon_years: int = 6):
        self.last_known = sessions[-1]
        proj, d = [], self.last_known + timedelta(days=1)
        end = self.last_known + timedelta(days=365 * horizon_years)
        while d <= end:
            if is_trading_day(d):
                proj.append(d)
            d += timedelta(days=1)
        self.days = np.array(sessions + proj, dtype="datetime64[D]")

    def trading_days(self, d: date, expiries: np.ndarray) -> np.ndarray:
        """Sessions s with d < s <= expiry (expiry day counts, today does not)."""
        e = np.asarray(expiries, dtype="datetime64[D]")
        return (np.searchsorted(self.days, e, side="right")
                - np.searchsorted(self.days, np.datetime64(d), side="right"))


def health_index(health_all: pd.DataFrame) -> dict:
    """{date: {(symbol, 'DD-Mon-YYYY'): row}} from the bhavcopy vol layer's health master."""
    out: dict = {}
    for row in health_all.itertuples(index=False):
        out.setdefault(row.date, {})[(row.symbol, row.expiry)] = row
    return out


# ---------------------------------------------------------------- one session


LOCAL_PARITY_PAIRS = 6
LOCAL_PARITY_MIN_PAIRS = 3
LOCAL_PARITY_MAX_SPREAD = 0.001


def _local_parity(opts: pd.DataFrame, S: float, DF: float) -> float:
    """Parity forward K + (C - P)/DF, median over the traded pairs nearest spot.

    For expiries the bhavcopy vol layer skips (it excludes dte = 1) and that have
    no futures - i.e. the day-before-expiry weekly. NaN unless at least 3 pairs
    agree within 0.1% (median absolute deviation / forward).
    """
    t = opts[(opts["contracts"] >= MIN_TRADED) & (opts["close"] > 0)]
    c = t[t["side"] == "CE"].drop_duplicates("strike").set_index("strike")["close"]
    p = t[t["side"] == "PE"].drop_duplicates("strike").set_index("strike")["close"]
    k = c.index.intersection(p.index)
    if len(k) < LOCAL_PARITY_MIN_PAIRS:
        return np.nan
    k = np.asarray(sorted(k, key=lambda x: abs(x - S))[:LOCAL_PARITY_PAIRS], dtype=float)
    f = k + (c.loc[k].to_numpy(float) - p.loc[k].to_numpy(float)) / DF
    F = float(np.median(f))
    return F if np.median(np.abs(f - F)) / F <= LOCAL_PARITY_MAX_SPREAD else np.nan


def _forwards(opts: pd.DataFrame, day: pd.DataFrame, session: str, spots: dict,
              health_day: dict, carry: dict, rate: float) -> pd.DataFrame:
    """Forward per (symbol, expiry) and the source used. Updates `carry` in place."""
    fut = day[(day["instrument"] == "FUT") & (day["contracts"] > 0)]
    fut_close = {(r.symbol, r.expiry): float(r.close) for r in fut.itertuples(index=False)}
    keys = opts[["symbol", "expiry", "dte"]].drop_duplicates()
    rows = []
    for k in keys.itertuples(index=False):
        S = spots.get(k.symbol)
        T = k.dte / 365.0
        F, src = np.nan, "none"
        if S is None or not np.isfinite(S):
            pass
        elif k.dte == 0:
            F, src = S, "expiry_day"
        else:
            h = health_day.get((k.symbol, fmt_expiry(k.expiry)))
            fc = fut_close.get((k.symbol, k.expiry))
            parity_ok = (h is not None and pd.isna(h.skipped) and h.forward > 0
                         and h.parity_r2 >= PARITY_MIN_R2
                         and (fc is None or abs(h.forward / fc - 1.0) <= PARITY_FUT_TOL))
            if parity_ok:
                F, src = float(h.forward), "parity"
            elif fc is not None and fc > 0:
                F, src = fc, "futures"
            else:
                sub = opts[(opts["symbol"] == k.symbol) & (opts["expiry"] == k.expiry)]
                F = _local_parity(sub, S, np.exp(-rate * T))
                src = "parity_local" if np.isfinite(F) else "none"
        rows.append({"symbol": k.symbol, "expiry": k.expiry, "dte": k.dte, "S": S,
                     "T": T, "forward": F, "fwd_source": src})
    fw = pd.DataFrame(rows)
    for sym in fw["symbol"].unique():
        g = fw[(fw["symbol"] == sym) & fw["fwd_source"].isin(["parity", "futures"]) & (fw["dte"] >= 7)]
        if len(g):
            c = np.log(g["forward"] / g["S"]) / g["T"]
            carry[sym] = float(np.clip(c.median(), -0.05, 0.20))
    need = (fw["fwd_source"] == "none") & fw["S"].notna() & (fw["dte"] > 0)
    for i in fw.index[need]:
        c = carry.get(fw.at[i, "symbol"])
        if c is not None:
            fw.at[i, "forward"] = fw.at[i, "S"] * np.exp(c * fw.at[i, "T"])
            fw.at[i, "fwd_source"] = "carry"
    return fw


def session_panel(day: pd.DataFrame, session: str, yahoo_close: dict, health_day: dict,
                  clock: SessionClock, rate: float, carry: dict,
                  nse_close: dict | None = None) -> pd.DataFrame:
    """All NIFTY/BANKNIFTY option contracts listed on one session."""
    d = date.fromisoformat(session)
    o = day[(day["instrument"] == "OPT") & day["symbol"].isin(SYMBOLS)
            & day["side"].isin(["CE", "PE"])].copy()
    if o.empty:
        return pd.DataFrame(columns=PANEL_COLS)
    # a label already past but still listed = expiry postponed to this session
    o["dte"] = [max((e - d).days, 0) for e in o["expiry"]]

    spots, spot_src = {}, {}
    for sym in SYMBOLS:
        u = day.loc[(day["symbol"] == sym) & day["underlying"].notna(), "underlying"]
        if len(u):
            spots[sym], spot_src[sym] = float(u.iloc[0]), "nse"
        elif session in yahoo_close.get(sym, {}):
            spots[sym], spot_src[sym] = float(yahoo_close[sym][session]), "yahoo"
        elif session in (nse_close or {}).get(sym, {}):
            spots[sym], spot_src[sym] = float(nse_close[sym][session]), "nse_index_file"
        elif (fs := expiring_future_settle(day, sym)) is not None:
            spots[sym], spot_src[sym] = fs, "fut_settle"

    fw = _forwards(o, day, session, spots, health_day, carry, rate)
    o = o.merge(fw[["symbol", "expiry", "forward", "fwd_source"]], on=["symbol", "expiry"], how="left")
    o["spot"] = o["symbol"].map(spots)
    o["spot_source"] = o["symbol"].map(spot_src).fillna("none")
    o["date"] = pd.Timestamp(d)
    o["expiry"] = pd.to_datetime(o["expiry"])
    o["T"] = o["dte"] / 365.0
    o["rate"] = rate
    o["df"] = np.exp(-rate * o["T"])
    o["tdte"] = np.maximum(clock.trading_days(d, o["expiry"].to_numpy(dtype="datetime64[D]")), 0)
    o["tdte_projected"] = o["expiry"].dt.date > clock.last_known

    traded = o["contracts"].fillna(0)
    o["quality"] = np.select([traded >= MIN_TRADED, traded > 0], ["traded", "thin"], "untraded")
    o["mark"] = o["close"].where((traded > 0) & (o["close"] > 0))

    is_call = (o["side"] == "CE").to_numpy()
    K = o["strike"].to_numpy(float)
    S = o["spot"].to_numpy(float)
    F = o["forward"].to_numpy(float)
    o["intrinsic"] = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        o["log_moneyness"] = np.log(K / F)

    T = o["T"].to_numpy(float)
    DF = o["df"].to_numpy(float)
    mark = o["mark"].to_numpy(float)
    iv = b76_iv(mark, F, K, T, DF, is_call)
    o["iv"] = iv
    fwd_intr = DF * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    o["iv_status"] = np.select(
        [o["dte"].to_numpy() == 0, ~np.isfinite(mark), ~np.isfinite(F), np.isfinite(iv),
         mark <= fwd_intr],
        ["expiry_day", "no_price", "no_forward", "ok", "below_intrinsic"], "outside_bounds")

    has = np.isfinite(iv)
    for col in ("delta", "gamma", "vega", "theta"):
        o[col] = np.nan
    if has.any():
        dl, gm, vg, th = b76_greeks(F[has], K[has], T[has], iv[has], DF[has], rate, is_call[has])
        o.loc[has, "delta"], o.loc[has, "gamma"], o.loc[has, "vega"], o.loc[has, "theta"] = dl, gm, vg, th

    for col in PANEL_COLS:
        if col not in o.columns:
            o[col] = np.nan
    return o[PANEL_COLS].reset_index(drop=True)


# ---------------------------------------------------------------- expiry labels


RELABEL_MAX_DAYS = 10
RELABEL_MIN_MATCH = 0.9


def _relabel_match(p: pd.DataFrame, old, new, d_old, d_new) -> tuple[float, int]:
    """Share of the old label's open positions on its last day that continue, with
    the same strike and side, under the new label on the next session.

    NSE sometimes reports change-in-OI against the old label and sometimes treats
    the relabelled contract as new (change = full OI), so a position continues if
    either OI - change == old OI, or OI moved by no more than that day's volume.
    With no open positions, the share of the old strikes listed under the new label.
    """
    k = ["strike", "side"]
    old_rows = p[(p["expiry"] == old) & (p["date"] == d_old)].set_index(k)
    b = p[(p["expiry"] == new) & (p["date"] == d_new)].set_index(k)
    a = old_rows.loc[old_rows["oi"] > 0, "oi"]
    if a.empty:
        return (float(old_rows.index.isin(b.index).mean()) if len(old_rows) else np.nan), 0
    j = b.reindex(a.index)
    ok = ((j["oi"] - j["chg_oi"] - a).abs() < 0.5) | ((j["oi"] - a).abs() <= j["contracts"].fillna(0))
    return float(ok.mean()), len(a)


def final_expiries(p: pd.DataFrame, sessions: list[date]) -> pd.DataFrame:
    """Map every published expiry label of one symbol to the session it settled.

    `p` needs date, expiry, strike, side, oi, chg_oi. Events:
      normal     last listed on the label date
      postponed  still listed after the label date; settled on the last listed day
      preponed   delisted before the label with no successor; settled on the last
                 listed day (the label date was a holiday)
      relabelled positions moved to a new label on the next session (OI continuity
                 checked); final expiry is the successor's
      live       still listed on the last cached session
    """
    sess = np.array(sorted(sessions), dtype="datetime64[D]")
    last_known = pd.Timestamp(sess[-1])
    span = p.groupby("expiry")["date"].agg(first="min", last="max").sort_index()
    rows = {}
    for E, (first, last) in span.iterrows():
        i = np.searchsorted(sess, np.datetime64(last.date()), side="right")
        nxt = pd.Timestamp(sess[i]) if i < len(sess) else None
        row = {"first_seen": first, "last_seen": last, "successor": pd.NaT,
               "match": np.nan, "n_open": 0}
        if last >= E:
            row.update(event="normal" if last == E else "postponed", settled=last)
        elif last == last_known:
            row.update(event="live", settled=pd.NaT)
        elif nxt > E:
            row.update(event="preponed", settled=last)
        else:
            cands = span[(span["first"] == nxt)
                         & (abs(span.index - E) <= pd.Timedelta(days=RELABEL_MAX_DAYS))].index
            scored = [(_relabel_match(p, E, c, last, nxt), c) for c in cands if c != E]
            scored = [(m, n, c) for (m, n), c in scored if np.isfinite(m)]
            best = max(scored, key=lambda t: t[0]) if scored else None
            if best and best[0] >= RELABEL_MIN_MATCH:
                row.update(event="relabelled", successor=best[2], match=best[0], n_open=best[1])
            else:
                row.update(event="unexplained", settled=pd.NaT,
                           match=best[0] if best else np.nan, n_open=best[1] if best else 0)
        rows[E] = row
    for E, row in rows.items():
        e, seen = E, set()
        while rows[e]["event"] == "relabelled" and e not in seen:
            seen.add(e)
            e = rows[e]["successor"]
        row["final_expiry"] = e if rows[e]["event"] != "relabelled" else pd.NaT
        if row["event"] == "relabelled":
            row["settled"] = rows[e].get("settled", pd.NaT)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "expiry"
    return out.reset_index()[["expiry", "event", "first_seen", "last_seen", "successor", "match",
                              "n_open", "final_expiry", "settled"]]


# ---------------------------------------------------------------- contract summary


def contract_summary(panel: pd.DataFrame, spot_at: dict, expiry_map: pd.DataFrame) -> pd.DataFrame:
    """One row per contract: listing, liquidity, first trade, and outcome at expiry.

    A contract is (symbol, final expiry, strike, side), so a series NSE relabelled
    is one contract. `expiry` is the final label, `settled` the session it settled,
    `labels` how many expiry labels it traded under. `spot_at` maps (symbol,
    'YYYY-MM-DD') -> index close; the final settlement value of an NSE index option
    is the index close on the settlement session.
    """
    m = expiry_map[["symbol", "expiry", "final_expiry", "settled", "event"]].rename(
        columns={"expiry": "label"})
    panel = panel.rename(columns={"expiry": "label"}).merge(m, on=["symbol", "label"], how="left")
    panel["expiry"] = panel["final_expiry"].fillna(panel["label"])
    key = ["symbol", "expiry", "strike", "side"]
    p = panel.sort_values(key + ["date"]).assign(
        _t=lambda x: x["contracts"] > 0, _t10=lambda x: x["contracts"] >= MIN_TRADED)
    g = p.groupby(key, sort=False)
    out = g.agg(first_date=("date", "min"), last_date=("date", "max"), n_days=("date", "size"),
                n_traded=("_t", "sum"), n_traded10=("_t10", "sum"),
                total_contracts=("contracts", "sum"), max_oi=("oi", "max"),
                labels=("label", "nunique"), settled=("settled", "max"),
                expiry_event=("event", "last")).reset_index()
    tr = p[p["mark"].notna()]
    first = tr.groupby(key, sort=False).first()[["date", "mark", "dte", "spot", "iv"]]
    first.columns = ["first_trade_date", "first_trade_price", "first_trade_dte",
                     "first_trade_spot", "first_trade_iv"]
    last = tr.groupby(key, sort=False).last()[["date", "mark", "dte"]]
    last.columns = ["last_trade_date", "last_trade_price", "last_trade_dte"]
    out = out.merge(first.reset_index(), on=key, how="left").merge(last.reset_index(), on=key, how="left")
    out["spot_at_expiry"] = [spot_at.get((s, e.strftime("%Y-%m-%d")), np.nan) if pd.notna(e) else np.nan
                             for s, e in zip(out["symbol"], out["settled"])]
    out["expired"] = out["spot_at_expiry"].notna()
    K, ST = out["strike"].to_numpy(float), out["spot_at_expiry"].to_numpy(float)
    out["payoff"] = np.where(out["side"] == "CE", np.maximum(ST - K, 0.0), np.maximum(K - ST, 0.0))
    out.loc[~out["expired"], "payoff"] = np.nan
    return out
