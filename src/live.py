"""Live NSE option-chain snapshots (notebook 02) as a contract panel, and a Market for the
Stage 5 simulator built from them.

The rules mirror the bhavcopy panel (src/contracts.py) so live and backtest numbers mean
the same thing:
  * Session = NSE trading day with chain files; files on non-trading days and files whose
    names are not `{YYYY-MM-DD}_{DD-Mon-YYYY}.csv` are ignored.
  * Price = mid of the snapshot's bid/ask (vol_model Layer 0: both sides quoted, spread <=
    50% of mid). "Traded" = such a quote and VOLUME >= MIN_TRADED contracts; only traded
    rows get a mark, as in the panel.
  * Forward: dte 0 -> spot; else the parity regression on near-ATM mids (vol_model Layer 1)
    when R^2 >= 0.99; else a local parity forward (median K + (C - P)/DF over the 6 traded
    pairs nearest spot, kept if they agree within 0.1%); else none (expiry unusable that day).
  * DF = exp(-r T), r = repo rate on the session (the parity slope is biased, see contracts).
    T = calendar days / 365; tdte = trading days d < s <= expiry.
  * Spot = the Yahoo close (identical to NSE's close, scripts/live_verify.py); the snapshot
    value only when Yahoo has no row yet. A session whose snapshot underlying differs from
    the close by more than SNAPSHOT_TOL_BP was scraped intraday: its prices are not
    end-of-day, so the session is dropped (flagged in `info`).
  * Expiry labels: if NSE moves a live expiry (holiday), the old label stops appearing and a
    new one within RELABEL_MAX_DAYS appears; the old label is mapped to the new one.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date

import numpy as np
import pandas as pd

from src import backtest as B
from src import contracts as ct
from src import vol_model as vm
from src.chain_parser import load_chain_csv
from src.trading_calendar import is_trading_day

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
REPO_CSV = os.path.join(PROJECT_ROOT, "config", "india_repo_rate.csv")
YAHOO = {"NIFTY": "nifty.csv", "BANKNIFTY": "banknifty.csv"}
CHAIN_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-[A-Za-z]{3}-\d{4})\.csv$")
MIN_TRADED = ct.MIN_TRADED
PARITY_MIN_R2 = 0.99
LOCAL_PAIRS, LOCAL_MIN_PAIRS, LOCAL_MAX_SPREAD = 6, 3, 0.001
RELABEL_MAX_DAYS = 10
SNAPSHOT_TOL_BP = 5.0


def chain_files(symbol: str) -> pd.DataFrame:
    d = os.path.join(RAW, symbol.lower(), "option_chain")
    rows = []
    for f in sorted(os.listdir(d)):
        m = CHAIN_RE.match(f)
        if not m:
            continue
        s = pd.Timestamp(m.group(1))
        if not is_trading_day(s.date()):
            continue
        rows.append(dict(date=s, expiry=pd.to_datetime(m.group(2), format="%d-%b-%Y"), path=os.path.join(d, f)))
    return pd.DataFrame(rows)


def yahoo_close(symbol: str) -> pd.Series:
    y = pd.read_csv(os.path.join(RAW, "underlying", YAHOO[symbol]), parse_dates=["date"])
    return y.set_index("date")["close"].sort_index()


def trading_days_between(d: pd.Timestamp, e: pd.Timestamp) -> int:
    return sum(1 for x in pd.date_range(d + pd.Timedelta(days=1), e) if is_trading_day(x.date()))


def _local_parity(per: pd.DataFrame, S: float, DF: float) -> float:
    t = per[per["ce_ok"] & per["pe_ok"] & (per["ce_vol"] >= MIN_TRADED) & (per["pe_vol"] >= MIN_TRADED)]
    if len(t) < LOCAL_MIN_PAIRS:
        return np.nan
    t = t.iloc[np.argsort(np.abs(t["strike"].to_numpy() - S))[:LOCAL_PAIRS]]
    f = t["strike"].to_numpy(float) + (t["ce_mid"].to_numpy(float) - t["pe_mid"].to_numpy(float)) / DF
    F = float(np.median(f))
    return F if np.median(np.abs(f - F)) / F <= LOCAL_MAX_SPREAD else np.nan


def load_file(path: str, symbol: str, d: pd.Timestamp, expiry: pd.Timestamp, rate: float) -> tuple[pd.DataFrame, dict]:
    meta_path = path[:-4] + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    S = float(meta.get("underlying_spot") or np.nan)
    per = vm.add_liquidity_flags(vm.build_per_strike(load_chain_csv(path)))
    dte = (expiry - d).days
    T = dte / 365.0
    DF = float(np.exp(-rate * T))
    info = dict(date=d, expiry=expiry, spot=S, dte=dte, scraped_at=meta.get("scraped_at_ist"))
    if dte == 0:
        F, src = S, "expiry_day"
    else:
        fw = vm.recover_forward_df(per, spot=S if np.isfinite(S) else None)
        if fw["ok"] and np.isfinite(fw["r2"]) and fw["r2"] >= PARITY_MIN_R2:
            F, src = float(fw["forward"]), "parity"
        else:
            F = _local_parity(per, S, DF) if np.isfinite(S) else np.nan
            src = "parity_local" if np.isfinite(F) else "none"
        info.update(parity_r2=fw["r2"], parity_pairs=fw["n_pairs"])
    info.update(forward=F, fwd_source=src)
    rows = []
    for side, tag in (("CE", "ce"), ("PE", "pe")):
        x = pd.DataFrame(dict(strike=per["strike"], side=side, bid=per[f"{tag}_bid"], ask=per[f"{tag}_ask"],
                              mid=per[f"{tag}_mid"], ltp=per[f"{tag}_ltp"], volume=per[f"{tag}_vol"],
                              oi=per[f"{tag}_oi"], quoted=per[f"{tag}_ok"].astype(bool)))
        rows.append(x)
    o = pd.concat(rows, ignore_index=True)
    o = o[o["strike"] > 0]
    o["traded"] = o["quoted"] & (o["volume"] >= MIN_TRADED)
    o["mark"] = np.where(o["traded"], o["mid"], np.nan)
    o["date"], o["expiry"], o["dte"], o["T"] = d, expiry, dte, T
    o["spot"], o["forward"], o["fwd_source"], o["rate"], o["df"] = S, F, src, rate, DF
    K, ic = o["strike"].to_numpy(float), (o["side"] == "CE").to_numpy()
    with np.errstate(all="ignore"):
        o["log_moneyness"] = np.log(K / F) if np.isfinite(F) else np.nan
        if dte > 0 and np.isfinite(F):
            iv = ct.b76_iv(o["mark"].to_numpy(float), np.full(len(o), F), K, np.full(len(o), T), np.full(len(o), DF), ic)
            dl, _, vg, _ = ct.b76_greeks(F, K, T, iv, DF, 0.0, ic)
        else:
            iv = dl = vg = np.full(len(o), np.nan)
    o["iv"], o["delta"], o["vega"] = iv, dl, vg
    return o, info


def expiry_relabels(files: pd.DataFrame) -> dict:
    """{old label: new label} for expiries NSE moved while listed."""
    seen = files.groupby("expiry")["date"].agg(["min", "max"])
    days = pd.DatetimeIndex(sorted(files["date"].unique()))
    if len(days) == 0:
        return {}
    out = {}
    for e, r in seen.iterrows():
        # a label that reached its own date, or whose date is not yet past, is not moved
        if r["max"] >= e or e > days[-1]:
            continue
        i = days.searchsorted(r["max"], side="right")
        if i >= len(days):
            continue
        nxt = days[i]
        later = seen[(seen["min"] == nxt) & (np.abs((seen.index - e).days) <= RELABEL_MAX_DAYS)]
        if len(later):
            out[e] = later.index[np.argmin(np.abs((later.index - e).days))]
    return out


def live_panel(symbol: str, since: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(contract rows, per-file info) for every live session of a symbol."""
    files = chain_files(symbol)
    if since is not None:
        files = files[files["date"] >= pd.Timestamp(since)]
    rates = ct.load_repo_rates(REPO_CSV)
    relabel = expiry_relabels(files)
    rows, infos = [], []
    for f in files.itertuples(index=False):
        o, info = load_file(f.path, symbol, f.date, f.expiry, ct.rate_on(rates, f.date.date()))
        fe = relabel.get(f.expiry, f.expiry)
        o["final_expiry"], info["final_expiry"] = fe, fe
        rows.append(o)
        infos.append(info)
    p = pd.concat(rows, ignore_index=True)
    info = pd.DataFrame(infos)
    y = yahoo_close(symbol)
    info["close"] = info["date"].map(y)
    info["snap_bp"] = (info["spot"] / info["close"] - 1) * 1e4
    info["session_ok"] = ~(info["snap_bp"].abs() > SNAPSHOT_TOL_BP)
    info["close_checked"] = info["close"].notna()
    td = {(d, e): trading_days_between(d, e) for d, e in info[["date", "expiry"]].drop_duplicates().itertuples(index=False)}
    p["tdte"] = [td[k] for k in zip(p["date"], p["expiry"])]
    info["tdte"] = [td[k] for k in zip(info["date"], info["expiry"])]
    p["symbol"] = symbol
    bad = set(info.loc[~info["session_ok"], "date"])
    p = p[~p["date"].isin(bad)].reset_index(drop=True)
    return p, info


class LiveMarket(B.Market):
    """The Stage 5 Market interface on live snapshots (mid quotes instead of closes)."""

    def __init__(self, symbol: str, panel: pd.DataFrame, info: pd.DataFrame, lot: float):
        self.symbol = symbol
        days = sorted(panel["date"].unique())
        all_days = [d for d in pd.date_range(days[0], days[-1]) if is_trading_day(d.date())]
        self.sessions = pd.DatetimeIndex(all_days)
        self.final_of = {e: fe for e, fe in info[["expiry", "final_expiry"]].drop_duplicates().itertuples(index=False)}
        self.settled = {fe: fe for fe in info["final_expiry"].unique()}
        snap = info.groupby("date")["spot"].median().reindex(self.sessions)
        y = yahoo_close(symbol).reindex(self.sessions)
        self.spot = y.fillna(snap)
        self.spot_source = pd.Series(np.where(y.notna(), "close", np.where(snap.notna(), "snapshot", "none")),
                                     index=self.sessions)
        self.dropped = sorted(info.loc[~info["session_ok"], "date"].unique())
        rates = ct.load_repo_rates(REPO_CSV)
        self.rate = pd.Series([ct.rate_on(rates, d.date()) for d in self.sessions], index=self.sessions)
        self.lot = pd.Series(float(lot), index=self.sessions)
        ok = info[(info["fwd_source"] != "none") & info["session_ok"]]
        self.fwd = {(d, fe): float(F) for d, fe, F in ok[["date", "final_expiry", "forward"]].itertuples(index=False)}
        tr = panel[panel["traded"] & panel["mark"].gt(0)]
        self.traded = tr
        self._build_slices(tr)

    def _build_slices(self, tr: pd.DataFrame):
        otm = np.where(tr["side"] == "CE", tr["log_moneyness"] >= 0, tr["log_moneyness"] <= 0)
        sm = tr[otm & tr["iv"].notna() & tr["fwd_source"].isin(["parity", "parity_local"])
                & (tr["iv"] > 0.01) & (tr["iv"] < 3)]
        self.smiles = {key: (g["log_moneyness"].to_numpy(float), g["iv"].to_numpy(float))
                       for key, g in sm.sort_values("log_moneyness").groupby(["date", "final_expiry"])}
        self.marks = {key: dict(zip(zip(g["strike"].to_numpy(float), g["side"].to_numpy()), g["mark"].to_numpy(float)))
                      for key, g in tr.groupby(["date", "final_expiry"])}
