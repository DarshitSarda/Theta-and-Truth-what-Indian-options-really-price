"""NSE F&O bhavcopy: download, normalise, and adapt to the live-chain pipeline.

The bhavcopy is NSE's official end-of-day file for the derivatives segment. It is
the only source of option-chain history before our live collection started, so it
is what the historical backfill runs on.

Two layouts exist:
  * legacy  (to 2024-07-05): fo{DD}{MON}{YYYY}bhav.csv.zip
        INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN, HIGH, LOW,
        CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH, OPEN_INT, CHG_IN_OI, TIMESTAMP
  * UDiFF   (from 2024-07-08): BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
        adds LastPric, UndrlygPric, NewBrdLotQty and the actual expiry date.

What the live chain has that the bhavcopy does not, and how each gap is handled:
  * No bid/ask. Options are priced at CLOSE, which for NSE derivatives is a
    volume-weighted price over the last 30 minutes, so calls and puts share a
    common time window and parity holds on it. CLOSE is the only price present
    across the full history, so it is the backfill price by construction.
  * Stale prints. An untraded contract carries the previous close forward, so a
    strike is only priced if it traded that day (`min_contracts`).
  * No spread. The SVI fit weights points by 1/(spread+0.02). We substitute a
    tick-quantisation proxy: a close print is off the true mid by about one tick,
    so its relative error is ~tick/price. Cheap wing options get down-weighted
    for the same reason wide-spread ones do in the live path.
  * Units. OI is in shares; the live chain reports contracts. We divide by the lot
    size -- read directly in UDiFF, inferred from futures turnover in legacy files
    (turnover / (contracts * price) = shares per contract).
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.chain_parser import compute_daily_metrics, max_pain
from src.expiry_utils import compute_dte, enrich_chain_metadata
from src.vol_model import reconcile_nse_iv, recover_forward_df

UDIFF_START = date(2024, 7, 8)
ARCHIVE = "https://nsearchives.nseindia.com/content"
OPTION_TICK = 0.05

LEGACY_INDEX_INSTRUMENTS = ("OPTIDX", "FUTIDX")
UDIFF_INDEX_INSTRUMENTS = ("IDO", "IDF")

NORMALISED_COLS = [
    "date", "instrument", "symbol", "expiry", "strike", "side",
    "open", "high", "low", "close", "settle", "last", "underlying",
    "contracts", "trades", "oi", "chg_oi", "lot_size",
]


# ---------------------------------------------------------------- download


def udiff_url(d: date) -> str:
    return f"{ARCHIVE}/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"


def legacy_url(d: date) -> str:
    mon = d.strftime("%b").upper()
    return f"{ARCHIVE}/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"


def cache_path(cache_dir: str, d: date) -> str:
    return os.path.join(cache_dir, f"{d:%Y}", f"fo_{d:%Y%m%d}.parquet")


def _missing_marker(cache_dir: str, d: date) -> str:
    return os.path.join(cache_dir, f"{d:%Y}", f"fo_{d:%Y%m%d}.missing")


def _fetch_zip_csv(session, url: str, timeout=(10, 90), retries: int = 3) -> Optional[pd.DataFrame]:
    """GET a zipped CSV. None on 404 (no session that day / not published)."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None
            if resp.status_code != 200 or resp.content[:2] != b"PK":
                raise IOError(f"status {resp.status_code} for {url}")
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            with z.open(z.namelist()[0]) as fh:
                return pd.read_csv(fh, low_memory=False)
        except Exception as exc:  # network hiccup, NSE throttling, truncated zip
            last_exc = exc
            time.sleep(min(2 ** attempt, 20))
    raise IOError(f"failed after {retries} attempts: {url} ({last_exc})")


def download_day(session, d: date, cache_dir: str, force: bool = False) -> Optional[str]:
    """Fetch one session's bhavcopy, keep only index derivatives, cache as parquet.

    Returns the cache path, or None when NSE has no file for that date (holiday).
    Stock derivatives are dropped at download time: they are ~85% of each file and
    irrelevant to the index backfill, and dropping them keeps 18 years of history
    in tens of MB instead of several GB.
    """
    out = cache_path(cache_dir, d)
    miss = _missing_marker(cache_dir, d)
    if not force and os.path.exists(out):
        return out
    if not force and os.path.exists(miss):
        return None
    os.makedirs(os.path.dirname(out), exist_ok=True)

    order = (udiff_url, legacy_url) if d >= UDIFF_START else (legacy_url, udiff_url)
    raw, fmt = None, None
    for make_url in order:
        raw = _fetch_zip_csv(session, make_url(d))
        if raw is not None:
            fmt = "udiff" if make_url is udiff_url else "legacy"
            break
    if raw is None:
        open(miss, "w").close()
        return None

    raw.columns = [str(c).strip() for c in raw.columns]
    if fmt == "udiff":
        raw = raw[raw["FinInstrmTp"].isin(UDIFF_INDEX_INSTRUMENTS)]
    else:
        raw = raw[raw["INSTRUMENT"].str.strip().isin(LEGACY_INDEX_INSTRUMENTS)]
        raw = raw.loc[:, [c for c in raw.columns if not c.startswith("Unnamed")]]
    raw = raw.copy()
    raw["_format"] = fmt
    raw["_session"] = d.isoformat()
    # mixed-type object columns (e.g. empty ISIN) break parquet; store as text
    for c in raw.columns:
        if raw[c].dtype == object:
            raw[c] = raw[c].astype(str)
    raw.to_parquet(out, index=False)
    return out


# ---------------------------------------------------------------- normalise


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalise_udiff(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "date": raw["_session"],
        "instrument": np.where(raw["FinInstrmTp"] == "IDO", "OPT", "FUT"),
        "symbol": raw["TckrSymb"].str.strip(),
        "expiry": pd.to_datetime(raw["FininstrmActlXpryDt"]).dt.date,
        "strike": _num(raw["StrkPric"]),
        "side": raw["OptnTp"].where(raw["OptnTp"].isin(["CE", "PE"]), ""),
        "open": _num(raw["OpnPric"]), "high": _num(raw["HghPric"]),
        "low": _num(raw["LwPric"]), "close": _num(raw["ClsPric"]),
        "settle": _num(raw["SttlmPric"]), "last": _num(raw["LastPric"]),
        "underlying": _num(raw["UndrlygPric"]),
        "contracts": _num(raw["TtlTradgVol"]),
        "trades": _num(raw["TtlNbOfTxsExctd"]),
        "oi": _num(raw["OpnIntrst"]), "chg_oi": _num(raw["ChngInOpnIntrst"]),
        "lot_size": _num(raw["NewBrdLotQty"]),
    })
    return out


def _infer_legacy_lot(raw: pd.DataFrame) -> dict:
    """Shares per contract, per symbol, from index-futures turnover.

    VAL_INLAKH is traded value in lakh rupees, so shares traded is
    VAL_INLAKH*1e5 / price and shares per contract follows. Using CLOSE as the
    average price is off by at most the intraday range (~1-2%), and lot sizes are
    multiples of 5, so rounding to the nearest 5 recovers them exactly.
    """
    fut = raw[raw["INSTRUMENT"].str.strip() == "FUTIDX"].copy()
    fut["CONTRACTS"] = _num(fut["CONTRACTS"])
    fut["VAL_INLAKH"] = _num(fut["VAL_INLAKH"])
    fut["CLOSE"] = _num(fut["CLOSE"])
    fut = fut[(fut["CONTRACTS"] > 0) & (fut["CLOSE"] > 0)]
    lots = {}
    for sym, g in fut.groupby(fut["SYMBOL"].str.strip()):
        est = (g["VAL_INLAKH"] * 1e5 / (g["CONTRACTS"] * g["CLOSE"])).median()
        if np.isfinite(est) and est > 0:
            lots[sym] = float(max(5.0, round(est / 5.0) * 5.0))
    return lots


LEGACY_ALIASES = {"OPTIONTYPE": "OPTION_TYP", "STRIKE_PRICE": "STRIKE_PR",
                  "EXPIRY": "EXPIRY_DT", "OPENINT": "OPEN_INT", "CHGINOI": "CHG_IN_OI"}


def _legacy_date(s: pd.Series) -> pd.Series:
    # Most files write "31-May-2012"; some 2012 files write "31-May-12".
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    return out.fillna(pd.to_datetime(s, format="%d-%b-%y", errors="coerce")).dt.date


def _normalise_legacy(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.rename(columns={k: v for k, v in LEGACY_ALIASES.items() if k in raw.columns})
    lots = _infer_legacy_lot(raw)
    inst = raw["INSTRUMENT"].str.strip()
    side = raw["OPTION_TYP"].str.strip()
    sym = raw["SYMBOL"].str.strip()
    out = pd.DataFrame({
        "date": raw["_session"],
        "instrument": np.where(inst == "OPTIDX", "OPT", "FUT"),
        "symbol": sym,
        "expiry": _legacy_date(raw["EXPIRY_DT"]),
        "strike": _num(raw["STRIKE_PR"]),
        "side": side.where(side.isin(["CE", "PE"]), ""),
        "open": _num(raw["OPEN"]), "high": _num(raw["HIGH"]),
        "low": _num(raw["LOW"]), "close": _num(raw["CLOSE"]),
        "settle": _num(raw["SETTLE_PR"]), "last": np.nan, "underlying": np.nan,
        "contracts": _num(raw["CONTRACTS"]), "trades": np.nan,
        "oi": _num(raw["OPEN_INT"]), "chg_oi": _num(raw["CHG_IN_OI"]),
        "lot_size": sym.map(lots).astype(float),
    })
    return out


def load_day(path: str) -> pd.DataFrame:
    """Cached parquet -> normalised frame; OI and chg_oi converted to contracts."""
    raw = pd.read_parquet(path)
    fmt = raw["_format"].iloc[0] if len(raw) else "udiff"
    df = _normalise_udiff(raw) if fmt == "udiff" else _normalise_legacy(raw)
    df["oi_shares"] = df["oi"]
    lot = df["lot_size"].where(df["lot_size"] > 0)
    df["oi"] = df["oi"] / lot
    df["chg_oi"] = df["chg_oi"] / lot
    df["source_format"] = fmt
    return df


# ---------------------------------------------------------------- adapters


def fmt_expiry(e: date) -> str:
    return e.strftime("%d-%b-%Y")


def option_expiries(day: pd.DataFrame, symbol: str, session: str,
                    max_dte: int = 75, max_n: int = 8) -> list:
    """Listed option expiries with 0 <= DTE <= max_dte, nearest first."""
    opts = day[(day["symbol"] == symbol) & (day["instrument"] == "OPT")]
    out = []
    for e in sorted(opts["expiry"].dropna().unique()):
        dte = compute_dte(session, fmt_expiry(e))
        if 0 <= dte <= max_dte:
            out.append(e)
    return out[:max_n]


def chain_tidy(day: pd.DataFrame, symbol: str, expiry: date) -> pd.DataFrame:
    """Bhavcopy rows -> the tidy long format `chain_parser.load_chain_csv` returns,
    so compute_daily_metrics and max_pain run unchanged."""
    sub = day[(day["symbol"] == symbol) & (day["instrument"] == "OPT")
              & (day["expiry"] == expiry) & (day["side"].isin(["CE", "PE"]))]
    return pd.DataFrame({
        "STRIKE": sub["strike"].to_numpy(float),
        "side": sub["side"].to_numpy(),
        "OI": sub["oi"].fillna(0).to_numpy(float),
        "CHNG IN OI": sub["chg_oi"].fillna(0).to_numpy(float),
        "VOLUME": sub["contracts"].fillna(0).to_numpy(float),
        "LTP": sub["close"].fillna(0).to_numpy(float),
    })


def price_error_proxy(price: pd.Series) -> pd.Series:
    """Stand-in for spread_pct: relative error of a close print ~ one tick."""
    return (2.0 * OPTION_TICK / price).clip(lower=0.01, upper=1.0)


def build_per_strike(day: pd.DataFrame, symbol: str, expiry: date,
                     price_field: str = "close", min_contracts: int = 10,
                     min_price: float = 0.5, max_error: float = 0.35) -> pd.DataFrame:
    """Per-strike frame with the same columns vol_model's Layer 0 produces
    (ce_mid/pe_mid, *_spread_pct, *_ok, ...), so Layers 1-2 run unchanged."""
    sub = day[(day["symbol"] == symbol) & (day["instrument"] == "OPT")
              & (day["expiry"] == expiry)]
    strikes = sorted(sub["strike"].dropna().unique())
    frame = pd.DataFrame({"strike": np.asarray(strikes, dtype=float)})
    for tag, side in (("ce", "CE"), ("pe", "PE")):
        s = sub[sub["side"] == side].drop_duplicates("strike").set_index("strike")
        s = s.reindex(frame["strike"])
        px = s[price_field].to_numpy(float)
        traded = s["contracts"].fillna(0).to_numpy(float)
        err = price_error_proxy(pd.Series(px)).to_numpy(float)
        ok = (traded >= min_contracts) & (px >= min_price) & (err <= max_error)
        frame[f"{tag}_bid"] = np.nan
        frame[f"{tag}_ask"] = np.nan
        frame[f"{tag}_mid"] = np.where(ok, px, np.nan)
        frame[f"{tag}_ltp"] = px
        frame[f"{tag}_iv_nse"] = np.nan
        frame[f"{tag}_oi"] = s["oi"].to_numpy(float)
        frame[f"{tag}_vol"] = traded
        frame[f"{tag}_spread_pct"] = err
        frame[f"{tag}_ok"] = ok
    return frame


def futures_close(day: pd.DataFrame, symbol: str, expiry: date) -> float:
    f = day[(day["symbol"] == symbol) & (day["instrument"] == "FUT") & (day["expiry"] == expiry)]
    return float(f["close"].iloc[0]) if len(f) else float("nan")


def expiring_future_settle(day: pd.DataFrame, symbol: str) -> Optional[float]:
    """Settle of the future expiring this session = NSE's final settlement price,
    i.e. the index close (exact on all 387 legacy expiry days checked vs Yahoo)."""
    f = day[(day["symbol"] == symbol) & (day["instrument"] == "FUT")
            & (pd.to_datetime(day["expiry"]) == pd.to_datetime(day["date"])) & (day["settle"] > 0)]
    return float(f["settle"].iloc[0]) if len(f) else None


def spot_for(day: pd.DataFrame, symbol: str, fallback: Optional[float]) -> Optional[float]:
    """UDiFF carries the index close; legacy does not, so use the OHLC history,
    and on an expiry day missing from it, the expiring future's final settle."""
    u = day.loc[(day["symbol"] == symbol) & day["underlying"].notna(), "underlying"]
    if len(u):
        return float(u.iloc[0])
    if fallback is not None:
        return fallback
    return expiring_future_settle(day, symbol)


def analyze_expiry(day: pd.DataFrame, symbol: str, expiry: date, session: str,
                   spot: Optional[float], **price_kw) -> dict:
    """Layers 0-2 for one bhavcopy expiry, mirroring vol_model.analyze_day."""
    dte = compute_dte(session, fmt_expiry(expiry))
    res = {"symbol": symbol, "date": session, "expiry": fmt_expiry(expiry),
           "dte": dte, "spot": spot, "T": dte / 365.0 if dte else float("nan"),
           "fut_close": futures_close(day, symbol, expiry)}
    if dte <= 1:
        res["skipped"] = f"dte={dte} too small for stable IV (excluded)"
        return res
    per = build_per_strike(day, symbol, expiry, **price_kw)
    fwd = recover_forward_df(per, spot=spot)
    res.update({"forward": fwd["forward"], "discount_factor": fwd["discount_factor"],
                "parity_pairs": fwd["n_pairs"], "parity_r2": fwd["r2"],
                "forward_ok": fwd["ok"], "forward_note": fwd["reason"]})
    if not fwd["ok"]:
        res["skipped"] = fwd["reason"]
        return res
    recon = reconcile_nse_iv(per, fwd["forward"], fwd["discount_factor"], res["T"])
    res["reconciliation"] = recon
    res["n_liquid"] = int(recon["liquid"].sum())
    return res


def daily_metric_rows(day: pd.DataFrame, symbol: str, session: str,
                      spot: Optional[float], expiries: list) -> list:
    """Same schema as data/processed/daily_metrics, plus walls and max pain."""
    rows = []
    front = expiries[0] if expiries else None
    for e in expiries:
        chain = chain_tidy(day, symbol, e)
        if chain.empty:
            continue
        m = compute_daily_metrics(chain, spot)
        piv = chain.pivot_table(index="STRIKE", columns="side", values="OI", aggfunc="sum").fillna(0)
        m["max_pain_strike"] = max_pain(chain)["max_pain_strike"]
        m["put_wall"] = float(piv["PE"].idxmax()) if "PE" in piv and piv["PE"].sum() > 0 else np.nan
        m["call_wall"] = float(piv["CE"].idxmax()) if "CE" in piv and piv["CE"].sum() > 0 else np.nan
        meta = enrich_chain_metadata({"date": session, "expiry": fmt_expiry(e)},
                                     is_front=(e == front))
        m.update({"symbol": symbol, "date": session, "expiry": fmt_expiry(e),
                  "dte": meta["dte"], "expiry_regime": meta["expiry_regime"],
                  "dte_bucket": meta["dte_bucket"], "is_front_expiry": meta["is_front_expiry"],
                  "signal_regime": meta["signal_regime"]})
        rows.append(m)
    return rows


def trading_sessions(underlying_csv: str, start: str, end: str) -> list:
    """Session calendar from the index OHLC history (it only has trading days)."""
    df = pd.read_csv(underlying_csv, usecols=["date"])
    s = sorted(d for d in df["date"].astype(str) if start <= d <= end)
    return [datetime.strptime(d, "%Y-%m-%d").date() for d in s]
