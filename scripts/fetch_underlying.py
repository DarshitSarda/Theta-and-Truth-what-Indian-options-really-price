"""Fetch / update daily OHLC for the underlying indices (Layer 5 realised vol).

Realised volatility is the index's own price path -- public daily OHLC,
independent of our option snapshots -- so we bootstrap a long history once and
top it up daily. Idempotent: merges freshly downloaded rows into the stored CSV,
de-duplicating by date and keeping the latest values.

Source: Yahoo Finance (^NSEI = NIFTY 50, ^NSEBANK = NIFTY Bank, ^INDIAVIX = India VIX).
Output: data/raw/underlying/{symbol}.csv  (columns: date, open, high, low, close)

India VIX is fetched on the same path but kept out of `TICKERS`: it is an
auxiliary series (a vol index, not a tradable underlying), so it must never be
picked up by `vrp.load_ohlc`, which would compute a meaningless realised vol of
a volatility index.

Pass a yfinance period as argv[1] to change the window (e.g. `max` to backfill).
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import yaml
import yfinance as yf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.trading_calendar import now_ist, resolve_session_date, session_status

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "underlying")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
AUX_TICKERS = {"INDIA_VIX": "^INDIAVIX"}
ALL_TICKERS = {**TICKERS, **AUX_TICKERS}
DEFAULT_PERIOD = "2y"


def load_symbols():
    try:
        with open(os.path.join(PROJECT_ROOT, "config", "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        syms = cfg.get("collection", {}).get("active_symbols")
        if syms:
            return [s for s in syms if s in TICKERS]
    except Exception:
        pass
    return list(TICKERS)


def _backfill_incomplete_close(ticker: str, out: pd.DataFrame) -> pd.DataFrame:
    """Yahoo often posts the latest India-index daily bar with O/H/L filled but
    Close still NaN for hours after NSE cash close. The quote endpoint usually
    already has the session last price — use it only when it clearly belongs to
    that incomplete bar (open matches and price sits inside H/L).
    """
    if out.empty:
        return out
    last = out.iloc[-1]
    if pd.notna(last["close"]):
        return out
    if not (pd.notna(last["open"]) and pd.notna(last["high"]) and pd.notna(last["low"])):
        return out
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return out
    px = info.get("regularMarketPrice")
    if px is None:
        return out
    open_ = float(last["open"])
    high = float(last["high"])
    low = float(last["low"])
    quote_open = info.get("open")
    # Same session: quote open matches history open (within a tick) and last
    # is inside the bar's range.
    if quote_open is not None and abs(float(quote_open) - open_) > 0.05:
        return out
    if not (low - 0.05 <= float(px) <= high + 0.05):
        return out
    out = out.copy()
    out.loc[out.index[-1], "close"] = float(px)
    return out


def fetch_one(ticker: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    h = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if h.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    h = h.rename(columns=str.lower).reset_index()
    h["date"] = pd.to_datetime(h["Date"] if "Date" in h.columns else h["index"]).dt.strftime("%Y-%m-%d")
    out = h[["date", "open", "high", "low", "close"]].copy()
    # During market hours Yahoo's bar for today is live, not a close.
    if session_status()["market_open"]:
        out = out[out["date"] != now_ist().date().isoformat()]
    out = _backfill_incomplete_close(ticker, out)
    out = out.dropna(subset=["close"])
    return out


def update_symbol(symbol: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    path = os.path.join(OUT_DIR, f"{symbol.lower()}.csv")
    new = fetch_one(ALL_TICKERS[symbol], period)
    if os.path.exists(path):
        old = pd.read_csv(path)
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    merged = (merged.dropna(subset=["close"])
              .drop_duplicates(subset=["date"], keep="last")
              .sort_values("date")
              .reset_index(drop=True))
    merged.to_csv(path, index=False)
    return merged


def main():
    period = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERIOD
    session = resolve_session_date()
    for symbol in load_symbols() + list(AUX_TICKERS):
        df = update_symbol(symbol, period)
        if df.empty:
            print(f"{symbol}: no data returned")
        else:
            print(f"{symbol}: {len(df)} rows  {df['date'].iloc[0]}..{df['date'].iloc[-1]}  "
                  f"last close={df['close'].iloc[-1]:.2f}")
            if df["date"].iloc[-1] < session:
                print(f"  [WARN] {symbol} has no close for session {session} yet; "
                      "Yahoo can lag - rerun in a while.")


if __name__ == "__main__":
    main()
