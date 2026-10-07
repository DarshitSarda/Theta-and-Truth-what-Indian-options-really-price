"""Parse NSE participant-wise OI reports (fao_participant_oi_DDMMYYYY.csv).

The raw file is the aggregate 4-bucket report: one row per client type
(Client / DII / FII / Pro / TOTAL) with long & short open-interest contract
counts across index/stock futures and index/stock options.

These helpers turn it into net positioning that is comparable day-to-day.
Note: this report is market-wide (all F&O), not per-index-symbol.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

CLIENT_ORDER = ["FII", "DII", "Pro", "Client", "TOTAL"]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def parse_participant_oi(path: str) -> pd.DataFrame:
    """Load one report into a tidy frame indexed by client type."""
    df = pd.read_csv(path, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]
    key = df.columns[0]
    df = df.rename(columns={key: "client_type"})
    df["client_type"] = df["client_type"].astype(str).str.strip()
    for c in df.columns:
        if c != "client_type":
            df[c] = _num(df[c])
    return df.set_index("client_type")


def net_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Net (long - short) contracts by client type for index futures & options."""
    out = pd.DataFrame(index=df.index)
    out["fut_idx_long"] = df["Future Index Long"]
    out["fut_idx_short"] = df["Future Index Short"]
    out["net_fut_idx"] = df["Future Index Long"] - df["Future Index Short"]
    out["net_ce_idx"] = df["Option Index Call Long"] - df["Option Index Call Short"]
    out["net_pe_idx"] = df["Option Index Put Long"] - df["Option Index Put Short"]
    # Net options delta lean: long call + short put = bullish; sign is directional.
    out["net_opt_idx_dir"] = out["net_ce_idx"] - out["net_pe_idx"]
    order = [c for c in CLIENT_ORDER if c in out.index]
    return out.loc[order]


def _path_for(directory: str, date: datetime) -> str:
    return os.path.join(directory, f"fao_participant_oi_{date.strftime('%d%m%Y')}.csv")


def latest_report(directory: str, on_or_before: datetime, lookback: int = 10):
    """Return (date, DataFrame) for the most recent report on/before a date."""
    for i in range(lookback + 1):
        d = on_or_before - timedelta(days=i)
        p = _path_for(directory, d)
        if os.path.isfile(p) and os.path.getsize(p) > 100:
            return d, parse_participant_oi(p)
    return None, None
