"""Days-to-expiry and regime tags for option-chain metadata."""
from __future__ import annotations

from datetime import date, datetime


def parse_trade_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_nse_expiry(value: str) -> date:
    """Parse NSE expiry text e.g. '28-Jul-2026'."""
    return datetime.strptime(value.strip(), "%d-%b-%Y").date()


def compute_dte(trade_date: str | date, expiry: str | date) -> int:
    if isinstance(trade_date, str):
        trade_date = parse_trade_date(trade_date)
    if isinstance(expiry, str):
        expiry = parse_nse_expiry(expiry)
    return (expiry - trade_date).days


def expiry_regime(dte: int) -> str:
    if dte <= 0:
        return "T-0"
    if dte == 1:
        return "T-1"
    if dte <= 7:
        return f"T-{dte}"
    return "T-8+"


def dte_bucket(dte: int) -> str:
    if dte <= 0:
        return "0"
    if dte == 1:
        return "1"
    if dte <= 4:
        return "2-4"
    if dte <= 10:
        return "5-10"
    return "11+"


def signal_regime(dte: int, is_front: bool) -> str:
    """How much to trust same-day positioning metrics."""
    if dte <= 0:
        return "expiry_day"
    if dte == 1:
        return "pre_expiry"
    if is_front and dte <= 4:
        return "front_week"
    if is_front:
        return "front_month"
    return "back_series"


def enrich_chain_metadata(meta: dict, *, is_front: bool | None = None) -> dict:
    """Add dte / regime fields to chain metadata dict (in place + return)."""
    trade_date = meta.get("date")
    expiry = meta.get("expiry")
    if not trade_date or not expiry:
        return meta

    dte = compute_dte(trade_date, expiry)
    front = bool(is_front) if is_front is not None else bool(meta.get("is_front_expiry", False))

    meta.update({
        "dte": dte,
        "expiry_regime": expiry_regime(dte),
        "dte_bucket": dte_bucket(dte),
        "is_front_expiry": front,
        "signal_regime": signal_regime(dte, front),
    })
    return meta
