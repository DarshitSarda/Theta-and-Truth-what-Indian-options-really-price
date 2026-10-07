"""Guards that stop the pipeline from writing a wrong or duplicate trading day.

Two independent checks:
  1. Before scraping: refuse while the market is live, and skip when final
     chains for the session are already on disk.
  2. After each download: if the chain is identical (OI and prices on every
     strike) to the previous session's file for the same expiry, NSE is still
     serving the old session - typically an unlisted holiday. The file is
     deleted and the run stops. This works even if the holiday calendar is wrong.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import date, datetime

from .chain_parser import load_chain_csv
from .trading_calendar import FINAL_OI_AFTER, IST, previous_trading_day, session_status


class MarketOpenError(RuntimeError):
    pass


class StaleChainError(RuntimeError):
    pass


def _chain_files(raw_dir: str, symbol: str, session: str) -> list[str]:
    pattern = os.path.join(raw_dir, symbol.lower(), "option_chain", f"{session}_*.csv")
    return [p for p in sorted(glob.glob(pattern)) if "Select" not in os.path.basename(p)]


def scraped_at(csv_path: str) -> datetime:
    """When a chain was scraped (IST): meta field if present, else file mtime."""
    meta_path = csv_path.replace(".csv", ".meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            stamp = json.load(f).get("scraped_at_ist")
        if stamp:
            return datetime.fromisoformat(stamp)
    return datetime.fromtimestamp(os.path.getmtime(csv_path), IST)


def is_final(csv_path: str, session: str) -> bool:
    cutoff = datetime.combine(date.fromisoformat(session), FINAL_OI_AFTER, IST)
    return scraped_at(csv_path) >= cutoff


def scrape_preflight(raw_dir: str, symbols: list[str], session: str, force: bool = False) -> bool:
    """Decide whether notebook 02 should scrape. Returns True to proceed.

    Raises MarketOpenError during market hours: NSE would serve live, unfinished
    data that has no correct session label.
    """
    st = session_status()
    if st["market_open"]:
        raise MarketOpenError(
            f"NSE market is open ({st['now_ist']:%H:%M} IST). Scrape after 15:30 IST, "
            "ideally after 21:00 IST when OI is final."
        )
    existing = {s: _chain_files(raw_dir, s, session) for s in symbols}
    have_all = all(existing.values())
    final = have_all and all(is_final(p, session) for ps in existing.values() for p in ps)
    if final and not force:
        print(f"Final chains for session {session} are already saved for {', '.join(symbols)}.")
        print("NSE has nothing newer (weekend/holiday or already scraped). Nothing to do.")
        print("Set FORCE_RESCRAPE = True only if you really want to overwrite them.")
        return False
    if have_all:
        print(f"Chains for {session} exist but were scraped before 21:00 IST (provisional) - re-scraping.")
    if st["provisional"]:
        print(f"[WARN] It is {st['now_ist']:%H:%M} IST: OI may still be provisional. "
              "Re-run after 21:00 IST for final OI.")
    return True


def _fingerprint(csv_path: str):
    ch = load_chain_csv(csv_path)
    cols = [c for c in ("STRIKE", "side", "OI", "CHNG IN OI", "LTP", "VOLUME") if c in ch.columns]
    return ch[cols].sort_values(["STRIKE", "side"]).reset_index(drop=True)


def previous_session_duplicate(csv_path: str, session: str) -> str | None:
    """Path of the previous session's file if `csv_path` is an exact copy of it."""
    name = os.path.basename(csv_path)
    expiry_part = name.split("_", 1)[1]
    prev = previous_trading_day(date.fromisoformat(session)).isoformat()
    prev_path = os.path.join(os.path.dirname(csv_path), f"{prev}_{expiry_part}")
    if not os.path.exists(prev_path):
        return None
    new, old = _fingerprint(csv_path), _fingerprint(prev_path)
    if new.shape != old.shape or new["OI"].sum() == 0:
        return None
    return prev_path if new.equals(old) else None


def reject_if_stale(csv_path: str, session: str) -> None:
    """Delete `csv_path` (+ meta) and raise if it duplicates the previous session."""
    dup = previous_session_duplicate(csv_path, session)
    if dup is None:
        return
    meta_path = csv_path.replace(".csv", ".meta.json")
    for p in (csv_path, meta_path):
        if os.path.exists(p):
            os.remove(p)
    raise StaleChainError(
        f"{os.path.basename(csv_path)} is identical to {os.path.basename(dup)}: NSE is still "
        f"serving the previous session, so {session} was probably not a trading day. "
        "The duplicate was deleted. If NSE declared an unscheduled holiday, run "
        "`python -c \"import sys; sys.path.insert(0,'.'); from src.trading_calendar import "
        "refresh_holidays; refresh_holidays()\"` and check config/nse_holidays.json."
    )
