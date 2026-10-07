"""NSE trading-session calendar: which session's data is on NSE right now.

Every pipeline stage labels its outputs with a *session date* (the NSE trading
day the data belongs to), never the computer's calendar date. The machine may
run in any timezone (the user is in New York, ~9.5-10.5 h behind IST), and NSE
keeps serving the last session's data on weekends and holidays, so "today" on
the local clock is wrong in both directions.

Rules (all in IST, which has no daylight saving):
  - trading day   = weekday and not an NSE F&O trading holiday
  - before 09:00 on a trading day, or any time on a non-trading day,
    NSE shows the previous trading session
  - 09:00-15:30 on a trading day the market is live (no finished session for today)
  - after 15:30 the session is today's; OI is provisional until ~21:00

Holidays come from NSE's own holiday-master API, cached in
config/nse_holidays.json. The cache keeps every year it has seen and is
refreshed when the current year is missing or the cache is over 30 days old
(NSE adds ad-hoc closures, e.g. election days).
"""
from __future__ import annotations

import json
import os
import time
import warnings
from datetime import date, datetime, time as dtime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), "IST")
MARKET_OPEN = dtime(9, 0)
MARKET_CLOSE = dtime(15, 30)
FINAL_OI_AFTER = dtime(21, 0)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLIDAY_CACHE = os.path.join(PROJECT_ROOT, "config", "nse_holidays.json")
HOLIDAY_API = "https://www.nseindia.com/api/holiday-master?type=trading"
HOLIDAY_PAGE = "https://www.nseindia.com/resources/exchange-communication-holidays"
CACHE_MAX_AGE_DAYS = 30

_holidays: set[date] | None = None


def now_ist() -> datetime:
    return datetime.now(IST)


def _fetch_nse_holidays() -> list[str]:
    import requests

    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": HOLIDAY_PAGE,
    })
    s.get(HOLIDAY_PAGE, timeout=15)
    time.sleep(1)
    r = s.get(HOLIDAY_API, timeout=20)
    r.raise_for_status()
    rows = r.json().get("FO") or []
    if not rows:
        raise ValueError("NSE holiday API returned no FO holidays")
    return sorted(
        datetime.strptime(x["tradingDate"], "%d-%b-%Y").date().isoformat() for x in rows
    )


def _read_cache() -> dict:
    if not os.path.exists(HOLIDAY_CACHE):
        return {"fetched_at": None, "holidays": []}
    with open(HOLIDAY_CACHE) as f:
        return json.load(f)


def refresh_holidays() -> list[str]:
    """Pull NSE's current-year list and merge it into the cache."""
    cache = _read_cache()
    fresh = _fetch_nse_holidays()
    fresh_years = {d[:4] for d in fresh}
    kept = [d for d in cache.get("holidays", []) if d[:4] not in fresh_years]
    cache = {
        "source": HOLIDAY_API + " (FO segment)",
        "fetched_at": now_ist().isoformat(timespec="seconds"),
        "holidays": sorted(set(kept) | set(fresh)),
    }
    with open(HOLIDAY_CACHE, "w") as f:
        json.dump(cache, f, indent=1)
    return cache["holidays"]


def holidays() -> set[date]:
    global _holidays
    if _holidays is not None:
        return _holidays
    cache = _read_cache()
    days = cache.get("holidays", [])
    year = str(now_ist().year)
    fetched = cache.get("fetched_at")
    stale = (
        fetched is None
        or not any(d.startswith(year) for d in days)
        or now_ist() - datetime.fromisoformat(fetched) > timedelta(days=CACHE_MAX_AGE_DAYS)
    )
    if stale:
        try:
            days = refresh_holidays()
        except Exception as exc:
            if any(d.startswith(year) for d in days):
                warnings.warn(f"Could not refresh NSE holiday list ({exc}); using cache from {fetched}.")
            else:
                warnings.warn(
                    f"No NSE holiday list for {year} and refresh failed ({exc}). "
                    "Falling back to weekends only - holidays will NOT be recognised."
                )
    _holidays = {date.fromisoformat(d) for d in days}
    return _holidays


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in holidays()


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def session_status(now: datetime | None = None) -> dict:
    """Which session NSE is showing at `now` (default: the current moment).

    Returns session (date), market_open (live, no finished session today),
    provisional (today's session closed but OI may not be final yet) and now_ist.
    """
    t = (now or now_ist()).astimezone(IST)
    d, clock = t.date(), t.time()
    market_open = provisional = False
    if not is_trading_day(d) or clock < MARKET_OPEN:
        session = previous_trading_day(d)
    elif clock < MARKET_CLOSE:
        session = previous_trading_day(d)
        market_open = True
    else:
        session = d
        provisional = clock < FINAL_OI_AFTER
    return {"session": session, "market_open": market_open,
            "provisional": provisional, "now_ist": t}


def resolve_session_date(override: str | None = None) -> str:
    """Session date (YYYY-MM-DD) every stage should label its outputs with.

    `override` (or env var NSE_SESSION_DATE) forces a specific past session,
    e.g. to reprocess one; it must be an NSE trading day.
    """
    override = override or os.environ.get("NSE_SESSION_DATE")
    if override:
        d = date.fromisoformat(override)
        if not is_trading_day(d):
            raise ValueError(f"{override} is not an NSE trading day (weekend or holiday).")
        return d.isoformat()
    return session_status()["session"].isoformat()


def describe_session(now: datetime | None = None) -> str:
    st = session_status(now)
    t, s = st["now_ist"], st["session"]
    msg = f"Session: {s.isoformat()} ({s.strftime('%a')}) | now {t.strftime('%Y-%m-%d %H:%M')} IST"
    if st["market_open"]:
        msg += " | MARKET OPEN - today's session not finished"
    elif st["provisional"]:
        msg += " | before 21:00 IST - OI may be provisional"
    elif s != t.date():
        msg += f" | {t.date().isoformat()} is not a finished trading session"
    return msg
