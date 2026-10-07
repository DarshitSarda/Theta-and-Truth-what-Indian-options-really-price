"""Download NSE participant-wise OI CSV (aggregate 4-bucket report)."""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime

import requests
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.trading_calendar import describe_session, is_trading_day, resolve_session_date
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    CONFIG["paths"].get("nse_reports", "data/raw/nse_reports"),
    "participant_oi",
)
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
    "Connection": "keep-alive",
}

ARCHIVE_BASE = "https://nsearchives.nseindia.com/content/nsccl"
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 45
MAX_RETRIES = CONFIG.get("scraping", {}).get("max_retries", 3)


class ReportNotPublishedError(FileNotFoundError):
    """Raised when NSE has not posted the report for the requested date yet."""


def nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    # Homepage often 403; derivatives page sets the cookies we need.
    s.get("https://www.nseindia.com/all-reports-derivatives", timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    time.sleep(1)
    return s


def _archive_url(date: datetime) -> str:
    ddmmyyyy = date.strftime("%d%m%Y")
    return f"{ARCHIVE_BASE}/fao_participant_oi_{ddmmyyyy}.csv"


def _out_path(date: datetime) -> str:
    ddmmyyyy = date.strftime("%d%m%Y")
    return os.path.join(OUT_DIR, f"fao_participant_oi_{ddmmyyyy}.csv")


def _check_available(session: requests.Session, url: str) -> int:
    """HEAD probe; NSE returns 404 quickly when missing, but GET may hang."""
    try:
        resp = session.head(url, timeout=(CONNECT_TIMEOUT, 15), allow_redirects=True)
        return resp.status_code
    except requests.RequestException as exc:
        raise TimeoutError(
            f"Could not reach NSE archives for {url}\n"
            f"Network error: {exc}\n"
            "Check your connection and retry in a few minutes."
        ) from exc


def _check_report_date(content: bytes, date: datetime) -> None:
    """The CSV title says 'as on Sep 30, 2026'; it must match the requested session."""
    m = re.search(rb"as on\s+([A-Za-z]{3}\s+\d{1,2},\s*\d{4})", content[:300])
    if not m:
        raise FileNotFoundError("Could not read the 'as on' date from the participant OI title row")
    report = datetime.strptime(m.group(1).decode().replace(" ,", ","), "%b %d, %Y").date()
    if report != date.date():
        raise FileNotFoundError(
            f"Report says 'as on {report}', but session {date.date()} was requested. Not saved."
        )


def _fetch_csv(session: requests.Session, url: str, date: datetime) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if resp.status_code == 404:
                raise ReportNotPublishedError(
                    f"Participant OI not published yet: {url}\n"
                    "NSE usually posts this file 15-45 minutes after market close "
                    "(~3:45-4:15 PM IST). Wait and rerun."
                )
            if resp.status_code != 200:
                raise FileNotFoundError(
                    f"Unexpected response for {url} (status {resp.status_code})"
                )
            content = resp.content.strip()
            if not content or b"Participant wise Open Interest" not in content:
                raise FileNotFoundError(f"Empty or invalid CSV from {url}")
            _check_report_date(content, date)
            return resp.content
        except ReportNotPublishedError:
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                wait = min(2 ** attempt, 20)
                print(f"  Attempt {attempt}/{MAX_RETRIES} failed ({exc.__class__.__name__}); retry in {wait}s...")
                time.sleep(wait)
    raise TimeoutError(
        f"Could not download participant OI after {MAX_RETRIES} attempts: {url}\n"
        f"Last error: {last_error}\n"
        "The report may not be on NSE yet — try again in 15-30 minutes."
    ) from last_error


def download_participant_oi(
    session: requests.Session,
    date: datetime | None = None,
    *,
    skip_existing: bool = True,
    wait_minutes: int = 0,
) -> str:
    date = date or datetime.strptime(resolve_session_date(), "%Y-%m-%d")
    out_path = _out_path(date)
    if skip_existing and os.path.isfile(out_path) and os.path.getsize(out_path) > 100:
        print(f"Already exists: {out_path}")
        return out_path

    url = _archive_url(date)
    deadline = time.time() + wait_minutes * 60

    while True:
        status = _check_available(session, url)
        if status == 404:
            if wait_minutes and time.time() < deadline:
                print("  Report not published yet; checking again in 2 min...")
                time.sleep(120)
                continue
            raise ReportNotPublishedError(
                f"Participant OI not published yet for {date.strftime('%d-%b-%Y')}: {url}\n"
                "Wait 15-30 minutes after market close and rerun, or use --wait-minutes 45."
            )
        if status == 200:
            break
        raise FileNotFoundError(f"Unexpected HEAD status {status} for {url}")

    content = _fetch_csv(session, url, date)
    with open(out_path, "wb") as f:
        f.write(content)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NSE participant-wise OI CSV.")
    parser.add_argument(
        "--date",
        help="Report date as DD-MM-YYYY (default: latest finished NSE session, in IST)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the session's file already exists",
    )
    parser.add_argument(
        "--wait-minutes",
        type=int,
        default=0,
        help="Poll every 2 min until the report appears (max minutes to wait)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.date:
        date = datetime.strptime(args.date, "%d-%m-%Y")
        if not is_trading_day(date.date()):
            sys.exit(f"{date:%d-%b-%Y} is not an NSE trading day (weekend or holiday); no report exists.")
    else:
        print(describe_session())
        date = datetime.strptime(resolve_session_date(), "%Y-%m-%d")

    session = nse_session()
    path = download_participant_oi(
        session, date, skip_existing=not args.force, wait_minutes=args.wait_minutes
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
