"""Backfill dte/expiry_regime on all .meta.json and rebuild daily_metrics."""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.chain_parser import compute_daily_metrics, load_chain_csv
from src.expiry_utils import compute_dte, enrich_chain_metadata

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
METRICS_DIR = os.path.join(DATA_PROCESSED, "daily_metrics")
CHAIN_DIR = os.path.join(DATA_PROCESSED, "option_chain")
ACTIVE = CONFIG.get("collection", {}).get("active_symbols", ["NIFTY", "BANKNIFTY"])


def _meta_paths():
    return sorted(glob.glob(os.path.join(DATA_RAW, "*", "option_chain", "*.meta.json")))


def backfill_metadata() -> int:
    paths = _meta_paths()
    groups: dict[tuple[str, str], list[tuple[str, dict]]] = {}

    for path in paths:
        with open(path) as f:
            meta = json.load(f)
        key = (meta.get("symbol", "").upper(), meta.get("date", ""))
        groups.setdefault(key, []).append((path, meta))

    updated = 0
    for (_symbol, _date), items in groups.items():
        dtes = [
            compute_dte(meta["date"], meta["expiry"]) if meta.get("expiry") else 9999
            for _path, meta in items
        ]
        min_dte = min(dtes) if dtes else 9999

        for (path, meta), dte in zip(items, dtes):
            enrich_chain_metadata(meta, is_front=(dte == min_dte))
            with open(path, "w") as f:
                json.dump(meta, f, indent=2)
            updated += 1

    return updated


def rebuild_metrics_for_symbol(symbol: str) -> int:
    pattern = os.path.join(DATA_RAW, symbol.lower(), "option_chain", "*.csv")
    paths = sorted(p for p in glob.glob(pattern) if "Select" not in p)
    by_date: dict[str, list] = {}

    for csv_path in paths:
        base = os.path.basename(csv_path)
        trade_date = base.split("_")[0]
        by_date.setdefault(trade_date, []).append(csv_path)

    days_written = 0
    for trade_date, day_paths in sorted(by_date.items()):
        rows = []
        for csv_path in day_paths:
            meta_path = csv_path.replace(".csv", ".meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            chain = load_chain_csv(csv_path)
            metrics = compute_daily_metrics(chain, spot=meta.get("underlying_spot"))
            metrics.update({
                "symbol": meta.get("symbol", symbol).upper(),
                "date": meta.get("date", trade_date),
                "expiry": meta.get("expiry"),
                "asset_class": meta.get("asset_class"),
                "index_name": meta.get("index_name"),
                "expiry_type": meta.get("expiry_type"),
                "dte": meta.get("dte"),
                "expiry_regime": meta.get("expiry_regime"),
                "dte_bucket": meta.get("dte_bucket"),
                "is_front_expiry": meta.get("is_front_expiry"),
                "signal_regime": meta.get("signal_regime"),
                "source_file": csv_path,
            })
            rows.append(metrics)

            tidy_path = csv_path.replace(DATA_RAW, CHAIN_DIR).replace(".csv", ".parquet")
            os.makedirs(os.path.dirname(tidy_path), exist_ok=True)
            chain.to_parquet(tidy_path, index=False)

        if rows:
            out = os.path.join(METRICS_DIR, f"{symbol.lower()}_{trade_date}.csv")
            pd.DataFrame(rows).to_csv(out, index=False)
            days_written += 1

    return days_written


def main():
    n = backfill_metadata()
    print(f"Updated {n} metadata files")
    for sym in ACTIVE:
        days = rebuild_metrics_for_symbol(sym)
        print(f"Rebuilt metrics for {sym}: {days} days")


if __name__ == "__main__":
    main()
