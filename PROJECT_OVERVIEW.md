# NSE F&O EOD Options Pipeline — Project Overview

Prepared as a handoff/review brief. Everything below reflects the repo state as of
session date **2026-09-07** (40 collected sessions).

---

## 1. Purpose and scope

A personal-research EOD data pipeline for Indian index options. The stated thesis is
that daily option-chain data is a **sensor layer** (positioning, open interest,
implied vol) rather than a trade-execution engine: signals are read off options, but
any execution is intended in futures/cash, not naked options.

- **Universe:** `config/config.yaml` carries a 211-name F&O watchlist, but
  `collection.active_symbols` is deliberately limited to `[NIFTY, BANKNIFTY]`
  (Sprint 1 = index only).
- **Frequency:** once per day, after market close (EOD snapshots only — no intraday
  time series, no tick data).
- **Not yet built:** backtests, strategy logic, position sizing, gamma/GEX layer,
  cross-sectional breadth across the 211 names.

---

## 2. Data sources and footprint

| Source | What | Mechanism | Current footprint |
|---|---|---|---|
| NSE option chain (website CSV export) | Full chain per expiry: OI, ΔOI, volume, NSE's IV, bid/ask/qty, LTP, CHNG | Selenium (`notebooks/02`), downloads NSE's own CSV | ~200 chain snapshots + sidecar `.meta.json` |
| NSE participant-wise OI (`fao_participant_oi_DDMMYYYY.csv`) | 4-bucket (FII/DII/Pro/Client) long+short OI in index/stock futures & options | `requests` against `nsearchives.nseindia.com` (`scripts/scrape_participant_oi.py`) | 40 daily reports |
| Yahoo Finance (`^NSEI`, `^NSEBANK`) | Daily OHLC for realized vol | `yfinance` (`scripts/fetch_underlying.py`) | 518 rows, 2024-08-06 → 2026-09-07 |

**Coverage:** 40 sessions, **2026-07-13 → 2026-09-07**, 2 symbols.
NIFTY carries 3 weekly expiries per day; BANKNIFTY 2 monthlies.

**Layout:**

```
data/raw/{symbol}/option_chain/{YYYY-MM-DD}_{DD-Mon-YYYY}.csv   + .meta.json
data/raw/nse_reports/participant_oi/fao_participant_oi_DDMMYYYY.csv
data/raw/underlying/{nifty,banknifty}.csv
data/processed/daily_metrics/{symbol}_{date}.csv
data/processed/option_chain/...                  # tidy long-format parquet
data/processed/vol_surface/{symbol}/{date}_{expiry}.parquet   # clean IV points
data/processed/vol_surface/svi/{symbol}_{date}.csv   + _svi_all.csv
data/processed/vol_surface/signals/{symbol}_{date}.csv + _signals_all.csv, _daily.csv
data/processed/vol_surface/vrp/{per_expiry.csv, _daily.csv}
```

The `.meta.json` sidecar is the authoritative record of trade date, symbol, expiry,
captured spot, DTE, and expiry-regime tags. Downstream layers read it rather than
re-deriving from filenames.

---

## 3. Architecture: layered pipeline

Each layer is a pure module in `src/` with a notebook for the live daily run and a
`scripts/backfill_*.py` twin that recomputes the same thing across all history. The
backfill scripts are the recovery path when anything upstream is corrected.

### Layer 0–2 — Chain hygiene, forward recovery, own implied vols (`src/vol_model.py`)

The core methodological choice: **do not trust NSE's published IV**; recompute it.

1. **Hygiene:** mid prices from bid/ask (never LTP — stale on illiquid strikes),
   liquidity filters, spread caps (`0.20` for DTE > 30, else `0.35`), OTM-wing
   selection only (NSE leaves deep-ITM IV blank anyway).
2. **Forward + discount factor from the chain itself** via put-call parity
   `C − P = DF·(F − K)` regressed across liquid near-ATM strikes. No assumed
   risk-free rate or dividend yield, so skew is not contaminated by a wrong rate.
3. **Black-76 inversion** (options on the forward, which is correct for European
   cash-settled index options) using a self-contained safeguarded
   Newton–bisection solver with no-arbitrage bound checks.

*Validation:* parity R² median **0.99993**; own IV vs NSE's published IV — median
mean-absolute difference **0.66 vol points**, correlation **0.99**. Days with
DTE ≤ 1 or bad parity are skipped (44 of 219 expiry-days).

### Layer 3 — Arbitrage-free SVI smile (`src/svi.py`)

Gatheral **raw SVI** fitted in total-implied-variance space,
`w(k) = a + b(ρ(k−m) + √((k−m)² + σ²))`, `k = ln(K/F)`.

- Fit is performed on **implied vol** (`√(w/T)`) with liquidity weights, so RMSE is
  directly in vol points.
- Multi-start trust-region least squares (`scipy.least_squares`) with a soft penalty
  enforcing non-negative minimum variance.
- Fit domain is **maturity-adaptive**: strikes within ~4 ATM standard deviations
  (`atm_iv·√T`), capped at `|k| ≤ 0.30`, widening if too few points survive.
- **Static no-arbitrage diagnostics:** min-variance ≥ 0, Gatheral butterfly
  `g(k) ≥ 0` over the traded span, and calendar monotonicity of total variance
  across expiries. Butterfly is judged on the traded region only; the wide-wing
  scan is reported for transparency but does not gate the flag.

*Validation:* 179 smiles fitted; RMSE median **0.30** vol points (max 0.54); median
64 IV points per smile; **1 of 179** rows fails the no-arb gate (NIFTY 2026-07-17).

### Layer 4 — Smile → trader signals (`src/vol_signals.py`)

Analytic reduction of each SVI smile at the forward:

- `atm_iv`, `atm_skew` (dσ/dk), `atm_curv` (d²σ/dk²)
- `rr25` — 25-delta risk reversal, `bf25` — 25-delta butterfly, both using
  **driftless forward deltas** `N(d₁)` (natural under Black-76, independent of DF)
- **Constant-maturity ATM IV** at 7/14/30 days by interpolating linearly in *total
  variance vs T* (arb-consistent, since variance is additive in time), with an
  `in_range` flag when the target tenor is outside the observed maturity span
- `term_slope_cmt_7_30` and `term_slope_raw`

Outputs one per-expiry row set plus a single dashboard row per symbol-day.

### Layer 5 — Realized vol and variance risk premium (`src/realized_vol.py`, `src/vrp.py`)

Realized-vol estimators, all annualized at 252 days: close-to-close, Parkinson,
Garman-Klass, Rogers-Satchell, Yang-Zhang. **Close-to-close is the headline** because
it includes overnight gaps, which is what option premium actually prices; Yang-Zhang
is the robust cross-check. Parkinson/GK/RS are intraday-only and understate index
vol, so they are deliberately excluded from VRP.

Two scored views of VRP = implied − realized (vol points):

1. **Per-expiry (expiry-aligned):** ATM IV observed on day *t* for expiry *E* vs
   realized vol over the path *(t, E]*. Scores only once the expiry passes. Doubles
   as the IV-as-forecast baseline.
2. **Fixed-horizon:** `cmt7` / `cmt30` vs forward realized over the next ~5 / ~21
   trading days. Scores once the window completes.

Plus a same-day rich/cheap gauge `iv_minus_trail20` (front ATM IV − trailing 20d
close-to-close RV), which is available every day and is what the daily read leans on.

*Current result:* **115 of 179** per-expiry observations scored; mean VRP (cc)
**+3.18 vol points**; "seller win" rate **90.4%**.

### Sensor layers outside the vol stack

- **`src/chain_parser.py`** — wide NSE CSV → tidy long format (CE/PE), then
  `compute_daily_metrics()`: PCR (OI and volume), OI centroids, total OI per side,
  ATM strike, and `composite_v0_chg_oi_atm_band` = ΔOI(calls) − ΔOI(puts) within
  ±5 strikes of ATM. Also `max_pain()`, computed properly as the settlement level
  minimising total ITM payout across the listed strike grid.
- **`src/participant_oi.py`** — parses the 4-bucket report into net positions
  (`net_fut_idx`, `net_ce_idx`, `net_pe_idx`, and a directional options lean
  `net_ce − net_pe`), with `latest_report()` resolving the most recent file on or
  before a date.
- **`src/expiry_utils.py`** — DTE, expiry regime (`T-0`, `T-1`, … `T-8+`), DTE
  buckets, and a `signal_regime` tag (`expiry_day` / `pre_expiry` / `front_week` /
  `front_month` / `back_series`) that flags how much to trust same-day positioning
  metrics.
- **`scripts/option_movers.py`** — best/worst CE/PE percentage movers per expiry.
  Uses hybrid logic because NSE's `CHNG` field is unreliable for EOD percentages
  (calls are frequently sign-inverted): it uses `CHNG/(LTP−CHNG)` when the sign
  agrees with the actual move vs prior session LTP, otherwise a true
  session-over-session `(today−yesterday)/yesterday`. Filters: volume ≥ 100,
  OI ≥ 100, base LTP ≥ 5.

---

## 4. Daily workflow

Per `Daily steps.txt`, run in order after close:

1. `notebooks/02_option_chain_scraper.ipynb` → Run All (Selenium; downloads chains)
2. `python scripts/scrape_participant_oi.py`
3. `notebooks/04_data_processing.ipynb` → Run All (tidy parquet + daily metrics)
4. `python scripts/fetch_underlying.py`
5. `notebooks/06_vol_surface.ipynb` → Run All (Layers 0–5; reuses chains from step 1)
6. `notebooks/07_dashboard.ipynb` → Run All (read-only consolidated view)

Optional: `python scripts/option_movers.py`.

The dashboard consolidates freshness checks, the vol/VRP snapshot, term structure,
OI walls / max pain / centroids, and participant OI into one numeric view. The daily
output is a discretionary written "verdict" (levels, pin/expiry mechanics, vol
rich/cheap, positioning), not a systematic signal.

---

## 5. Known issues, limitations, and open questions for review

**Correctness / engineering**

1. **Date stamping uses `datetime.now()`** in `notebooks/01_setup.ipynb` (`TODAY`),
   `04`, and `06`. Running the pipeline on a non-trading day (or late, after
   midnight) stamps chains with the run date rather than the session date. This bit
   once already: a weekend run labelled Friday 2026-09-04 data as 2026-09-06, which
   also produced wrong DTEs. It was repaired by renaming raw files, fixing the
   `.meta.json` `date`/`source_file`, deleting the misdated processed artifacts, and
   re-running all five backfill scripts. **A session-date override (or a trading
   calendar) is the obvious hardening.**
2. **No test suite.** No `tests/` directory, no assertions beyond inline canary
   checks in the scraper notebook.
3. **Scraper requires manual notebook execution** (Selenium, non-headless per
   config). No scheduler, no retry orchestration at the pipeline level.
4. 20 orphan chain CSVs from 2026-03-21 / 2026-07-11 exploratory scrapes have no
   `.meta.json` and are outside the 2026-07-13+ analysis window. Harmless but
   untracked.
5. `fetch_underlying.py` carries a workaround: Yahoo's latest daily bar can have a
   NaN close while `regularMarketPrice` is populated, which silently left the
   underlying stale. It now backfills the incomplete bar from the quote.

**Statistical**

6. **VRP observations are heavily overlapping** — per-expiry windows for 3
   concurrent NIFTY weeklies share the same underlying path, and fixed-horizon
   windows roll daily. The "90.4% seller win rate" over 115 observations is
   therefore nowhere near 115 independent trials. Effective sample size and
   autocorrelation-aware standard errors have not been computed.
7. **Sample is small** — 40 sessions. Rolling 20-day z-scores were intentionally
   deferred until ~35–40 sessions and are still not wired in. Planned approach is
   single-feature-first (`composite_v0` or `iv−rv20`) and DTE-aware, since the same
   raw value means different things at 1 DTE vs 25 DTE.
8. **The mean VRP of +3.18 vp is measured over a single, quiet regime.** No
   high-vol regime is in the sample.

**Interpretive**

9. **Participant OI is market-wide across all F&O**, not per-index-symbol. Treating
   FII index-futures net position as a NIFTY-specific signal is an approximation.
10. **`composite_v0` is a ΔOI flow measure, not buyer/seller attribution.** OI change
    alone cannot distinguish writing from unwinding without trade-side data.
11. **Max pain and OI walls are descriptive, not causal.** They are used as magnets
    and pin/breakout reference levels; no dealer-hedging model backs them.
12. **No gamma / GEX layer.** Wanted eventually as a thin dashboard add-on (gamma
    walls specifically), not as a full dealer-positioning model. Not built.

**Roadmap not started**

13. Phase 2 breadth across the 211-name watchlist (and the dispersion idea that
    motivated it) is unstarted. The long-run edge was expected to come from breadth,
    so the current index-only scope is a deliberate but material narrowing.

---

## 6. What a reviewer would most usefully check

- Is the parity-based forward/DF recovery sound, and are the liquidity gates and
  spread caps reasonable, or are they discarding informative wing data?
- Is the maturity-adaptive SVI fit domain (`4·atm_iv·√T`, capped at `|k| ≤ 0.30`)
  defensible, especially at 1–4 DTE where the smile is most distorted?
- Is constant-maturity interpolation linear in total variance the right choice given
  only 2–3 expiries per day, and is the clamp-and-flag behaviour outside the
  observed span acceptable?
- Are the driftless forward deltas for `rr25`/`bf25` being computed and reported
  consistently?
- How should the overlapping-window VRP statistics be corrected before any of this
  is treated as evidence of an edge?
- Are the daily discretionary reads over-reading OI walls and max pain relative to
  what 40 sessions of data can support?
