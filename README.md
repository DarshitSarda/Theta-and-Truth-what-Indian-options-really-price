# Theta & Truth

**What Indian index options really price: an 18-year empirical study of NIFTY and BANKNIFTY options,
from raw exchange data to a realistic, cost-aware trading test.**

Theta & Truth is a personal quant research project. It builds its own data (a daily option-chain
pipeline plus the full NSE bhavcopy archive from 2008), fits its own volatility surfaces and a
Bates jump-diffusion model, and then asks one question with increasing realism:

> Are Indian index options systematically overpriced, and can that be turned into money
> after costs, lot sizes and realistic execution?

Every test was written down (rules frozen and hashed) before it was run, uses only information
available at the time, and is corrected for multiple testing. Negative results are reported as
prominently as positive ones.

---

## Findings at a glance

| Question | Answer |
|---|---|
| Are options overpriced? | **Yes, consistently.** Implied variance is 1.2-1.6x realised variance in every era since 2008; a delta-hedged option buyer loses 32-47% of premium. A variance seller wins 67-72% of windows. |
| Did the 2020 retail boom or SEBI's Nov-2024 F&O rules change that? | **No.** Difference-in-differences (BANKNIFTY vs NIFTY, 144 placebo dates): no change in any pricing outcome. |
| Can a pricing model pick the "most overpriced" strikes? | **No.** Bates rich/cheap rankings are statistically real (t 8.7 in the holdout) but worth ~Rs 3.8 per Rs 100 of premium against ~Rs 6 of costs. Ridge and gradient-boosting strike pickers (walk-forward, 1-5 year windows) do not beat selling everything. |
| Does option positioning (PCR, OI walls, max pain, skew, ...) time market bounces? | **No.** 13 features, 18 years, circular-shift placebos, Bonferroni holdout: nothing passes. |
| Can an incrementally learning model forecast volatility better than the market? | **No.** A daily-refitted HAR / GARCH / implied blend does not beat implied volatility out of sample (2018-2026, Holm p = 1.0). The market already prices what is forecastable. |
| Does a hedged monthly short straddle make money in an idealised backtest? | Yes: +6.2%/yr over cash 2018-2026 (Sharpe 0.58), +1.4%/yr 2008-2017. |
| ...with whole lots, Zerodha costs, next-day VWAP execution? | **No, at any account size (Rs 10 lakh to 5 crore).** The one-day hedging lag alone costs 3-5%/yr. |
| ...hedging near the close (exploratory)? | +3.8 to +5.6%/yr over cash 2018-2026 with zero extra slippage; breakeven with 2 bp; negative with 5 bp. **The edge is real but thin and execution-bound.** |

**Bottom line:** the market is efficient at pricing information but charges a steady insurance
premium. That premium can be harvested only with tight execution and cost control, not by
out-predicting the market.

---

## What is in the repository

### Data layer
- **Live daily pipeline** (`notebooks/02`, `04`, `06`, `07`, `scripts/live_daily.py`): Selenium
  scraper for NSE option chains, participant-wise OI, underlying prices; tidy parquet storage with
  per-snapshot metadata and trading-calendar date guards.
- **Historical layer** (`src/bhavcopy.py`, `scripts/backfill_bhavcopy.py`): every NSE F&O
  bhavcopy since 2008 (legacy and UDiFF formats), validated (`scripts/validate_bhavcopy.py`),
  plus a daily VWAP table reconstructed from traded value (`scripts/build_vwap.py`).

### Models
- **Own implied vols** (`src/vol_model.py`): forward and discount factor recovered from put-call
  parity, Black-76 inversion; parity R^2 median 0.9999.
- **Arbitrage-free SVI smiles** (`src/svi.py`) and smile signals (`src/vol_signals.py`): ATM IV,
  25-delta risk reversal and butterfly, constant-maturity term structure.
- **Bates stochastic-volatility jump-diffusion** (`src/bates.py`, `scripts/stage3_*`): COS pricing,
  daily calibration across expiries, validated against QuantLib.
- **Real-world (P) model** (`scripts/stage4_*`): past-only GJR-GARCH-t with a calibration factor;
  attribution of the premium to diffusion, jumps and events.
- **Backtester** (`src/backtest.py`): daily mark-to-market, smile-based marks, era-correct STT,
  exchange fees and stamp duty (incl. the Finance Act 2026 changes), stress cost scenarios.
- **Realistic execution simulator** (`scripts/realistic_engine.py`): whole lots with each
  contract's own lot size, next-day VWAP fills, futures vs synthetic-futures hedging, hedge
  bands, collateral interest, approximate margin; synthetic-market self-test and look-ahead
  scramble test.

### Studies (each script holds its pre-declared rules in its header)
| Stage | Script | Result file |
|---|---|---|
| 1. Option returns by moneyness / tenor | `scripts/stage1_option_returns.py` | `data/processed/stage1/_report.txt` |
| 3. Bates calibration | `scripts/stage3_*.py` | `data/processed/stage3/_report.txt` |
| 4. Risk-neutral vs real-world | `scripts/stage4_build.py`, `stage4_report.py` | `data/processed/stage4/_report.txt` |
| 5. Portfolio backtest | `scripts/stage5_backtest.py`, `stage5_report.py`, `stage5_audit.py` | `data/processed/stage5/` |
| Bounce event study | `scripts/bounce_event_study.py` | `data/processed/analysis/` |
| SEBI 2024 rules (DiD) | `scripts/sebi2024_did.py` | `data/processed/sebi2024/` |
| Strike picker (ML) | `scripts/strike_picker*.py` | `data/processed/strike_picker/` |
| Realistic execution | `scripts/realistic_engine.py`, `realistic_nearclose.py` | `data/processed/realistic/` |
| Forecast ledger (online learning) | `scripts/forecast_ledger.py` | `data/processed/ledger/` |

### Live layer
- **Paper trading** (`scripts/paper_engine.py`): the frozen Stage 5 engine run on live chains,
  with an append-only, hash-stamped journal (history is never silently rewritten).
- **Bates view** (`scripts/live_bates.py`): daily priced-vs-real-world move per expiry and
  rich/cheap strikes, shown in the dashboard (`notebooks/07_dashboard.ipynb`).

---

## Methodology standards
- Rules frozen before each run (a hashed copy is saved next to the results).
- No look-ahead: past-only parameters, trailing windows, decisions at the close and fills later;
  automated truncation / scramble checks.
- Discovery period (mostly 2008-2017) and holdout (2018 onwards); Newey-West standard errors;
  Holm or Bonferroni correction; circular-shift and shuffled-date placebos.
- Synthetic-market tests for every simulator before it touches real data.

## Running it
```bash
pip install -r requirements.txt
```
The daily steps are in `Daily steps.txt`. Data is not included in the repository (about 1.7 GB);
the backfill scripts rebuild it from NSE's public archives.

## Disclaimer
Research only. Nothing here is investment advice.
