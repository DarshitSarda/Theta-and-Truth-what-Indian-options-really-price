"""Daily Bates view of the live option chains (descriptive, for the dashboard).

For each symbol and live session: the Stage 3 calibration (src/calibrate.py: same option
filter, thinning, models, event rule) on the live snapshot (src/live.py), warm-started from
the previous live fit, else the last Stage 3 fit before the session (past only). From the
best fit (Bates + events if identified, else Bates) and the Stage 4 P model:
  expiry view   per fitted expiry: market ATM IV, the move the market prices to expiry
                (sqrt of Q expected quadratic variation), the move our P model expects
                (GJR-GARCH-t x past-only calibration factor), the gap in vol points, and
                the share of Q variance from jumps/events
  rich / cheap  every traded OTM option (|delta| 0.02-0.50) of the fitted expiries:
                S_P = ln(Bates value at the P variance level / mid), ranked within its
                side x delta bucket. The only Bates ranking that survived Stage 5's
                one-session-skip test; its money edge (~Rs 4 per Rs 100, holdout) is below
                trading costs, so this is context, not a trade list. (The fitted-surface
                residual failed the skip test and is not shown.)
  events        scheduled events (config/india_events.csv, once known) inside the listed
                expiries, the event-day move priced by the fit when identified, and the
                historical record of that event type (Stage 4)
P model: closes = NSE close (panel) then Yahoo (identical, scripts/live_verify.py); monthly
GARCH parameters reused from Stage 4 when fitted on the same returns, refitted otherwise.
Calibration factor = sum realised / sum forecast over Stage 4 rows settled before the session.

  python scripts/live_bates.py            fit missing recent sessions, write the views
  python scripts/live_bates.py --verify   also fit every live session the bhavcopy covers
                                          and compare with the Stage 3 fits
Outputs data/processed/live_bates/: fits.parquet (cache), expiry_view.csv,
richcheap/{SYMBOL}_{date}.csv, events_upcoming.csv, _latest.txt, _verify.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src import bates as bt  # noqa: E402
from src import calibrate as cb  # noqa: E402
from src import live  # noqa: E402
from src import pmodel as pm  # noqa: E402
from src.backtest import PANEL  # noqa: E402
from src.trading_calendar import is_trading_day  # noqa: E402
import stage4_build as s4  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "data", "processed", "live_bates")
FITS = os.path.join(OUT, "fits.parquet")
S4 = os.path.join(PROJECT_ROOT, "data", "processed", "stage4")
CATCH_UP = 5            # default run fits missing sessions among the last 5 live sessions
MAX_PREV_GAP = 10
BUCKETS = [(0.02, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50)]
LINES: list[str] = []


def out(s=""):
    print(s)
    LINES.append(str(s))


# ---------------------------------------------------------------- P model

class LiveP:
    def __init__(self, symbol: str):
        sp = pd.read_parquet(PANEL, columns=["date", "spot"], filters=[("symbol", "=", symbol)]).groupby("date")["spot"].first()
        y = live.yahoo_close(symbol)
        c = pm.close_series(y, sp)
        c = pd.concat([c, y[y.index > c.index.max()]]).sort_index()
        self.r = np.log(c).diff().dropna()
        cached = pd.read_parquet(os.path.join(S4, f"garch_params_{symbol}.parquet"))
        rows, self.refit = [], []
        for m in pd.date_range(pm.FIRST_FIT, self.r.index.max() + pd.offsets.MonthBegin(1), freq="MS"):
            n = int((self.r.index < m).sum())
            row = cached[cached["month"] == m]
            if len(row) and int(row["n_obs"].iloc[0]) == n:
                rows.append(row.iloc[0].drop(labels=["symbol"], errors="ignore").to_dict())
            else:
                rows.append(pm.fit_month(self.r, m))
                self.refit.append(m)
        self.params = pd.DataFrame(rows)
        self.state = pm.daily_state(self.r, self.params).set_index("date")
        self.naive_tab = pm.trailing_rv(self.r)
        o = pd.read_parquet(os.path.join(S4, "outcomes.parquet"))
        self.o4 = o[o["symbol"] == symbol].dropna(subset=["rv", "p_total"])

    def forecast(self, t, n) -> float:
        if t not in self.state.index or n <= 0:
            return np.nan
        return pm.variance_forecast(self.state.loc[t], n)

    def naive(self, t, n) -> float:
        return float(n * self.naive_tab.get(t, np.nan)) if n > 0 else np.nan

    def cal(self, d) -> float:
        x = self.o4[self.o4["settled"] < d]
        return float(x["rv"].sum() / x["p_total"].sum()) if len(x) >= s4.MIN_CAL_ROWS else np.nan


# ---------------------------------------------------------------- fitting

def all_trading_days(start="2008-01-01", years_ahead=2) -> pd.DatetimeIndex:
    end = pd.Timestamp.today() + pd.DateOffset(years=years_ahead)
    return pd.DatetimeIndex([d for d in pd.date_range(start, end) if is_trading_day(d.date())])


def stage3_prev(fits3: pd.DataFrame, d) -> dict:
    f = fits3[fits3["date"] < d]
    if f.empty or (d - f["date"].max()).days > MAX_PREV_GAP:
        return {}
    f = f[f["date"] == f["date"].max()]
    return {r.model: np.array([getattr(r, f"p_{k}") for k in cb.MODELS[r.model]], float) for r in f.itertuples()}


def fit_rows(symbol, d, ch, fits) -> list[dict]:
    rows = []
    for m, f in fits.items():
        if not np.isfinite(f.x).all():
            rows.append(dict(symbol=symbol, date=d, model=m, status=f.status, n_opt=ch.n))
            continue
        err = cb.iv_errors(m, f.x, ch)
        rows.append(dict(symbol=symbol, date=d, model=m, n_opt=ch.n, n_exp=len(ch.expiries),
                         t_min=ch.expiries[0].T, t_max=ch.expiries[-1].T,
                         n_events=max(e.n_events for e in ch.expiries), atm_iv=np.sqrt(ch.atm_var()),
                         rmse_iv=float(np.sqrt(np.nanmean(err ** 2))), max_abs_iv=float(np.nanmax(np.abs(err))),
                         status=f.status, secs=f.secs, **{f"p_{k}": float(v) for k, v in zip(cb.MODELS[m], f.x)}))
    return rows


def chain_input(lp: pd.DataFrame) -> pd.DataFrame:
    return cb.select_options(lp.assign(contracts=lp["volume"]))


def fit_symbol(symbol: str, lp: pd.DataFrame, sessions: list, ev: pd.DataFrame, cache: pd.DataFrame) -> list[dict]:
    fits3 = s4.all_fits(symbol)
    sel = chain_input(lp)
    by_day = dict(tuple(sel.groupby("date")))
    rows = []
    mine = cache[cache["symbol"] == symbol] if len(cache) else cache
    for d in sessions:
        day = by_day.get(d)
        ch = cb.build_chain(symbol, d, day, ev) if day is not None else None
        if ch is None:
            print(f"  {symbol} {d.date()}: no usable chain", flush=True)
            continue
        prev = {}
        done = pd.concat([mine, pd.DataFrame(rows)], ignore_index=True) if rows else mine
        if len(done):
            f = done[(done["date"] < d) & done["p_v0"].notna()]
            if len(f) and (d - f["date"].max()).days <= MAX_PREV_GAP:
                f = f[f["date"] == f["date"].max()]
                prev = {r["model"]: np.array([r[f"p_{k}"] for k in cb.MODELS[r["model"]]], float) for _, r in f.iterrows()}
        if not prev:
            prev = stage3_prev(fits3, d)
        t0 = time.time()
        rows += fit_rows(symbol, d, ch, cb.fit_day(ch, prev))
        print(f"  {symbol} {d.date()}: fitted {ch.n} options / {len(ch.expiries)} expiries in {time.time() - t0:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------- views

def dbucket(a):
    for lo, hi in BUCKETS:
        if lo <= a < hi or (hi == 0.50 and a == 0.50):
            return f"{lo:.2f}-{hi:.2f}"
    return None


def views(symbol: str, d, lp: pd.DataFrame, fr: pd.Series, P: LiveP, ev: pd.DataFrame):
    sel = chain_input(lp[lp["date"] == d])
    cal = P.cal(d)
    exp_rows, rc = [], []
    for e, g in sel.groupby("expiry"):
        T = float(g["T"].iloc[0])
        if not (fr["t_min"] - 1e-9 <= T <= fr["t_max"] + 1e-9):
            continue
        n_ev = int(((ev["known_from"] <= d) & (ev["session"] > d) & (ev["session"] <= e)).sum()) \
            if fr["model"] == "bates_ev" else 0
        p = s4.params_of(fr, n_ev)
        qd, qj, qe = s4.qv_parts(T, p)
        qv = qd + qj + qe
        n_s = live.trading_days_between(d, e)
        pc = P.forecast(d, n_s) * cal
        pn = P.naive(d, n_s)
        F, DF = float(g["forward"].iloc[0]), float(g["df"].iloc[0])
        allx = lp[(lp["date"] == d) & (lp["expiry"] == e) & lp["iv"].notna()]
        atm = allx.iloc[np.argmin(np.abs(allx["log_moneyness"].to_numpy()))] if len(allx) else None
        exp_rows.append(dict(symbol=symbol, date=d, expiry=e, dte=int(g["dte"].iloc[0]), sessions=n_s, F=F,
                             atm_iv=float(atm["iv"]) if atm is not None else np.nan, model=fr["model"], n_events=n_ev,
                             q_move=np.sqrt(qv), p_move=np.sqrt(pc) if pc > 0 else np.nan,
                             naive_move=np.sqrt(pn) if pn > 0 else np.nan,
                             q_vol=np.sqrt(qv / T), p_vol=np.sqrt(pc / T) if pc > 0 else np.nan,
                             jump_share=(qj + qe) / qv, event_share=qe / qv, cal=cal))
        q, _ = s4.at_level(p, T, pc, "j")
        if q is None:
            continue
        K, ic = g["strike"].to_numpy(float), (g["side"] == "CE").to_numpy()
        fair = bt.price_cos(F, K, T, q, DF, ic)
        x = g[["expiry", "strike", "side", "mark", "bid", "ask", "iv", "delta", "volume", "oi"]].copy()
        x["fair_p"] = fair
        x["s_p"] = np.log(fair / x["mark"])
        x["bucket"] = [dbucket(abs(v)) for v in x["delta"]]
        rc.append(x)
    rc = pd.concat(rc, ignore_index=True) if rc else pd.DataFrame()
    if len(rc):
        rc["rank_in_cell"] = rc.groupby(["expiry", "side", "bucket"])["s_p"].rank(pct=True)
        rc["n_in_cell"] = rc.groupby(["expiry", "side", "bucket"])["s_p"].transform("size")
    return pd.DataFrame(exp_rows), rc


def event_view(symbol, d, ev, fr, lp):
    ev4 = pd.read_parquet(os.path.join(S4, "events.parquet"))
    ev4 = ev4[ev4["symbol"] == symbol]
    listed = lp.loc[lp["date"] == d, "expiry"].max()
    up = ev[(ev["session"] > d)].copy()
    rows = []
    for _, e in up.iterrows():
        h = ev4[ev4["type"] == e["type"]]
        priced = np.sqrt(h["q_day_var"] + h["psi"].fillna(0) ** 2)
        rows.append(dict(symbol=symbol, date=d, event=e["date"], type=e["type"], note=e.get("note", ""),
                         known=bool(e["known_from"] <= d), inside_listed=bool(e["session"] <= listed),
                         psi_today=float(fr["p_psi"]) if fr["model"] == "bates_ev" and e["known_from"] <= d
                         and e["session"] <= listed else np.nan,
                         hist_n=len(h), hist_priced_med=float(priced.median()) if len(h) else np.nan,
                         hist_realised_med=float(h["r"].abs().median()) if len(h) else np.nan))
    return pd.DataFrame(rows)


def summarize(ex: pd.DataFrame, rc: pd.DataFrame, evv: pd.DataFrame, fr, symbol, d, P: LiveP):
    out(f"\n{symbol} - {d.date()}: Bates fit {fr['model']}, {int(fr['n_opt'])} options / {int(fr['n_exp'])} expiries,"
        f" fit error {fr['rmse_iv']:.2f} vol pts (RMS)")
    if P.refit:
        out(f"  P model months refitted on current data: {[m.strftime('%Y-%m') for m in P.refit]}")
    if len(ex):
        out("  expiry        dte  ATM IV   priced move   P-model move   gap (vol pts)   jumps/events share")
        for r in ex.itertuples():
            out(f"  {r.expiry.date()}  {r.dte:4d}  {r.atm_iv * 100:6.2f}   +/-{r.q_move * 100:5.2f}%      +/-{r.p_move * 100:5.2f}%"
                f"        {(r.q_vol - r.p_vol) * 100:+6.2f}          {r.jump_share:.0%}")
    if len(rc):
        e = ex.iloc[np.argmin(np.abs(ex["dte"] - 30))]["expiry"] if len(ex) else rc["expiry"].iloc[0]
        x = rc[(rc["expiry"] == e) & (rc["n_in_cell"] >= 3)]
        out(f"  rich/cheap within side x delta bucket, expiry {e.date()} (S_P = ln(Bates real-world value / mid))")
        for side in ("PE", "CE"):
            y = x[x["side"] == side].sort_values("s_p")
            if y.empty:
                continue
            fmt = lambda r: f"{r.strike:g} (d {abs(r.delta):.2f}, mid {r.mark:.1f}, S_P {r.s_p:+.2f})"
            rich = y.groupby("bucket").head(1)
            cheap = y.groupby("bucket").tail(1)
            out(f"    {side} richest per bucket: " + "; ".join(fmt(r) for r in rich.itertuples()))
            out(f"    {side} cheapest per bucket: " + "; ".join(fmt(r) for r in cheap.itertuples()))
    if len(evv):
        for r in evv.itertuples():
            state = ("not public yet" if not r.known else "outside the listed expiries" if not r.inside_listed
                     else f"priced event-day move +/-{r.psi_today * 100:.1f}%" if np.isfinite(r.psi_today)
                     else "inside listed expiries, not identified by today's fit")
            out(f"  event {r.event.date()} {r.type}: {state}; history ({r.hist_n}): priced +/-{r.hist_priced_med * 100:.1f}%"
                f" vs realised |move| {r.hist_realised_med * 100:.1f}% (medians)")
    else:
        out("  events: none scheduled in config/india_events.csv after this session")


def verify(cache: pd.DataFrame, lps: dict, ev: pd.DataFrame):
    out("=" * 100 + "\nVERIFY: live-snapshot fits vs Stage 3 bhavcopy fits, same sessions\n" + "=" * 100)
    for sym in ("NIFTY", "BANKNIFTY"):
        a = s4.best_fits(cache[cache["symbol"] == sym].dropna(subset=["p_v0"]))
        b = s4.best_fits(s4.all_fits(sym))
        both = a.index.intersection(b.index)
        if len(both) == 0:
            out(f"{sym}: no common sessions")
            continue
        a, b = a.loc[both], b.loc[both]
        q30 = lambda f: np.array([sum(s4.qv_parts(30 / 365, s4.params_of(r, 0))) for _, r in f.iterrows()])
        qa, qb = q30(a), q30(b)
        out(f"{sym}: {len(both)} sessions; same model chosen {np.mean(a['model'] == b['model']):.0%}")
        out(f"  ATM IV live - bhavcopy (vol pts): median {np.median((a['atm_iv'] - b['atm_iv']) * 100):+.2f},"
            f" p90 |diff| {np.quantile(np.abs(a['atm_iv'] - b['atm_iv']) * 100, 0.9):.2f}")
        out(f"  fit error RMS (vol pts): live median {a['rmse_iv'].median():.2f} vs bhavcopy {b['rmse_iv'].median():.2f}")
        out(f"  30-day priced vol sqrt(Q QV/T): live/bhavcopy ratio median {np.median(np.sqrt(qa / qb)):.3f},"
            f" p10-p90 {np.quantile(np.sqrt(qa / qb), 0.1):.3f}-{np.quantile(np.sqrt(qa / qb), 0.9):.3f}; corr {np.corrcoef(qa, qb)[0, 1]:.3f}")
        for k in ("rho", "lam", "mu_j", "delta"):
            out(f"  {k:5s}: live median {a[f'p_{k}'].median():+.3f} vs bhavcopy {b[f'p_{k}'].median():+.3f}; corr {a[f'p_{k}'].corr(b[f'p_{k}']):.2f}")
        # what the dashboard shows: the rich/cheap ranking of the same live options under each fit
        lp = lps[sym]
        P = LiveP(sym)
        cors, same, sp_gap = [], [], []
        for d in both:
            _, ra = views(sym, d, lp, a.loc[d], P, ev)
            _, rb = views(sym, d, lp, b.loc[d], P, ev)
            if ra.empty or rb.empty:
                continue
            m = ra.merge(rb[["expiry", "strike", "side", "s_p"]], on=["expiry", "strike", "side"], suffixes=("", "_b"))
            sp_gap.append(float((m["s_p"] - m["s_p_b"]).abs().median()))
            for _, g in m.groupby(["expiry", "side", "bucket"]):
                if len(g) >= 4:
                    cors.append(g["s_p"].rank().corr(g["s_p_b"].rank()))
                    same.append(g["s_p"].idxmin() == g["s_p_b"].idxmin())
        out(f"  rich/cheap ranks within side x delta cells, same live options, live fit vs Stage 3 fit:"
            f" rank corr median {np.nanmedian(cors):.2f} (p10 {np.nanquantile(cors, 0.1):.2f}, {len(cors)} cells);"
            f" same richest option {np.mean(same):.0%}; |S_P difference| median {np.median(sp_gap):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "richcheap"), exist_ok=True)
    cache = pd.read_parquet(FITS) if os.path.exists(FITS) else pd.DataFrame()
    ev = cb.event_sessions(cb.load_events(), all_trading_days())
    all_ex, lps = [], {}
    for sym in ("NIFTY", "BANKNIFTY"):
        lp, info = live.live_panel(sym)
        lps[sym] = lp
        sessions = sorted(lp["date"].unique())
        have = set(cache.loc[cache["symbol"] == sym, "date"]) if len(cache) else set()
        if a.verify:
            last3 = s4.all_fits(sym)["date"].max()
            todo = [d for d in sessions if d <= last3 and d not in have] + [d for d in sessions[-CATCH_UP:] if d > last3 and d not in have]
        else:
            todo = [d for d in sessions[-CATCH_UP:] if d not in have]
        print(f"{sym}: fitting {len(todo)} sessions", flush=True)
        rows = fit_symbol(sym, lp, todo, ev, cache)
        if rows:
            cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            cache.to_parquet(FITS)
        best = s4.best_fits(cache[(cache["symbol"] == sym)].dropna(subset=["p_v0"]))
        P = LiveP(sym)
        for d in best.index:
            ex, rc = views(sym, d, lp, best.loc[d], P, ev)
            all_ex.append(ex)
            f = os.path.join(OUT, "richcheap", f"{sym}_{d.date()}.csv")
            if len(rc) and not os.path.exists(f):
                rc.to_csv(f, index=False)
        d = best.index.max()
        if d != sessions[-1]:
            out(f"\n{sym}: WARNING latest live session {sessions[-1].date()} has no Bates fit; showing {d.date()}")
        ex, rc = views(sym, d, lp, best.loc[d], P, ev)
        evv = event_view(sym, d, ev, best.loc[d], lp)
        evv.to_csv(os.path.join(OUT, f"events_upcoming_{sym}.csv"), index=False)
        summarize(ex, rc, evv, best.loc[d], sym, d, P)
    pd.concat(all_ex, ignore_index=True).to_csv(os.path.join(OUT, "expiry_view.csv"), index=False)
    with open(os.path.join(OUT, "_latest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    if a.verify:
        LINES.clear()
        verify(cache, lps, ev)
        with open(os.path.join(OUT, "_verify.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
