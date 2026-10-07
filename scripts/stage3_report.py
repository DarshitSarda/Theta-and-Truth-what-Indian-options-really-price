"""Stage 3 evaluation (rules fixed before the full run was inspected).

GATES (must pass, else stop and investigate)
  C0 pricer mode     calibration uses the one-series (puts + parity) pricer; on 300 random final
                     fits it must equal the validated two-sided pricer: max |diff|/vega < 1e-6 vol pt
  C1 recovery        24 real chain grids (both symbols, spread over the years); truth = that day's
                     fitted Bates parameters; synthetic market IVs = truth + N(0, 0.3 vol pt) noise;
                     Bates calibrated from cold starts. Pass if on >= 90% of chains (a) the fit
                     error is <= 0.36 vol pt (1.2 x noise) and (b) fitted vs true IV on a dense
                     grid across each expiry's observed strike range differs by <= 0.3 vol pt RMS.
                     (Amended after the pilot: the first version used K/F 0.85-1.15, which tests
                     extrapolation far beyond the data on short expiries; that is now reported
                     separately, not gated. Threshold unchanged.)
                     Parameter recovery is reported (identifiability), not gated.
  C2 coverage        >= 97% of sessions with a chain have finite Heston, Merton and Bates fits,
                     and Bates finished with a converged status on >= 97%
  C3 nesting         Bates objective <= min(Heston, Merton) objective (+1e-6 relative) on >= 99%
  C4 IV coverage     model prices without an implied vol < 0.5% of options

FINDINGS (reported, no pass/fail)
  - fit quality by model x era x symbol: in-sample, hidden-strike and next-day RMSE (vol pts)
  - "earns its keep": paired t-test of hidden-strike and next-day RMSE, Bates vs Heston, Bates vs
    Merton, Bates+events vs Bates; significant (p < 0.01, Bonferroni over 6 tests) in BOTH the
    discovery (< 2018) and holdout (>= 2018) periods
  - parameter behaviour: share of days at a bound, daily changes, typical values by era
  - systematic misfit: mean Bates IV error by moneyness x maturity
  - scheduled events: implied event move psi before each event vs the realised move

Writes data/processed/stage3/_report.txt
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import bates as bt  # noqa: E402
from src import calibrate as cb  # noqa: E402
from src.contracts import b76_iv  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
DISCOVERY_END = pd.Timestamp("2018-01-01")
ERAS = [("2008-13", "2008-01-01"), ("2014-19", "2014-01-01"), ("2020-Nov24", "2020-01-01"), ("Nov24-now", "2024-11-20")]
NOISE = 0.3
RESULTS = []
rng = np.random.default_rng(3)


def gate(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def sec(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def era_of(d: pd.Series) -> pd.Series:
    out = pd.Series(ERAS[0][0], index=d.index)
    for name, start in ERAS[1:]:
        out[d >= start] = name
    return out


def load(base):
    f = pd.concat([pd.read_parquet(x) for x in glob.glob(os.path.join(base, "fits", "*.parquet"))], ignore_index=True)
    f["era"] = era_of(f["date"])
    return f


_PANELS = {}


def chain_for(symbol, date):
    if symbol not in _PANELS:
        p = cb.load_symbol(symbol)
        _PANELS[symbol] = (p, cb.event_sessions(cb.load_events(), pd.DatetimeIndex(p["date"].unique())))
    p, ev = _PANELS[symbol]
    return cb.build_chain(symbol, date, p[p["date"] == date], ev)


def xvec(row, model):
    return np.array([row[f"p_{k}"] for k in cb.MODELS[model]], float)


# ---------------------------------------------------------------- gates

def c0_pricer(f):
    sec("C0 one-series pricer vs validated two-sided pricer")
    b = f[(f["model"].isin(["bates", "heston", "bates_ev"])) & f["p_v0"].notna()]
    worst = 0.0
    for _, r in b.sample(min(300, len(b)), random_state=1).iterrows():
        ch = chain_for(r["symbol"], r["date"])
        x = xvec(r, r["model"])
        a = cb.model_prices(r["model"], x, ch, puts_only=True)
        c = cb.model_prices(r["model"], x, ch, puts_only=False)
        for e, pa, pc in zip(ch.expiries, a, c):
            worst = max(worst, float(np.max(np.abs(pa - pc) / e.vega)))
    gate("C0 pricer mode", worst < 1e-6, f"max |diff|/vega {worst:.1e} vol pt")


def c1_recovery(f):
    sec("C1 synthetic recovery (truth = fitted Bates; noise 0.3 vol pt; cold starts)")
    b = f[(f["model"] == "bates") & f["p_v0"].notna()].copy()
    b["y"] = b["date"].dt.year
    picks = []
    for sym in b["symbol"].unique():
        s = b[b["symbol"] == sym]
        ys = sorted(s["y"].unique())
        for y in ys[:: max(1, len(ys) // 12)][:12]:
            picks.append(s[s["y"] == y].sample(1, random_state=y).iloc[0])
    rows = []
    for r in picks:
        ch = chain_for(r["symbol"], r["date"])
        xt = xvec(r, "bates")
        true_p = cb.model_prices("bates", xt, ch, puts_only=False)
        syn = []
        for e, tp in zip(ch.expiries, true_p):
            n = len(e.K)
            iv_t = b76_iv(tp, np.full(n, e.F), e.K, np.full(n, e.T), np.full(n, e.DF), e.is_call)
            iv_s = iv_t + rng.normal(0, NOISE / 100, n)
            px = bt.black76(e.F, e.K, e.T, iv_s, e.DF, e.is_call)
            syn.append(cb.Expiry(e.expiry, e.F, e.DF, e.T, e.K, e.is_call, px, e.vega, iv_s, e.n_events))
        sch = cb.Chain(ch.symbol, ch.date, syn)
        fit = cb.fit_day(sch, None)["bates"]
        def grid_err(K, e):
            ic = K >= e.F
            n = len(K)
            pt = bt.price_cos(e.F, K, e.T, cb.to_params("bates", xt), e.DF, ic)
            pf = bt.price_cos(e.F, K, e.T, cb.to_params("bates", fit.x), e.DF, ic)
            it = b76_iv(pt, np.full(n, e.F), K, np.full(n, e.T), np.full(n, e.DF), ic)
            iff = b76_iv(pf, np.full(n, e.F), K, np.full(n, e.T), np.full(n, e.DF), ic)
            return (iff - it) * 100

        dense = np.concatenate([grid_err(np.linspace(e.K.min(), e.K.max(), 31), e) for e in ch.expiries])
        extra = np.concatenate([grid_err(e.F * np.linspace(0.85, 1.15, 31), e) for e in ch.expiries])
        rows.append(dict(symbol=r["symbol"], date=r["date"].date(), n=ch.n, fit_rmse=fit.rmse_vega,
                         dense_rms=float(np.sqrt(np.nanmean(dense ** 2))),
                         extrap_rms=float(np.sqrt(np.nanmean(extra ** 2))),
                         **{f"t_{k}": v for k, v in zip(cb.MODELS["bates"], xt)},
                         **{f"f_{k}": v for k, v in zip(cb.MODELS["bates"], fit.x)}))
    t = pd.DataFrame(rows)
    print(t[["symbol", "date", "n", "fit_rmse", "dense_rms", "extrap_rms"]].round(3).to_string(index=False))
    print("(dense = observed strike range of each expiry, gated; extrap = K/F 0.85-1.15, information only)")
    rec = pd.DataFrame({k: [np.median(np.abs(t[f"f_{k}"] - t[f"t_{k}"])), np.median(np.abs(t[f"t_{k}"]))]
                        for k in cb.MODELS["bates"]}, index=["median |fit - truth|", "median |truth|"])
    print("\nParameter recovery (identifiability):\n" + rec.round(4).to_string())
    a = (t["fit_rmse"] <= 1.2 * NOISE).mean()
    d = (t["dense_rms"] <= NOISE).mean()
    gate("C1 recovery", a >= 0.9 and d >= 0.9,
         f"fit at noise level on {a:.0%}, true smile reproduced (<= {NOISE} vol pt) on {d:.0%} of {len(t)} chains")


def c2_c4(f):
    sec("C2-C4 coverage, nesting, IV coverage")
    base = f[f["model"].isin(["heston", "merton", "bates"])]
    ok = base.pivot_table(index=["symbol", "date"], columns="model", values="p_v0", aggfunc="first").notna().all(axis=1)
    b = f[f["model"] == "bates"]
    conv = (b["status"] > 0).mean()
    gate("C2 coverage", ok.mean() >= 0.97 and conv >= 0.97,
         f"all three fitted on {ok.mean():.2%} of {len(ok)} sessions; Bates converged on {conv:.2%}")
    w = base.pivot_table(index=["symbol", "date"], columns="model", values="cost")
    nest = (w["bates"] <= np.minimum(w["heston"], w["merton"]) * (1 + 1e-6)).mean()
    gate("C3 nesting", nest >= 0.99, f"Bates <= both sub-models on {nest:.2%}")
    nan = f["iv_nan"].sum() / f["n_opt"].sum()
    gate("C4 IV coverage", nan < 0.005, f"{nan:.3%} of model prices without IV")


# ---------------------------------------------------------------- findings

def fit_quality(f):
    sec("FIT QUALITY (median RMSE, vol points): in-sample / hidden strikes / next day")
    g = f.groupby(["symbol", "era", "model"], observed=True).agg(
        days=("rmse_iv", "size"), insample=("rmse_iv", "median"), hidden=("hold_rmse", "median"),
        next_day=("next_rmse", "median"), p90_insample=("rmse_iv", lambda s: s.quantile(0.9)))
    print(g.round(3).to_string())


def keep_tests(f):
    sec("DOES THE RICHER MODEL EARN ITS KEEP? paired differences (simpler - richer), vol pts")
    pairs = [("heston", "bates"), ("merton", "bates"), ("bates", "bates_ev")]
    rows = []
    for col in ("hold_rmse", "next_rmse"):
        w = f.pivot_table(index=["symbol", "date"], columns="model", values=col)
        for a, b in pairs:
            if a not in w or b not in w:
                continue
            d = (w[a] - w[b]).dropna()
            dates = d.index.get_level_values("date")
            for per, m in (("discovery <2018", dates < DISCOVERY_END), ("holdout >=2018", dates >= DISCOVERY_END)):
                x = d[m]
                if len(x) < 5:
                    continue
                t, p = stats.ttest_1samp(x, 0.0)
                rows.append(dict(test=col.replace("_rmse", ""), simpler=a, richer=b, period=per, n=len(x),
                                 mean_gain=x.mean(), median_gain=x.median(), richer_wins=(x > 0).mean(), t=t,
                                 p_bonf=min(1.0, p * 6)))
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    print("\nverdict (significant gain p_bonf<0.01 in both periods, both tests):")
    for a, b in pairs:
        s = t[(t["simpler"] == a) & (t["richer"] == b)]
        ok = len(s) == 4 and ((s["p_bonf"] < 0.01) & (s["mean_gain"] > 0)).all()
        print(f"  {b} over {a}: {'EARNS ITS KEEP' if ok else 'not established'}")


def params(f):
    sec("PARAMETERS: share of days at a bound")
    for m in cb.MODELS:
        s = f[f["model"] == m]
        if not len(s):
            continue
        hit = {k: s["at_bounds"].fillna("").str.split(",").apply(lambda l: k in l).mean() for k in cb.MODELS[m]}
        print(f"  {m:9s} " + "  ".join(f"{k} {v:.0%}" for k, v in hit.items()))
    sec("BATES PARAMETERS by era (median) and day-to-day stability (median |change|)")
    b = f[f["model"] == "bates"].sort_values(["symbol", "date"])
    cols = [f"p_{k}" for k in cb.MODELS["bates"]]
    print(b.groupby(["symbol", "era"])[cols].median().round(3).to_string())
    ch = b.groupby("symbol")[cols].diff().abs()
    ch = ch[b["prev_gap"].le(5).to_numpy()]
    print("\nmedian |daily change|:\n" + ch.median().round(4).to_string())
    print("\nimplied quantities (median): ATM-vol sqrt(v0) %, jump crash size e^mu-1 %, jumps/year")
    q = pd.DataFrame({"vol_now": np.sqrt(b["p_v0"]) * 100, "vol_longrun": np.sqrt(b["p_theta"]) * 100,
                      "jump_mean_pct": np.expm1(b["p_mu_j"]) * 100, "jumps_per_year": b["p_lam"],
                      "symbol": b["symbol"], "era": b["era"]})
    print(q.groupby(["symbol", "era"]).median().round(2).to_string())


def misfit(base):
    sec("SYSTEMATIC MISFIT: mean Bates IV error (model - market, vol pts) by moneyness x maturity")
    r = pd.concat([pd.read_parquet(x, filters=[("model", "=", "bates")])
                   for x in glob.glob(os.path.join(base, "resid", "*.parquet"))], ignore_index=True)
    r["k_b"] = pd.cut(r["k"], [-1, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 1])
    r["T_b"] = pd.cut(r["T"] * 365, [0, 7, 14, 35, 100])
    print(r.pivot_table(index="k_b", columns="T_b", values="err_iv", aggfunc="mean", observed=True).round(2).to_string())


def events(f):
    sec("SCHEDULED EVENTS: implied event move (psi) on the sessions before, vs realised move")
    e = f[(f["model"] == "bates_ev") & f["p_psi"].notna()] if "p_psi" in f else pd.DataFrame()
    if e.empty:
        print("  no identified event days")
        return
    evs = cb.load_events()
    rows = []
    for sym in e["symbol"].unique():
        spot = pd.read_parquet(cb.PANEL, columns=["date", "spot"], filters=[("symbol", "=", sym)]).groupby("date")["spot"].first()
        es = cb.event_sessions(evs, pd.DatetimeIndex(spot.index))
        for _, ev in es.iterrows():
            before = e[(e["symbol"] == sym) & (e["date"] < ev["session"]) & (e["date"] >= ev["session"] - pd.Timedelta(days=14))]
            if before.empty or ev["session"] not in spot.index:
                continue
            i = spot.index.get_loc(ev["session"])
            realised = np.log(spot.iloc[i] / spot.iloc[i - 1]) if i > 0 else np.nan
            bb = f[(f["model"] == "bates") & (f["symbol"] == sym) & f["date"].isin(before["date"])]
            rows.append(dict(symbol=sym, event=ev["date"].date(), type=ev["type"], days=len(before),
                             psi_pct=before["p_psi"].median() * 100, realised_pct=realised * 100,
                             gain_vs_bates=(bb["rmse_iv"].median() - before["rmse_iv"].median())))
    t = pd.DataFrame(rows)
    print(t.round(2).to_string(index=False))
    if len(t) > 3:
        print(f"\n  median implied event sd {t['psi_pct'].median():.2f}%, median |realised| {t['realised_pct'].abs().median():.2f}%,"
              f" corr(psi, |realised|) {np.corrcoef(t['psi_pct'], t['realised_pct'].abs())[0, 1]:.2f}  (n={len(t)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--skip-recovery", action="store_true")
    a = ap.parse_args()
    base = os.path.join(PROJECT_ROOT, "data", "processed", "stage3", *(["pilot"] if a.pilot else []))
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        f = load(base)
        print(f"{f[['symbol', 'date']].drop_duplicates().shape[0]} sessions, {len(f)} fits, {f['date'].min().date()} .. {f['date'].max().date()}")
        c0_pricer(f)
        if not a.skip_recovery:
            c1_recovery(f)
        c2_c4(f)
        fit_quality(f)
        keep_tests(f)
        params(f)
        misfit(base)
        events(f)
        sec("GATES")
        for n, ok, d in RESULTS:
            print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}")
    with open(os.path.join(base, "_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())


if __name__ == "__main__":
    main()
