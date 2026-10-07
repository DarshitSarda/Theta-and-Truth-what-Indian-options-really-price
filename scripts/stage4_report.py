"""Stage 4 report: does the market's (Q) distribution match what happened, and where does
the option buyer's loss come from?

Pre-declared tests (discovery = sessions before 2018, holdout = 2018 on; P-based tests
start 2009). Inference: all means / ratios use scores aggregated by settlement month with a
Newey-West (lag 2 months) variance, because outcome windows overlap.
  T1  PIT of the realised settlement under Q. H1a var(z) < 1 (Q too wide), H1b left-tail
      (u < 5%) frequency < 5%, H1c mean(z) > 0 (realised above Q's centre). One-sided,
      Bonferroni over symbol x target x hypothesis (12) within each period.
  T2  Variance premium: sum E^Q[QV] / sum RV > 1. One-sided, Bonferroni over 4 per period.
      Also Q vs the past-only P forecasts (calibrated GARCH = primary, naive trailing RV).
  T3  Jump share: Q's jump + event share of expected QV vs the realised share of variance
      on threshold-jump days (3.5 / 4 / 5 sigma). Descriptive (definitions differ).
  T4  Events: Q event-session variance (normal day + psi^2) vs realised squared return,
      clustered by event date.
  T5  Stage 1 attribution, per side x delta bucket x horizon: buyer return R - 1
      (R = DF payoff / mark) = variance-level part (B - 1, B = q_{v}{l} / q_a, the market
      price with Q's model moved to the P variance level) + residual (R - B: shape / tail /
      luck). H5a mean B < 1, H5b residual != 0. BH over cells within each period.
      Added after the first run showed the original choice (diffusive-only rescaling, "d")
      squeezes the body: three variants d / p / j at two P levels; the primary one is
      the variant whose PITs are closest to uniform (mean KS distance over symbol x
      target) in the discovery period; the holdout is reported for it unchanged.
  T2b (descriptive, added) terminal dispersion: sum x^2 vs sum RV and vs Q's variance.
Assumption checks A1-A10 follow.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

D = os.path.join(PROJECT_ROOT, "data", "processed", "stage4")
RAW = os.path.join(PROJECT_ROOT, "data", "raw", "underlying")
HOLDOUT = pd.Timestamp("2018-01-01")
ERAS = [("2009-13", "2009-01-01", "2014-01-01"), ("2014-19", "2014-01-01", "2020-01-01"),
        ("2020-Nov24", "2020-01-01", "2024-11-20"), ("Nov24-now", "2024-11-20", "2100-01-01")]
LAG = 2
LINES: list[str] = []


def out(s=""):
    print(s)
    LINES.append(str(s))


def period(d: pd.Series) -> pd.Series:
    return pd.Series(np.where(d < HOLDOUT, "disc", "hold"), index=d.index)


CANDS = [f"{v}{l}" for l in "cn" for v in "dpj"]


def era(d: pd.Series) -> pd.Series:
    e = pd.Series("2008", index=d.index)
    for name, a, b in ERAS:
        e[(d >= a) & (d < b)] = name
    return e


# ---------------------------------------------------------------- HAC on month clusters

def _hac_var(scores: pd.Series) -> float:
    """Newey-West variance of the sum of monthly scores (index = month, sorted, gaps = 0)."""
    s = scores.groupby(level=0).sum()
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"), fill_value=0.0).to_numpy()
    v = float(s @ s)
    for l in range(1, LAG + 1):
        v += 2 * (1 - l / (LAG + 1)) * float(s[l:] @ s[:-l])
    return max(v, 0.0)


def hac_mean(y, month):
    y = pd.Series(np.asarray(y, float), index=month)
    y = y[np.isfinite(y.to_numpy())]
    n = len(y)
    if n < 10:
        return np.nan, np.nan, n
    m = y.mean()
    return m, np.sqrt(_hac_var(y - m)) / n, n


def hac_ratio(a, b, month):
    """sum a / sum b with delta-method HAC se of the log ratio."""
    a = pd.Series(np.asarray(a, float), index=month)
    b = pd.Series(np.asarray(b, float), index=month)
    ok = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
    a, b = a[ok], b[ok]
    if len(a) < 10:
        return np.nan, np.nan, len(a)
    R = a.sum() / b.sum()
    se_log = np.sqrt(_hac_var(a / a.sum() - b / b.sum()))
    return R, se_log, len(a)


def months(d) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.DatetimeIndex(d), freq="M")


def p_one(t, greater=True):
    return stats.norm.sf(t) if greater else stats.norm.cdf(t)


def bh(p: pd.Series) -> pd.Series:
    p = p.dropna()
    n = len(p)
    o = p.sort_values()
    q = (o * n / np.arange(1, n + 1))[::-1].cummin()[::-1].clip(upper=1)
    return q.reindex(p.index)


# ---------------------------------------------------------------- data

def load():
    o = pd.read_parquet(os.path.join(D, "outcomes.parquet"))
    o["qv"] = o.q_diff + o.q_jump + o.q_event
    o["z"] = stats.norm.ppf(o.u_q.clip(1e-6, 1 - 1e-6))
    for c in CANDS:
        o[f"z_{c}"] = stats.norm.ppf(o[f"u_{c}"].clip(1e-6, 1 - 1e-6))
    o["per"] = period(o.date)
    o["era"] = era(o.date)
    o["mon"] = months(o.settled)
    a = pd.read_parquet(os.path.join(D, "attrib.parquet"))
    a["R"] = a.df * a.payoff / a.mark
    for c in CANDS:
        a[f"B_{c}"] = a[f"q_{c}"] / a.q_a
    a["fit"] = a.q_a / a.mark
    a["hz"] = pd.cut(a.tdte, [1, 5, 15, 100], labels=["2-5", "6-15", "16+"])
    a["per"] = period(a.date)
    a["era"] = era(a.date)
    a["mon"] = months(a.settled)
    e = pd.read_parquet(os.path.join(D, "events.parquet"))
    s = pd.read_parquet(os.path.join(D, "pstate.parquet"))
    return o, a, e, s


# ---------------------------------------------------------------- tests

def t1(o):
    out("=" * 100)
    out("T1  PIT of the realised settlement under the market (Q) distribution")
    out("    z = Phi^-1(u). Calibrated forecasts give mean 0, var 1, tail frequencies = nominal.")
    out("=" * 100)
    rows = []
    for (per, sym, tg), g in o.groupby(["per", "symbol", "target"]):
        mo = g.mon
        mz, sz, n = hac_mean(g.z, mo)
        vz, svz, _ = hac_mean(g.z ** 2 - 1, mo)
        lt, slt, _ = hac_mean((g.u_q < .05) - .05, mo)
        rt, srt, _ = hac_mean((g.u_q > .95) - .05, mo)
        l1, _, _ = hac_mean((g.u_q < .01).astype(float), mo)
        r1, _, _ = hac_mean((g.u_q > .99).astype(float), mo)
        mid, _, _ = hac_mean(((g.u_q > .25) & (g.u_q < .75)).astype(float), mo)
        rows.append(dict(per=per, symbol=sym, target=tg, n=n, months=g.mon.nunique(), mean_z=mz, se=sz,
                         p_H1c=p_one(mz / sz), var_z=1 + vz, se_v=svz, p_H1a=p_one(vz / svz, False),
                         lt5=lt + .05, p_H1b=p_one(lt / slt, False), rt5=rt + .05, p_rt=2 * p_one(abs(rt / srt)),
                         lt1=l1, rt1=r1, mid50=mid))
    r = pd.DataFrame(rows)
    for per, g in r.groupby("per"):
        for h in ["p_H1a", "p_H1b", "p_H1c"]:
            r.loc[g.index, h + "_bonf"] = (g[h] * 12).clip(upper=1)
    out(r.round(4).to_string(index=False))
    out("  lt5/rt5 = share with u < 5% / u > 95%; lt1/rt1 the 1% tails; mid50 = share in the central 50%.")
    out("  p_rt is two-sided (right tail was not pre-declared).")
    out("\n  By era (both targets pooled, per symbol):")
    e = o.groupby(["symbol", "era"]).apply(lambda g: pd.Series(dict(
        n=len(g), mean_z=g.z.mean(), var_z=(g.z ** 2).mean(), lt1=(g.u_q < .01).mean(), lt5=(g.u_q < .05).mean(),
        mid50=((g.u_q > .25) & (g.u_q < .75)).mean(), rt5=(g.u_q > .95).mean(), rt1=(g.u_q > .99).mean())),
        include_groups=False)
    out(e.round(4).to_string())
    return r


def t2(o):
    out("\n" + "=" * 100)
    out("T2  Variance premium: market expected quadratic variation vs realised (sum r^2 to settlement)")
    out("    Ratios of sums; ann_vol = sqrt(sum / sum T). P_cal = calibrated GARCH (primary), naive = trailing RV.")
    out("=" * 100)
    x = o.dropna(subset=["p_cal", "p_naive", "rv"])
    rows = []
    for keys, g in list(x.groupby(["per", "symbol", "target"])) + list(x.groupby(["era", "symbol", "target"])):
        R, sl, n = hac_ratio(g.qv, g.rv, g.mon)
        Rp, slp, _ = hac_ratio(g.qv, g.p_cal, g.mon)
        Rn, sln, _ = hac_ratio(g.qv, g.p_naive, g.mon)
        rows.append(dict(group=keys[0], symbol=keys[1], target=keys[2], n=n,
                         Q_vol=np.sqrt(g.qv.sum() / g["T"].sum()), RV_vol=np.sqrt(g.rv.sum() / g["T"].sum()),
                         Pcal_vol=np.sqrt(g.p_cal.sum() / g["T"].sum()), Q_RV=R, se_log=sl, p_H2=p_one(np.log(R) / sl),
                         Q_Pcal=Rp, p_QP=p_one(np.log(Rp) / slp), Q_naive=Rn, p_Qn=p_one(np.log(Rn) / sln),
                         Pcal_RV=g.p_cal.sum() / g.rv.sum(), naive_RV=g.p_naive.sum() / g.rv.sum(),
                         med_Q_RV=(g.qv / g.rv).median(), share_Q_gt_RV=(g.qv > g.rv).mean()))
    r = pd.DataFrame(rows)
    per = r.group.isin(["disc", "hold"])
    r.loc[per, "p_H2_bonf"] = (r.loc[per, "p_H2"] * 4).clip(upper=1)
    out(r.round(4).to_string(index=False))
    out("  Q_RV > 1: the market expected more variance than was realised. Q_Pcal: vs the past-only P forecast.")
    out("  Pcal_RV / naive_RV: P-forecast bias (1 = unbiased). med/share: per-row (skewed RV, so medians > means).")
    return r


def t2b(o):
    out("\n  T2b terminal dispersion (options pay on the end point, RV adds up daily moves):")
    x = o.dropna(subset=["rv"])
    rows = []
    for keys, g in list(x.groupby(["per", "symbol", "target"])):
        for lab, h in [("all", g), ("ex2020", g[g.date.dt.year != 2020])]:
            a, sa, n = hac_ratio(h.x ** 2, h.rv, h.mon)
            b, sb, _ = hac_ratio(h.q_var, h.x ** 2, h.mon)
            rows.append(dict(per=keys[0], symbol=keys[1], target=keys[2], sample=lab, n=n, x2_RV=a, se_log=sa,
                             Qvar_x2=b, se_log2=sb, p_Q_gt_x2=p_one(np.log(b) / sb)))
    out(pd.DataFrame(rows).round(4).to_string(index=False))
    out("  x2_RV > 1: end-point moves bigger than the sum of daily variances (trending). Qvar_x2 > 1: market's")
    out("  terminal variance above the realised terminal second moment.")


def choose_variant(o):
    out("\n" + "=" * 100)
    out("Variant choice for T5: PIT uniformity (mean KS distance over symbol x target; lower = better)")
    out("=" * 100)
    rows = []
    for per, g in o.groupby("per"):
        for c in ["q"] + CANDS:
            u = g["u_q" if c == "q" else f"u_{c}"]
            ks = [stats.kstest(uu.dropna(), "uniform").statistic for _, uu in u.groupby([g.symbol, g.target])]
            z = g["z" if c == "q" else f"z_{c}"]
            rows.append(dict(per=per, cand=c, ks=np.mean(ks), var_z=np.nanmean(z ** 2), lt5=(u < .05).mean(),
                             rt5=(u > .95).mean(), lt1=(u < .01).mean(), rt1=(u > .99).mean()))
    r = pd.DataFrame(rows)
    out(r.round(4).to_string(index=False))
    d = r[(r.per == "disc") & (r.cand != "q")]
    best = d.loc[d.ks.idxmin(), "cand"]
    qks = r[(r.per == "disc") & (r.cand == "q")].ks.iloc[0]
    out(f"  chosen (discovery): {best}  (Q itself: {qks:.4f})")
    out("  floor hits for d (Q jumps + events alone exceed the P forecast): "
        f"cal {o.floor_c.mean():.2%}, naive {o.floor_n.mean():.2%}")
    return best


def t3(o):
    out("\n" + "=" * 100)
    out("T3  Jump share of variance: market (Q) jump + event share vs realised share on threshold-jump days")
    out("=" * 100)
    rows = []
    for (per, sym, tg), g in o.dropna(subset=["rv"]).groupby(["per", "symbol", "target"]):
        rows.append(dict(per=per, symbol=sym, target=tg, n=len(g),
                         Q_jump=g.q_jump.sum() / g.qv.sum(), Q_event=g.q_event.sum() / g.qv.sum(),
                         **{f"real_{c}s": g[f"rv_jump_{c}"].sum() / g.rv.sum() for c in ["3.5", "4.0", "5.0"]},
                         Q_jump_vol=np.sqrt((g.q_jump + g.q_event).sum() / g["T"].sum()),
                         real_jump4_vol=np.sqrt(g["rv_jump_4.0"].sum() / g["T"].sum())))
    out(pd.DataFrame(rows).round(4).to_string(index=False))
    out("  Q's 'jump' is the Bates jump component (it also carries the smile's skew); realised jump days are")
    out("  daily returns beyond c GARCH sigmas (they include that day's diffusive move). Read as orders of magnitude.")


def t4(e):
    out("\n" + "=" * 100)
    out("T4  Scheduled events: market-implied event-session variance vs realised")
    out("=" * 100)
    x = e.dropna(subset=["psi"]).copy()
    x["q_ev"] = x.q_day_var + x.psi ** 2
    x["r2"] = x.r ** 2
    x["q_sd"] = np.sqrt(x.q_ev)
    out(x[["symbol", "event", "type", "psi", "q_sd", "r", "p_day_var"]].assign(
        p_sd=np.sqrt(x.p_day_var)).drop(columns="p_day_var").round(4).to_string(index=False))
    mo = pd.PeriodIndex(x.event, freq="D")
    for name, g in [("all", x), ("budget", x[x.type == "budget"]), ("election", x[x.type == "election"])]:
        gm = pd.PeriodIndex(g.event, freq="M")
        R, sl, n = hac_ratio(g.q_ev, g.r2, gm)
        out(f"  {name:9s} n={n:3d} events={g.event.nunique():3d}  sum Q var / sum r^2 = {R:.3f} (se log {sl:.3f})"
            f"  median Q sd {g.q_sd.median():.4f}  median |r| {g.r.abs().median():.4f}"
            f"  share |r| > Q sd {(g.r.abs() > g.q_sd).mean():.2f}  corr(Q sd,|r|) {np.corrcoef(g.q_sd, g.r.abs())[0, 1]:.2f}")
    out("  Under a calibrated normal, |r| > sd about 32% of the time; median |r| = 0.67 sd.")
    out(f"  Events without an event-identified fit (psi missing): {e.psi.isna().sum()}; psi at 0: {(e.psi == 0).sum()}")
    _ = mo


def t5(a, best):
    alt = best[0] + ("n" if best[1] == "c" else "c")
    out("\n" + "=" * 100)
    out("T5  Where does the option buyer's return come from? (Stage 1 entries, tdte >= 2, inside fitted maturities)")
    out(f"    buyer = R - 1 (R = DF payoff / mark). var_level = B - 1 (B = q_{best} / q_a: the market price with Q's")
    out("    model moved to the P variance level). residual = R - B (shape / tail / luck). share = var_level / buyer.")
    out(f"    var_level_alt uses the other P level ({alt}).")
    out("=" * 100)
    x = a.copy()
    x["B"], x["Bn"] = x[f"B_{best}"], x[f"B_{alt}"]
    x = x.dropna(subset=["B", "R"])
    x = x[np.isfinite(x.B) & (x.q_a > 0)]
    out("  Bracket (pooled by period x side): var_level under each variant and P level")
    br = x.groupby(["per", "side"]).apply(lambda g: pd.Series({c: (g[f"B_{c}"] - 1).mean() for c in CANDS}
                                                              | {"buyer": (g.R - 1).mean()}), include_groups=False)
    out(br.round(3).to_string())
    rows = []
    for keys, g in x.groupby(["per", "side", "dbucket", "hz"], observed=True):
        mR, sR, n = hac_mean(g.R - 1, g.mon)
        mB, sB, _ = hac_mean(g.B - 1, g.mon)
        mD, sD, _ = hac_mean(g.R - g.B, g.mon)
        mBn, _, _ = hac_mean(g.Bn - 1, g.mon)
        rows.append(dict(per=keys[0], side=keys[1], dbucket=keys[2], hz=keys[3], n=n, buyer=mR, se_buyer=sR,
                         var_level=mB, se_vl=sB, p_H5a=p_one(mB / sB, False), residual=mD, se_res=sD,
                         p_H5b=2 * p_one(abs(mD / sD)), share=mB / mR if mR < 0 else np.nan, var_level_alt=mBn,
                         fit_med=g.fit.median()))
    r = pd.DataFrame(rows)
    for per, g in r.groupby("per"):
        r.loc[g.index, "q_H5a"] = bh(g.p_H5a)
        r.loc[g.index, "q_H5b"] = bh(g.p_H5b)
    out(r.round(3).to_string(index=False))
    out("\n  Pooled by side and period:")
    for keys, g in x.groupby(["per", "side"]):
        mR, sR, n = hac_mean(g.R - 1, g.mon)
        mB, sB, _ = hac_mean(g.B - 1, g.mon)
        mD, sD, _ = hac_mean(g.R - g.B, g.mon)
        mBn, sBn, _ = hac_mean(g.Bn - 1, g.mon)
        out(f"  {keys[0]} {keys[1]} n={n:6d}  buyer {mR:+.3f} ({sR:.3f})  var_level {mB:+.3f} ({sB:.3f})"
            f"  residual {mD:+.3f} ({sD:.3f})  var_level_alt {mBn:+.3f} ({sBn:.3f})")
    out("\n  By era (side pooled over buckets):")
    for keys, g in x.groupby(["era", "side"]):
        mR, sR, n = hac_mean(g.R - 1, g.mon)
        mB, sB, _ = hac_mean(g.B - 1, g.mon)
        mD, sD, _ = hac_mean(g.R - g.B, g.mon)
        out(f"  {keys[0]:11s} {keys[1]} n={n:6d}  buyer {mR:+.3f} ({sR:.3f})  var_level {mB:+.3f} ({sB:.3f})"
            f"  residual {mD:+.3f} ({sD:.3f})")
    return r


# ---------------------------------------------------------------- assumption checks

def checks(o, a, e, s):
    out("\n" + "=" * 100)
    out("ASSUMPTION CHECKS")
    out("=" * 100)
    # A1
    for sym, fn in [("NIFTY", "nifty.csv"), ("BANKNIFTY", "banknifty.csv")]:
        y = pd.read_csv(os.path.join(RAW, fn), parse_dates=["date"]).set_index("date")["close"]
        sp = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "processed", "contracts", "panel"),
                             columns=["date", "spot"], filters=[("symbol", "=", sym)]).groupby("date")["spot"].first()
        j = pd.concat([y, sp], axis=1, join="inner").dropna()
        d = (j.iloc[:, 0] / j.iloc[:, 1] - 1).abs()
        out(f"A1 {sym}: Yahoo vs panel close on {len(j)} common days: max |rel diff| {d.max():.2e}, "
            f"days > 0.1%: {(d > 1e-3).sum()}")
    # A2
    files = glob.glob(os.path.join(PROJECT_ROOT, "data", "processed", "contracts", "panel", "**", "*.parquet"), recursive=True)
    p = pd.concat([pd.read_parquet(f, columns=["date", "expiry", "settle", "spot"]).assign(
        symbol=f.split("symbol=")[1].split(os.sep)[0]) for f in files], ignore_index=True)
    p = p[(p.date == p.expiry) & p.settle.notna() & (p.settle > 0)]
    g = (p.settle / p.spot - 1).abs().groupby([p.symbol, p.expiry]).median()
    out(f"A2 settlement = closing spot: {len(g)} expiries with an expiry-day settle (NSE reports the index "
        f"settlement price there); median |settle/spot-1| {g.median():.2e}; > 0.01%: {(g > 1e-4).sum()}; "
        f"> 0.1%: {(g > 1e-3).sum()}")
    # A3
    v = pd.read_csv(os.path.join(RAW, "india_vix.csv"), parse_dates=["date"]).set_index("date")["close"] / 100
    n = o[(o.symbol == "NIFTY")].drop_duplicates("date").set_index("date")
    j = pd.concat([np.sqrt(n.q30 / (30 / 365)).rename("q"), v.rename("vix")], axis=1, join="inner").dropna()
    j["era"] = era(j.index.to_series())
    out(f"A3 NIFTY model 30d vol vs India VIX: n={len(j)} corr {j.q.corr(j.vix):.3f}, median model/VIX "
        f"{(j.q / j.vix).median():.3f}; by era: {j.groupby('era').apply(lambda g: round((g.q / g.vix).median(), 3), include_groups=False).to_dict()}")
    out("   (model 30d excludes events; VIX includes them and uses the full strip, so small gaps are expected)")
    # A4
    x = a[a.q_a > 0]
    out("A4 model price / market mark on Stage 1 entries (fit quality where we reprice):")
    out(x.groupby(["side", "dbucket", "hz"], observed=True).fit.quantile([.05, .5, .95]).unstack().round(3).to_string())
    # A5
    out("A5 P-forecast bias, sum RV / sum forecast (1 = unbiased), and QLIKE (lower = better):")
    y = o.dropna(subset=["p_cal", "p_naive", "rv"])
    y = y[y.rv > 0]
    rows = []
    for (sym, per), g in y.groupby(["symbol", "per"]):
        d = dict(symbol=sym, per=per, n=len(g))
        for c in ["p_total", "p_cal", "p_naive", "qv"]:
            d[f"bias_{c}"] = g.rv.sum() / g[c].sum()
            d[f"qlike_{c}"] = np.mean(g.rv / g[c] - np.log(g.rv / g[c]) - 1)
        rows.append(d)
    out(pd.DataFrame(rows).round(3).to_string(index=False))
    out("   by year (sum RV / sum P_cal):")
    out(y.groupby([y.symbol, y.date.dt.year]).apply(lambda g: g.rv.sum() / g.p_cal.sum(), include_groups=False)
        .unstack(0).round(2).T.to_string())
    # A6 in T3
    out("A6 jump threshold sensitivity: see T3 (3.5 / 4 / 5 sigma columns).")
    # A7
    o2 = o.copy()
    o2["below"] = o2.x < o2.k_lo
    o2["above"] = o2.x > o2.k_hi
    out("A7 outcomes beyond the strikes quoted on day t (tail probability is model extrapolation there):")
    out(o2.groupby(["symbol", "target"])[["below", "above"]].mean().round(4).to_string())
    # A8
    out("A8 excluding poorly fitted days (rmse_iv > 1.5 vol points):")
    good = o[o.fit_rmse <= 1.5]
    out(f"   dropped {1 - len(good) / len(o):.2%} of rows")
    for (sym, tg), g in good.groupby(["symbol", "target"]):
        gg = g.dropna(subset=["rv"])
        out(f"   {sym:9s} {tg:5s} var_z {np.mean(g.z ** 2):.3f} mean_z {g.z.mean():+.3f} lt5 {(g.u_q < .05).mean():.3f} "
            f"rt5 {(g.u_q > .95).mean():.3f}  Q/RV {gg.qv.sum() / gg.rv.sum():.3f}")
    # A9
    out("A9 overnight share of close-to-close variance (Yahoo opens; years with a real opening print only):")
    for sym, fn in [("NIFTY", "nifty.csv"), ("BANKNIFTY", "banknifty.csv")]:
        y = pd.read_csv(os.path.join(RAW, fn), parse_dates=["date"]).set_index("date")
        on = np.log(y.open / y.close.shift())
        cc = np.log(y.close / y.close.shift())
        stale = (on.abs() < 1e-6).groupby(y.index.year).mean()
        ok_years = stale[stale < 0.05].index
        m = y.index.year.isin(ok_years) & on.notna()
        sh = (on[m] ** 2).groupby(y.index[m].year).sum() / (cc[m] ** 2).groupby(y.index[m].year).sum()
        out(f"   {sym}: years used {list(ok_years)[:1]}..{list(ok_years)[-1:]} ({len(ok_years)}); overnight share "
            f"overall {(on[m] ** 2).sum() / (cc[m] ** 2).sum():.3f}; by year {sh.round(2).to_dict()}")
    # A10
    out("A10 P-model PIT: see the variant-choice table (PITs of Q's model at each P level).")


def main():
    o, a, e, s = load()
    out(f"Stage 4 report. outcomes {len(o)} rows ({o.date.min().date()} .. {o.date.max().date()}), "
        f"attribution {len(a)} trades, events {len(e)}")
    out(f"u_q exactly 0/1 (< 1e-6 or > 1-1e-6): {((o.u_q < 1e-6) | (o.u_q > 1 - 1e-6)).sum()}")
    t1(o)
    t2(o)
    t2b(o)
    t3(o)
    t4(e)
    best = choose_variant(o)
    t5(a, best)
    checks(o, a, e, s)
    with open(os.path.join(D, "_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
    main()
