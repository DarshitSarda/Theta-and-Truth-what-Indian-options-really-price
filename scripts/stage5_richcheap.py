"""Stage 5 core test (from the original plan): do Bates rich/cheap rankings predict
realised option returns beyond a naive IV-minus-RV ranking?  Rules fixed before running.

Universe   Stage 4 attribution set = Stage 1 entries (OTM, |delta| 0.02-0.50, tdte 2-42,
           traded, entered at the close) whose expiry is inside that day's Bates fit.
Signals    (all "+" = cheap for a buyer; all known at the entry close)
  S_Q  ln(q_a / mark)     market vs that day's fitted Bates surface (Q residual)
  S_P  ln(q_jc / mark)    market vs Bates moved to the calibrated P variance level
                          (Stage 4's chosen variant jc): Bates' expected-payoff view
  S_N  sigma_N - iv       naive: trailing 22-session RV over the option's horizon
                          (sqrt(p_naive / T)) minus the option's own IV
Outcome    hbuy_gross: Stage 1 delta-hedged buyer return per Rs 100 of premium (gross).
Cross-section  (symbol, entry date, final expiry), >= 12 options. Within it, options are
           compared only with the same side x delta bucket (cell fixed effects), so a
           signal cannot win just by preferring a wing or a side.
Primary statistic  per cross-section, OLS of rank(outcome) on rank(S_B) and rank(S_N)
           with cell fixed effects (ranks scaled to [-0.5, 0.5]); b_B = Bates' partial
           rank slope beyond the naive signal. Averaged by settlement month; HAC
           (Newey-West, 2 lags) t on the monthly series; one-sided p for b_B > 0.
Decisions  holdout (settled >= 2018): R1 = S_P, R2 = S_Q; Holm at 5% over {R1, R2}.
           Bates earns a role in option selection only if a signal passes Holm, has the
           same sign in discovery, and its tradable long-short is positive in holdout:
           within cross-section, residualise S_B on S_N (+ cells), buy the top third,
           sell the bottom third, Rs 100 premium per option; LS_net = mean hbuy_net(long)
           + mean hsell_net(short) (Stage 1 costs; also stress).
           Otherwise Bates stays descriptive-only for rich/cheap.
Descriptive  stand-alone rank ICs of S_Q, S_P, S_N; unhedged (buy_gross) version; eras.

Validity check (added after the first run, before looking at it): signal and outcome
share the entry mark, so a noisy close would make any mark-based signal look predictive.
Skip test: signal on session t, outcome bought at the contract's next-session traded
mark, hedged from t+1 with the same entry-IV deltas (Stage 1 hedge minus its first step):
y_skip = (payoff - mark_{t+1} + hedge - h_1) / mark_{t+1} * 100. If the Bates partial slope
dies here, the main result is a microstructure artifact.
Exploratory (post hoc): for someone who sells anyway, hedged seller net return of the
Bates-rich vs Bates-cheap third (residual signal).

Exploratory hedge benchmark (not a decision): engine straddles re-hedged with the
sticky-moneyness smile delta (Black-76 delta + vega x dsigma/dF, dsigma/dF = -sigma'(k)/F)
vs the Black-76 smile-IV delta and Bates MV delta already run in Stage 5 (H1).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import backtest as B  # noqa: E402
from src import contracts as ct  # noqa: E402

S1 = os.path.join(PROJECT_ROOT, "data", "processed", "stage1", "trades.parquet")
S4 = os.path.join(PROJECT_ROOT, "data", "processed", "stage4", "attrib.parquet")
D5 = os.path.join(PROJECT_ROOT, "data", "processed", "stage5")
HOLDOUT = pd.Timestamp("2018-01-01")
MIN_N = 12
LINES: list[str] = []


def out(s=""):
    print(s)
    LINES.append(str(s))


def hac(x: np.ndarray, lags=2):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return np.nan, np.nan, np.nan, n
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    t = x.mean() / np.sqrt(v / n)
    from scipy.stats import norm
    return x.mean(), t, 1 - norm.cdf(t), n


def rk(x: np.ndarray) -> np.ndarray:
    return (pd.Series(x).rank().to_numpy() - 0.5) / len(x) - 0.5


def demean(x: np.ndarray, cell: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return (s - s.groupby(cell).transform("mean")).to_numpy()


def load() -> pd.DataFrame:
    a = pd.read_parquet(S4)
    t = pd.read_parquet(S1, columns=["tid", "iv", "T", "hbuy_gross", "hbuy_net", "hsell_net", "hbuy_net_stress",
                                     "hsell_net_stress", "buy_gross"])
    a = a.merge(t, on="tid", how="left")
    a = a[(a["mark"] > 0) & (a["q_a"] > 0) & (a["q_jc"] > 0) & (a["T"] > 0) & a["iv"].notna() & a["hbuy_gross"].notna()]
    a["S_Q"] = np.log(a["q_a"] / a["mark"])
    a["S_P"] = np.log(a["q_jc"] / a["mark"])
    a["S_N"] = np.sqrt(a["p_naive"] / a["T"]) - a["iv"]
    a["cell"] = a["side"].astype(str) + a["dbucket"].astype(str)
    a = a[np.isfinite(a[["S_Q", "S_P", "S_N"]]).all(axis=1)]
    return a


def add_skip(a: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    import stage1_option_returns as s1
    paths = s1.forward_paths()
    paths = paths.sort_values(["symbol", "final_expiry", "date"])
    paths["d1"] = paths.groupby(["symbol", "final_expiry"])["date"].shift(-1)
    paths["F1"] = paths.groupby(["symbol", "final_expiry"])["forward"].shift(-1)
    a = a.merge(paths[["symbol", "final_expiry", "date", "d1", "F1"]], on=["symbol", "final_expiry", "date"], how="left")
    em = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "processed", "contracts", "_expiry_map.parquet"))
    q = []
    for sym in ("NIFTY", "BANKNIFTY"):
        x = pd.read_parquet(B.PANEL, columns=["date", "expiry", "strike", "side", "mark", "contracts"],
                            filters=[("symbol", "=", sym), ("dte", "<=", 70), ("contracts", ">=", ct.MIN_TRADED)])
        x["symbol"] = sym
        q.append(x)
    q = pd.concat(q).merge(em[["symbol", "expiry", "final_expiry"]], on=["symbol", "expiry"])
    q = q[q["mark"] > 0].drop_duplicates(["symbol", "final_expiry", "date", "strike", "side"])
    q = q.rename(columns={"date": "d1", "mark": "mark1"})[["symbol", "final_expiry", "d1", "strike", "side", "mark1"]]
    a = a.merge(q, on=["symbol", "final_expiry", "d1", "strike", "side"], how="left")
    t = pd.read_parquet(S1, columns=["tid", "hedge", "forward", "delta"]).rename(columns={"delta": "delta_s1"})
    a = a.merge(t, on="tid", how="left")
    h1 = -a["delta_s1"] * (a["F1"] - a["forward"])
    ok = a["mark1"].notna() & (a["d1"] < a["settled"])
    a["y_skip"] = np.where(ok, (a["payoff"] - a["mark1"] + a["hedge"] - h1) / a["mark1"] * 100, np.nan)
    return a


TAGS = {"hbuy_gross": "", "buy_gross": "_u", "y_skip": "_s"}


def cross_sections(a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sym, d, fe), g in a.groupby(["symbol", "date", "final_expiry"], sort=False):
        if len(g) < MIN_N:
            continue
        cell = g["cell"].to_numpy()
        row = dict(symbol=sym, date=d, settled=g["settled"].iloc[0], n=len(g))
        rN = demean(rk(g["S_N"].to_numpy()), cell)
        for yname, tag in TAGS.items():
            gg = g if yname != "y_skip" else g[g["y_skip"].notna()]
            if len(gg) < MIN_N:
                continue
            c = gg["cell"].to_numpy()
            rN = demean(rk(gg["S_N"].to_numpy()), c)
            ry = demean(rk(gg[yname].to_numpy()), c)
            if ry @ ry == 0:
                continue
            row[f"ic_N{tag}"] = np.corrcoef(ry, rN)[0, 1] if rN @ rN > 0 else np.nan
            for sb in ("S_Q", "S_P"):
                rB = demean(rk(gg[sb].to_numpy()), c)
                X = np.column_stack([rB, rN])
                if np.linalg.matrix_rank(X) < 2:
                    continue
                b = np.linalg.lstsq(X, ry, rcond=None)[0]
                row[f"b_{sb}{tag}"] = b[0]
                row[f"bN_{sb}{tag}"] = b[1]
                row[f"ic_{sb}{tag}"] = np.corrcoef(ry, rB)[0, 1]
        for sb in ("S_Q", "S_P"):
            xB = demean(g[sb].to_numpy(), cell)
            xN = demean(g["S_N"].to_numpy(), cell)
            e = xB - (xB @ xN / (xN @ xN)) * xN if xN @ xN > 0 else xB
            q = pd.Series(e).rank(pct=True).to_numpy()
            top, bot = q > 2 / 3, q <= 1 / 3
            if top.sum() < 2 or bot.sum() < 2:
                continue
            row[f"ls_gross_{sb}"] = g["hbuy_gross"].to_numpy()[top].mean() - g["hbuy_gross"].to_numpy()[bot].mean()
            row[f"ls_net_{sb}"] = g["hbuy_net"].to_numpy()[top].mean() + g["hsell_net"].to_numpy()[bot].mean()
            row[f"ls_stress_{sb}"] = (g["hbuy_net_stress"].to_numpy()[top].mean()
                                      + g["hsell_net_stress"].to_numpy()[bot].mean())
            hs = g["hsell_net"].to_numpy()
            row[f"sell_rich_{sb}"], row[f"sell_cheap_{sb}"] = hs[bot].mean(), hs[top].mean()
            row[f"sell_all_{sb}"] = hs.mean()
            ys = g["y_skip"].to_numpy()
            if np.isfinite(ys[top]).sum() >= 2 and np.isfinite(ys[bot]).sum() >= 2:
                row[f"ls_skip_{sb}"] = np.nanmean(ys[top]) - np.nanmean(ys[bot])
        rows.append(row)
    return pd.DataFrame(rows)


def monthly(cs: pd.DataFrame, col: str, mask) -> pd.Series:
    x = cs.loc[mask & cs[col].notna()]
    return x.groupby(x["settled"].dt.to_period("M"))[col].mean()


def fmt(m, t, p, n):
    return f"{m:+.4f} (t {t:+.2f}, p {p:.3f}, {n} mo)"


def report(cs: pd.DataFrame):
    disc, hold = cs["settled"] < HOLDOUT, cs["settled"] >= HOLDOUT
    out("=" * 100 + "\nRICH/CHEAP: Bates rankings vs naive IV-minus-RV, within side x delta-bucket cells\n" + "=" * 100)
    out(f"cross-sections: {len(cs)} (discovery {disc.sum()}, holdout {hold.sum()}); options per cross-section median "
        f"{cs['n'].median():.0f}")
    out("\nStand-alone rank IC (hedged buyer return): + = signal's 'cheap' options earned more")
    for c in ("ic_N", "ic_S_Q", "ic_S_P"):
        out(f"  {c:7s} discovery {fmt(*hac(monthly(cs, c, disc)))} | holdout {fmt(*hac(monthly(cs, c, hold)))}")
    out("\nPartial rank slope beyond the naive signal (primary; b_B > 0 = Bates adds information)")
    res = {}
    for r, sb in (("R1", "S_P"), ("R2", "S_Q")):
        dd, hh = hac(monthly(cs, f"b_{sb}", disc)), hac(monthly(cs, f"b_{sb}", hold))
        nd, nh = hac(monthly(cs, f"bN_{sb}", disc)), hac(monthly(cs, f"bN_{sb}", hold))
        res[r] = (sb, dd, hh)
        out(f"  {r} {sb}: Bates b  discovery {fmt(*dd)} | holdout {fmt(*hh)}")
        out(f"         naive b  discovery {fmt(*nd)} | holdout {fmt(*nh)}")
    out("\nTradable long-short (residual Bates signal, top vs bottom third), Rs per Rs100 premium per leg")
    for sb in ("S_P", "S_Q"):
        for c in ("ls_gross", "ls_net", "ls_stress"):
            col = f"{c}_{sb}"
            out(f"  {sb} {c:9s} discovery {fmt(*hac(monthly(cs, col, disc)))} | holdout {fmt(*hac(monthly(cs, col, hold)))}")
    out("\nVALIDITY: one-session skip (signal at t, bought at t+1's traded mark)")
    out(f"  cross-sections with >= {MIN_N} skip outcomes: {cs['b_S_P_s'].notna().sum()}")
    for c in ("ic_N_s", "ic_S_Q_s", "ic_S_P_s", "b_S_P_s", "bN_S_P_s", "b_S_Q_s", "bN_S_Q_s"):
        out(f"  {c:9s} discovery {fmt(*hac(monthly(cs, c, disc)))} | holdout {fmt(*hac(monthly(cs, c, hold)))}")
    for sb in ("S_P", "S_Q"):
        out(f"  {sb} gross long-short, skip basis: discovery {fmt(*hac(monthly(cs, f'ls_skip_{sb}', disc)))}"
            f" | holdout {fmt(*hac(monthly(cs, f'ls_skip_{sb}', hold)))}")
    out("\nEXPLORATORY (post hoc): hedged SELLER net return per Rs100 by residual Bates third")
    for sb in ("S_P", "S_Q"):
        for lab, m in (("discovery", disc), ("holdout", hold)):
            r_, c_, a_ = (hac(monthly(cs, f"sell_{k}_{sb}", m))[0] for k in ("rich", "cheap", "all"))
            d_ = (cs[f"sell_rich_{sb}"] - cs[f"sell_cheap_{sb}"])
            out(f"  {sb} {lab}: sell rich third {r_:+.2f} | all {a_:+.2f} | cheap third {c_:+.2f};"
                f" rich - cheap {fmt(*hac(monthly(cs.assign(_d=d_), '_d', m)))}")
    out("\nUnhedged outcome (descriptive)")
    for c in ("ic_N_u", "ic_S_Q_u", "ic_S_P_u", "b_S_P_u", "b_S_Q_u"):
        out(f"  {c:9s} discovery {fmt(*hac(monthly(cs, c, disc)))} | holdout {fmt(*hac(monthly(cs, c, hold)))}")
    out("\nBy era (partial Bates slope, hedged)")
    eras = [("2008-2012", "2008", "2013"), ("2013-2017", "2013", "2018"), ("2018-2021", "2018", "2022"),
            ("2022-now", "2022", "2100")]
    for lab, a, b in eras:
        m = (cs["settled"] >= a) & (cs["settled"] < b)
        out(f"  {lab}: S_P {fmt(*hac(monthly(cs, 'b_S_P', m)))} | S_Q {fmt(*hac(monthly(cs, 'b_S_Q', m)))}"
            f" | naive IC {fmt(*hac(monthly(cs, 'ic_N', m)))}")
    out("\nDECISION (holdout, Holm 5% over R1/R2)")
    order = sorted(res, key=lambda r: res[r][2][2])
    alive = True
    for i, r in enumerate(order):
        sb, dd, hh = res[r]
        thr = 0.05 / (len(order) - i)
        ok = alive and np.isfinite(hh[2]) and hh[2] < thr
        alive = ok
        ls = hac(monthly(cs, f"ls_net_{sb}", hold))
        role = ok and dd[0] > 0 and ls[0] > 0
        out(f"  {r} {sb}: holdout p {hh[2]:.3f} vs Holm {thr:.3f} -> {'PASS' if ok else 'fail'}; discovery sign "
            f"{'+' if dd[0] > 0 else '-'}; holdout LS net {ls[0]:+.2f} -> Bates role in selection: {'YES' if role else 'no'}")


def smile_delta_check():
    out("\n" + "=" * 100 + "\nEXPLORATORY hedge benchmark: sticky-moneyness smile delta vs Black-76 smile-IV delta\n" + "=" * 100)
    pos = pd.read_parquet(os.path.join(D5, "positions.parquet"))
    day = pd.read_parquet(os.path.join(D5, "daily.parquet"))
    st = B.spread_table(os.path.join(D5, "_spread_table.parquet"))
    costs = [B.Costs(st)]
    h = 0.01

    def smile_delta(mk):
        def f(d, fe, K, sides, s):
            K = np.asarray(K, float)
            ic = np.asarray(sides) == "CE"
            if s.T <= 0:
                return None
            sig = mk.iv_at(s, K)
            dl, _, vg, _ = ct.b76_greeks(s.F, K, s.T, sig, s.DF, 0.0, ic)
            slope = (mk.iv_at(s, K * np.exp(h)) - mk.iv_at(s, K * np.exp(-h))) / (2 * h)
            return dl + vg * 100 * (-slope / s.F)
        return f

    res = []
    for sym in ("NIFTY", "BANKNIFTY"):
        mk = B.Market(sym)
        mk.settled = {f: s for f, s in mk.settled.items() if s <= mk.sessions[-1]}
        p = pos[(pos.symbol == sym) & (pos.strategy == "eng_straddle_bs")]
        for _, r in p.iterrows():
            K = np.array([float(k) for k in r.strikes.split(",")])
            sd = np.array(r.sides.split(","))
            q = -r.qty * np.ones(len(K))
            ps = B.Position("eng_straddle_sm", sym, r.entry, r.exit, r.final_expiry, K, sd, q, "mv")
            x = B.simulate(ps, mk, costs, smile_delta(mk))
            if len(x):
                x["pid"] = r.pid
                res.append(x)
    sm = pd.concat(res, ignore_index=True)
    base = day[day.strategy.isin(["eng_straddle_bs", "eng_straddle_mv"])]
    tot = lambda g: (g.option + g.hedge).groupby(g.date).sum()
    series = {"bs": tot(base[base.strategy == "eng_straddle_bs"]), "mv": tot(base[base.strategy == "eng_straddle_mv"]),
              "smile": tot(sm)}
    df = pd.DataFrame(series).dropna()
    for lab, m in (("discovery", df.index < HOLDOUT), ("holdout", df.index >= HOLDOUT)):
        x = df[m]
        out(f"  {lab}: daily hedged P&L sd (gross, book units) BS {x.bs.std():.5f} | Bates MV {x.mv.std():.5f} | "
            f"sticky-moneyness {x.smile.std():.5f}; mean BS {x.bs.mean() * 252:+.4f}/yr, smile {x.smile.mean() * 252:+.4f}/yr")
        d = (x.smile ** 2 - x.bs ** 2).resample("ME").sum()
        out(f"    squared-P&L difference smile - BS (monthly, HAC): {fmt(*hac(d.to_numpy()))}  (+ = smile delta worse)")


def main():
    a = add_skip(load())
    out(f"options in test: {len(a):,}; with a next-session traded mark before settlement: {a['y_skip'].notna().mean():.1%}")
    cs = cross_sections(a)
    cs.to_parquet(os.path.join(D5, "richcheap_cs.parquet"))
    report(cs)
    smile_delta_check()
    with open(os.path.join(D5, "_richcheap.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
