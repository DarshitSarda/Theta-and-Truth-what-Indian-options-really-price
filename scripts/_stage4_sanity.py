import os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "stage4")
pd.set_option("display.width", 220, "display.max_columns", 40)
o = pd.read_parquet(os.path.join(D, "outcomes.parquet"))
print(o.isna().mean().round(3)[lambda s: s > 0])
print("first date with p_total / cal:", o.dropna(subset=["p_total"]).groupby("symbol").date.min().to_dict(),
      o.dropna(subset=["cal"]).groupby("symbol").date.min().to_dict())
print("cal by year:", o.groupby([o.symbol, o.date.dt.year]).cal.median().unstack(0).round(3).to_string())
print("floor counts:", o.floor_b.sum(), o.floor_bc.sum())
o["qv"] = o.q_diff + o.q_jump + o.q_event
print("sum-ratio Q/RV, P/RV, Pcal/RV by symbol,target:")
g = o.dropna(subset=["p_cal"]).groupby(["symbol", "target"])
print(g.apply(lambda x: pd.Series(dict(n=len(x), q=x.qv.sum() / x.rv.sum(), p=x.p_total.sum() / x.rv.sum(),
                                        pc=x.p_cal.sum() / x.rv.sum(), jshare_q=(x.q_jump + x.q_event).sum() / x.qv.sum(),
                                        jshare_rv4=x["rv_jump_4.0"].sum() / x.rv.sum()))).round(3))
for c in ["u_q", "u_b", "u_bc"]:
    print(c, o[c].quantile([.01, .05, .25, .5, .75, .95, .99]).round(4).to_dict())
a = pd.read_parquet(os.path.join(D, "attrib.parquet"))
print(len(a), a.isna().mean().round(3)[lambda s: s > 0])
for c in ["q_b", "q_bc"]:
    print(c, "/q_a quantiles", (a[c] / a.q_a).quantile([.05, .25, .5, .75, .95]).round(3).to_dict())
e = pd.read_parquet(os.path.join(D, "events.parquet"))
e["q_ev_var"] = e.q_day_var + e.psi ** 2
print(e.round(5).to_string())
