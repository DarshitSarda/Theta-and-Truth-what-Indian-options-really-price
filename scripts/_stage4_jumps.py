import os
import numpy as np
import pandas as pd
from scipy import stats

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "stage4")
s = pd.read_parquet(os.path.join(D, "pstate.parquet"))
for sym in ["NIFTY", "BANKNIFTY"]:
    g = pd.read_parquet(os.path.join(D, f"garch_params_{sym}.parquet"))
    print(sym, "nu median", round(g.nu.median(), 2), "persistence", round((g.alpha + g.beta + g.gamma / 2).median(), 4))
    x = s[s.symbol == sym].copy()
    z = x.r / np.sqrt(x.sigma2)
    nu = g.nu.median()
    for c in [3.0, 3.5, 4.0, 5.0]:
        sd = np.sqrt(nu / (nu - 2))
        exp_rate = 2 * stats.t.sf(c * sd, nu)
        j = x[np.abs(z) > c]
        print(f"  c={c}: n={len(j)}  per yr={len(j) / (len(x) / 250):.2f}  t-model expects/yr={exp_rate * 250:.2f}  "
              f"first3={list(j.date.dt.date[:3])}")
    print("  |z| top 12:", x.assign(z=z).reindex(z.abs().sort_values(ascending=False).index)[["date", "r", "z"]].head(12).round(3).to_string())
    print("  std(z) by year:", z.groupby(x.date.dt.year).std().round(2).to_dict())
