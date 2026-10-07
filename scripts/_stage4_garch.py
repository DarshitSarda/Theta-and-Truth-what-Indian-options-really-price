import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "processed", "stage4")
s = pd.read_parquet(os.path.join(D, "pstate.parquet"))
for sym in ["NIFTY", "BANKNIFTY"]:
    g = pd.read_parquet(os.path.join(D, f"garch_params_{sym}.parquet"))
    phi = g.alpha + g.beta + g.gamma / 2
    g["lr_vol"] = np.sqrt(g.omega / (1 - phi) * 252) / 100
    g["phi"] = phi
    print(sym, g.iloc[::12][["month", "omega", "alpha", "gamma", "beta", "nu", "phi", "lr_vol"]].round(4).to_string())
    x = s[s.symbol == sym]
    z2 = x.eps ** 2 / x.sigma2
    print(" mean z^2 by year (1-day forecast bias):", z2.groupby(x.date.dt.year).mean().round(2).to_dict())
    print(" realised ann vol by year:", (np.sqrt((x.r ** 2).groupby(x.date.dt.year).mean() * 252)).round(3).to_dict())
    print(" mean GARCH ann vol by year:", (np.sqrt(x.sigma2.groupby(x.date.dt.year).mean() * 252)).round(3).to_dict())

print(pq.read_schema(os.path.join(ROOT, "data", "processed", "contracts", "panel")).names)
y = pd.read_csv(os.path.join(ROOT, "data", "raw", "underlying", "nifty.csv"), parse_dates=["date"])
y["gap"] = np.log(y.open / y.close.shift())
print("NIFTY yahoo open==prev close share by year:", (y.gap.abs() < 1e-6).groupby(y.date.dt.year).mean().round(2).to_dict())
