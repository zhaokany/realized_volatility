import math
import pandas as pd
import numpy as np
from rv.utils.ts_plot import tsplot
from scipy.stats.mstats import winsorize

DATA_FILE_PATH = "../../data/stockdata3.csv"

if __name__ == "__main__":
    raw_data = pd.read_csv(DATA_FILE_PATH)
    raw_xs = raw_data[["a"]]
    raw_xs["log"] = np.log(raw_xs["a"])
    raw_xs["return"] = raw_xs["log"].diff()
    xs = winsorize(raw_xs["return"].loc[1:20000].values, 0.01).data
    tsplot(xs, lags=30)