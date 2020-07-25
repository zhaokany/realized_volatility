import math
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.stats.mstats import winsorize

from rv.datamodels.daily_movement import OneDay

DAYS_IN_A_YEAR = 365

class VolatilitySignatureCalculator:
    def __init__(self, max_lag):
        self.max_lag = max_lag

    def calculate(self, data: pd.DataFrame, overall_volatility: float):
        overall_variance = overall_volatility**2
        df = data.copy()
        lags = range(1, self.max_lag, 1)
        for lag in lags:
            column_name = "return_{}".format(lag)
            df[column_name] = df['return'].shift(lag)
        df.dropna(inplace=True)
        covariances = df.cov().iloc[0]
        volatility_signatures = []
        for i in lags:
            cumulative_sum = 0.0
            for j in range(1, i):
                cumulative_sum += (1.0 - j / i) * covariances.iloc[j]
            volatility_signatures.append(np.sqrt(overall_variance + 2.0 * cumulative_sum))
        return volatility_signatures

class RealizedVolatilityCalculator:
    def __init__(self, winsorize_limits):
        self.winsozir_limits = winsorize_limits

    def daily_return_samples(self, one_day_data: OneDay, sample_interval: int=2):
        assert sample_interval >= 1, "sample interval must be bigger than 1"
        intraday_data = one_day_data.get_intraday_prices()
        sampled_data = np.array(intraday_data[::sample_interval])
        sampled_returns = np.diff(sampled_data) / sampled_data[:-1]
        return sampled_returns

    def total_variance(self, daily_data: 'OrderedDict[int, OneDay]', start_index: int, end_index: int, sample_interval=1):
        total_returns = [self.daily_return_samples(daily_data[i], sample_interval) for i in range(start_index, end_index)]
        total_returns = np.concatenate(total_returns)
        return np.var(winsorize(total_returns, self.winsozir_limits).data) * len(total_returns)

    def realized_vol(self, daily_data: 'OrderedDict[int, OneDay]', end_index: int, tenor_in_days: int,
                            intraday_fraction: float,
                            sample_interval: int=1):
        start_index = end_index - tenor_in_days
        assert start_index >= 0, "start_index can't be negative"
        variance = self.total_variance(daily_data, start_index, end_index, sample_interval)
        time = intraday_fraction * tenor_in_days / DAYS_IN_A_YEAR
        return math.sqrt(variance / time)

    def realized_vols(self, daily_data: 'OrderedDict[int, OneDay]', tenor_in_days: int,
                                intraday_fraction: float,
                                sample_interval: int = 1):
        min_end_index = tenor_in_days
        max_end_index = len(daily_data)
        realized_volatilities = []
        for end_index in range(min_end_index, max_end_index):
            rv = self.realized_vol(daily_data, end_index, tenor_in_days, intraday_fraction, sample_interval)
            realized_volatilities.append(rv)
        return realized_volatilities