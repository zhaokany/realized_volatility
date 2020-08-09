import math
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from rv.datamodels.daily_movement import OneDay
from rv.preprocessing import MINUTE_IN_TRADING_DAY

RETURN = 'return'
CUM_RETURN = "CUM_RETURN"
DAYS_IN_A_YEAR = 365
TRADING_DAY_IN_YEAR = 252
MINUTE_IN_HALF_TRADING_DAY = int(MINUTE_IN_TRADING_DAY / 2)
LAGS = [1, 5, 10, 15, 30, 60, 120, 195]
EWMA_SPAN = 22


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
        self.n_pca_components = 3
        self.variance_ratio_threshold = 0.95

    # def daily_return_samples(self, one_day_data: OneDay, sample_interval: int=2):
    #     assert sample_interval >= 1, "sample interval must be bigger than 1"
    #     intraday_data = one_day_data.get_intraday_prices()
    #     sampled_data = np.array(intraday_data[::sample_interval])
    #     sampled_returns = np.diff(sampled_data) / sampled_data[:-1]
    #     return sampled_returns
    #
    # def total_variance(self, daily_data: 'OrderedDict[int, OneDay]', start_index: int, end_index: int, sample_interval=1):
    #     total_returns = [self.daily_return_samples(daily_data[i], sample_interval) for i in range(start_index, end_index)]
    #     total_returns = np.concatenate(total_returns)
    #     return np.var(winsorize(total_returns, self.winsozir_limits).data) * len(total_returns)
    #
    # def realized_vol(self, daily_data: 'OrderedDict[int, OneDay]', end_index: int, tenor_in_days: int,
    #                         intraday_fraction: float,
    #                         sample_interval: int=1):
    #     start_index = end_index - tenor_in_days
    #     assert start_index >= 0, "start_index can't be negative"
    #     variance = self.total_variance(daily_data, start_index, end_index, sample_interval)
    #     time = intraday_fraction * tenor_in_days / DAYS_IN_A_YEAR
    #     return math.sqrt(variance / time)
    #
    # def realized_vols(self, daily_data: 'OrderedDict[int, OneDay]', tenor_in_days: int,
    #                             intraday_fraction: float,
    #                             sample_interval: int = 1):
    #     min_end_index = tenor_in_days
    #     max_end_index = len(daily_data)
    #     realized_volatilities = []
    #     for end_index in range(min_end_index, max_end_index):
    #         rv = self.realized_vol(daily_data, end_index, tenor_in_days, intraday_fraction, sample_interval)
    #         realized_volatilities.append(rv)
    #     return realized_volatilities

    def calculate_realized_volatilities(self, daily_data: 'OrderedDict[int, OneDay]', intraday_fraction: float) -> pd.DataFrame:
        realized_variances = []

        for _, one_day in daily_data.items():
            prices_and_returns = one_day.intraday_prices
            column_names = [f"SAMPLE_{tau}" for tau in LAGS]
            for i, tau in enumerate(LAGS):
                prices_and_returns[column_names[i]] = prices_and_returns[RETURN].shift(-tau - 1)
            covariance_matrix = prices_and_returns[column_names].dropna().cov().iloc[0, :]
            overall_variance = np.var(prices_and_returns[RETURN])
            realized_variances_per_day = []
            for i in range(len(LAGS)):
                summed_covariances = 0.0
                for j in range(1, i + 1):
                    summed_covariances += (1.0 - j / i) * covariance_matrix.iloc[j - 1]
                realized_variance = overall_variance + 2.0 * summed_covariances
                realized_variances_per_day.append(realized_variance)
            realized_variances.append(realized_variances_per_day)

        realized_variances = np.array(realized_variances)
        means = np.mean(realized_variances, axis=0)
        stds = np.std(realized_variances, axis=0)
        standardized_rv = StandardScaler().fit_transform(np.array(realized_variances))
        pca = PCA(n_components=self.n_pca_components)
        principal_components = pca.fit_transform(standardized_rv)
        assert np.sum(pca.explained_variance_ratio_) > self.variance_ratio_threshold, "Explained variance ratio too small"
        emwa_principal_components = pd.DataFrame(principal_components).ewm(span=EWMA_SPAN).mean().loc[len(principal_components)-1].values
        fitted_standarzied_varianced = pca.components_.T.dot(emwa_principal_components)
        fitted_variances = np.multiply(fitted_standarzied_varianced, stds) + means
        vol_times = [tau / MINUTE_IN_TRADING_DAY * intraday_fraction / TRADING_DAY_IN_YEAR for tau in LAGS]
        fitted_realized_volatility = np.sqrt(fitted_variances / vol_times)
        return fitted_realized_volatility