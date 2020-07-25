from collections import OrderedDict
from typing import Tuple
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

from rv.datamodels.daily_movement import OneDay, DAY, TIMESTR, PRICE
from rv.calculator import RealizedVolatilityCalculator

RETURN = 'return'

def drop_zeros(df: pd.DataFrame) -> pd.DataFrame:
    df_tmp = df.replace(0, np.nan)
    return df_tmp.dropna()

def prepare(data_file_path: str, column_name: str) -> 'OrderedDict[int, OneDay]':
    raw_data = pd.read_csv(data_file_path)
    stock_data = raw_data[[DAY, TIMESTR, column_name]].groupby(DAY)
    daily_data = OrderedDict()
    previous_open = None
    previous_close = None
    counter = 0
    for group, grouped in stock_data:
        open = grouped.iloc[0][column_name]
        close = grouped.iloc[-1][column_name]

        if not np.isnan(open) and not np.isnan(close):
            if previous_open is not None and previous_close is not None:
                if not np.isnan(previous_open) and not np.isnan(previous_close):
                    grouped_data = drop_zeros(grouped)
                    intraday_data = grouped_data.drop(columns=[DAY])

                    hourly_data = grouped_data[grouped_data[TIMESTR].str.contains(":00:00")][column_name].values

                    one_day = OneDay(open, close, previous_open, previous_close, intraday_data.rename(columns={
                        TIMESTR: TIMESTR,
                        column_name: PRICE,
                    }), hourly_data)
                    daily_data[counter] = one_day
                    counter += 1
            previous_open = open
            previous_close = close

    return daily_data

def prepare_minute(daily_data: 'OrderedDict[int, OneDay]') -> pd.DataFrame:
    minute_data = []

    for _, one_day in daily_data.items():
        minute_data.append(one_day.intraday_prices[PRICE].values)
    df = pd.DataFrame({PRICE: np.concatenate(minute_data)})
    df[RETURN] = np.log(df[PRICE]).diff()
    return df.dropna()

def prepared_hourly(daily_data: 'OrderedDict[int, OneDay]', overnight_fraction: float) -> pd.DataFrame:
    overnight_hourly_time_increment = np.ceil(overnight_fraction * 24.0)
    hourly_price = []
    hourly_time_increment = []

    for _, one_day in daily_data.items():
        hourly_price.append(one_day.hourly_prices)
        houly_time_increment_per_day = np.ones(len(one_day.hourly_prices))
        houly_time_increment_per_day[0] = overnight_hourly_time_increment
        hourly_time_increment.append(houly_time_increment_per_day)
    df = pd.DataFrame({'hour_increment': np.concatenate(hourly_time_increment), PRICE: np.concatenate(hourly_price)})
    df[RETURN] = np.log(df[PRICE]).diff()
    return df.dropna()


def prepare_daily(daily_data: 'OrderedDict[int, OneDay]') -> pd.DataFrame:
    daily_return = []

    for _, one_day in daily_data.items():
        daily_return.append(np.log(one_day.close_price / one_day.open_price))
    df = pd.DataFrame({RETURN: daily_return})
    return df.dropna()

def calculate_intraday_fraction(daily_data: 'OrderedDict[int, OneDay]', limits: float) -> Tuple[float, float]:
    open_to_close_returns = [(v.close_price - v.open_price) / v.open_price for k, v in daily_data.items()]
    close_to_open_returns = [(v.open_price - v.previous_close) / v.previous_close for k, v in daily_data.items()]

    open_to_close_variances = np.var(winsorize(open_to_close_returns, limits).data)
    close_to_open_variances = np.var(winsorize(close_to_open_returns, limits).data)
    intraday_fraction = open_to_close_variances / (open_to_close_variances + close_to_open_variances)
    overnight_fraction = 1.0 - intraday_fraction
    return intraday_fraction, overnight_fraction

def calculate_realized_vols(calculator: RealizedVolatilityCalculator, daily_data: 'OrderedDict[int, OneDay]',
                           intraday_fraction: float, tenor_in_days: int, sample_interval: int = 1):
    min_end_index = tenor_in_days
    max_end_index = len(daily_data)
    realized_volatilities = []
    for end_index in range(min_end_index, max_end_index):
        rv = calculator.realized_vol(daily_data, end_index, tenor_in_days, intraday_fraction, sample_interval)
        realized_volatilities.append(rv)
    return pd.DataFrame({'rv_0': realized_volatilities})

def generate_rv_features(df: pd.DataFrame, tenor_in_days: int):
    for i in range(0, 50):
        df[f'f{i}'] = df['rv_0'].shift(periods=10+i)
    return df.dropna().rename(columns={'rv_0': 'y'})
