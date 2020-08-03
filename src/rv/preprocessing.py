from collections import OrderedDict
from typing import Tuple
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from scipy import stats



from rv.datamodels.daily_movement import OneDay, DAY, TIMESTR, PRICE

RETURN = 'return'
MINUTE_IN_TRADING_DAY = 60 * (16 - 9.5) + 1


def fill_zeros_with_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    df_tmp = df.replace(0, np.nan)
    return df_tmp.interpolate()


# def drop_zeros(df: pd.DataFrame) -> pd.DataFrame:
#     df_tmp = df.replace(0, np.nan)
#     return df_tmp.dropna()


def prepare(df: pd.DataFrame) -> 'OrderedDict[int, OneDay]':
    stock_data = df.groupby(DAY)
    daily_data = OrderedDict()
    previous_open = None
    previous_close = None
    counter = 0
    for group, grouped in stock_data:
        grouped = fill_zeros_with_interpolation(grouped)
        open = grouped.iloc[0][PRICE]
        close = grouped.iloc[-1][PRICE]

        if not np.isnan(open) and not np.isnan(close):
            if previous_open is not None and previous_close is not None:
                if not np.isnan(previous_open) and not np.isnan(previous_close):
                    intraday_data = grouped.drop(columns=[DAY])
                    if len(intraday_data) == MINUTE_IN_TRADING_DAY:
                        hourly_data = grouped[grouped[TIMESTR].str.contains(":00:00")][PRICE].values
                        one_day = OneDay(open, close, previous_open, previous_close, intraday_data, hourly_data)
                        daily_data[counter] = one_day
                        counter += 1
            previous_open = open
            previous_close = close

    return daily_data


def clean_data(daily_data: 'OrderedDict[int, OneDay]') -> 'OrderedDict[int, OneDay]':
    cleaned_data = remove_intraday_outliers(remove_overnight_jumps(daily_data))
    print("Data cleaned")
    return cleaned_data


def remove_overnight_jumps(daily_data: 'OrderedDict[int, OneDay]') -> 'OrderedDict[int, OneDay]':
    cum_adjustment = 0.0

    overnight_returns = [np.log(v.open_price / v.previous_close) for k, v in daily_data.items()]
    stdev = np.std(overnight_returns)
    stdev_threshold = stdev * 1.5
    relative_threahold = 0.0
    for day, daily in daily_data.items():
        adjusted_intraday_prices = daily.intraday_prices
        if cum_adjustment != 0.0:
            daily.open_price += cum_adjustment
            daily.close_price += cum_adjustment
            daily.previous_open += cum_adjustment
            daily.previous_close += cum_adjustment
        ratio = daily.open_price / daily.previous_close
        if np.abs(ratio - 1.0) > relative_threahold and np.abs(np.log(ratio)) > stdev_threshold:
            new_adjustment = daily.previous_close - daily.open_price
            daily.open_price += new_adjustment
            daily.close_price += new_adjustment
            cum_adjustment += new_adjustment
            print(f"new_adjustment/cum_adjustment at day {day} with amount {new_adjustment}/{cum_adjustment}")
        adjusted_intraday_prices[PRICE] += cum_adjustment
        daily.intraday_prices = adjusted_intraday_prices
    return daily_data


def remove_intraday_outliers(daily_data: 'OrderedDict[int, OneDay]') -> 'OrderedDict[int, OneDay]':
    for day, daily in daily_data.items():
        df = daily.intraday_prices.loc[daily.intraday_prices[PRICE] > 0.0]
        mean = df[PRICE].mean()
        stdev = df[PRICE].std()
        df2 = df.loc[((df[PRICE]-mean) / stdev).abs() < 3.0].copy()
        df2.loc[:, "log_price"] = np.log(df2[PRICE])
        df2.loc[:, "prev_log_price"] = df2["log_price"].shift(1)
        df2.loc[:, RETURN] = df2["log_price"] - df2["prev_log_price"]
        daily.intraday_prices = df2.dropna()
    return daily_data


def prepare_minute(daily_data: 'OrderedDict[int, OneDay]') -> pd.DataFrame:
    minute_data = []

    for _, one_day in daily_data.items():
        df = one_day.intraday_prices
        df[RETURN] = np.log(df[PRICE]).diff()
        minute_data.append(df[RETURN].dropna().values)
    return np.concatenate(minute_data)


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


# def calculate_realized_vols(calculator: RealizedVolatilityCalculator, daily_data: 'OrderedDict[int, OneDay]',
#                            intraday_fraction: float, tenor_in_days: int, sample_interval: int = 1):
#     min_end_index = tenor_in_days
#     max_end_index = len(daily_data)
#     realized_volatilities = []
#     for end_index in range(min_end_index, max_end_index):
#         rv = calculator.realized_vol(daily_data, end_index, tenor_in_days, intraday_fraction, sample_interval)
#         realized_volatilities.append(rv)
#     return pd.DataFrame({'rv_0': realized_volatilities})


def generate_rv_features(df: pd.DataFrame, tenor_in_days: int):
    for i in range(0, 50):
        df[f'f{i}'] = df['rv_0'].shift(periods=10+i)
    return df.dropna().rename(columns={'rv_0': 'y'})
