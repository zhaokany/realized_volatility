import pandas as pd

from rv.datamodels.daily_movement import OneDay, DAY, TIMESTR, PRICE
from rv.preprocessing import prepare, clean_data, calculate_intraday_fraction
from rv.datamodels.config import Config
from rv.calculator import RealizedVolatilityCalculator


if __name__ == "__main__":
    configs = {
        "a": Config(0.05, 3.0, 0.05),
        "b": Config(0.05, 3.0, 0.05),
        "c": Config(0.05, 3.0, 0.05),
        "d": Config(0.05, 3.0, 0.05),
        "e": Config(0.05, 3.0, 0.05),
        "f": Config(0.005, 3.0, 0.05),
    }
    all_data = pd.read_csv("../data/stockdata3.csv")
    for stock_name in ["a", "b", "c", "d", "e", "f"]:
        config = configs[stock_name]
        print("=== ===")
        print(f"Working for stock {stock_name}")
        data = all_data[[DAY, TIMESTR, stock_name]].rename(columns={stock_name: PRICE})
        daily_data = prepare(data)
        daily_data = clean_data(daily_data, config.overnight_adjustment_std_multiplier, config.overnight_relative_threahold)
        print(f"Data cleaned for stock {stock_name}")
        intraday_fraction, overnight_fraction = calculate_intraday_fraction(daily_data, config.winsorization_limit)
        print(f"intraday_fraction={intraday_fraction}")

        calculator = RealizedVolatilityCalculator(config.winsorization_limit)
        rv = calculator.calculate_realized_volatilities(daily_data, intraday_fraction)
        print(f"Estimated volatility is {rv[-1]} for stock {stock_name}")
        print(f"Done for stock {stock_name}")
