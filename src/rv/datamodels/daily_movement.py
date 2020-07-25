from dataclasses import dataclass
from typing import Sequence
import pandas as pd

DAY = "day"
TIMESTR = "timestr"
PRICE = "price"

@dataclass
class OneDay:
    open_price: float
    close_price: float
    previous_open: float
    previous_close: float
    intraday_prices: pd.DataFrame
    hourly_prices: Sequence[float]

    def get_intraday_prices(self):
        return self.intraday_prices[PRICE].values

    def get_hourly_prices(self):
        return self.hourly_prices

