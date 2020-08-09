from dataclasses import dataclass


@dataclass
class Config:
    winsorization_limit: float
    overnight_adjustment_std_multiplier: float
    overnight_relative_threahold: float
