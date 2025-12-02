# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
    AnnotationType,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from pykalman import KalmanFilter


class KalmanSlopeStrategy(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "15m"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {"60": 0.01, "30": 0.02, "0": 0.04}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = True
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.05  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Return on investment parameters
    enter_long_ror = DecimalParameter(
        low=0, high=1, default=0.0, decimals=1, space="enter", optimize=True, load=True
    )
    exit_long_ror = DecimalParameter(
        low=-1, high=0, default=0.0, decimals=1, space="exit", optimize=True, load=True
    )
    enter_short_ror = DecimalParameter(
        low=-1, high=0, default=0.0, decimals=1, space="enter", optimize=True, load=True
    )
    exit_short_ror = DecimalParameter(
        low=0, high=1, default=0.0, decimals=1, space="exit", optimize=True, load=True
    )

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                "kalman": {"color": "red"},
            },
            "subplots": {
                # Subplots - each dict defines one additional plot
                "returns": {
                    "ror_kalman": {"color": "red"},
                }
            },
        }

    def version(self) -> str | None:
        return super().version()

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return 5.0

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # Kalman Filter
        close_prices = dataframe["close"].bfill().values.astype(float)
        kf = KalmanFilter(
            transition_matrices=[[1]],
            observation_matrices=[[1]],
            initial_state_mean=close_prices[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01,
        )
        filtered_state_means, _ = kf.filter(close_prices)
        dataframe["kalman"] = filtered_state_means.flatten()
        dataframe["ror_kalman"] = dataframe["kalman"].pct_change()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["ror_kalman"] > self.enter_long_ror.value)
                & (dataframe["ror_kalman"].shift(1) > 0)
                # & (dataframe["ror_kalman"].shift(2) > 0)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                (dataframe["ror_kalman"] < self.enter_short_ror.value)
                & (dataframe["ror_kalman"].shift(1) < 0)
                # & (dataframe["ror_kalman"].shift(2) < 0)
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["ror_kalman"] < self.exit_long_ror.value)
                # & (dataframe["ror_kalman"].shift(1) < 0)
                # & (dataframe["ror_kalman"].shift(2) < 0)
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                (dataframe["ror_kalman"] > self.exit_short_ror.value)
                # & (dataframe["ror_kalman"].shift(1) > 0)
                # & (dataframe["ror_kalman"].shift(2) > 0)
                & (dataframe["volume"] > 0)
            ),
            "exit_short",
        ] = 1

        return dataframe
