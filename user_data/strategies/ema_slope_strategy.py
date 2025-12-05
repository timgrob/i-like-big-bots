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


class EmaSlopeStrategy(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "15m"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {"0": 1.0}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.03

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 8

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # EMA timeperiod strategy parameter
    ema_timeperiod = IntParameter(
        low=2, high=90, default=8, space="timeperiod", optimize=True, load=True
    )

    # Volume threshold strategy parameter
    volume_thershold = DecimalParameter(
        low=0.0,
        high=50.0,
        default=5e6,
        decimals=2,
        space="enter",
        optimize=True,
        load=True,
    )

    # Return on investment strategy parameters
    enter_long_ror = DecimalParameter(
        low=0.0,
        high=1.0,
        default=0.002,
        decimals=3,
        space="enter",
        optimize=True,
        load=True,
    )
    exit_long_ror = DecimalParameter(
        low=-1,
        high=0.0,
        default=0.0,
        decimals=2,
        space="exit",
        optimize=True,
        load=True,
    )
    enter_short_ror = DecimalParameter(
        low=-1.0,
        high=0.0,
        default=-0.002,
        decimals=3,
        space="enter",
        optimize=True,
        load=True,
    )
    exit_short_ror = DecimalParameter(
        low=0.0,
        high=1.0,
        default=-0.0,
        decimals=2,
        space="exit",
        optimize=True,
        load=True,
    )

    # Aroon strategy parameters
    enter_aroon_up = DecimalParameter(
        low=0.0,
        high=100.0,
        default=90.0,
        decimals=1,
        space="enter",
        optimize=True,
        load=True,
    )
    enter_aroon_down = DecimalParameter(
        low=0.0,
        high=100.0,
        default=90.0,
        decimals=1,
        space="enter",
        optimize=True,
        load=True,
    )
    exit_aroon_up = DecimalParameter(
        low=0.0,
        high=100.0,
        default=60.0,
        decimals=1,
        space="exit",
        optimize=True,
        load=True,
    )
    exit_aroon_down = DecimalParameter(
        low=0.0,
        high=100.0,
        default=60.0,
        decimals=1,
        space="exit",
        optimize=True,
        load=True,
    )

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                "ema8": {"color": "blue"},
            },
            "subplots": {
                # Subplots - each dict defines one additional plot
                "returns": {
                    "ror_ema8": {"color": "blue"},
                },
                "aroon": {
                    "aroonup": {"color": "orange"},
                    "aroondown": {"color": "blue"},
                },
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
        # Aroon, Aroon Oscillator
        aroon = ta.AROON(dataframe)
        dataframe["aroonup"] = aroon["aroonup"]
        dataframe["aroondown"] = aroon["aroondown"]

        # EMA - Parameter Range
        for val in self.ema_timeperiod.range:
            dataframe[f"ema{val}"] = ta.EMA(dataframe, timeperiod=val)
            dataframe[f"ror_ema{val}"] = dataframe[f"ema{val}"].pct_change()

        # Past three volume mean
        dataframe["volume_mean"] = np.mean(
            [dataframe["volume"].shift(i) for i in range(3)]
        )

        print("====== datafame ======")
        print(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    dataframe[f"ror_ema{self.ema_timeperiod.value}"]
                    > self.enter_long_ror.value
                )
                & (dataframe[f"ror_ema{self.ema_timeperiod.value}"].shift(1) > 0)
                & (dataframe["aroonup"] > dataframe["aroondown"])
                & (dataframe["aroonup"] > self.enter_aroon_up.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                (
                    dataframe[f"ror_ema{self.ema_timeperiod.value}"]
                    < self.enter_short_ror.value
                )
                & (dataframe[f"ror_ema{self.ema_timeperiod.value}"].shift(1) < 0)
                & (dataframe["aroondown"] > dataframe["aroonup"])
                & (dataframe["aroondown"] > self.enter_aroon_down.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (
                        dataframe[f"ror_ema{self.ema_timeperiod.value}"]
                        < self.exit_long_ror.value
                    )
                    | (dataframe["aroonup"] < dataframe["aroondown"])
                    | (dataframe["aroonup"] < self.exit_aroon_up.value)
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                (
                    (
                        dataframe[f"ror_ema{self.ema_timeperiod.value}"]
                        > self.exit_short_ror.value
                    )
                    | (dataframe["aroondown"] < dataframe["aroonup"])
                    | (dataframe["aroondown"] < self.exit_aroon_down.value)
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_short",
        ] = 1

        return dataframe
