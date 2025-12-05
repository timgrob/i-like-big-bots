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
from technical import qtpylib


class ThreeCandlePatternStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {"0": 1.0}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.8

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = False

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 3

    # # Strategy parameters
    # enter_long = IntParameter(
    #     low=0, high=100, default=100, space="enter", optimize=True, load=True
    # )
    # enter_short = IntParameter(
    #     low=-100, high=0, default=-100, space="enter", optimize=True, load=True
    # )

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                "ema8": {"color": "yellow"},
            },
            "subplots": {
                # Subplots - each dict defines one additional plot
                "Long": {
                    "CDL3WHITESOLDIERS": {"color": "blue"},
                },
                "Short": {
                    "CDL3BLACKCROWS": {"color": "red"},
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
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA - Exponential Moving Average
        dataframe["ema8"] = ta.EMA(dataframe, timeperiod=8)

        # Chart type
        # ------------------------------------
        # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["ha_open"] = heikinashi["open"]
        dataframe["ha_close"] = heikinashi["close"]
        dataframe["ha_high"] = heikinashi["high"]
        dataframe["ha_low"] = heikinashi["low"]

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Three White Soldiers: values [0, 100]
        # dataframe["CDL3WHITESOLDIERS"] = ta.CDL3WHITESOLDIERS(dataframe)

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # Three Black Chrows: values [0, 100]
        # dataframe["CDL3BLACKCROWS"] = ta.CDL3BLACKCROWS(dataframe)

        # print("======== dataframe ======")
        # print(dataframe.tail())

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe.loc[
        #     (
        #         (dataframe["CDL3WHITESOLDIERS"] >= self.enter_long.value)
        #         & (dataframe["volume"] > 0)
        #     ),
        #     "enter_long",
        # ] = 1
        dataframe.loc[
            (
                (dataframe["close"].shift(0) > dataframe["open"].shift(0))
                & (dataframe["close"].shift(1) > dataframe["open"].shift(1))
                & (dataframe["close"].shift(2) > dataframe["open"].shift(2))
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        # dataframe.loc[
        #     (
        #         (dataframe["CDL3BLACKCROWS"] <= self.enter_short.value)
        #         & (dataframe["volume"] > 0)
        #     ),
        #     "enter_short",
        # ] = 1
        dataframe.loc[
            (
                (dataframe["close"].shift(0) < dataframe["open"].shift(0))
                & (dataframe["close"].shift(1) < dataframe["open"].shift(1))
                & (dataframe["close"].shift(2) < dataframe["open"].shift(2))
                & (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        print("====== dataframe =====")
        print(dataframe)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
