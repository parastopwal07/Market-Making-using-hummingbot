#!/usr/bin/env python

import time
from decimal import Decimal
from typing import Tuple

import pandas as pd
import pandas_ta as ta
from pydantic import Field, validator

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class AVTMMConfig(BaseClientModel):
    """Configuration parameters for the AVTMM strategy"""

    connector_name: str = Field(default="binance_paper_trade")
    trading_pair: str = Field(default="BTC-USDT")
    order_amount: Decimal = Field(default=Decimal("0.01"))

    atr_period: int = Field(default=14)
    bb_period: int = Field(default=20)
    trend_period_fast: int = Field(default=20)
    trend_period_slow: int = Field(default=50)

    base_bid_spread: Decimal = Field(default=Decimal("0.005"))
    base_ask_spread: Decimal = Field(default=Decimal("0.005"))

    max_inventory_pct: float = Field(default=0.4)
    order_refresh_time: int = Field(default=60)

    @validator("order_amount", "base_bid_spread", "base_ask_spread", pre=True, allow_reuse=True)
    def validate_decimals(cls, v):
        return Decimal(str(v))


class AVTMMStrategy(ScriptStrategyBase):
    """
    Adaptive Volatility-Trend Market Making strategy
    Combines volatility indicators, trend analysis, and inventory risk management
    """

    def __init__(self, config: AVTMMConfig):
        super().__init__()
        self.config = config

        # Initialize candles for technical analysis
        self.candles = CandlesFactory.get_candle(
            connector=self.config.connector_name, trading_pair=self.config.trading_pair, interval="1h", max_records=100
        )

        # State management
        self.processing = False
        self.last_update = 0
        self.logger().info(f"Initialized AVTMM on {self.config.connector_name} for {self.config.trading_pair}")

    async def on_tick(self):
        current_time = int(time.time())
        if (current_time - self.last_update < self.config.order_refresh_time) or self.processing:
            return

        self.processing = True
        try:
            await self._process_strategy()
        except Exception as e:
            self.logger().error(f"Error: {str(e)}", exc_info=True)
        finally:
            self.processing = False
            self.last_update = current_time

    async def _process_strategy(self):
        if not self.candles.is_ready:
            self.logger().info("Waiting for candle data...")
            return

        mid_price = await self.connectors[self.config.connector_name].get_mid_price(self.config.trading_pair)
        if not mid_price:
            return

        df = self._calculate_indicators()
        if df is None or df.empty:
            return

        latest = df.iloc[-1]
        volatility = latest.get("atr", 0) / latest["close"] if latest["close"] else 0.01
        trend = 1 if latest.get("ma_fast", 0) > latest.get("ma_slow", 0) else -1

        bid_spread, ask_spread = await self._calculate_spreads(mid_price, volatility, trend)
        await self._place_orders(mid_price, bid_spread, ask_spread)

    def _calculate_indicators(self) -> pd.DataFrame:
        """Calculate technical indicators from candle data"""
        try:
            df = self.candles.candles_df.copy()
            df["atr"] = ta.atr(df.high, df.low, df.close, length=self.config.atr_period)
            bbands = ta.bbands(df.close, length=self.config.bb_period)
            df = pd.concat([df, bbands], axis=1)
            df["ma_fast"] = ta.sma(df.close, self.config.trend_period_fast)
            df["ma_slow"] = ta.sma(df.close, self.config.trend_period_slow)
            return df
        except Exception as e:
            self.logger().error(f"Indicator error: {str(e)}")
            return None

    async def _calculate_spreads(self, mid_price: Decimal, volatility: float, trend: int) -> Tuple[Decimal, Decimal]:
        """Calculate dynamic spreads based on market conditions"""
        try:
            base, quote = self.config.trading_pair.split("-")
            base_bal = await self.connectors[self.config.connector_name].get_balance(base)
            quote_bal = await self.connectors[self.config.connector_name].get_balance(quote)

            total_value = base_bal + (quote_bal / mid_price)
            base_pct = base_bal / total_value if total_value else 0

            # Base spreads with volatility adjustment
            vol_multiplier = Decimal(1 + volatility * 2)
            bid_spread = self.config.base_bid_spread * vol_multiplier
            ask_spread = self.config.base_ask_spread * vol_multiplier

            # Trend adjustment
            trend_multiplier = Decimal(1.2 if trend == 1 else 0.8)
            bid_spread *= trend_multiplier
            ask_spread *= 2 - trend_multiplier

            # Inventory adjustment
            if base_pct > self.config.max_inventory_pct:
                bid_spread *= Decimal(1.5)
                ask_spread *= Decimal(0.8)
            elif base_pct < (1 - self.config.max_inventory_pct):
                bid_spread *= Decimal(0.8)
                ask_spread *= Decimal(1.5)

            return bid_spread, ask_spread
        except Exception as e:
            self.logger().error(f"Spread calculation error: {str(e)}")
            return self.config.base_bid_spread, self.config.base_ask_spread

    async def _place_orders(self, mid_price: Decimal, bid_spread: Decimal, ask_spread: Decimal):
        """Place orders with calculated spreads"""
        bid_price = mid_price * (1 - bid_spread)
        ask_price = mid_price * (1 + ask_spread)

        await self.cancel_all_orders(self.config.connector_name)

        order_args = {
            "connector_name": self.config.connector_name,
            "trading_pair": self.config.trading_pair,
            "amount": self.config.order_amount,
            "order_type": "LIMIT",
        }

        await self.place_order(**order_args, is_buy=True, price=bid_price)
        await self.place_order(**order_args, is_buy=False, price=ask_price)

        self.logger().info(f"Placed orders | Bid: {bid_price:.2f} | Ask: {ask_price:.2f}")


def main():
    return AVTMMStrategy(AVTMMConfig())
