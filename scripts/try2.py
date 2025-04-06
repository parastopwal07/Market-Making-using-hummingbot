import logging
from decimal import Decimal
from typing import Dict, List

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class SimplePMM(ScriptStrategyBase):
    """
    Combined strategy using:
    - Volatility spreads (NATR)
    - Trend detection (EMA crossover)
    - Inventory balancing
    """

    # Base configuration
    bid_spread = 0.0001
    ask_spread = 0.0001
    order_refresh_time = 15
    order_amount = 0.01
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice

    # Indicator parameters
    natr_length = 14
    ema_fast = 12
    ema_slow = 26

    # Inventory management
    max_inventory_ratio = 0.5  # 50% allocation

    # Candles config
    candles = CandlesFactory.get_candle(
        CandlesConfig(connector="binance", trading_pair=trading_pair, interval="5m", max_records=100)
    )

    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()
        self.create_timestamp = 0

    def on_stop(self):
        self.candles.stop()

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp and self.candles.is_ready:
            self.cancel_all_orders()
            self.update_spreads()
            proposal = self.create_proposal()
            self.place_orders(proposal)
            self.create_timestamp = self.current_timestamp + self.order_refresh_time

    def update_spreads(self):
        """Update spreads based on volatility and inventory"""
        df = self.candles.candles_df

        # Volatility component
        df.ta.natr(length=self.natr_length, append=True)
        natr_value = df[f"NATR_{self.natr_length}"].iloc[-1]
        self.bid_spread = natr_value * 0.8
        self.ask_spread = natr_value * 1.2

        # Trend component
        df.ta.ema(length=self.ema_fast, append=True)
        df.ta.ema(length=self.ema_slow, append=True)
        if df[f"EMA_{self.ema_fast}"].iloc[-1] > df[f"EMA_{self.ema_slow}"].iloc[-1]:
            self.bid_spread *= 1.2  # Wider spreads in uptrend
        else:
            self.ask_spread *= 1.2  # Wider spreads in downtrend

        # Inventory component
        base_bal = self.connectors[self.exchange].get_balance(self.trading_pair.split("-")[0])
        quote_bal = self.connectors[self.exchange].get_balance(self.trading_pair.split("-")[1])
        total_value = base_bal + (quote_bal / self.get_mid_price())
        inventory_ratio = base_bal / total_value if total_value > 0 else 0

        if inventory_ratio > self.max_inventory_ratio:
            self.bid_spread *= 1.5  # Discourage buys
        elif inventory_ratio < (1 - self.max_inventory_ratio):
            self.ask_spread *= 1.5  # Discourage sells

    def get_mid_price(self):
        return self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)

    def create_proposal(self) -> List[OrderCandidate]:
        mid_price = self.get_mid_price()
        buy_price = mid_price * Decimal(1 - self.bid_spread)
        sell_price = mid_price * Decimal(1 + self.ask_spread)

        return [
            OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=Decimal(self.order_amount),
                price=buy_price,
            ),
            OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=Decimal(self.order_amount),
                price=sell_price,
            ),
        ]

    def place_orders(self, proposal: List[OrderCandidate]):
        for order in proposal:
            if order.order_side == TradeType.BUY:
                self.buy(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=order.amount,
                    price=order.price,
                    order_type=OrderType.LIMIT,
                )
            else:
                self.sell(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=order.amount,
                    price=order.price,
                    order_type=OrderType.LIMIT,
                )

    def cancel_all_orders(self):
        for order in self.get_active_orders(self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def format_status(self) -> str:
        status = []
        status.append(f"| Spreads | Bid: {self.bid_spread*10000:.1f}bps | Ask: {self.ask_spread*10000:.1f}bps |")
        status.append(
            f"| Inventory Ratio | Current: {self.get_inventory_ratio():.2%} | Max: {self.max_inventory_ratio:.0%} |"
        )
        return "\n".join(status)

    def get_inventory_ratio(self):
        base = self.connectors[self.exchange].get_balance(self.trading_pair.split("-")[0])
        quote = self.connectors[self.exchange].get_balance(self.trading_pair.split("-")[1])
        total = base + (quote / self.get_mid_price())
        return base / total if total > 0 else 0
