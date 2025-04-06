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
        # Check if candles data is available instead of using is_ready
        if len(self.candles.candles_df) == 0:
            self.logger().info("Waiting for candles data...")
            return
            
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp
    
    def create_proposal(self) -> List[OrderCandidate]:
        """Create buy and sell order proposals"""
        ref_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        
        # Calculate buy and sell prices with spreads
        buy_price = ref_price * Decimal(1 - self.bid_spread)
        sell_price = ref_price * Decimal(1 + self.ask_spread)
        
        # Create order candidates
        buy_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal(self.order_amount),
            price=buy_price
        )
        
        sell_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=Decimal(self.order_amount),
            price=sell_price
        )
        
        return [buy_order, sell_order]
    
    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust order proposals to available budget"""
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted
    
    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place orders from the proposal"""
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)
    
    def place_order(self, connector_name: str, order: OrderCandidate):
        """Place individual order"""
        if order.order_side == TradeType.SELL:
            self.sell(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        elif order.order_side == TradeType.BUY:
            self.buy(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
    
    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)
    
    def format_status(self) -> str:
        """Returns status of the current strategy"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        
        lines = []
        
        # Display balances
        balance_df = self.get_balance_df()
        lines.extend(["", "Balances:"] + ["  " + line for line in balance_df.to_string(index=False).split("\n")])
        
        # Display active orders
        try:
            df = self.active_orders_df()
            lines.extend(["", "Orders:"] + ["  " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "No active maker orders."])
        
        # Display candles data if available
        if len(self.candles.candles_df) > 0:
            lines.extend(["", f"Candles: {self.candles.name} | Interval: {self.candles.interval}", ""])
            lines.extend(["  " + line for line in self.candles.candles_df.tail(5).to_string(index=False).split("\n")])
        
        return "\n".join(lines)
