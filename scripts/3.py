import logging
from decimal import Decimal
from typing import Dict, List

import numpy as np
import pandas as pd

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class EnhancedPMMStrategy(ScriptStrategyBase):
    """
    Enhanced Pure Market Making Strategy incorporating:
    - Dynamic volatility-based spread adjustment
    - Trend-following price shifts
    - Inventory management
    - Risk management framework
    """
    
    # Basic settings
    bid_spread = Decimal("0.0005")  # 5 basis points
    ask_spread = Decimal("0.0005")  # 5 basis points
    order_refresh_time = 30  # Increased to reduce cancellations
    order_amount = Decimal("0.001")  # Reduced to ensure orders can be placed
    trading_pair = "BTC-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    
    # Define markets - THIS WAS MISSING
    markets = {"binance_paper_trade": {"BTC-USDT"}}
    
    # Parse base and quote from trading pair
    base, quote = trading_pair.split('-')
    
    # Candles params
    candle_exchange = "binance"
    candles_interval = "1m"
    candles_length = 30
    max_records = 100  # Reduced to improve performance
    
    # Spread params - volatility based
    bid_spread_scalar = Decimal("80")
    ask_spread_scalar = Decimal("80")
    
    # Price shift params - trend based
    max_shift_spread = Decimal("0.0002")
    trend_scalar = Decimal("-0.8")  # Negative means we go against extreme RSI
    price_multiplier = Decimal("1.0")
    
    # Inventory management params
    target_ratio = Decimal("0.5")
    inventory_scalar = Decimal("0.7")
    inventory_multiplier = Decimal("1.0")
    
    # Risk management
    max_position_size = Decimal("0.05")  # Maximum position size as % of portfolio
    daily_loss_limit = Decimal("0.02")   # 2% daily loss limit
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        # Initialize candles
        self.candles = CandlesFactory.get_candle(CandlesConfig(
            connector=self.candle_exchange,
            trading_pair=self.trading_pair,
            interval=self.candles_interval,
            max_records=self.max_records))
        
        self.candles.start()
        self.daily_pnl = Decimal("0")
        self.current_day = None
        self.trade_count = 0
        self.winning_trades = 0
        self.total_fees_paid = Decimal("0")
        self.brokerage_fee = Decimal("0.001")  # 0.1% fee
        self.create_timestamp = 0
        self.reference_price = Decimal("0")
        self.orig_price = Decimal("0")
    
    def on_stop(self):
        self.candles.stop()
    
    def on_tick(self):
        if not self.ready_to_trade:
            return
            
        if self.create_timestamp <= self.current_timestamp:
            try:
                self.cancel_all_orders()
                self.update_multipliers()
                proposal: List[OrderCandidate] = self.create_proposal()
                proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
                self.place_orders(proposal_adjusted)
                self.create_timestamp = self.order_refresh_time + self.current_timestamp
            except Exception as e:
                self.logger().error(f"Error in on_tick: {str(e)}", exc_info=True)
    
    def update_multipliers(self):
        try:
            # Get candles dataframe
            candles_df = self.candles.candles_df
            if candles_df.empty:
                self.logger().warning("Candles dataframe is empty")
                return
                
            # Calculate NATR for volatility (simplified calculation)
            high = candles_df['high']
            low = candles_df['low']
            close = candles_df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(self.candles_length).mean()
            natr = atr / candles_df['close']
            
            # Calculate RSI
            delta = candles_df['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            
            avg_gain = gain.rolling(self.candles_length).mean()
            avg_loss = loss.rolling(self.candles_length).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get latest values
            latest_natr = Decimal(str(natr.iloc[-1])) if not pd.isna(natr.iloc[-1]) else Decimal("0.001")
            latest_rsi = Decimal(str(rsi.iloc[-1])) if not pd.isna(rsi.iloc[-1]) else Decimal("50")
            
            # Update spreads based on volatility
            self.bid_spread = max(Decimal("0.0005"), latest_natr * self.bid_spread_scalar)
            self.ask_spread = max(Decimal("0.0005"), latest_natr * self.ask_spread_scalar)
            
            # Trend-based price shift (RSI)
            self.price_multiplier = (latest_rsi - Decimal("50")) / Decimal("50") * self.max_shift_spread * self.trend_scalar
            
            # Inventory-based price shift
            base_bal = self.connectors[self.exchange].get_balance(self.base)
            current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            base_bal_in_quote = base_bal * current_price
            quote_bal = self.connectors[self.exchange].get_balance(self.quote)
            
            total_value = base_bal_in_quote + quote_bal
            if total_value > Decimal("0"):
                current_ratio = base_bal_in_quote / total_value
            else:
                current_ratio = Decimal("0")
                
            inventory_delta = ((self.target_ratio - current_ratio) / self.target_ratio)
            self.inventory_multiplier = inventory_delta * self.max_shift_spread * self.inventory_scalar
            
            # Calculate reference price with both adjustments
            self.orig_price = current_price
            self.reference_price = current_price * (Decimal("1") + self.price_multiplier) * (Decimal("1") + self.inventory_multiplier)
            
        except Exception as e:
            self.logger().error(f"Error in update_multipliers: {str(e)}", exc_info=True)
            # Use default values if calculation fails
            current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            self.reference_price = current_price
            self.orig_price = current_price
    
    def create_proposal(self) -> List[OrderCandidate]:
        try:
            # Make sure orders are not tighter than the best bid/ask
            current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            
            if self.reference_price == Decimal("0"):
                self.reference_price = current_price
            
            buy_price = current_price * (Decimal("1") - self.bid_spread)
            sell_price = current_price * (Decimal("1") + self.ask_spread)
            
            # Create order candidates
            buy_order = OrderCandidate(
                trading_pair=self.trading_pair, 
                is_maker=True, 
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY, 
                amount=self.order_amount, 
                price=buy_price
            )
            
            sell_order = OrderCandidate(
                trading_pair=self.trading_pair, 
                is_maker=True, 
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL, 
                amount=self.order_amount, 
                price=sell_price
            )
            
            return [buy_order, sell_order]
        except Exception as e:
            self.logger().error(f"Error in create_proposal: {str(e)}", exc_info=True)
            return []
    
    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        try:
            proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=False)
            return proposal_adjusted
        except Exception as e:
            self.logger().error(f"Error in adjust_proposal_to_budget: {str(e)}", exc_info=True)
            return []
    
    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)
    
    def place_order(self, connector_name: str, order: OrderCandidate):
        try:
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
        except Exception as e:
            self.logger().error(f"Error in place_order: {str(e)}", exc_info=True)
    
    def cancel_all_orders(self):
        try:
            for order in self.get_active_orders(connector_name=self.exchange):
                self.cancel(self.exchange, order.trading_pair, order.client_order_id)
        except Exception as e:
            self.logger().error(f"Error in cancel_all_orders: {str(e)}", exc_info=True)
    
    def did_fill_order(self, event: OrderFilledEvent):
        try:
            # Calculate fees
            fee_amount = event.price * event.amount * self.brokerage_fee
            self.total_fees_paid += fee_amount
            
            # Track trade statistics
            self.trade_count += 1
            
            # Log trade information
            msg = (f"{event.trade_type.name} {event.amount} {event.trading_pair} at {event.price}, Fee: {fee_amount} {self.quote}")
            self.logger().info(msg)
            self.notify_hb_app_with_timestamp(msg)
        except Exception as e:
            self.logger().error(f"Error in did_fill_order: {str(e)}", exc_info=True)
    
    def format_status(self) -> str:
        """Returns status of the current strategy with detailed metrics"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        
        lines = []
        
        try:
            # Balance information
            balance_df = self.get_balance_df()
            lines.extend(["", "Balances:"] + ["  " + line for line in balance_df.to_string(index=False).split("\n")])
            
            # Active orders
            active_orders = self.get_active_orders(connector_name=self.exchange)
            if len(active_orders) > 0:
                df = self.active_orders_df()
                lines.extend(["", "Orders:"] + ["  " + line for line in df.to_string(index=False).split("\n")])
            else:
                lines.extend(["", "No active maker orders."])
            
            # Current price and spread information
            current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            
            # Display spread information
            lines.extend(["\n----------------------------------------------------------------------\n"])
            lines.extend(["Strategy Information:"])
            lines.extend([f"  Bid Spread: {float(self.bid_spread) * 10000:.2f} bps"])
            lines.extend([f"  Ask Spread: {float(self.ask_spread) * 10000:.2f} bps"])
            lines.extend([f"  Current Price: {float(current_price):.2f}"])
            lines.extend([f"  Reference Price: {float(self.reference_price):.2f}"])
            
            # Display trade statistics
            lines.extend(["\n----------------------------------------------------------------------\n"])
            lines.extend(["Trade Statistics:"])
            lines.extend([f"  Total Trades: {self.trade_count}"])
            lines.extend([f"  Total Fees Paid: {float(self.total_fees_paid):.6f} {self.quote}"])
            
            return "\n".join(lines)
        except Exception as e:
            self.logger().error(f"Error in format_status: {str(e)}", exc_info=True)
            return "Error generating status report."

