from decimal import Decimal

import numpy as np
import pandas as pd

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class EnhancedBTCMarketMaker(ScriptStrategyBase):
    """
    Enhanced market making strategy optimized for BTC-USDT with high volatility (~$90 per minute).
    Features:
    - Dynamic spreads based on volatility (NATR)
    - Multi-level grid orders for better fill rates
    - Advanced trend detection with EMA crossover and RSI
    - Smart inventory management with target ratio
    - Volatility-based position sizing
    """
    
    # Base configuration - wider spreads to account for high volatility
    base_bid_spread = Decimal("0.0015")  # 0.15%
    base_ask_spread = Decimal("0.0015")  # 0.15%
    
    # Order refresh time - shorter to respond to 1-minute volatility
    order_refresh_time = 15  # seconds
    
    # Trading parameters
    trading_pair = "BTC-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    
    # Indicator parameters
    natr_length = 14
    ema_fast = 8     # Faster EMA to better capture 1-min movements
    ema_medium = 20  # Medium timeframe
    ema_slow = 50    # Longer timeframe for trend confirmation
    rsi_length = 14
    
    # Inventory management
    target_inventory_ratio = Decimal("0.5")  # 50% BTC, 50% USDT
    max_inventory_ratio = Decimal("0.75")    # Cap BTC at 75% of portfolio
    min_inventory_ratio = Decimal("0.25")    # Minimum 25% BTC
    
    # Portfolio allocation per order
    base_order_amount_fraction = Decimal("0.01")  # 1% of portfolio per order
    
    # Candle feed - using 1m candles for faster response
    candles = CandlesFactory.get_candle(
        CandlesConfig(
            connector="binance",
            trading_pair=trading_pair,
            interval="1m",  # 1-minute candles for more responsive strategy
            max_records=100
        )
    )
    
    markets = {exchange: {trading_pair}}
    
    def __init__(self, connectors: dict):
        super().__init__(connectors)
        self.candles.start()
        self.create_timestamp = 0
        self.avg_minute_move = Decimal("90")  # Observed average BTC movement in 1 min
        
    def on_stop(self):
        self.candles.stop()
        
    def on_tick(self):
        if len(self.candles.candles_df) == 0:
            self.logger().info("Waiting for candles data...")
            return
            
        if self.create_timestamp <= self.current_timestamp:
            try:
                self.cancel_all_orders()
                proposal = self.create_proposal()
                proposal_adjusted = self.adjust_proposal_to_budget(proposal)
                self.place_orders(proposal_adjusted)
            except Exception as e:
                self.logger().error(f"Error during order creation/execution: {e}")
            self.create_timestamp = self.current_timestamp + self.order_refresh_time
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for decision making"""
        df = df.copy()
        
        # EMAs for trend detection
        df["ema_fast"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_medium"] = df["close"].ewm(span=self.ema_medium, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()
        
        # ATR calculation
        df["prev_close"] = df["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = (df["high"] - df["prev_close"]).abs()
        df["tr3"] = (df["low"] - df["prev_close"]).abs()
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["true_range"].rolling(window=self.natr_length).mean()
        df["natr"] = (df["atr"] / df["close"]) * 100
        
        # RSI calculation
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_length).mean()
        avg_loss = loss.rolling(window=self.rsi_length).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        return df
    
    def create_proposal(self):
        """Create a multi-level grid order proposal based on market conditions"""
        # Get reference price
        try:
            ref_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, self.price_source
            )
        except ValueError:
            self.logger().warning("No order book, falling back to last candle close.")
            ref_price = Decimal(str(self.candles.candles_df.iloc[-1]["close"]))
        
        # Calculate indicators
        df = self.candles.candles_df.copy()
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        
        # Extract indicator values
        ema_fast_val = Decimal(str(latest["ema_fast"]))
        ema_medium_val = Decimal(str(latest["ema_medium"]))
        ema_slow_val = Decimal(str(latest["ema_slow"]))
        natr_val = Decimal(str(latest["natr"]))
        rsi_val = Decimal(str(latest["rsi"]))
        
        self.logger().info(
            f"Indicators - EMA Fast: {ema_fast_val:.2f}, Medium: {ema_medium_val:.2f}, "
            f"Slow: {ema_slow_val:.2f}, NATR: {natr_val:.2f}%, RSI: {rsi_val:.2f}"
        )
        
        # Calculate trend strength and direction
        trend_strength = abs(ema_fast_val - ema_slow_val) / ema_slow_val
        strong_trend = trend_strength > Decimal("0.005")
        bullish = (ema_fast_val > ema_medium_val > ema_slow_val) and rsi_val < Decimal("70")
        bearish = (ema_fast_val < ema_medium_val < ema_slow_val) and rsi_val > Decimal("30")
        
        # Dynamic spread adjustments based on volatility and trend
        vol_factor = Decimal("1") + (natr_val / Decimal("20")) + (trend_strength * Decimal("0.3"))
        
        if bullish:
            # Bullish: narrower bid, wider ask
            bid_spread_adj = self.base_bid_spread * Decimal("0.9")
            ask_spread_adj = self.base_ask_spread * Decimal("1.1")
        elif bearish:
            # Bearish: wider bid, narrower ask
            bid_spread_adj = self.base_bid_spread * Decimal("1.1")
            ask_spread_adj = self.base_ask_spread * Decimal("0.9")
        else:
            # Neutral: balanced spreads
            bid_spread_adj = self.base_bid_spread
            ask_spread_adj = self.base_ask_spread
        
        # Apply volatility factor to spreads
        bid_spread_final = bid_spread_adj * vol_factor
        ask_spread_final = ask_spread_adj * vol_factor
        
        self.logger().info(
            f"Spreads - Base: {self.base_bid_spread:.4f}/{self.base_ask_spread:.4f}, "
            f"Final: {bid_spread_final:.4f}/{ask_spread_final:.4f}, Vol Factor: {vol_factor:.2f}"
        )
        
        # Portfolio analysis for inventory management
        total_usdt = self.connectors[self.exchange].get_balance("USDT")
        total_btc = self.connectors[self.exchange].get_balance("BTC")
        total_portfolio_value_usdt = total_usdt + (total_btc * ref_price)
        
        if total_portfolio_value_usdt > 0:
            current_btc_ratio = (total_btc * ref_price) / total_portfolio_value_usdt
        else:
            current_btc_ratio = Decimal("0")
        
        # Calculate inventory skew factor (0.2 to 1.8 range)
        inventory_skew = Decimal("1") + ((self.target_inventory_ratio - current_btc_ratio) * Decimal("2"))
        inventory_skew = max(min(inventory_skew, Decimal("1.8")), Decimal("0.2"))
        
        # Calculate dynamic order size based on volatility
        volatility_ratio = natr_val / Decimal("10")  # Normalize NATR
        base_order_amount = (total_portfolio_value_usdt * self.base_order_amount_fraction) / ref_price
        adjusted_order_amount = base_order_amount * (Decimal("1") - (volatility_ratio * Decimal("0.5")))
        
        # Apply inventory skew to order amounts
        order_amount_buy = adjusted_order_amount * inventory_skew
        order_amount_sell = adjusted_order_amount * (Decimal("2") - inventory_skew)
        
        self.logger().info(
            f"Portfolio - BTC: {total_btc:.6f}, USDT: {total_usdt:.2f}, "
            f"BTC Ratio: {current_btc_ratio:.2%}, Inventory Skew: {inventory_skew:.2f}"
        )
        
        # Create grid orders
        proposal = []
        
        # Only place buy orders if below max inventory
        if current_btc_ratio < self.max_inventory_ratio and order_amount_buy > 0:
            # Create 3 buy orders at increasing depths
            for i in range(3):
                depth_multiplier = Decimal("1") + (Decimal(str(i)) * Decimal("0.5"))
                grid_bid_spread = bid_spread_final * depth_multiplier
                grid_price = ref_price * (Decimal("1") - grid_bid_spread)
                grid_amount = order_amount_buy * (Decimal("1") - (Decimal(str(i)) * Decimal("0.3")))
                
                if grid_amount > 0:
                    buy_order = OrderCandidate(
                        trading_pair=self.trading_pair,
                        is_maker=True,
                        order_type=OrderType.LIMIT,
                        order_side=TradeType.BUY,
                        amount=grid_amount,
                        price=grid_price
                    )
                    proposal.append(buy_order)
        
        # Only place sell orders if above min inventory
        if current_btc_ratio > self.min_inventory_ratio and order_amount_sell > 0:
            # Create 3 sell orders at increasing heights
            for i in range(3):
                depth_multiplier = Decimal("1") + (Decimal(str(i)) * Decimal("0.5"))
                grid_ask_spread = ask_spread_final * depth_multiplier
                grid_price = ref_price * (Decimal("1") + grid_ask_spread)
                grid_amount = order_amount_sell * (Decimal("1") - (Decimal(str(i)) * Decimal("0.3")))
                
                if grid_amount > 0:
                    sell_order = OrderCandidate(
                        trading_pair=self.trading_pair,
                        is_maker=True,
                        order_type=OrderType.LIMIT,
                        order_side=TradeType.SELL,
                        amount=grid_amount,
                        price=grid_price
                    )
                    proposal.append(sell_order)
        
        # Safety check - don't trade during extreme volatility
        if natr_val > Decimal("15"):
            self.logger().warning(f"Extreme volatility detected (NATR: {natr_val}%) - reducing order sizes")
            for order in proposal:
                order.amount = order.amount * Decimal("0.5")
        
        return proposal
    
    def adjust_proposal_to_budget(self, proposal):
        """Adjust orders to available budget"""
        return self.connectors[self.exchange].budget_checker.adjust_candidates(
            proposal, all_or_none=False
        )
    
    def place_orders(self, proposal):
        """Place orders from the proposal"""
        for order in proposal:
            if order.order_side == TradeType.SELL:
                self.sell(
                    connector_name=self.exchange,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    order_type=order.order_type,
                    price=order.price
                )
            else:
                self.buy(
                    connector_name=self.exchange,
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
        """Returns status of the strategy for monitoring"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
            
        lines = []
        
        # Display account balances
        balance_df = self.get_balance_df()
        lines.extend(["", "Balances:"] + ["  " + line for line in balance_df.to_string(index=False).split("\n")])
        
        # Display active orders
        try:
            df = self.active_orders_df()
            lines.extend(["", "Orders:"] + ["  " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "No active maker orders."])
            
        # Display latest candle information
        if len(self.candles.candles_df) > 0:
            lines.extend(["", f"Candles: {self.candles.name} | Interval: {self.candles.interval}", ""])
            lines.extend(["  " + line for line in self.candles.candles_df.tail(5).to_string(index=False).split("\n")])
            
        return "\n".join(lines)

