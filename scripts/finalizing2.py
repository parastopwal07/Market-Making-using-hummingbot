from decimal import Decimal

import pandas as pd

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class EnhancedBTCMarketMaker(ScriptStrategyBase):
    """
    Enhanced market making strategy optimized for BTC-USDT with high volatility (~$90 per minute).
    Features:
    - Multiple grid orders for higher trade frequency
    - Dynamic spreads based on volatility
    - Advanced trend detection with EMA crossover
    - Smart inventory management
    """
    
    # Base configuration - slightly wider spreads for better profitability
    base_bid_spread = Decimal("0.0015")  # 0.10%
    base_ask_spread = Decimal("0.0015")  # 0.10%
    
    # Order refresh time - shorter to increase trade frequency
    order_refresh_time = 15  # seconds
    
    # Trading parameters
    trading_pair = "BTC-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    
    # Indicator parameters
    natr_length = 14
    ema_fast = 5     # Faster EMA to better capture 1-min movements
    ema_medium = 15  # Medium timeframe
    ema_slow = 30    # Longer timeframe for trend confirmation
    
    # Inventory management
    target_inventory_ratio = Decimal("0.5")  # 50% BTC, 50% USDT
    max_inventory_ratio = Decimal("0.75")    # Cap BTC at 75% of portfolio
    min_inventory_ratio = Decimal("0.25")    # Minimum 25% BTC
    
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
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
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
        bullish = (ema_fast_val > ema_medium_val > ema_slow_val)
        bearish = (ema_fast_val < ema_medium_val < ema_slow_val)
        
        # Dynamic spread adjustments based on volatility and trend
        vol_factor = Decimal("1") + (natr_val / Decimal("25")) + (trend_strength * Decimal("0.25"))
        
        if bullish:
            # Bullish: narrower bid, wider ask
            bid_spread_adj = self.base_bid_spread * Decimal("0.95")
            ask_spread_adj = self.base_ask_spread * Decimal("1.05")
        elif bearish:
            # Bearish: wider bid, narrower ask
            bid_spread_adj = self.base_bid_spread * Decimal("1.05")
            ask_spread_adj = self.base_ask_spread * Decimal("0.95")
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
        
        # Increase base order amount to 3% of portfolio (up from 1%)
        base_order_amount_fraction = Decimal("0.03")
        dynamic_base_amount_btc = (total_portfolio_value_usdt * base_order_amount_fraction) / ref_price
        
        # Number of grid levels - increase from default 1 to 8 levels
        num_grid_levels = 8
        
        # Calculate total allocation per side (buy/sell)
        if bullish:
            # Bullish: allocate more to buys
            buy_allocation = dynamic_base_amount_btc * (Decimal("1.2") + trend_strength)
            sell_allocation = dynamic_base_amount_btc * Decimal("0.8")
        else:
            # Bearish: allocate more to sells
            buy_allocation = dynamic_base_amount_btc * Decimal("0.8")
            sell_allocation = dynamic_base_amount_btc * (Decimal("1.2") + trend_strength)
        
        # Apply inventory management
        if current_btc_ratio > self.target_inventory_ratio:
            # We have more BTC than desired - reduce buys, increase sells
            if current_btc_ratio >= self.max_inventory_ratio:
                buy_scale = Decimal("0.2")  # Still allow some buys (20%)
            else:
                fraction = (
                    (self.max_inventory_ratio - current_btc_ratio)
                    / (self.max_inventory_ratio - self.target_inventory_ratio)
                )
                buy_scale = Decimal("0.2") + (fraction * Decimal("0.8"))  # Scale from 20% to 100%
            
            sell_scale = Decimal("1.5")  # Increase sells by 50%
            buy_allocation *= buy_scale
            sell_allocation *= sell_scale
        
        if current_btc_ratio < self.target_inventory_ratio:
            # We have less BTC than desired - increase buys, reduce sells
            if current_btc_ratio <= self.min_inventory_ratio:
                sell_scale = Decimal("0.2")  # Still allow some sells (20%)
            else:
                fraction = (
                    (current_btc_ratio - self.min_inventory_ratio)
                    / (self.target_inventory_ratio - self.min_inventory_ratio)
                )
                sell_scale = Decimal("0.2") + (fraction * Decimal("0.8"))  # Scale from 20% to 100%
            
            buy_scale = Decimal("1.5")  # Increase buys by 50%
            buy_allocation *= buy_scale
            sell_allocation *= sell_scale
        
        self.logger().info(f"BTC ratio in portfolio: {current_btc_ratio:.2%}")
        self.logger().info(f"Total allocations => Buy: {buy_allocation}, Sell: {sell_allocation}")
        
        # Calculate spread step size for grid
        bid_step = bid_spread_final / Decimal(str(num_grid_levels))
        ask_step = ask_spread_final / Decimal(str(num_grid_levels))
        
        # Calculate amount distribution (more near mid price, less further away)
        total_weight = sum(range(1, num_grid_levels + 1))
        buy_weights = [i/total_weight for i in range(num_grid_levels, 0, -1)]
        sell_weights = [i/total_weight for i in range(num_grid_levels, 0, -1)]
        
        # Create grid orders
        proposal = []
        
        # Create buy orders
        for i in range(num_grid_levels):
            level_spread = bid_spread_final * (Decimal("0.5") + (Decimal(str(i)) / Decimal(str(num_grid_levels))))
            buy_price = ref_price * (Decimal("1") - level_spread)
            buy_amount = buy_allocation * Decimal(str(buy_weights[i]))
            
            if buy_amount > 0:
                buy_order = OrderCandidate(
                    trading_pair=self.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.BUY,
                    amount=buy_amount,
                    price=buy_price
                )
                proposal.append(buy_order)
        
        # Create sell orders
        for i in range(num_grid_levels):
            level_spread = ask_spread_final * (Decimal("0.5") + (Decimal(str(i)) / Decimal(str(num_grid_levels))))
            sell_price = ref_price * (Decimal("1") + level_spread)
            sell_amount = sell_allocation * Decimal(str(sell_weights[i]))
            
            if sell_amount > 0:
                sell_order = OrderCandidate(
                    trading_pair=self.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.SELL,
                    amount=sell_amount,
                    price=sell_price
                )
                proposal.append(sell_order)
        
        self.logger().info(f"Created {len(proposal)} orders in total")
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
