import logging
from decimal import Decimal
from typing import Dict, List

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class AdvancedBTCUSDTMarketMaking(ScriptStrategyBase):
    """
    Advanced Market Making Strategy for BTC-USDT
    
    This strategy combines:
    1. Volatility-based spread adjustment using NATR
    2. Trend-based price shifting using RSI
    3. Inventory management to maintain target ratio
    4. Dynamic order sizing based on market conditions
    5. Stop-loss protection for risk management
    """
    
    # Base parameters
    bid_spread = 0.0001
    ask_spread = 0.0001
    order_refresh_time = 15
    order_amount = 0.001  # Smaller amount for BTC
    create_timestamp = 0
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    
    base, quote = trading_pair.split("-")
    
    # Candles parameters
    candle_exchange = "binance"
    candles_interval = "5m"  # Using 5-minute candles for better trend detection
    candles_length = 30
    max_records = 1000
    
    # Volatility parameters
    volatility_window = 20
    bid_spread_scalar = 150  # Higher for BTC due to its volatility
    ask_spread_scalar = 100
    
    # Trend parameters
    trend_window = 14
    max_shift_spread = 0.0001  # Max price shift based on trend
    trend_scalar = -1  # Negative to counter trend (mean reversion)
    
    # Inventory parameters
    target_ratio = 0.5  # Target 50% BTC, 50% USDT
    current_ratio = 0.5
    inventory_delta = 0
    inventory_scalar = 2  # Stronger inventory correction
    inventory_multiplier = 0
    
    # Risk management parameters
    max_position_size = 0.1  # Maximum BTC position size
    stop_loss_pct = 0.0628  # 6.28% maximum drawdown
    initial_portfolio_value = 0
    
    # Dynamic order sizing
    min_order_amount = 0.0005
    max_order_amount = 0.005
    volume_window = 10
    
    # Price variables
    orig_price = 1
    reference_price = 1
    price_multiplier = 1
    last_trade_price = 0
    
    # Initialize candles
    candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=candles_interval,
        max_records=max_records
    ))
    
    # Define markets
    markets = {exchange: {trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()
        # Initialize portfolio value for stop loss tracking
        self.initial_portfolio_value = self.get_portfolio_value()
        self.log_with_clock(logging.INFO, f"Initial portfolio value: {self.initial_portfolio_value}")
    
    def on_stop(self):
        self.candles.stop()
    
    def on_tick(self):
        # Check stop loss first
        current_value = self.get_portfolio_value()
        drawdown = 1 - (current_value / self.initial_portfolio_value)
        
        if drawdown > self.stop_loss_pct:
            self.log_with_clock(logging.WARNING, f"Stop loss triggered! Drawdown: {drawdown:.2%}")
            self.cancel_all_orders()
            return
        
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            self.update_multipliers()
            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp
    
    def get_portfolio_value(self):
        """Calculate total portfolio value in USDT"""
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        base_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        base_value = base_bal * base_price
        quote_bal = self.connectors[self.exchange].get_balance(self.quote)
        return float(base_value + quote_bal)
    
    def get_candles_with_features(self):
        """Add technical indicators to candles dataframe"""
        candles_df = self.candles.candles_df
        
        # Volatility indicators
        candles_df.ta.natr(length=self.volatility_window, scalar=1, append=True)
        candles_df['bid_spread_bps'] = candles_df[f"NATR_{self.volatility_window}"] * self.bid_spread_scalar * 10000
        candles_df['ask_spread_bps'] = candles_df[f"NATR_{self.volatility_window}"] * self.ask_spread_scalar * 10000
        
        # Trend indicators
        candles_df.ta.rsi(length=self.trend_window, append=True)
        
        # Volume analysis for order sizing
        candles_df.ta.sma(close=candles_df["volume"], length=self.volume_window, append=True, suffix=f"_volume_{self.volume_window}")
        
        # Bollinger Bands for volatility
        candles_df.ta.bbands(length=20, append=True)
        
        # MACD for trend confirmation
        candles_df.ta.macd(append=True)
        
        return candles_df
    
    def update_multipliers(self):
        """Update all strategy multipliers based on market conditions"""
        candles_df = self.get_candles_with_features()
        
        # Update spreads based on volatility
        natr = candles_df[f"NATR_{self.volatility_window}"].iloc[-1]
        self.bid_spread = natr * self.bid_spread_scalar
        self.ask_spread = natr * self.ask_spread_scalar
        
        # Trend-based price shift
        rsi = candles_df[f"RSI_{self.trend_window}"].iloc[-1]
        self.price_multiplier = (rsi - 50) / 50 * self.max_shift_spread * self.trend_scalar
        
        # Volume-based order sizing
        current_volume = candles_df["volume"].iloc[-1]
        avg_volume = candles_df[f"SMA_volume_{self.volume_window}"].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        self.order_amount = max(
            self.min_order_amount,
            min(self.max_order_amount, self.min_order_amount * volume_ratio)
        )
        
        # Inventory management
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        base_bal_in_quote = base_bal * self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        quote_bal = self.connectors[self.exchange].get_balance(self.quote)
        
        # Calculate current inventory ratio
        total_value = float(base_bal_in_quote + quote_bal)
        self.current_ratio = float(base_bal_in_quote / total_value) if total_value > 0 else 0.5
        
        # Calculate inventory delta and multiplier
        delta = ((self.target_ratio - self.current_ratio) / self.target_ratio) if self.target_ratio > 0 else 0
        self.inventory_delta = max(-1, min(1, delta))
        self.inventory_multiplier = self.inventory_delta * self.max_shift_spread * self.inventory_scalar
        
        # MACD trend confirmation
        macd = candles_df["MACD_12_26_9"].iloc[-1]
        macd_signal = candles_df["MACDs_12_26_9"].iloc[-1]
        macd_trend = 1 if macd > macd_signal else -1
        
        # Adjust trend scalar based on MACD confirmation
        if (rsi > 50 and macd_trend > 0) or (rsi < 50 and macd_trend < 0):
            # Trend confirmed by MACD, strengthen the signal
            self.price_multiplier *= 1.2
        
        # Define shifted reference price combining trend and inventory
        self.orig_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        self.reference_price = self.orig_price * Decimal(str(1 + self.price_multiplier)) * Decimal(str(1 + self.inventory_multiplier))
        
        # Store last trade price for ping-pong strategy
        self.last_trade_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.LastTrade)
    
    def create_proposal(self) -> List[OrderCandidate]:
        """Create buy and sell order proposals with adjusted prices"""
        # Get best bid/ask from the order book
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        
        # Calculate buy and sell prices
        buy_price = min(self.reference_price * Decimal(1 - self.bid_spread), best_bid)
        sell_price = max(self.reference_price * Decimal(1 + self.ask_spread), best_ask)
        
        # Adjust order sizes based on inventory
        buy_amount = Decimal(self.order_amount)
        sell_amount = Decimal(self.order_amount)
        
        # If significantly off target ratio, skew order sizes
        if abs(self.inventory_delta) > 0.2:
            skew_factor = 1 + min(abs(self.inventory_delta), 0.5)
            if self.inventory_delta > 0:  # Too little base asset
                buy_amount *= skew_factor
                sell_amount /= skew_factor
            else:  # Too much base asset
                buy_amount /= skew_factor
                sell_amount *= skew_factor
        
        # Create order candidates
        buy_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=buy_amount,
            price=buy_price
        )
        
        sell_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=sell_amount,
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
    
    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order fill events"""
        msg = (f"{event.trade_type.name} {round(event.amount, 4)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
        
        # Update portfolio value after trade
        current_value = self.get_portfolio_value()
        self.log_with_clock(logging.INFO, f"Portfolio value: {current_value} (Change: {(current_value/self.initial_portfolio_value - 1):.2%})")
    
    def format_status(self) -> str:
        """Returns status of the current strategy and displays candles feed info"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        
        lines = []
        
        # Display balances
        balance_df = self.get_balance_df()
        lines.extend(["", " Balances:"] + [" " + line for line in balance_df.to_string(index=False).split("\n")])
        
        # Display active orders
        try:
            df = self.active_orders_df()
            lines.extend(["", " Orders:"] + [" " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", " No active maker orders."])
        
        # Display portfolio metrics
        current_value = self.get_portfolio_value()
        drawdown = 1 - (current_value / self.initial_portfolio_value)
        
        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend([" Portfolio Metrics:"])
        lines.extend([f" Current Value: {current_value:.2f} USDT | Initial Value: {self.initial_portfolio_value:.2f} USDT"])
        lines.extend([f" Performance: {(current_value/self.initial_portfolio_value - 1):.2%} | Drawdown: {drawdown:.2%}"])
        lines.extend([f" Stop Loss Threshold: {self.stop_loss_pct:.2%}"])
        
        # Display spreads
        ref_price = self.reference_price
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        best_bid_spread = (ref_price - best_bid) / ref_price
        best_ask_spread = (best_ask - ref_price) / ref_price
        
        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend([" Spreads:"])
        lines.extend([f" Bid Spread (bps): {self.bid_spread * 10000:.4f} | Best Bid Spread (bps): {best_bid_spread * 10000:.4f}"])
        lines.extend([f" Ask Spread (bps): {self.ask_spread * 10000:.4f} | Best Ask Spread (bps): {best_ask_spread * 10000:.4f}"])
        
        # Display price shifts
        trend_price_shift = Decimal(self.price_multiplier) * Decimal(self.reference_price)
        inventory_price_shift = Decimal(self.inventory_multiplier) * Decimal(self.reference_price)
        
        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend([" Price Shifts:"])
        lines.extend([f" Max Shift (bps): {self.max_shift_spread * 10000:.4f}"])
        lines.extend([f" Trend Scalar: {self.trend_scalar:.1f} | Trend Multiplier (bps): {self.price_multiplier * 10000:.4f} | Trend Price Shift: {trend_price_shift:.4f}"])
        lines.extend([f" Target Inventory Ratio: {self.target_ratio:.4f} | Current Inventory Ratio: {self.current_ratio:.4f} | Inventory Delta: {self.inventory_delta:.4f}"])
        lines.extend([f" Inventory Multiplier (bps): {self.inventory_multiplier * 10000:.4f} | Inventory Price Shift: {inventory_price_shift:.4f}"])
        lines.extend([f" Orig Price: {self.orig_price:.4f} | Reference Price: {self.reference_price:.4f}"])
        
        # Display order sizing
        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend([" Order Sizing:"])
        lines.extend([f" Current Order Size: {self.order_amount:.6f} BTC | Min: {self.min_order_amount:.6f} | Max: {self.max_order_amount:.6f}"])
        
        # Display candles data
        lines.extend(["\n----------------------------------------------------------------------\n"])
        candles_df = self.get_candles_with_features()
        lines.extend([f" Candles: {self.candles.name} | Interval: {self.candles.interval}", ""])
        lines.extend([" " + line for line in candles_df.tail().iloc[::-1].to_string(index=False).split("\n")])
        
        return "\n".join(lines)

