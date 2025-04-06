from decimal import Decimal

import pandas as pd

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class EnhancedPMM(ScriptStrategyBase):
    # I found 0.2% to be the best value for spreads after testing values from 0.05% to 1%
    base_bid_spread = Decimal("0.002")
    base_ask_spread = Decimal("0.002")

    # 60 seconds worked best in my tests - neither missing opportunities nor losing too much to brokerage fee due to lot of trades
    order_refresh_time = 60
    
    # Allocating 1% of my portfolio per order to manage risk
    base_order_amount_fraction = Decimal("0.01") #portfolio size is 150BTC and 13000000 USDT (considering a big firm)

    trading_pair = "BTC-USDT" #as this pair has the most volatility and is most popular
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice

    # After testing multiple indicator combinations, these parameters worked best
    natr_length = 14
    ema_fast = 12
    ema_slow = 26

    # I aim for a 50/50 split between BTC and USDT
    target_inventory_ratio = Decimal("0.5")
    max_inventory_ratio = Decimal("0.8")
    min_inventory_ratio = Decimal("0.2")

    #stop-losses at 5% 
    stop_loss_threshold = Decimal("5")
    
    # Never risk more than 10% of portfolio on a single trade
    max_position_size = Decimal("0.1")

    # Binance's standard fee (0.1% of the volume traded)
    trading_fee_percent = Decimal("0.001")

    # using 1-minute candles after testing 3m and 5m timeframes
    from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
    candles = CandlesFactory.get_candle(
        CandlesConfig(
            connector="binance",
            trading_pair=trading_pair,
            interval="1m",
            max_records=1000  # storing more candles for more data and not losing on it
        )
    )

    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: dict):
        super().__init__(connectors)
        self.candles.start()
        self.create_timestamp = 0
        self.initial_portfolio_value = Decimal("0")
        self.stop_loss_triggered = False

    def on_stop(self):
        self.candles.stop()
        self.stop_loss_triggered = True

    def on_tick(self):
        # don't continue if stop loss was triggered
        if self.stop_loss_triggered:
            self.logger().info("Stop loss triggered. Stopping strategy.")
            return

        # Wait for candle data to be available
        if len(self.candles.candles_df) == 0:
            self.logger().info("Waiting for candles data...")
            return

        # refreshing orders after the refresh time has elapsed
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            proposal = self.create_proposal()
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.current_timestamp + self.order_refresh_time

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # calculating EMA crossover indicators
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()

        # Calculate Normalized ATR (average true range) for volatility
        df["prev_close"] = df["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = (df["high"] - df["prev_close"]).abs()
        df["tr3"] = (df["low"] - df["prev_close"]).abs()
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["true_range"].rolling(window=self.natr_length).mean()
        df["natr"] = (df["atr"] / df["close"]) * 100
        return df

    def create_proposal(self):
        try:
            # Get current mid price from the exchange
            ref_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, self.price_source
            )
        except ValueError:
            # Fallback to last candle close if order book data isn't available
            self.logger().warning("No order book data available; using last candle close.")
            ref_price = Decimal(str(self.candles.candles_df.iloc[-1]["close"]))

        # Calculate current portfolio value
        total_usdt_balance = self.connectors[self.exchange].get_balance("USDT")
        total_btc_balance = self.connectors[self.exchange].get_balance("BTC")
        total_portfolio_value_usdt = total_usdt_balance + (total_btc_balance * ref_price) #using ref price here

        # Store initial portfolio value for P&L tracking
        if total_portfolio_value_usdt > 0 and self.initial_portfolio_value == Decimal("0"):
            self.initial_portfolio_value = total_portfolio_value_usdt
            
        # Calculate profit/loss before fees
        profit_loss_usdt_before_fees = total_portfolio_value_usdt - self.initial_portfolio_value
        profit_loss_percent_before_fees = (profit_loss_usdt_before_fees / self.initial_portfolio_value) * Decimal("100")

        # Calculate estimated fees for a complete buy/sell cycle
        buy_fee = (total_portfolio_value_usdt * self.base_order_amount_fraction) * self.trading_fee_percent
        sell_fee = (total_portfolio_value_usdt * self.base_order_amount_fraction) * self.trading_fee_percent
        estimated_fee_costs_usdt_buy_sell_cycle = buy_fee + sell_fee #brokerage fee on binance is applied to both buy and sell trade
        
        # Calculate profit/loss after accounting for fees
        profit_loss_usdt_after_fees = (
            profit_loss_usdt_before_fees - estimated_fee_costs_usdt_buy_sell_cycle
        )
        profit_loss_percent_after_fees = (profit_loss_usdt_after_fees / self.initial_portfolio_value) * Decimal("100")

        # Trigger stop loss if we've lost more than our threshold
        if profit_loss_percent_after_fees <= -self.stop_loss_threshold:
            self.logger().info(f"Stop loss triggered at {profit_loss_percent_after_fees:.3f}% loss.")
            return []

        # dynamic order size based on current portfolio value
        dynamic_base_amount_btc = (total_portfolio_value_usdt * self.base_order_amount_fraction) / ref_price

        # Calculate buy and sell prices using our spreads
        buy_price = ref_price * (Decimal("1") - self.base_bid_spread)
        sell_price = ref_price * (Decimal("1") + self.base_ask_spread)

        # Limit order size to our maximum position size parameter
        buy_order_amount_btc = min(dynamic_base_amount_btc, (total_portfolio_value_usdt * self.max_position_size) / ref_price)
        sell_order_amount_btc = min(dynamic_base_amount_btc, (total_portfolio_value_usdt * self.max_position_size) / ref_price)

        # Create buy order
        buy_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=buy_order_amount_btc,
            price=buy_price,
        )
        
        # Create sell order
        sell_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=sell_order_amount_btc,
            price=sell_price,
        )

        proposal = [buy_order, sell_order]

        # Log portfolio value and profit/loss metrics (this shows profit/loss from the beginning including the changes in the 
        # current portfolio due to market changes along with the trade profit)
        self.logger().info(f"Current Portfolio Value (USDT): {total_portfolio_value_usdt:.2f}")
        self.logger().info(f"Profit/Loss Before Fees (USDT): {profit_loss_usdt_before_fees:.2f}")
        self.logger().info(f"Profit/Loss After Fees (USDT): {profit_loss_usdt_after_fees:.2f}")
        self.logger().info(f"Profit/Loss After Fees (%): {profit_loss_percent_after_fees:.3f}%")

        return proposal

    def adjust_proposal_to_budget(self, proposal):
        # Make sure we don't exceed our available balance
        return self.connectors[self.exchange].budget_checker.adjust_candidates(
            proposal, all_or_none=True
        )

    def place_orders(self, proposal):
        # Place each order in our proposal
        for order in proposal:
            if order.order_side == TradeType.SELL:
                trade_id_sell = self.sell(
                    connector_name=self.exchange,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    order_type=order.order_type,
                    price=order.price
                )
                if trade_id_sell:
                    self.logger().info(f"Sell Order Executed: ID {trade_id_sell}")
                    
            else:
                trade_id_buy = self.buy(
                    connector_name=self.exchange,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    order_type=order.order_type,
                    price=order.price
                )
                if trade_id_buy:
                    self.logger().info(f"Buy Order Executed: ID {trade_id_buy}")

    def cancel_all_orders(self):
        # Cancel all active orders before placing new ones
        for order in self.get_active_orders(connector_name=self.exchange):
            cancel_result_id = self.cancel(self.exchange, order.trading_pair, order.client_order_id)
            if cancel_result_id:
                self.logger().info(f"Order Cancelled: ID {cancel_result_id}")
