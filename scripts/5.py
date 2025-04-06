from decimal import Decimal

import pandas as pd

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class EnhancedPMM(ScriptStrategyBase):
    # Base spreads (you can adjust these)
    base_bid_spread = Decimal("0.002")  # 0.1%
    base_ask_spread = Decimal("0.002")  # 0.1%

    order_refresh_time = 60
    base_order_amount_fraction = Decimal("0.01")  # 1% of portfolio in each order, for example

    trading_pair = "BTC-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice

    # Indicator parameters
    natr_length = 14
    ema_fast = 12
    ema_slow = 26

    # Inventory management
    target_inventory_ratio = Decimal("0.5")  # want 50% of your portfolio in BTC
    max_inventory_ratio = Decimal("0.8")    # won't let BTC exceed 80% of portfolio
    min_inventory_ratio = Decimal("0.2")    # won't let BTC drop below 20% of portfolio

    # Risk Management Parameters
    stop_loss_threshold = Decimal("0.05")  # 5% stop loss
    max_position_size = Decimal("0.1")     # 10% of portfolio per trade

    # Candle feed for real-time data from binance (1m intervals)
    from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
    candles = CandlesFactory.get_candle(
        CandlesConfig(
            connector="binance",
            trading_pair=trading_pair,
            interval="1m",
            max_records=1000  # Increased to store more candles
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
        if self.stop_loss_triggered:
            self.logger().info("Stop loss triggered. Stopping strategy.")
            return

        if len(self.candles.candles_df) == 0:
            self.logger().info("Waiting for candles data...")
            return

        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            proposal = self.create_proposal()
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.current_timestamp + self.order_refresh_time

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()

        df["prev_close"] = df["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = (df["high"] - df["prev_close"]).abs()
        df["tr3"] = (df["low"] - df["prev_close"]).abs()
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["true_range"].rolling(window=self.natr_length).mean()
        df["natr"] = (df["atr"] / df["close"]) * 100
        return df

    def create_proposal(self):
        # 1) Get a reference price
        try:
            ref_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, self.price_source
            )
        except ValueError:
            self.logger().warning("No order book, falling back to last candle close.")
            ref_price = Decimal(str(self.candles.candles_df.iloc[-1]["close"]))

        # 2) Calculate indicators
        df = self.candles.candles_df.copy()
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        ema_fast_val = Decimal(str(latest["ema_fast"]))
        ema_slow_val = Decimal(str(latest["ema_slow"]))
        natr_val = Decimal(str(latest["natr"]))

        self.logger().info(
            f"EMA Fast: {ema_fast_val}, EMA Slow: {ema_slow_val}, NATR: {natr_val}%"
        )

        # 3) Calculate the "trend strength" to scale order sizes/spreads
        if ema_slow_val > 0:
            trend_strength = abs(ema_fast_val - ema_slow_val) / ema_slow_val
        else:
            trend_strength = Decimal("0")

        # 4) Dynamic spread adjustments
        vol_factor = Decimal("1") + (natr_val / Decimal("30")) + (trend_strength * Decimal("0.2"))
        if ema_fast_val > ema_slow_val:
            bid_spread_adj = self.base_bid_spread * Decimal("0.95")
            ask_spread_adj = self.base_ask_spread * Decimal("1.05")
        else:
            bid_spread_adj = self.base_bid_spread * Decimal("1.05")
            ask_spread_adj = self.base_ask_spread * Decimal("0.95")

        bid_spread_final = bid_spread_adj * vol_factor
        ask_spread_final = ask_spread_adj * vol_factor

        self.logger().info(
            f"Final spreads - Bid: {bid_spread_final:.5f}, Ask: {ask_spread_final:.5f}"
        )

        # 5) Dynamic order amount: fraction of total portfolio
        total_usdt = self.connectors[self.exchange].get_balance("USDT")
        total_btc = self.connectors[self.exchange].get_balance("BTC")
        total_portfolio_value_usdt = total_usdt + (total_btc * ref_price)
        if total_portfolio_value_usdt > 0:
            current_btc_ratio = (total_btc * ref_price) / total_portfolio_value_usdt
        else:
            current_btc_ratio = Decimal("0")

        # base_order_amount in BTC terms:
        dynamic_base_amount_btc = (total_portfolio_value_usdt * self.base_order_amount_fraction) / ref_price

        # Next, scale buy/sell amounts depending on bullish or bearish
        if ema_fast_val > ema_slow_val:
            order_amount_buy = dynamic_base_amount_btc * (Decimal("1") + trend_strength)
            order_amount_sell = dynamic_base_amount_btc
        else:
            order_amount_buy = dynamic_base_amount_btc
            order_amount_sell = dynamic_base_amount_btc * (Decimal("1") + trend_strength)

        # 6) Partial skew logic
        if current_btc_ratio > self.target_inventory_ratio:
            if current_btc_ratio >= self.max_inventory_ratio:
                buy_scale = Decimal("0.0")
            else:
                fraction = (
                    (self.max_inventory_ratio - current_btc_ratio)
                    / (self.max_inventory_ratio - self.target_inventory_ratio)
                )
                buy_scale = fraction
            order_amount_buy *= buy_scale

        if current_btc_ratio < self.target_inventory_ratio:
            if current_btc_ratio <= self.min_inventory_ratio:
                sell_scale = Decimal("0.0")
            else:
                fraction = (
                    (current_btc_ratio - self.min_inventory_ratio)
                    / (self.target_inventory_ratio - self.min_inventory_ratio)
                )
                sell_scale = fraction
            order_amount_sell *= sell_scale

        self.logger().info(f"BTC ratio in portfolio: {current_btc_ratio:.2%}")
        self.logger().info(f"Order amounts (in BTC) => Buy: {order_amount_buy}, Sell: {order_amount_sell}")

        # 7) Calculate buy/sell prices
        buy_price = ref_price * (Decimal("1") - bid_spread_final)
        sell_price = ref_price * (Decimal("1") + ask_spread_final)

        # 8) Create orders
        buy_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=order_amount_buy,
            price=buy_price
        )
        sell_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=order_amount_sell,
            price=sell_price
        )

        # Build proposal, only if amounts > 0
        proposal = []
        if buy_order.amount > 0:
            proposal.append(buy_order)
        if sell_order.amount > 0:
            proposal.append(sell_order)

        # Log portfolio value and profit/loss
        if self.initial_portfolio_value == Decimal("0"):
            self.initial_portfolio_value = total_portfolio_value_usdt

        current_portfolio_value = total_portfolio_value_usdt
        profit_loss = current_portfolio_value - self.initial_portfolio_value
        profit_loss_percent = (profit_loss / self.initial_portfolio_value) * Decimal("100")

        self.logger().info(f"Current Portfolio Value (USDT): {current_portfolio_value:.2f}")
        self.logger().info(f"Profit/Loss (USDT): {profit_loss:.2f}")
        self.logger().info(f"Profit/Loss (%): {profit_loss_percent:.3f}%")

        # Risk Management: Stop Loss
        if profit_loss_percent <= -self.stop_loss_threshold:
            self.logger().info(f"Stop loss triggered at {profit_loss_percent:.3f}% loss.")
            self.stop_loss_triggered = True
            self.on_stop()

        # Risk Management: Position Sizing
        for order in proposal:
            if order.amount > (total_portfolio_value_usdt * self.max_position_size) / ref_price:
                order.amount = (total_portfolio_value_usdt * self.max_position_size) / ref_price

        return proposal

    def adjust_proposal_to_budget(self, proposal):
        return self.connectors[self.exchange].budget_checker.adjust_candidates(
            proposal, all_or_none=True
        )

    def place_orders(self, proposal):
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
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)
