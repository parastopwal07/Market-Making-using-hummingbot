from decimal import Decimal

import pandas as pd

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class ImprovedPMM(ScriptStrategyBase):
    # Base parameters (make these bigger to avoid constant losses from tight spreads)
    base_bid_spread = Decimal("0.002")  # 0.1%
    base_ask_spread = Decimal("0.002")  # 0.1%

    # Increase or decrease to see how it affects your PnL
    order_refresh_time = 15
    # We'll keep a base order amount
    base_order_amount = Decimal("0.005")

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

    # Candle feed for real-time data from binance (5m intervals)
    from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig, CandlesFactory
    candles = CandlesFactory.get_candle(
        CandlesConfig(
            connector="binance",
            trading_pair=trading_pair,
            interval="5m",
            max_records=100
        )
    )

    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: dict):
        super().__init__(connectors)
        self.candles.start()
        self.create_timestamp = 0

    def on_stop(self):
        self.candles.stop()

    def on_tick(self):
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

        self.logger().info(f"EMA Fast: {ema_fast_val}, EMA Slow: {ema_slow_val}, NATR: {natr_val}%")

        # 3) Dynamic spread adjustments
        # Increase volatility factor to have bigger adjustments
        vol_factor = Decimal("1") + (natr_val / Decimal("20"))  # more aggressive than /100

        if ema_fast_val > ema_slow_val:
            # Bullish: narrower bid spread, wider ask
            bid_spread_adj = self.base_bid_spread * Decimal("0.8")
            ask_spread_adj = self.base_ask_spread * Decimal("1.2")
        else:
            # Bearish: narrower ask, wider bid
            bid_spread_adj = self.base_bid_spread * Decimal("1.2")
            ask_spread_adj = self.base_ask_spread * Decimal("0.8")

        bid_spread_final = bid_spread_adj * vol_factor
        ask_spread_final = ask_spread_adj * vol_factor

        self.logger().info(f"Final spreads - Bid: {bid_spread_final:.5f}, Ask: {ask_spread_final:.5f}")

        # 4) Dynamic order amount
        # Example: if bullish, buy more. If strongly bullish, buy even more.
        # You can refine the logic (e.g. difference between EMA fast and slow).
        if ema_fast_val > ema_slow_val:
            order_amount_buy = self.base_order_amount * Decimal("2")  # double buy size
            order_amount_sell = self.base_order_amount
        else:
            order_amount_buy = self.base_order_amount
            order_amount_sell = self.base_order_amount * Decimal("2")  # double sell size

        # 5) Inventory-based skew (simplified)
        #   - If your BTC ratio is above max, reduce buy amount or skip buys entirely.
        #   - If your BTC ratio is below min, reduce sell amount or skip sells entirely.
        #   - For real use, you'd measure total portfolio in USDT + BTC*price.
        total_usdt = self.connectors[self.exchange].get_balance("USDT")
        total_btc = self.connectors[self.exchange].get_balance("BTC")
        total_portfolio_value_usdt = total_usdt + (total_btc * ref_price)
        if total_portfolio_value_usdt > 0:
            current_btc_ratio = (total_btc * ref_price) / total_portfolio_value_usdt
        else:
            current_btc_ratio = Decimal("0")

        self.logger().info(f"BTC ratio in portfolio: {current_btc_ratio:.2%}")

        # If we have too much BTC, reduce buy amount to 0 or near 0
        if current_btc_ratio > self.max_inventory_ratio:
            order_amount_buy = Decimal("0")
        # If we have too little BTC, reduce sells
        if current_btc_ratio < self.min_inventory_ratio:
            order_amount_sell = Decimal("0")

        # 6) Calculate buy/sell prices
        buy_price = ref_price * (Decimal("1") - bid_spread_final)
        sell_price = ref_price * (Decimal("1") + ask_spread_final)

        # 7) Create orders
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

        # If either amount is 0, you can optionally skip creating that side.
        proposal = []
        if buy_order.amount > 0:
            proposal.append(buy_order)
        if sell_order.amount > 0:
            proposal.append(sell_order)

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
