import pandas as pd

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class CustomMarketMaker(ScriptStrategyBase):
    markets = {"binance_paper_trade": {"BTC-USDT"}}  

    fast_ema_period = 12
    slow_ema_period = 26
    atr_period = 14
    base_order_size = 0.001  # 0.001 BTC per order

    def on_tick(self):
        trading_pair = "BTC-USDT"
        exchange = "binance_paper_trade"

        # ✅ Fetch historical candles
        candles = self.connectors[exchange].get_historical_klines(
            trading_pair=trading_pair,
            interval="5m",
            limit=self.slow_ema_period + self.atr_period
        )

        # ✅ Ensure enough data
        if not candles or len(candles) < self.slow_ema_period:
            self.logger().warning("Not enough candle data")
            return  

        # ✅ Convert to DataFrame
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["close"] = df["close"].astype(float)

        # ✅ Compute indicators
        df["EMA_fast"] = df["close"].ewm(span=self.fast_ema_period).mean()
        df["EMA_slow"] = df["close"].ewm(span=self.slow_ema_period).mean()
        df["ATR"] = df["high"].sub(df["low"]).rolling(window=self.atr_period).mean()

        # ✅ Dynamic spread based on ATR
        current_price = df["close"].iloc[-1]
        spread = max(0.001, min(0.5, df["ATR"].iloc[-1] / current_price))  # Ensuring spread isn't too small or large
        bid_price, ask_price = current_price * (1 - spread / 2), current_price * (1 + spread / 2)

        # ✅ Cancel existing orders
        self.cancel_all_orders()

        # ✅ Place new limit orders
        self.place_limit_order(exchange, trading_pair, True, bid_price, self.base_order_size)
        self.place_limit_order(exchange, trading_pair, False, ask_price, self.base_order_size)

        self.logger().info(f"Placed Orders: Bid {bid_price:.2f}, Ask {ask_price:.2f}, Spread: {spread:.4f}")


