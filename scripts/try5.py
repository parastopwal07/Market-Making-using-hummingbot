from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class EnhancedPMMStrategy(ScriptStrategyBase):
    # Define markets as a class variable (not inside __init__)
    markets = {"binance_paper_trade": {"BTC-USDT"}}
    
    def __init__(self, connectors=None):
        super().__init__(connectors)
        self.logger().info("Strategy initialized")
    
    def on_tick(self):
        self.logger().info("On tick called")

