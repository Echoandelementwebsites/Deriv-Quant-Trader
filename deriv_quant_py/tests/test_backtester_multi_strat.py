import unittest
from unittest.mock import MagicMock, AsyncMock
import pandas as pd
import numpy as np
from deriv_quant_py.core.backtester import Backtester

class TestBacktesterMultiStrat(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.backtester = Backtester(self.mock_client)

    def create_synthetic_data(self, length=5000):
        # Create a DataFrame with clear Trend and Reversal signals
        # 1. Trend: Rising price with MACD > Signal
        # 2. Reversal: High RSI then drop

        dates = pd.date_range(start='2023-01-01', periods=length, freq='1min')
        df = pd.DataFrame({
            'epoch': dates,
            'open': np.linspace(100, 200, length),
            'high': np.linspace(100, 200, length) + 1,
            'low': np.linspace(100, 200, length) - 1,
            'close': np.linspace(100, 200, length) # Strong uptrend
        })

        # Add some volatility for BB and RSI
        noise = np.random.normal(0, 0.5, length)
        df['close'] = df['close'] + noise
        df['high'] = df['close'] + 0.5
        df['low'] = df['close'] - 0.5

        return df

    def test_run_wfa_optimization_runs(self):
        # Verify that WFA runs without error and returns a result
        # We need enough data for Train(3000) + Test(500)
        df = self.create_synthetic_data(length=4000)

        result = self.backtester.run_wfa_optimization(df)

        # Since it's random/linear data, it might not find a profitable strategy,
        # but it shouldn't crash.
        # If result is None, that's valid (no profitable strategy).
        # If result is dict, check structure.

        if result:
            self.assertIn('strategy_type', result)
            self.assertIn('config', result)
            self.assertIn('Expectancy', result)
            print(f"Found strategy: {result['strategy_type']} EV={result['Expectancy']}")
        else:
            print("No profitable strategy found (Expected for linear noise data)")

    def test_eval_trend_ha_logic(self):
        # Test specific trend logic helper
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105] * 20, # Short repeating pattern
            'open': [99, 100, 101, 102, 103, 104] * 20,
            'high': [105] * 120,
            'low': [95] * 120
        })

        params = {
            'ema_period': 50,
            'duration': 3
        }

        # We need enough data for EMA(50)
        # 120 rows is enough

        wins, losses, signals = self.backtester._eval_trend_ha(df, params)

        # Just ensure it returns integers
        self.assertIsInstance(wins, (int, np.integer))
        self.assertIsInstance(losses, (int, np.integer))
        self.assertIsInstance(signals, (int, np.integer))

if __name__ == '__main__':
    unittest.main()
