import unittest
import pandas as pd
import numpy as np
from deriv_quant_py.utils.indicators import calculate_chop

class TestIndicators(unittest.TestCase):
    def test_calculate_chop(self):
        # Create synthetic data
        # Trending data (should have low CHOP)
        length = 100
        closes = pd.Series(np.linspace(10, 100, length))
        highs = closes + 1
        lows = closes - 1

        # CHOP is usually high for sideways, low for trend.
        # However, CHOP requires ATR and High-Low range.
        # Let's just verify it returns a Series and not None.
        chop = calculate_chop(highs, lows, closes, length=14)

        self.assertIsNotNone(chop)
        self.assertIsInstance(chop, pd.Series)
        # First 14 values will be NaN
        self.assertTrue(np.isnan(chop.iloc[0]))
        self.assertFalse(np.isnan(chop.iloc[-1]))

if __name__ == '__main__':
    unittest.main()
