import unittest
from unittest.mock import MagicMock, patch
from deriv_quant_py.core.executor import TradeExecutor
from deriv_quant_py.shared_state import state

class TestRiskLogic(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_db = MagicMock()
        self.executor = TradeExecutor(self.mock_client, self.mock_db)

    def test_calculate_stake_percentage(self):
        # Setup State
        state.update_balance(1000.0)
        state.set_risk_settings(percentage=1.0, limit=50.0)

        # 1% of 1000 = 10
        stake = self.executor._calculate_stake("R_100")
        self.assertEqual(stake, 10.0)

        # Change Percentage
        state.set_risk_settings(percentage=2.5, limit=50.0)
        stake = self.executor._calculate_stake("R_100")
        self.assertEqual(stake, 25.0) # 2.5% of 1000 = 25

    def test_calculate_stake_min_value(self):
        # Setup State for very low balance
        state.update_balance(10.0)
        state.set_risk_settings(percentage=1.0, limit=50.0)

        # 1% of 10 = 0.10. Min is 0.35
        stake = self.executor._calculate_stake("R_100")
        self.assertEqual(stake, 0.35)

if __name__ == '__main__':
    unittest.main()
