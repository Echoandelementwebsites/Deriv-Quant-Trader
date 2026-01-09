import unittest
from unittest.mock import MagicMock
from deriv_quant_py.shared_state import SharedState

class TestSharedState(unittest.TestCase):
    def test_singleton(self):
        s1 = SharedState()
        s2 = SharedState()
        self.assertIs(s1, s2)

    def test_update_scanner(self):
        s = SharedState()
        data = {'Forex': [{'symbol': 'EURUSD'}]}
        s.update_scanner(data)
        self.assertEqual(s.get_scanner_data(), data)

    def test_settings(self):
        s = SharedState()
        s.set_trading_active(True)
        self.assertTrue(s.is_trading_active())

if __name__ == '__main__':
    unittest.main()
