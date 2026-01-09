import unittest
from unittest.mock import MagicMock
from deriv_quant_py.shared_state import state

class TestSpread(unittest.TestCase):
    def test_spread_update(self):
        state.update_spread("R_100", 0.5)
        self.assertEqual(state.get_spread("R_100"), 0.5)
        self.assertEqual(state.get_spread("R_NONE"), 0.0)

if __name__ == '__main__':
    unittest.main()
