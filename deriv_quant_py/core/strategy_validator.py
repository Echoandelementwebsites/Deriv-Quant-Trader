import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from dataclasses import dataclass
from deriv_quant_py.core.metrics import MetricsEngine

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    passed: bool
    reason: str
    metrics: dict = None

class StrategyValidator:
    def __init__(self):
        pass

    def validate_strategy(self, df: pd.DataFrame, config: dict) -> ValidationResult:
        """
        Validates a strategy configuration against the Gauntlet criteria.
        """
        try:
            # 1. Parse & Execute
            call, put = self.parse_and_execute(df, config)
            if call is None or put is None:
                return ValidationResult(False, "Execution Failed: Signals are None")

            if call.sum() == 0 and put.sum() == 0:
                return ValidationResult(False, "No Signals Generated")

            # 2. Base Performance (Full Data)
            # Use duration 1 for validation (standardization)
            metrics = MetricsEngine.calculate_performance(df, call, put, duration=1)
            if not metrics:
                return ValidationResult(False, "Insufficient Trades (<5)")

            win_rate = metrics['win_rate']
            ev = metrics['ev']

            # CRITERIA 1: Hard Fail if Win Rate < 56%
            if win_rate < 56.0:
                return ValidationResult(False, f"Win Rate too low: {win_rate:.1f}% < 56%")

            # CRITERIA 2: Regime Check
            regime_type = config.get('regime_type', 'TREND').upper()

            # Calculate ADX for Regime Slicing
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_df is None:
                adx = pd.Series(0, index=df.index)
            else:
                adx = adx_df.iloc[:, 0].fillna(0)

            # Define Masks
            is_trend = adx > 25
            is_chop = adx <= 25

            # Calculate PnL Series (1.0 for win, -1.0 for loss, approx)
            # We can use metrics['pnl_series'] which has real payout logic
            pnl_series = metrics['pnl_series']

            if regime_type == 'TREND':
                regime_pnl = pnl_series[is_trend].sum()
                if regime_pnl < 0:
                    return ValidationResult(False, f"Negative PnL in TREND regime (ADX>25): {regime_pnl:.2f}")
            elif regime_type == 'RANGE':
                regime_pnl = pnl_series[is_chop].sum()
                if regime_pnl < 0:
                    return ValidationResult(False, f"Negative PnL in RANGE regime (ADX<=25): {regime_pnl:.2f}")

            # If all passed
            return ValidationResult(True, "Passed", metrics)

        except Exception as e:
            logger.error(f"Validation Error: {e}")
            return ValidationResult(False, f"Runtime Error: {str(e)}")

    def parse_and_execute(self, df: pd.DataFrame, config: dict):
        """
        Executes the JSON config using pandas_ta and df.eval.
        """
        df = df.copy()

        # 1. Calculate Indicators
        indicators = config.get('indicators', [])
        # Sanitize indicators list for pandas_ta Strategy
        # pandas_ta expects [{"kind": "rsi", "length": 14}, ...]

        if indicators:
            CustomStrategy = ta.Strategy(
                name="AI_Strategy",
                ta=indicators
            )
            df.ta.strategy(CustomStrategy)

        # 2. Logic Evaluation
        logic = config.get('entry_logic', {})
        call_logic = logic.get('call', 'False')
        put_logic = logic.get('put', 'False')

        # Clean strings (security check?)
        # For now, we assume simple pandas eval syntax.
        # Ensure boolean output
        try:
            call_signal = df.eval(call_logic)
            put_signal = df.eval(put_logic)

            # Ensure they are Series (scalars might be returned if logic is "True")
            if isinstance(call_signal, bool):
                call_signal = pd.Series(call_signal, index=df.index)
            if isinstance(put_signal, bool):
                put_signal = pd.Series(put_signal, index=df.index)

            return call_signal.fillna(False), put_signal.fillna(False)

        except Exception as e:
            logger.error(f"Eval Error: {e}")
            raise e
