import pandas as pd
import numpy as np

class MetricsEngine:
    @staticmethod
    def calculate_performance(df, call_signal, put_signal, duration, symbol=""):
        """
        Calculates advanced performance metrics for a strategy.
        Returns None if less than 5 trades.
        """
        # 1. Calculate Standard Wins/Losses (Vectorized)
        future_close = df['close'].shift(-duration)

        # Apply Safety Layer (Crash/Boom protection) if needed
        if symbol and 'CRASH' in symbol: put_signal = put_signal & False
        if symbol and 'BOOM' in symbol: call_signal = call_signal & False
        if symbol and 'RDBULL' in symbol: put_signal = put_signal & False
        if symbol and 'RDBEAR' in symbol: call_signal = call_signal & False

        # Determine Payout
        # 'R_' are Volatility Indices, '1HZ' are Jump/Step/Crash/Boom usually have diff payouts?
        # The logic from Backtester was: payout = 0.94 if ('R_' in symbol or '1HZ' in symbol) else 0.85
        payout = 0.94 if ('R_' in symbol or '1HZ' in symbol) else 0.85

        # Vectorized PnL Series (1 = Win, -1 = Loss, 0 = No Trade)
        pnl = pd.Series(0.0, index=df.index)

        # Wins (+payout)
        wins_mask = (call_signal & (future_close > df['close'])) | (put_signal & (future_close < df['close']))
        pnl[wins_mask] = payout

        # Losses (-1.0)
        losses_mask = (call_signal & (future_close <= df['close'])) | (put_signal & (future_close >= df['close']))
        pnl[losses_mask] = -1.0

        total_trades = wins_mask.sum() + losses_mask.sum()
        if total_trades < 5: return None

        # 2. Calculate Metrics
        win_rate = wins_mask.sum() / total_trades
        loss_rate = losses_mask.sum() / total_trades

        # Expectancy (EV) - The Core Metric
        ev = (win_rate * payout) - (loss_rate * 1.0)

        # Kelly Criterion (Full Kelly) -> % of bankroll to risk
        # f* = (bp - q) / b
        # b = payout, p = win_rate, q = 1-win_rate
        if ev > 0:
            kelly = ((payout * win_rate) - (1 - win_rate)) / payout
            kelly = max(0, kelly) # Floor at 0
        else:
            kelly = 0.0

        # Max Drawdown (MDD) via Cumulative Returns
        # We simulate a compounding account to find the deepest % drop
        equity_curve = (1 + (pnl * 0.02)).cumprod() # Assume 2% fixed stake for MDD calc
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min() # Negative float (e.g. -0.15 for 15% DD)

        return {
            'ev': ev,               # Still the primary sorter
            'kelly': kelly,         # Sizing recommendation
            'max_drawdown': max_drawdown, # Risk grade
            'win_rate': win_rate * 100,
            'signals': int(total_trades),
            'wins': int(wins_mask.sum()),
            'losses': int(losses_mask.sum()),
            'pnl_series': pnl # For aggregation
        }
