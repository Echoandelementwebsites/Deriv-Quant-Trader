import pandas as pd
from deriv_quant_py.utils.indicators import calculate_ema, calculate_rsi, calculate_adx, detect_patterns
from deriv_quant_py.config import Config

class TripleConfluenceStrategy:
    def __init__(self):
        self.ema_period = Config.EMA_PERIOD
        self.rsi_period = Config.RSI_PERIOD
        self.rsi_ob = Config.RSI_OB
        self.rsi_os = Config.RSI_OS
        self.adx_threshold = Config.ADX_THRESHOLD

    def analyze(self, candles: list, params: dict = None):
        """
        Analyzes a list of candles to produce a signal.
        candles: list of dicts {open, high, low, close, epoch}
        params: dict (optional) overrides for 'rsi_period', 'ema_period'
        """
        # Determine periods
        ema_p = params.get('ema_period', self.ema_period) if params else self.ema_period
        rsi_p = params.get('rsi_period', self.rsi_period) if params else self.rsi_period

        if len(candles) < ema_p + 5:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        # Calculate Indicators
        ema = calculate_ema(df['close'], ema_p)
        rsi = calculate_rsi(df['close'], rsi_p)
        adx = calculate_adx(df['high'], df['low'], df['close'], 14)

        # Get values for the last closed candle (index -1)
        # Assuming the 'candles' list is historical + current open?
        # Usually strategy runs on confirmed close.
        # But if 'ticks' stream builds candles, the last one might be forming.
        # Let's assume the input is "last N completed candles".

        last_idx = df.index[-1]
        current_price = df['close'].iloc[-1]

        val_ema = ema.iloc[-1]
        val_rsi = rsi.iloc[-1]
        val_adx = adx.iloc[-1] if adx is not None else 0

        # Pattern Recognition
        pattern = detect_patterns(df['open'], df['high'], df['low'], df['close'])

        signal = None
        reason = ""

        # Logic: Triple Confluence + ADX
        # 1. ADX Filter (Market must be trending)
        if val_adx > self.adx_threshold:

            # 2. Pattern + Trend + Momentum (Reversal)
            # Reversal Logic:
            # Bullish: Pattern is BULL, Price < EMA (Oversold territory?? No, reversal usually means fading the trend or catching the dip in uptrend)
            # Wait, the prompt said:
            # Condition B (Trend): Compare Current Price vs EMA (200 period).
            # Condition C (Momentum): RSI < 30 (Bullish), RSI > 70 (Bearish).

            # Standard "Pullback" Reversal in Uptrend:
            # Price > EMA (Overall Uptrend) AND RSI < 30 (Local Oversold) -> BUY

            # Or Standard "Reversal" of Downtrend:
            # Price < EMA (Overall Downtrend) but RSI < 30? No that's catching a falling knife.

            # Let's stick to the prompt's implied logic coupled with the JS.
            # JS Logic was: BULL Pattern + Price > EMA + RSI < 70 (Trend Following).
            # User specifically requested: "Classic Reversal (RSI < 30 Bull, > 70 Bear)".

            # Interpretation of User's Reversal:
            # BUY when RSI < 30 (Oversold)
            # SELL when RSI > 70 (Overbought)
            # But what about EMA?
            # Usually:
            # Buy Reversal: Price is far below EMA? Or Price is above EMA but dipped?
            # Let's assume "Trend Pullback" which is safest:
            # BULL: Price > EMA (Long term trend UP) + RSI < 30 (Short term dip) + Bull Pattern
            # BEAR: Price < EMA (Long term trend DOWN) + RSI > 70 (Short term spike) + Bear Pattern

            is_bull_trend = current_price > val_ema
            is_bear_trend = current_price < val_ema

            is_oversold = val_rsi < self.rsi_os
            is_overbought = val_rsi > self.rsi_ob

            if pattern == 'BULL' and is_bull_trend and is_oversold:
                signal = 'CALL'
                reason = f"Bull Pattern + Price > EMA + RSI({val_rsi:.2f}) < {self.rsi_os}"
            elif pattern == 'BEAR' and is_bear_trend and is_overbought:
                signal = 'PUT'
                reason = f"Bear Pattern + Price < EMA + RSI({val_rsi:.2f}) > {self.rsi_ob}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'ema': val_ema,
                'rsi': val_rsi,
                'adx': val_adx,
                'pattern': pattern
            }
        } if signal else None
