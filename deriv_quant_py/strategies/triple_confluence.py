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
        params: dict (optional) overrides:
            'rsi_period', 'ema_period', 'rsi_vol_window'
        """
        # Determine periods
        ema_p = params.get('ema_period', self.ema_period) if params else self.ema_period
        rsi_p = params.get('rsi_period', self.rsi_period) if params else self.rsi_period
        rsi_vol_window = params.get('rsi_vol_window', 100) if params else 100

        if len(candles) < max(ema_p, rsi_vol_window) + 5:
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

        # New: Choppiness Index (Regime Filter)
        # Using default length 14, but could be parameterized
        from deriv_quant_py.utils.indicators import calculate_chop
        chop = calculate_chop(df['high'], df['low'], df['close'], 14)

        # Dynamic RSI Thresholds
        rsi_rolling = rsi.rolling(window=rsi_vol_window)
        rsi_mean = rsi_rolling.mean()
        rsi_std = rsi_rolling.std()

        # Last values
        last_idx = df.index[-1]
        current_price = df['close'].iloc[-1]

        val_ema = ema.iloc[-1]
        val_rsi = rsi.iloc[-1]
        val_adx = adx.iloc[-1] if adx is not None else 0
        val_chop = chop.iloc[-1] if chop is not None else 50 # Default to neutral

        # Pattern Recognition
        pattern = detect_patterns(df['open'], df['high'], df['low'], df['close'])

        # Dynamic Bands (2 Sigma)
        # We need the bands at the current moment
        dynamic_ob = rsi_mean.iloc[-1] + (2 * rsi_std.iloc[-1])
        dynamic_os = rsi_mean.iloc[-1] - (2 * rsi_std.iloc[-1])

        signal = None
        reason = ""

        # Logic:
        # 0. Chop Filter (Regime)
        # Danger Zone: CHOP < 38 (Strong Trend) -> Reject Reversals
        if val_chop < 38:
            return None

        # 1. ADX Filter (Legacy, keeping as secondary confirmation or removing?)
        # User prompt said "Replace fixed thresholds... Add Choppiness Filter".
        # It didn't explicitly say remove ADX, but the logic flow implies CHOP is the new regime filter.
        # However, to avoid regression, let's keep ADX > 25 check if it was working,
        # OR assume CHOP handles the "Trending vs Ranging" better.
        # The prompt says: "High CHOP (> 60): Enable Reversal... Low CHOP (< 38): Disable."
        # It says "Since we are trading Reversals... we face highest risk during Parabolic moves... Add CHOP filter."
        # I will prioritize CHOP. If CHOP >= 38, we allow.

        # 2. Dynamic Trigger + Safety Rails
        # Buy Trigger: RSI < Dynamic Low
        # Buy Safety: RSI < 45 (Don't buy near 50 even if volatility is squeezed)

        # Sell Trigger: RSI > Dynamic High
        # Sell Safety: RSI > 55 (Don't sell near 50)

        is_bull_trigger = val_rsi < dynamic_os
        is_bull_safe = val_rsi < 45

        is_bear_trigger = val_rsi > dynamic_ob
        is_bear_safe = val_rsi > 55

        # 3. Trend Context (EMA)
        # "Triple Confluence" usually implies Trend Alignment.
        # Buy Reversal (Pullback) usually means Price > EMA (Trend is Up, catching dip).
        is_bull_trend = current_price > val_ema
        is_bear_trend = current_price < val_ema

        if pattern == 'BULL' and is_bull_trend and is_bull_trigger and is_bull_safe:
            signal = 'CALL'
            reason = f"Bull Pattern + Price > EMA + RSI({val_rsi:.2f}) < Dyn({dynamic_os:.2f})"
        elif pattern == 'BEAR' and is_bear_trend and is_bear_trigger and is_bear_safe:
            signal = 'PUT'
            reason = f"Bear Pattern + Price < EMA + RSI({val_rsi:.2f}) > Dyn({dynamic_ob:.2f})"

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
