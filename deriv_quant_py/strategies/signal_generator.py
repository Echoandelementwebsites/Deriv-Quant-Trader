import pandas as pd
import pandas_ta as ta
import json
from deriv_quant_py.utils.indicators import calculate_ema, calculate_rsi, calculate_adx, detect_patterns, calculate_chop
from deriv_quant_py.config import Config

class SignalGenerator:
    def __init__(self):
        self.ema_period = Config.EMA_PERIOD
        self.rsi_period = Config.RSI_PERIOD
        # Default fallback config if nothing passed

    def analyze(self, candles: list, params: dict = None):
        """
        Analyzes a list of candles to produce a signal.
        candles: list of dicts {open, high, low, close, epoch}
        params: dict from DB (StrategyParams), containing 'strategy_type', 'config_json', etc.
        """
        if not candles:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        strategy_type = 'REVERSAL' # Default
        config = {}

        # Parse params
        if params:
            # Check for new schema
            if 'strategy_type' in params and params['strategy_type']:
                strategy_type = params['strategy_type']

            # Check for config_json or flattened legacy params
            if 'config_json' in params and params['config_json']:
                try:
                    if isinstance(params['config_json'], str):
                         config = json.loads(params['config_json'])
                    else:
                         config = params['config_json']
                except:
                    pass

            # Merge flattened params for legacy compatibility (REVERSAL uses these top-level)
            if strategy_type == 'REVERSAL':
                 if 'rsi_period' in params: config['rsi_period'] = params['rsi_period']
                 if 'ema_period' in params: config['ema_period'] = params['ema_period']
                 if 'rsi_vol_window' in params: config['rsi_vol_window'] = params['rsi_vol_window']

        # Dispatch
        if strategy_type == 'TREND':
            return self._analyze_trend(df, config)
        elif strategy_type == 'BREAKOUT':
            return self._analyze_breakout(df, config)
        else:
            return self._analyze_reversal(df, config)

    def _analyze_reversal(self, df, config):
        # Default config
        ema_p = config.get('ema_period', self.ema_period)
        rsi_p = config.get('rsi_period', self.rsi_period)
        rsi_vol_window = config.get('rsi_vol_window', 100)

        if len(df) < max(ema_p, rsi_vol_window) + 5:
            return None

        # Indicators
        ema = calculate_ema(df['close'], ema_p)
        rsi = calculate_rsi(df['close'], rsi_p)
        chop = calculate_chop(df['high'], df['low'], df['close'], 14)

        # Dynamic Bands
        rsi_rolling = rsi.rolling(window=rsi_vol_window)
        rsi_mean = rsi_rolling.mean()
        rsi_std = rsi_rolling.std()

        # Current Values
        current_price = df['close'].iloc[-1]
        val_ema = ema.iloc[-1]
        val_rsi = rsi.iloc[-1]
        val_chop = chop.iloc[-1] if chop is not None else 50

        dynamic_ob = rsi_mean.iloc[-1] + (2 * rsi_std.iloc[-1])
        dynamic_os = rsi_mean.iloc[-1] - (2 * rsi_std.iloc[-1])

        pattern = detect_patterns(df['open'], df['high'], df['low'], df['close'])

        signal = None
        reason = ""

        # Logic 0: Chop Filter (Reject Strong Trends)
        if val_chop < 38:
            return None

        # Logic 1: Reversal
        is_bull_trigger = val_rsi < dynamic_os
        is_bull_safe = val_rsi < 45
        is_bull_trend = current_price > val_ema

        is_bear_trigger = val_rsi > dynamic_ob
        is_bear_safe = val_rsi > 55
        is_bear_trend = current_price < val_ema

        if pattern == 'BULL' and is_bull_trend and is_bull_trigger and is_bull_safe:
            signal = 'CALL'
            reason = f"Reversal: Bull Pattern + Price > EMA + RSI({val_rsi:.2f}) < Dyn({dynamic_os:.2f})"
        elif pattern == 'BEAR' and is_bear_trend and is_bear_trigger and is_bear_safe:
            signal = 'PUT'
            reason = f"Reversal: Bear Pattern + Price < EMA + RSI({val_rsi:.2f}) > Dyn({dynamic_ob:.2f})"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'ema': val_ema,
                'rsi': val_rsi,
                'pattern': pattern,
                'strategy': 'REVERSAL'
            }
        } if signal else None

    def _analyze_trend(self, df, config):
        # Config: macd_fast, macd_slow, ema_period
        fast = int(config.get('macd_fast', 12))
        slow = int(config.get('macd_slow', 26))
        ema_p = int(config.get('ema_period', 50))

        if len(df) < max(slow, ema_p) + 5:
            return None

        # MACD
        macd_df = ta.macd(df['close'], fast=fast, slow=slow, signal=9)
        if macd_df is None or macd_df.empty:
            return None

        macd_val = macd_df.iloc[-1, 0]
        signal_val = macd_df.iloc[-1, 2]

        # EMA
        ema = ta.ema(df['close'], length=ema_p)
        val_ema = ema.iloc[-1]

        current_price = df['close'].iloc[-1]

        signal = None
        reason = ""

        # Logic: Follow Trend
        # Long: MACD > Signal AND Price > EMA
        if macd_val > signal_val and current_price > val_ema:
             signal = 'CALL'
             reason = f"Trend: MACD > Signal + Price > EMA({ema_p})"

        # Short: MACD < Signal AND Price < EMA
        elif macd_val < signal_val and current_price < val_ema:
             signal = 'PUT'
             reason = f"Trend: MACD < Signal + Price < EMA({ema_p})"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'macd': macd_val,
                'macd_signal': signal_val,
                'ema': val_ema,
                'strategy': 'TREND'
            }
        } if signal else None

    def _analyze_breakout(self, df, config):
        # Config: bb_length, bb_std
        length = int(config.get('bb_length', 20))
        std = float(config.get('bb_std', 2.0))

        if len(df) < length + 5:
            return None

        bb = ta.bbands(df['close'], length=length, std=std)
        if bb is None or bb.empty:
            return None

        lower = bb.iloc[-1, 0]
        upper = bb.iloc[-1, 2]

        current_price = df['close'].iloc[-1]

        signal = None
        reason = ""

        # Logic: Breakout
        if current_price > upper:
            signal = 'CALL'
            reason = f"Breakout: Price {current_price:.2f} > UpperBB {upper:.2f}"
        elif current_price < lower:
            signal = 'PUT'
            reason = f"Breakout: Price {current_price:.2f} < LowerBB {lower:.2f}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'bb_upper': upper,
                'bb_lower': lower,
                'strategy': 'BREAKOUT'
            }
        } if signal else None
