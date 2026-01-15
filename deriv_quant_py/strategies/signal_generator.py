import pandas as pd
import pandas_ta as ta
import numpy as np
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
        if strategy_type == 'SUPERTREND':
            return self._analyze_supertrend(df, config)
        elif strategy_type == 'BB_REVERSAL':
            return self._analyze_bb_reversal(df, config)
        elif strategy_type == 'TREND_HEIKIN_ASHI':
            return self._analyze_trend_ha(df, config)
        elif strategy_type == 'BREAKOUT':
            return self._analyze_breakout(df, config)
        elif strategy_type == 'TREND': # Legacy MACD
            return self._analyze_trend(df, config)
        else: # Default/Legacy REVERSAL
            return self._analyze_reversal(df, config)

    def _analyze_supertrend(self, df, config):
        # Concept: ATR Trailing Stop.
        # Params: length (7, 10, 14), multiplier (2.0, 3.0, 4.0)
        length = int(config.get('length', 10))
        multiplier = float(config.get('multiplier', 3.0))

        if len(df) < length + 5:
            return None

        # Supertrend
        # pandas_ta supertrend returns DataFrame with columns: SUPERT_{len}_{mult}, SUPERTd_{len}_{mult}, SUPERTl_{len}_{mult}, SUPERTs_{len}_{mult}
        st = ta.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
        if st is None or st.empty:
            return None

        # Identify the main Supertrend line column (usually the first one)
        # Or specifically: f"SUPERT_{length}_{multiplier}"
        st_col = f"SUPERT_{length}_{multiplier}"
        st_line = st[st_col]

        # ADX Filter > 20
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is None or adx.empty:
             val_adx = 0
        else:
             val_adx = adx.iloc[-1, 0] # ADX_14

        current_price = df['close'].iloc[-1]
        val_st = st_line.iloc[-1]

        signal = None
        reason = ""

        # Logic:
        # Long: Close > ST Line (Trend Green) AND ADX > 20
        # Short: Close < ST Line (Trend Red) AND ADX > 20

        if val_adx > 20:
            if current_price > val_st:
                signal = 'CALL'
                reason = f"Supertrend: Price {current_price:.2f} > ST {val_st:.2f} & ADX {val_adx:.1f} > 20"
            elif current_price < val_st:
                signal = 'PUT'
                reason = f"Supertrend: Price {current_price:.2f} < ST {val_st:.2f} & ADX {val_adx:.1f} > 20"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'supertrend': val_st,
                'adx': val_adx,
                'strategy': 'SUPERTREND'
            }
        } if signal else None

    def _analyze_bb_reversal(self, df, config):
        # Concept: Mean Reversion (Forex/OTC)
        # Params: bb_length (20), bb_std (2.0, 2.5), rsi_period (7, 14)
        bb_length = int(config.get('bb_length', 20))
        bb_std = float(config.get('bb_std', 2.0))
        rsi_p = int(config.get('rsi_period', 14))

        if len(df) < max(bb_length, rsi_p) + 5:
            return None

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=bb_length, std=bb_std)
        if bb is None or bb.empty:
            return None
        lower = bb.iloc[:, 0]
        upper = bb.iloc[:, 2]

        # RSI
        rsi = ta.rsi(df['close'], length=rsi_p)
        if rsi is None or rsi.empty:
            return None

        # ADX Filter < 25 (Not trending)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        val_adx = adx.iloc[-1, 0] if (adx is not None and not adx.empty) else 50

        current_price = df['close'].iloc[-1]
        val_lower = lower.iloc[-1]
        val_upper = upper.iloc[-1]
        val_rsi = rsi.iloc[-1]

        signal = None
        reason = ""

        # Logic:
        # Long: Close < Lower Band AND RSI < 30 AND ADX < 25
        # Short: Close > Upper Band AND RSI > 70 AND ADX < 25

        if val_adx < 25:
            if current_price < val_lower and val_rsi < 30:
                signal = 'CALL'
                reason = f"BB Reversion: Price < Lower & RSI {val_rsi:.1f} < 30 & ADX {val_adx:.1f} < 25"
            elif current_price > val_upper and val_rsi > 70:
                signal = 'PUT'
                reason = f"BB Reversion: Price > Upper & RSI {val_rsi:.1f} > 70 & ADX {val_adx:.1f} < 25"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'bb_upper': val_upper,
                'bb_lower': val_lower,
                'rsi': val_rsi,
                'adx': val_adx,
                'strategy': 'BB_REVERSAL'
            }
        } if signal else None

    def _analyze_trend_ha(self, df, config):
        # Heikin Ashi Trend (Same as Backtester)
        # Params: ema_period
        ema_p = int(config.get('ema_period', 50))

        if len(df) < ema_p + 5:
            return None

        # EMA
        ema = ta.ema(df['close'], length=ema_p)
        val_ema = ema.iloc[-1]

        # ADX (Fixed 14, Thresh 25)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        val_adx = adx.iloc[-1, 0] if (adx is not None and not adx.empty) else 0

        # Heikin Ashi
        ha_df = ta.ha(df['open'], df['high'], df['low'], df['close'])
        if ha_df is None or ha_df.empty:
            return None

        # Last Candle HA
        ha_open = ha_df['HA_open'].iloc[-1]
        ha_high = ha_df['HA_high'].iloc[-1]
        ha_low = ha_df['HA_low'].iloc[-1]
        ha_close = ha_df['HA_close'].iloc[-1]

        current_price = df['close'].iloc[-1]

        signal = None
        reason = ""

        # Logic
        strong_trend = val_adx > 25

        # Long: HA Green (Close > Open) & Flat Bottom (Low == Open) & ADX > 25 & Price > EMA
        ha_green = ha_close > ha_open
        ha_flat_bottom = np.isclose(ha_low, ha_open)

        # Short: HA Red (Close < Open) & Flat Top (High == Open) & ADX > 25 & Price < EMA
        ha_red = ha_close < ha_open
        ha_flat_top = np.isclose(ha_high, ha_open)

        if strong_trend:
            if ha_green and ha_flat_bottom and current_price > val_ema:
                signal = 'CALL'
                reason = f"HA Trend: Green Flat Bottom + Price > EMA + ADX {val_adx:.1f}"
            elif ha_red and ha_flat_top and current_price < val_ema:
                signal = 'PUT'
                reason = f"HA Trend: Red Flat Top + Price < EMA + ADX {val_adx:.1f}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'ha_close': ha_close,
                'ema': val_ema,
                'adx': val_adx,
                'strategy': 'TREND_HEIKIN_ASHI'
            }
        } if signal else None

    def _analyze_breakout(self, df, config):
        # Volatility Squeeze Breakout (BB inside KC)
        # Params: None (Fixed BB 20/2, KC 20/1.5) or could be passed.
        # Backtester uses fixed params, so we stick to that unless config overrides.

        bb_length = int(config.get('bb_length', 20))
        bb_std = float(config.get('bb_std', 2.0))
        kc_length = int(config.get('kc_length', 20))
        kc_scalar = float(config.get('kc_scalar', 1.5))

        if len(df) < max(bb_length, kc_length) + 5:
            return None

        # BB
        bb = ta.bbands(df['close'], length=bb_length, std=bb_std)
        if bb is None or bb.empty: return None
        bb_lower = bb.iloc[:, 0]
        bb_upper = bb.iloc[:, 2]

        # KC
        kc = ta.kc(df['high'], df['low'], df['close'], length=kc_length, scalar=kc_scalar)
        if kc is None or kc.empty: return None
        kc_lower = kc.iloc[:, 0]
        kc_upper = kc.iloc[:, 2]

        # Squeeze Logic: BB inside KC
        is_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # We need previous candle to be in squeeze
        prev_squeeze = is_squeeze.iloc[-2]

        current_price = df['close'].iloc[-1]
        val_bb_upper = bb_upper.iloc[-1]
        val_bb_lower = bb_lower.iloc[-1]

        signal = None
        reason = ""

        # Breakout Signals
        # Long: Previous was Squeeze AND Close > BB Upper
        if prev_squeeze and current_price > val_bb_upper:
            signal = 'CALL'
            reason = f"Squeeze Breakout: Prev Squeeze & Price {current_price:.2f} > BB Upper {val_bb_upper:.2f}"

        # Short: Previous was Squeeze AND Close < BB Lower
        elif prev_squeeze and current_price < val_bb_lower:
            signal = 'PUT'
            reason = f"Squeeze Breakout: Prev Squeeze & Price {current_price:.2f} < BB Lower {val_bb_lower:.2f}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'bb_upper': val_bb_upper,
                'bb_lower': val_bb_lower,
                'prev_squeeze': bool(prev_squeeze),
                'strategy': 'BREAKOUT'
            }
        } if signal else None

    def _analyze_reversal(self, df, config):
        # Legacy: Dynamic RSI Reversal
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
        # Legacy MACD Trend
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
