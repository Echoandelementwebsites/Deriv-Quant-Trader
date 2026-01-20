import pandas as pd
import pandas_ta as ta
import numpy as np
import json
from deriv_quant_py.utils.indicators import calculate_ema, calculate_rsi, calculate_adx, detect_patterns, calculate_chop
from deriv_quant_py.config import Config
import importlib.util
import os
import logging

logger = logging.getLogger(__name__)

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
        # 1. Check for Ensemble Mode
        if config.get('mode') == 'ENSEMBLE':
            result = self._analyze_ensemble(df, config.get('members', []))
        elif strategy_type == 'AI_GENERATED':
            result = self._analyze_ai_generated(df, params)
        elif strategy_type == 'SUPERTREND':
            result = self._analyze_supertrend(df, config)
        elif strategy_type == 'BB_REVERSAL':
            result = self._analyze_bb_reversal(df, config)
        elif strategy_type == 'TREND_HEIKIN_ASHI':
            result = self._analyze_trend_ha(df, config)
        elif strategy_type == 'BREAKOUT':
            result = self._analyze_breakout(df, config)
        elif strategy_type == 'ICHIMOKU':
            result = self._analyze_ichimoku(df, config)
        elif strategy_type == 'EMA_CROSS':
            result = self._analyze_ema_cross(df, config)
        elif strategy_type == 'PARABOLIC_SAR':
            result = self._analyze_psar(df, config)
        elif strategy_type == 'EMA_PULLBACK':
            result = self._analyze_ema_pullback(df, config)
        elif strategy_type == 'MTF_TREND':
            result = self._analyze_mtf_trend(df, config)
        elif strategy_type == 'STREAK_EXHAUSTION':
            result = self._analyze_streak_exhaustion(df, config)
        elif strategy_type == 'VOL_SQUEEZE':
            result = self._analyze_vol_squeeze(df, config)
        elif strategy_type == 'TREND': # Legacy MACD
            result = self._analyze_trend(df, config)
        else: # Default/Legacy REVERSAL
            result = self._analyze_reversal(df, config)

        # SAFETY LAYER: Crash/Boom/Reset Filter
        if result and result['signal']:
            symbol = params.get('symbol', '')

            # Reject Puts on Bullish Assets
            if ('CRASH' in symbol or 'RDBULL' in symbol) and result['signal'] == 'PUT':
                return None

            # Reject Calls on Bearish Assets
            if ('BOOM' in symbol or 'RDBEAR' in symbol) and result['signal'] == 'CALL':
                return None

        return result

    def _analyze_ai_generated(self, df, params):
        symbol = params.get('symbol')
        if not symbol: return None

        try:
            module_path = f"deriv_quant_py/strategies/generated/{symbol}_ai.py"

            # Check if file exists
            if not os.path.exists(module_path):
                return None

            # Dynamic Import
            spec = importlib.util.spec_from_file_location(f"{symbol}_ai", module_path)
            if spec is None: return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Execute logic (Vectorized)
            call_series, put_series = module.strategy_logic(df)

            # Get last signal
            if call_series.empty or put_series.empty:
                return None

            is_call = call_series.iloc[-1]
            is_put = put_series.iloc[-1]
            current_price = df['close'].iloc[-1]

            if is_call:
                return {
                    'signal': 'CALL',
                    'reason': f"AI Generated Strategy ({symbol}) Triggered CALL",
                    'price': current_price,
                    'analysis': {'strategy': 'AI_GENERATED'}
                }
            elif is_put:
                return {
                    'signal': 'PUT',
                    'reason': f"AI Generated Strategy ({symbol}) Triggered PUT",
                    'price': current_price,
                    'analysis': {'strategy': 'AI_GENERATED'}
                }

            return None

        except Exception as e:
            logger.error(f"Error executing AI strategy in SignalGenerator for {symbol}: {e}")
            return None

    def _analyze_ensemble(self, df, members):
        # Run all member strategies
        votes = []
        analyses = {}

        for i, member_config in enumerate(members):
            strat_type = member_config.get('strategy_type')

            # Dispatch (Reuse existing methods)
            res = None
            if strat_type == 'SUPERTREND': res = self._analyze_supertrend(df, member_config)
            elif strat_type == 'BB_REVERSAL': res = self._analyze_bb_reversal(df, member_config)
            elif strat_type == 'TREND_HEIKIN_ASHI': res = self._analyze_trend_ha(df, member_config)
            elif strat_type == 'BREAKOUT': res = self._analyze_breakout(df, member_config)
            elif strat_type == 'ICHIMOKU': res = self._analyze_ichimoku(df, member_config)
            elif strat_type == 'EMA_CROSS': res = self._analyze_ema_cross(df, member_config)
            elif strat_type == 'PARABOLIC_SAR': res = self._analyze_psar(df, member_config)
            elif strat_type == 'EMA_PULLBACK': res = self._analyze_ema_pullback(df, member_config)
            elif strat_type == 'MTF_TREND': res = self._analyze_mtf_trend(df, member_config)
            elif strat_type == 'STREAK_EXHAUSTION': res = self._analyze_streak_exhaustion(df, member_config)
            elif strat_type == 'VOL_SQUEEZE': res = self._analyze_vol_squeeze(df, member_config)

            if res and res['signal']:
                votes.append(res['signal'])
                analyses[f'member_{i}_{strat_type}'] = res['analysis']
            else:
                votes.append(None) # Abstain

        # Unanimous Consent Logic
        # If any member says None, NO TRADE.
        # If members conflict (CALL vs PUT), NO TRADE.
        if None in votes: return None
        if not votes: return None # No members?

        if 'CALL' in votes and 'PUT' in votes: return None

        # If we got here, everyone agrees
        return {
            'signal': votes[0],
            'reason': f"Ensemble Agreement ({len(votes)} strats)",
            'price': df['close'].iloc[-1],
            'analysis': analyses
        }

    def _analyze_supertrend(self, df, config):
        # Concept: ATR Trailing Stop.
        # Params: length (7, 10, 14), multiplier (2.0, 3.0, 4.0), trend_ema, adx_threshold
        length = int(config.get('length', 10))
        multiplier = float(config.get('multiplier', 3.0))
        trend_ema_len = int(config.get('trend_ema', 200))
        adx_threshold = int(config.get('adx_threshold', 20))

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

        # ADX Filter
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is None or adx.empty:
             val_adx = 0
        else:
             val_adx = adx.iloc[-1, 0] # ADX_14

        # NEW: Major Trend Filter (EMA dynamic)
        ema_trend = ta.ema(df['close'], length=trend_ema_len)
        if ema_trend is None or ema_trend.empty:
            # Fallback if EMA cannot be calculated (not enough data)
            val_ema_trend = df['close'].iloc[-1]
        else:
            # Handle start where EMA is NaN
            ema_trend = ema_trend.fillna(df['close'])
            val_ema_trend = ema_trend.iloc[-1]

        current_price = df['close'].iloc[-1]
        val_st = st_line.iloc[-1]

        signal = None
        reason = ""

        # Logic:
        # Long: Close > ST Line (Trend Green) AND ADX > Thresh AND Price > EMA
        # Short: Close < ST Line (Trend Red) AND ADX > Thresh AND Price < EMA

        if val_adx > adx_threshold:
            if current_price > val_st and current_price > val_ema_trend:
                signal = 'CALL'
                reason = f"Supertrend: Price {current_price:.2f} > ST {val_st:.2f} & ADX {val_adx:.1f} > {adx_threshold} & Price > EMA {trend_ema_len}"
            elif current_price < val_st and current_price < val_ema_trend:
                signal = 'PUT'
                reason = f"Supertrend: Price {current_price:.2f} < ST {val_st:.2f} & ADX {val_adx:.1f} > {adx_threshold} & Price < EMA {trend_ema_len}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'supertrend': val_st,
                'adx': val_adx,
                'ema_200': val_ema_trend, # Key supertrend specific
                'strategy': 'SUPERTREND'
            }
        } if signal else None

    def _analyze_bb_reversal(self, df, config):
        # Concept: Mean Reversion (Forex/OTC)
        # Params: bb_length (20), bb_std (2.0, 2.5), rsi_period (7, 14), stoch_oversold, stoch_overbought
        bb_length = int(config.get('bb_length', 20))
        bb_std = float(config.get('bb_std', 2.0))
        rsi_p = int(config.get('rsi_period', 14))
        stoch_os = int(config.get('stoch_oversold', 20))
        stoch_ob = int(config.get('stoch_overbought', 80))

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

        # NEW: Stochastic (14, 3, 3)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        val_stoch_k = 50
        if stoch is not None and not stoch.empty:
             val_stoch_k = stoch.iloc[-1, 0]

        current_price = df['close'].iloc[-1]
        val_lower = lower.iloc[-1]
        val_upper = upper.iloc[-1]
        val_rsi = rsi.iloc[-1]

        signal = None
        reason = ""

        # Logic:
        # Long: Close < Lower Band AND RSI < 30 AND ADX < 25 AND Stoch K < Oversold
        # Short: Close > Upper Band AND RSI > 70 AND ADX < 25 AND Stoch K > Overbought

        # Hardcoded RSI limits (30/70) kept for legacy parity unless we want to param them too
        # Prompt said "Replace fixed Stoch < 20 with Stoch < params['stoch_oversold']"

        if val_adx < 25:
            if current_price < val_lower and val_rsi < 30 and val_stoch_k < stoch_os:
                signal = 'CALL'
                reason = f"BB Reversion: Price < Lower & RSI {val_rsi:.1f} < 30 & Stoch K {val_stoch_k:.1f} < {stoch_os}"
            elif current_price > val_upper and val_rsi > 70 and val_stoch_k > stoch_ob:
                signal = 'PUT'
                reason = f"BB Reversion: Price > Upper & RSI {val_rsi:.1f} > 70 & Stoch K {val_stoch_k:.1f} > {stoch_ob}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'bb_upper': val_upper,
                'bb_lower': val_lower,
                'rsi': val_rsi,
                'adx': val_adx,
                'stoch_k': val_stoch_k,
                'strategy': 'BB_REVERSAL'
            }
        } if signal else None

    def _analyze_trend_ha(self, df, config):
        # Heikin Ashi Trend (Same as Backtester)
        # Params: ema_period, adx_threshold, rsi_max
        ema_p = int(config.get('ema_period', 50))
        adx_threshold = int(config.get('adx_threshold', 25))
        rsi_max = int(config.get('rsi_max', 75))

        if len(df) < ema_p + 5:
            return None

        # EMA
        ema = ta.ema(df['close'], length=ema_p)
        val_ema = ema.iloc[-1]

        # ADX (Fixed 14, Dynamic Thresh)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        val_adx = adx.iloc[-1, 0] if (adx is not None and not adx.empty) else 0

        # NEW: ADX Slope (Rising)
        # Check current vs previous ADX
        # Handle short dataframe edge case
        adx_slope_rising = False
        val_adx_prev = 0
        if adx is not None and len(adx) >= 2:
            val_adx_prev = adx.iloc[-2, 0]
            adx_slope_rising = val_adx > val_adx_prev

        # NEW: RSI
        rsi = ta.rsi(df['close'], length=14)
        val_rsi = rsi.iloc[-1] if (rsi is not None and not rsi.empty) else 50

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
        strong_trend = (val_adx > adx_threshold) & adx_slope_rising

        # For Short, we use 100 - rsi_max as the lower bound
        rsi_min_short = 100 - rsi_max

        # Long: HA Green (Close > Open) & Flat Bottom & ADX > Thresh & Rising & Price > EMA & RSI < Max
        ha_green = ha_close > ha_open
        ha_flat_bottom = np.isclose(ha_low, ha_open)

        # Short: HA Red (Close < Open) & Flat Top & ADX > Thresh & Rising & Price < EMA & RSI > Min
        ha_red = ha_close < ha_open
        ha_flat_top = np.isclose(ha_high, ha_open)

        if strong_trend:
            if ha_green and ha_flat_bottom and current_price > val_ema and val_rsi < rsi_max:
                signal = 'CALL'
                reason = f"HA Trend: Green Flat Bottom + Price > EMA + ADX {val_adx:.1f} Rising + RSI {val_rsi:.1f} < {rsi_max}"
            elif ha_red and ha_flat_top and current_price < val_ema and val_rsi > rsi_min_short:
                signal = 'PUT'
                reason = f"HA Trend: Red Flat Top + Price < EMA + ADX {val_adx:.1f} Rising + RSI {val_rsi:.1f} > {rsi_min_short}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'ha_close': ha_close,
                'ema': val_ema,
                'adx': val_adx,
                'adx_prev': val_adx_prev,
                'rsi': val_rsi,
                'strategy': 'TREND_HEIKIN_ASHI'
            }
        } if signal else None

    def _analyze_breakout(self, df, config):
        # Volatility Squeeze Breakout (BB inside KC)
        # Params: rsi_entry_bull, rsi_entry_bear
        rsi_entry_bull = int(config.get('rsi_entry_bull', 55))
        rsi_entry_bear = int(config.get('rsi_entry_bear', 45))

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

        # NEW: RSI Momentum
        rsi = ta.rsi(df['close'], length=14)
        val_rsi = rsi.iloc[-1] if (rsi is not None and not rsi.empty) else 50

        current_price = df['close'].iloc[-1]
        val_bb_upper = bb_upper.iloc[-1]
        val_bb_lower = bb_lower.iloc[-1]

        signal = None
        reason = ""

        # Breakout Signals
        # Long: Previous was Squeeze AND Close > BB Upper AND RSI > Bull Thresh
        if prev_squeeze and current_price > val_bb_upper and val_rsi > rsi_entry_bull:
            signal = 'CALL'
            reason = f"Squeeze Breakout: Prev Squeeze & Price > BB Upper & RSI {val_rsi:.1f} > {rsi_entry_bull}"

        # Short: Previous was Squeeze AND Close < BB Lower AND RSI < Bear Thresh
        elif prev_squeeze and current_price < val_bb_lower and val_rsi < rsi_entry_bear:
            signal = 'PUT'
            reason = f"Squeeze Breakout: Prev Squeeze & Price < BB Lower & RSI {val_rsi:.1f} < {rsi_entry_bear}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'bb_upper': val_bb_upper,
                'bb_lower': val_bb_lower,
                'prev_squeeze': bool(prev_squeeze),
                'rsi': val_rsi,
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

    def _analyze_ichimoku(self, df, config):
        # Ichimoku Cloud Breakout
        # Params: tenkan, kijun, senkou_b
        tenkan_len = int(config.get('tenkan', 9))
        kijun_len = int(config.get('kijun', 26))
        senkou_b_len = int(config.get('senkou_b', 52))

        if len(df) < max(tenkan_len, kijun_len, senkou_b_len) + 5:
            return None

        # Ichimoku
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'],
                               tenkan=tenkan_len, kijun=kijun_len, senkou=senkou_b_len)
        if ichimoku is None: return None

        # Parse data
        data = ichimoku[0]
        # Current values
        # Access by index: ISA, ISB, ITS, IKS
        span_a = data.iloc[:, 0]
        span_b = data.iloc[:, 1]
        tenkan = data.iloc[:, 2]
        kijun = data.iloc[:, 3]

        current_price = df['close'].iloc[-1]

        val_span_a = span_a.iloc[-1]
        val_span_b = span_b.iloc[-1]
        val_tenkan = tenkan.iloc[-1]
        val_kijun = kijun.iloc[-1]

        # Previous values (for transition check)
        val_span_a_prev = span_a.iloc[-2]
        val_span_b_prev = span_b.iloc[-2]
        val_tenkan_prev = tenkan.iloc[-2]
        val_kijun_prev = kijun.iloc[-2]
        val_price_prev = df['close'].iloc[-2]

        # Logic:
        # Long: Price > Cloud (Span A & B) AND Tenkan > Kijun
        # Trigger: Transition from False to True

        cloud_top = np.maximum(val_span_a, val_span_b)
        cloud_bottom = np.minimum(val_span_a, val_span_b)

        cloud_top_prev = np.maximum(val_span_a_prev, val_span_b_prev)
        cloud_bottom_prev = np.minimum(val_span_a_prev, val_span_b_prev)

        long_cond = (current_price > cloud_top) and (val_tenkan > val_kijun)
        long_cond_prev = (val_price_prev > cloud_top_prev) and (val_tenkan_prev > val_kijun_prev)

        short_cond = (current_price < cloud_bottom) and (val_tenkan < val_kijun)
        short_cond_prev = (val_price_prev < cloud_bottom_prev) and (val_tenkan_prev < val_kijun_prev)

        signal = None
        reason = ""

        if long_cond and not long_cond_prev:
            signal = 'CALL'
            reason = f"Ichimoku: Price {current_price:.2f} > Cloud & Tenkan > Kijun (Transition)"
        elif short_cond and not short_cond_prev:
            signal = 'PUT'
            reason = f"Ichimoku: Price {current_price:.2f} < Cloud & Tenkan < Kijun (Transition)"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'span_a': val_span_a,
                'span_b': val_span_b,
                'tenkan': val_tenkan,
                'kijun': val_kijun,
                'strategy': 'ICHIMOKU'
            }
        } if signal else None

    def _analyze_ema_cross(self, df, config):
        # EMA Crossover (Golden Cross)
        # Params: ema_fast, ema_slow
        fast_len = int(config.get('ema_fast', 9))
        slow_len = int(config.get('ema_slow', 50))

        if len(df) < max(fast_len, slow_len) + 5:
            return None

        ema_fast = ta.ema(df['close'], length=fast_len)
        ema_slow = ta.ema(df['close'], length=slow_len)

        val_fast = ema_fast.iloc[-1]
        val_slow = ema_slow.iloc[-1]
        val_fast_prev = ema_fast.iloc[-2]
        val_slow_prev = ema_slow.iloc[-2]

        current_price = df['close'].iloc[-1]

        signal = None
        reason = ""

        # Logic:
        # Long: Fast crosses above Slow (Fast > Slow and Prev_Fast <= Prev_Slow)
        if val_fast > val_slow and val_fast_prev <= val_slow_prev:
            signal = 'CALL'
            reason = f"EMA Cross: Fast({fast_len}) {val_fast:.2f} crossed above Slow({slow_len}) {val_slow:.2f}"

        # Short: Fast crosses below Slow (Fast < Slow and Prev_Fast >= Prev_Slow)
        elif val_fast < val_slow and val_fast_prev >= val_slow_prev:
            signal = 'PUT'
            reason = f"EMA Cross: Fast({fast_len}) {val_fast:.2f} crossed below Slow({slow_len}) {val_slow:.2f}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'ema_fast': val_fast,
                'ema_slow': val_slow,
                'strategy': 'EMA_CROSS'
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

    def _analyze_psar(self, df, config):
        # Parabolic SAR for Step/Jump
        # Params: af, max_af, adx_threshold
        af = float(config.get('af', 0.02))
        max_af = float(config.get('max_af', 0.2))
        adx_thresh = int(config.get('adx_threshold', 20))

        if len(df) < 50: # PSAR needs some history to stabilize
            return None

        # PSAR
        psar = ta.psar(df['high'], df['low'], df['close'], af0=af, af=af, max_af=max_af)
        if psar is None: return None
        # Coalesce
        psar_combined = psar.iloc[:, 0].fillna(psar.iloc[:, 1])

        # ADX
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        val_adx = adx.iloc[-1, 0] if (adx is not None and not adx.empty) else 0

        trend_ok = val_adx > adx_thresh

        current_price = df['close'].iloc[-1]
        val_psar = psar_combined.iloc[-1]

        signal = None
        reason = ""

        if trend_ok:
            if current_price > val_psar:
                signal = 'CALL'
                reason = f"PSAR: Price {current_price:.2f} > SAR {val_psar:.2f} & ADX {val_adx:.1f} > {adx_thresh}"
            elif current_price < val_psar:
                signal = 'PUT'
                reason = f"PSAR: Price {current_price:.2f} < SAR {val_psar:.2f} & ADX {val_adx:.1f} > {adx_thresh}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'psar': val_psar,
                'adx': val_adx,
                'strategy': 'PARABOLIC_SAR'
            }
        } if signal else None

    def _analyze_ema_pullback(self, df, config):
        # EMA Pullback
        # Params: ema_trend, ema_pullback, rsi_limit
        ema_t_len = int(config.get('ema_trend', 200))
        ema_p_len = int(config.get('ema_pullback', 50))
        rsi_lim = int(config.get('rsi_limit', 60))

        if len(df) < max(ema_t_len, ema_p_len) + 5:
            return None

        ema_t = ta.ema(df['close'], length=ema_t_len)
        ema_p = ta.ema(df['close'], length=ema_p_len)
        rsi = ta.rsi(df['close'], length=14)

        if ema_t is None or ema_p is None or rsi is None: return None

        val_ema_t = ema_t.iloc[-1]
        val_ema_p = ema_p.iloc[-1]
        val_rsi = rsi.iloc[-1]

        current_price = df['close'].iloc[-1]
        current_open = df['open'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_high = df['high'].iloc[-1]

        signal = None
        reason = ""

        # Long: Trend Up, Touched Pullback, Bounce Up, RSI Safe
        trend_up = current_price > val_ema_t
        touched_support = current_low <= val_ema_p
        bounce_up = (current_price > current_open) and (current_price > val_ema_p)
        safe_rsi = val_rsi < rsi_lim

        if trend_up and touched_support and bounce_up and safe_rsi:
            signal = 'CALL'
            reason = f"EMA Pullback: Bounce off EMA({ema_p_len}) in Uptrend + RSI Safe"

        # Short: Trend Down, Touched Resist, Bounce Down, RSI Safe
        trend_down = current_price < val_ema_t
        touched_resist = current_high >= val_ema_p
        bounce_down = (current_price < current_open) and (current_price < val_ema_p)
        safe_rsi_short = val_rsi > (100 - rsi_lim)

        if trend_down and touched_resist and bounce_down and safe_rsi_short:
            signal = 'PUT'
            reason = f"EMA Pullback: Bounce off EMA({ema_p_len}) in Downtrend + RSI Safe"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'ema_trend': val_ema_t,
                'ema_pullback': val_ema_p,
                'rsi': val_rsi,
                'strategy': 'EMA_PULLBACK'
            }
        } if signal else None

    def _analyze_mtf_trend(self, df, config):
        # MTF Trend
        # Params: mtf_ema, local_ema
        mtf_len = int(config.get('mtf_ema', 1000))
        local_len = int(config.get('local_ema', 50))

        if len(df) < max(mtf_len, local_len) + 5:
            # Not enough history
            return None

        ema_mtf = ta.ema(df['close'], length=mtf_len)
        ema_local = ta.ema(df['close'], length=local_len)
        rsi = ta.rsi(df['close'], length=14)

        if ema_mtf is None or ema_local is None or rsi is None: return None

        val_mtf = ema_mtf.iloc[-1]
        val_local = ema_local.iloc[-1]
        val_rsi = rsi.iloc[-1]

        current_price = df['close'].iloc[-1]

        signal = None
        reason = ""

        # Long
        if current_price > val_mtf and current_price > val_local and val_rsi > 50:
            signal = 'CALL'
            reason = f"MTF Trend: Price > EMA({mtf_len}) & EMA({local_len}) & RSI > 50"

        # Short
        elif current_price < val_mtf and current_price < val_local and val_rsi < 50:
            signal = 'PUT'
            reason = f"MTF Trend: Price < EMA({mtf_len}) & EMA({local_len}) & RSI < 50"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'ema_mtf': val_mtf,
                'ema_local': val_local,
                'rsi': val_rsi,
                'strategy': 'MTF_TREND'
            }
        } if signal else None

    def _analyze_streak_exhaustion(self, df, config):
        streak_len = int(config.get('streak_length', 7))
        rsi_thresh = int(config.get('rsi_threshold', 80))

        if len(df) < streak_len + 2: return None

        # Check Streak
        # Get last 'streak_len' candles
        subset = df.iloc[-streak_len:]
        is_all_green = (subset['close'] > subset['open']).all()
        is_all_red = (subset['close'] < subset['open']).all()

        # Check RSI (2)
        rsi = ta.rsi(df['close'], length=2)
        if rsi is None: return None
        cur_rsi = rsi.iloc[-1]

        current_price = df['close'].iloc[-1]

        signal = None
        reason = ""

        if is_all_green and cur_rsi > rsi_thresh:
            signal = 'PUT'
            reason = f"Streak {streak_len} Green + RSI {cur_rsi:.1f} > {rsi_thresh}"
        elif is_all_red and cur_rsi < (100 - rsi_thresh):
            signal = 'CALL'
            reason = f"Streak {streak_len} Red + RSI {cur_rsi:.1f} < {100-rsi_thresh}"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_price,
            'analysis': {
                'streak_len': streak_len,
                'rsi_2': cur_rsi,
                'strategy': 'STREAK_EXHAUSTION'
            }
        } if signal else None

    def _analyze_vol_squeeze(self, df, config):
        lookback = int(config.get('squeeze_lookback', 20))
        bb_len = int(config.get('bb_length', 20))
        bb_std = float(config.get('bb_std', 2.0))

        if len(df) < lookback + 5: return None

        bb = ta.bbands(df['close'], length=bb_len, std=bb_std)
        if bb is None: return None
        upper = bb.iloc[:, 2]
        lower = bb.iloc[:, 0]

        bb_width = (upper - lower) / df['close']

        # Check if we are/were in a squeeze (Low Volatility)
        # We check the last 3 candles for a minimum width event
        # Note: 'recent_width' slice logic needs care.
        # last 3: -3, -2, -1.
        recent_width = bb_width.iloc[-3:]

        # History excluding recent 3
        hist_min = bb_width.iloc[-(lookback+3):-3].min()

        # Squeeze if any recent width <= historical min + tolerance
        is_squeezing = (recent_width <= hist_min + 0.00001).any()

        current_close = df['close'].iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]

        signal = None
        reason = ""

        if is_squeezing:
            if current_close > current_upper:
                signal = 'CALL'
                reason = "Vol Squeeze Breakout: Recent Squeeze + Break Upper"
            elif current_close < current_lower:
                signal = 'PUT'
                reason = "Vol Squeeze Breakout: Recent Squeeze + Break Lower"

        return {
            'signal': signal,
            'reason': reason,
            'price': current_close,
            'analysis': {
                'is_squeezing': bool(is_squeezing),
                'bb_upper': current_upper,
                'bb_lower': current_lower,
                'strategy': 'VOL_SQUEEZE'
            }
        } if signal else None
