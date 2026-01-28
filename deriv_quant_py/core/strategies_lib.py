import pandas as pd
import pandas_ta as ta
import numpy as np
import importlib.util
import os
import logging

logger = logging.getLogger(__name__)

def gen_signals_trend_ha(df, params):
    df = df.copy()
    ema_p = int(params.get('ema_period', 50))
    adx_threshold = int(params.get('adx_threshold', 25))
    rsi_max = int(params.get('rsi_max', 75))

    df['EMA'] = ta.ema(df['close'], length=ema_p)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is None or adx_df.empty: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    df['ADX'] = adx_df.iloc[:, 0]
    df['ADX_Prev'] = df['ADX'].shift(1)
    adx_rising = df['ADX'] > df['ADX_Prev']
    df['RSI'] = ta.rsi(df['close'], length=14)
    ha_df = ta.ha(df['open'], df['high'], df['low'], df['close'])
    if ha_df is None or ha_df.empty: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    ha_green = ha_df['HA_close'] > ha_df['HA_open']
    ha_red = ha_df['HA_close'] < ha_df['HA_open']
    ha_flat_bottom = np.isclose(ha_df['HA_low'], ha_df['HA_open'])
    ha_flat_top = np.isclose(ha_df['HA_high'], ha_df['HA_open'])
    strong_trend = (df['ADX'] > adx_threshold) & adx_rising
    rsi_min_short = 100 - rsi_max

    call_signal = ha_green & ha_flat_bottom & strong_trend & (df['close'] > df['EMA']) & (df['RSI'] < rsi_max)
    put_signal = ha_red & ha_flat_top & strong_trend & (df['close'] < df['EMA']) & (df['RSI'] > rsi_min_short)

    return call_signal.fillna(False), put_signal.fillna(False)

def gen_signals_breakout(df, params):
    df = df.copy()
    rsi_entry_bull = int(params.get('rsi_entry_bull', 55))
    rsi_entry_bear = int(params.get('rsi_entry_bear', 45))
    bb_length = 20
    bb_std = 2.0
    kc_length = 20
    kc_scalar = 1.5

    bb = ta.bbands(df['close'], length=bb_length, std=bb_std)
    if bb is None or bb.empty: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    bb_lower = bb.iloc[:, 0]
    bb_upper = bb.iloc[:, 2]

    kc = ta.kc(df['high'], df['low'], df['close'], length=kc_length, scalar=kc_scalar)
    if kc is None or kc.empty: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    kc_lower = kc.iloc[:, 0]
    kc_upper = kc.iloc[:, 2]

    is_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    prev_squeeze = is_squeeze.shift(1).fillna(False)
    rsi = ta.rsi(df['close'], length=14)

    call_signal = prev_squeeze & (df['close'] > bb_upper) & (rsi > rsi_entry_bull)
    put_signal = prev_squeeze & (df['close'] < bb_lower) & (rsi < rsi_entry_bear)
    return call_signal.fillna(False), put_signal.fillna(False)

def gen_signals_supertrend(df, params):
    df = df.copy()
    length = params['length']
    multiplier = params['multiplier']
    trend_ema_len = int(params.get('trend_ema', 200))
    adx_threshold = int(params.get('adx_threshold', 20))

    st = ta.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
    if st is None or st.empty: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    st_line = st.iloc[:, 0]

    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is None or adx_df.empty: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    adx = adx_df.iloc[:, 0]
    trend_filter = adx > adx_threshold

    ema_trend = ta.ema(df['close'], length=trend_ema_len)
    ema_trend = ema_trend.fillna(df['close'])

    call_signal = (df['close'] > st_line) & trend_filter & (df['close'] > ema_trend)
    put_signal = (df['close'] < st_line) & trend_filter & (df['close'] < ema_trend)
    return call_signal.fillna(False), put_signal.fillna(False)

def gen_signals_bb_reversal(df, params):
    df = df.copy()
    bb_length = params['bb_length']
    bb_std = params['bb_std']
    rsi_period = params['rsi_period']
    stoch_os = int(params.get('stoch_oversold', 20))
    stoch_ob = int(params.get('stoch_overbought', 80))

    bb = ta.bbands(df['close'], length=bb_length, std=bb_std)
    if bb is None or bb.empty: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    lower = bb.iloc[:, 0]
    upper = bb.iloc[:, 2]

    rsi = ta.rsi(df['close'], length=rsi_period)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is None: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    adx = adx_df.iloc[:, 0]

    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
    if stoch is None: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    stoch_k = stoch.iloc[:, 0]

    range_filter = adx < 25
    call_signal = (df['close'] < lower) & (rsi < 30) & (stoch_k < stoch_os) & range_filter
    put_signal = (df['close'] > upper) & (rsi > 70) & (stoch_k > stoch_ob) & range_filter
    return call_signal.fillna(False), put_signal.fillna(False)

def gen_signals_ichimoku(df, params):
    df = df.copy()
    tenkan_len = int(params.get('tenkan', 9))
    kijun_len = int(params.get('kijun', 26))
    senkou_b_len = int(params.get('senkou_b', 52))

    ichimoku = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=tenkan_len, kijun=kijun_len, senkou=senkou_b_len)
    if ichimoku is None: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    data = ichimoku[0]
    span_a, span_b, tenkan, kijun = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], data.iloc[:, 3]

    cloud_top = np.maximum(span_a, span_b)
    cloud_bottom = np.minimum(span_a, span_b)

    long_cond = (df['close'] > cloud_top) & (tenkan > kijun)
    short_cond = (df['close'] < cloud_bottom) & (tenkan < kijun)
    call_signal = long_cond & (~long_cond.shift(1).fillna(False))
    put_signal = short_cond & (~short_cond.shift(1).fillna(False))
    return call_signal.fillna(False), put_signal.fillna(False)

def gen_signals_ema_cross(df, params):
    df = df.copy()
    fast_len = int(params.get('ema_fast', 9))
    slow_len = int(params.get('ema_slow', 50))
    ema_fast = ta.ema(df['close'], length=fast_len)
    ema_slow = ta.ema(df['close'], length=slow_len)
    fast_gt_slow = ema_fast > ema_slow
    fast_lt_slow = ema_fast < ema_slow
    call_signal = fast_gt_slow & (~fast_gt_slow.shift(1).fillna(False))
    put_signal = fast_lt_slow & (~fast_lt_slow.shift(1).fillna(False))
    return call_signal.fillna(False), put_signal.fillna(False)

def gen_signals_psar(df, params):
    df = df.copy()
    af = float(params['af'])
    max_af = float(params['max_af'])
    adx_thresh = int(params.get('adx_threshold', 20))
    psar = ta.psar(df['high'], df['low'], df['close'], af0=af, af=af, max_af=max_af)
    if psar is None: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    psar_combined = psar.iloc[:, 0].fillna(psar.iloc[:, 1])
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    adx = adx_df.iloc[:, 0] if adx_df is not None else 0
    trend_ok = adx > adx_thresh
    call_signal = (df['close'] > psar_combined) & trend_ok
    put_signal = (df['close'] < psar_combined) & trend_ok
    return call_signal.fillna(False), put_signal.fillna(False)

def gen_signals_ema_pullback(df, params):
    df = df.copy()
    ema_t_len = int(params['ema_trend'])
    ema_p_len = int(params['ema_pullback'])
    rsi_lim = int(params['rsi_limit'])
    df['EMA_Trend'] = ta.ema(df['close'], length=ema_t_len)
    df['EMA_Pullback'] = ta.ema(df['close'], length=ema_p_len)
    df['RSI'] = ta.rsi(df['close'], length=14)
    if df['EMA_Trend'] is None or df['EMA_Pullback'] is None: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    trend_up = df['close'] > df['EMA_Trend']
    touched_support = df['low'] <= df['EMA_Pullback']
    bounce_up = (df['close'] > df['open']) & (df['close'] > df['EMA_Pullback'])
    safe_rsi = df['RSI'] < rsi_lim
    call_signal = trend_up & touched_support & bounce_up & safe_rsi

    trend_down = df['close'] < df['EMA_Trend']
    touched_resist = df['high'] >= df['EMA_Pullback']
    bounce_down = (df['close'] < df['open']) & (df['close'] < df['EMA_Pullback'])
    safe_rsi_short = df['RSI'] > (100 - rsi_lim)
    put_signal = trend_down & touched_resist & bounce_down & safe_rsi_short
    return call_signal.fillna(False), put_signal.fillna(False)

def gen_signals_mtf_trend(df, params):
    df = df.copy()
    mtf_len = int(params['mtf_ema'])
    local_len = int(params['local_ema'])
    df['EMA_MTF'] = ta.ema(df['close'], length=mtf_len)
    df['EMA_Local'] = ta.ema(df['close'], length=local_len)
    df['RSI'] = ta.rsi(df['close'], length=14)
    if df['EMA_MTF'] is None: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    call_signal = (df['close'] > df['EMA_MTF']) & (df['close'] > df['EMA_Local']) & (df['RSI'] > 50)
    put_signal = (df['close'] < df['EMA_MTF']) & (df['close'] < df['EMA_Local']) & (df['RSI'] < 50)
    return call_signal.fillna(False), put_signal.fillna(False)

def gen_signals_streak_exhaustion(df, params):
    df = df.copy()
    streak_len = int(params.get('streak_length', 7))
    rsi_thresh = int(params.get('rsi_threshold', 80))

    # 1 = Green, -1 = Red
    direction = np.where(df['close'] > df['open'], 1, -1)
    direction_series = pd.Series(direction, index=df.index)

    # Rolling Min/Max to detect streaks
    roll_min = direction_series.rolling(window=streak_len).min() # All 1s -> Min is 1
    roll_max = direction_series.rolling(window=streak_len).max() # All -1s -> Max is -1

    rsi = ta.rsi(df['close'], length=2) # Fast RSI

    # PUT if Streak Green (1) & Overbought
    put_signal = (roll_min == 1) & (rsi > rsi_thresh)
    # CALL if Streak Red (-1) & Oversold
    call_signal = (roll_max == -1) & (rsi < (100 - rsi_thresh))

    return call_signal.fillna(False), put_signal.fillna(False)

def gen_signals_vol_squeeze(df, params):
    df = df.copy()
    lookback = int(params.get('squeeze_lookback', 20))
    bb_len = int(params.get('bb_length', 20))
    bb_std = float(params.get('bb_std', 2.0))

    # BB
    bb = ta.bbands(df['close'], length=bb_len, std=bb_std)
    if bb is None: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    upper = bb.iloc[:, 2]
    lower = bb.iloc[:, 0]

    # BB Width & Squeeze
    bb_width = (upper - lower) / df['close']
    # Check if current width is the lowest in 'lookback' candles
    min_width = bb_width.rolling(window=lookback).min()
    is_squeeze = bb_width <= min_width + 0.00001 # Floating point tolerance

    # Trigger: Squeeze occurred recently (last 3 candles) AND Price breaks band
    recent_squeeze = is_squeeze.rolling(window=3).max() > 0

    call_signal = recent_squeeze & (df['close'] > upper)
    put_signal = recent_squeeze & (df['close'] < lower)

    return call_signal.fillna(False), put_signal.fillna(False)

def dispatch_signal(strat_type, df, params):
    # 1. DeepSeek Strategy Dispatch (New Suffix)
    if strat_type.endswith('_ai'):
        try:
            # Load file: deriv_quant_py/strategies/generated/{strat_type}.py
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            module_path = os.path.join(base_dir, "deriv_quant_py", "strategies", "generated", f"{strat_type}.py")

            # Robust path check: if base_dir ends with deriv_quant_py, don't append it again
            if base_dir.endswith('deriv_quant_py'):
                 module_path = os.path.join(base_dir, "strategies", "generated", f"{strat_type}.py")

            if not os.path.exists(module_path):
                 # Try another path guess (if run from root)
                 module_path = os.path.abspath(f"deriv_quant_py/strategies/generated/{strat_type}.py")

            spec = importlib.util.spec_from_file_location(strat_type, module_path)
            if spec is None: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            return module.strategy_logic(df)
        except Exception as e:
            logger.error(f"Error executing AI strategy {strat_type}: {e}")
            return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    # 2. Legacy AI Generated Strategy Dispatch
    if strat_type == 'AI_GENERATED':
        symbol = params.get('symbol')
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            module_path = os.path.join(base_dir, "deriv_quant_py", "strategies", "generated", f"{symbol}_ai.py")

            if base_dir.endswith('deriv_quant_py'):
                 module_path = os.path.join(base_dir, "strategies", "generated", f"{symbol}_ai.py")

            if not os.path.exists(module_path):
                 module_path = os.path.abspath(f"deriv_quant_py/strategies/generated/{symbol}_ai.py")

            # Dynamic Import
            spec = importlib.util.spec_from_file_location(f"{symbol}_ai", module_path)
            if spec is None: return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Execute logic
            return module.strategy_logic(df)
        except Exception as e:
            # Log error and fallback
            logger.error(f"Error executing AI strategy for {symbol}: {e}")
            return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    if strat_type == 'BB_REVERSAL': return gen_signals_bb_reversal(df, params)
    elif strat_type == 'SUPERTREND': return gen_signals_supertrend(df, params)
    elif strat_type == 'TREND_HEIKIN_ASHI': return gen_signals_trend_ha(df, params)
    elif strat_type == 'BREAKOUT': return gen_signals_breakout(df, params)
    elif strat_type == 'ICHIMOKU': return gen_signals_ichimoku(df, params)
    elif strat_type == 'EMA_CROSS': return gen_signals_ema_cross(df, params)
    elif strat_type == 'PARABOLIC_SAR': return gen_signals_psar(df, params)
    elif strat_type == 'EMA_PULLBACK': return gen_signals_ema_pullback(df, params)
    elif strat_type == 'MTF_TREND': return gen_signals_mtf_trend(df, params)
    elif strat_type == 'STREAK_EXHAUSTION': return gen_signals_streak_exhaustion(df, params)
    elif strat_type == 'VOL_SQUEEZE': return gen_signals_vol_squeeze(df, params)
    return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
