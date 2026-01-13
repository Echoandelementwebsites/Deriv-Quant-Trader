import pandas as pd
import pandas_ta as ta

def calculate_ema(closes: pd.Series, length: int):
    return ta.ema(closes, length=length)

def calculate_rsi(closes: pd.Series, length: int):
    return ta.rsi(closes, length=length)

def calculate_adx(highs: pd.Series, lows: pd.Series, closes: pd.Series, length: int):
    # pandas_ta.adx returns a DataFrame with ADX_14, DMP_14, DMN_14
    adx_df = ta.adx(highs, lows, closes, length=length)
    if adx_df is not None and not adx_df.empty:
        return adx_df[f"ADX_{length}"]
    return None

def detect_patterns(opens: pd.Series, highs: pd.Series, lows: pd.Series, closes: pd.Series):
    """
    Detects patterns for the LAST candle in the series.
    Returns 'BULL', 'BEAR', or None.
    """
    # pandas_ta cdls patterns return 0 (no pattern), 100 (bull), -100 (bear)

    # Bullish Engulfing
    engulfing = ta.cdl_pattern(opens, highs, lows, closes, name="engulfing")
    # Hammer
    hammer = ta.cdl_pattern(opens, highs, lows, closes, name="hammer")
    # 3 Black Crows (Bear)
    crows = ta.cdl_pattern(opens, highs, lows, closes, name="3blackcrows")

    # Check last value
    last_idx = closes.index[-1]

    # Handle DataFrame return from cdl_pattern
    if engulfing is not None and not engulfing.empty:
        engulfing_val = engulfing['CDL_ENGULFING'].loc[last_idx]
    else:
        engulfing_val = 0

    if hammer is not None and not hammer.empty:
        hammer_val = hammer['CDL_HAMMER'].loc[last_idx]
    else:
        hammer_val = 0

    if crows is not None and not crows.empty:
        crows_val = crows['CDL_3BLACKCROWS'].loc[last_idx]
    else:
        crows_val = 0

    is_bull_engulfing = engulfing_val > 0
    is_hammer = hammer_val > 0
    is_bear_engulfing = engulfing_val < 0
    is_crows = crows_val < 0

    if is_bull_engulfing or is_hammer:
        return 'BULL'
    if is_bear_engulfing or is_crows:
        return 'BEAR'

    return None

def calculate_chop(highs: pd.Series, lows: pd.Series, closes: pd.Series, length: int = 14):
    """
    Calculates the Choppiness Index (CHOP).
    Values < 38 indicate a strong trend (Danger Zone for reversals).
    Values > 61 indicate a choppy market (Ideal for reversals).
    """
    chop = ta.chop(highs, lows, closes, length=length)
    return chop
