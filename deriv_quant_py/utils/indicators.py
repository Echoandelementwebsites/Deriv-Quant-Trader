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
    engulfing = ta.cdl_engulfing(opens, highs, lows, closes)
    # Hammer
    hammer = ta.cdl_hammer(opens, highs, lows, closes)
    # 3 Black Crows (Bear)
    crows = ta.cdl_3blackcrows(opens, highs, lows, closes)

    # Check last value
    last_idx = closes.index[-1]

    is_bull_engulfing = engulfing.loc[last_idx] > 0 if engulfing is not None else False
    is_hammer = hammer.loc[last_idx] > 0 if hammer is not None else False
    is_bear_engulfing = engulfing.loc[last_idx] < 0 if engulfing is not None else False
    is_crows = crows.loc[last_idx] < 0 if crows is not None else False

    if is_bull_engulfing or is_hammer:
        return 'BULL'
    if is_bear_engulfing or is_crows:
        return 'BEAR'

    return None
