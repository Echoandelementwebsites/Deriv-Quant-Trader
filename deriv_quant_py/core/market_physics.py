import numpy as np
import pandas as pd
import pandas_ta as ta

class MarketPhysics:
    @staticmethod
    def get_asset_dna(df: pd.DataFrame) -> dict:
        # 1. Hurst Exponent (Trend Persistence)
        # Use simple Aggregated Variance method or R/S analysis proxy
        lags = range(2, 20)
        tau = [np.sqrt(np.std(df['close'].diff(lag))) for lag in lags]
        # Calculate slope of log-log plot (simplified)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0 # Approximation

        # 2. Chaos / Noise (Efficiency Ratio)
        # Kaufman's Efficiency Ratio (Signal / Noise)
        change = (df['close'] - df['close'].shift(10)).abs()
        volatility = (df['close'] - df['close'].shift(1)).abs().rolling(10).sum()
        er = change / volatility
        noise_index = 1.0 - er.iloc[-1] # 1.0 = Pure Noise

        # 3. Volatility Stability (Vol of Vol)
        returns = df['close'].pct_change()
        vol = returns.std()
        vol_vol = returns.rolling(20).std().std()

        return {
            "hurst": float(hurst),
            "noise_index": float(noise_index),
            "volatility": float(vol),
            "vol_of_vol": float(vol_vol),
            "regime": "TRENDING" if hurst > 0.55 else "MEAN_REVERSION"
        }

    @staticmethod
    def get_dominant_cycle(df: pd.DataFrame) -> int:
        """Finds the strongest autocorrelation lag (The Market's Heartbeat)."""
        try:
            returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            # Check lags 5 to 60
            correlations = [returns.autocorr(lag=i) for i in range(5, 60)]
            best_lag = np.argmax(np.abs(correlations)) + 5
            return int(best_lag)
        except:
            return 14 # Fallback

    @staticmethod
    def get_price_action_tape(df: pd.DataFrame, lookback=15) -> str:
        """Generates a text-based visual of the last N candles."""
        subset = df.iloc[-lookback:].copy()
        base = subset['open'].iloc[0]
        tape = []
        for i, row in subset.iterrows():
            o = (row['open'] / base - 1) * 1000
            h = (row['high'] / base - 1) * 1000
            l = (row['low'] / base - 1) * 1000
            c = (row['close'] / base - 1) * 1000
            shape = "Green" if c > o else "Red"
            if abs(c - o) < abs(h - l) * 0.1: shape = "Doji"
            tape.append(f"T{i}: {shape} (O:{o:.1f}, H:{h:.1f}, L:{l:.1f}, C:{c:.1f})")
        return "\n".join(tape)

    @staticmethod
    def get_market_structure(df: pd.DataFrame, window=50) -> dict:
        """
        Identifies Support, Resistance, and Range position.
        """
        recent = df.iloc[-window:]
        resistance = recent['high'].max()
        support = recent['low'].min()
        current = df['close'].iloc[-1]

        # Position in Range (0.0 to 1.0)
        range_pos = (current - support) / (resistance - support) if resistance != support else 0.5

        # Structure Tag
        if range_pos > 0.9: structure = "Testing Resistance"
        elif range_pos < 0.1: structure = "Testing Support"
        elif 0.4 < range_pos < 0.6: structure = "Equilibrium/Mid-Range"
        else: structure = "In Range"

        return {
            "resistance_price": resistance,
            "support_price": support,
            "range_position": range_pos,
            "structure_tag": structure
        }

    @staticmethod
    def perform_deep_analysis(df: pd.DataFrame) -> dict:
        """Deep Dive with Adaptive Metrics."""
        dna = MarketPhysics.get_asset_dna(df)
        cycle = MarketPhysics.get_dominant_cycle(df)
        tape = MarketPhysics.get_price_action_tape(df)
        structure = MarketPhysics.get_market_structure(df)

        # Calculate Trend Strength using the DYNAMIC cycle length
        adx = ta.adx(df['high'], df['low'], df['close'], length=cycle)
        adx_val = adx.iloc[-1, 0] if adx is not None and not adx.empty else 0

        return {
            **dna,
            "dominant_cycle": cycle,
            "price_action_tape": tape,
            "trend_strength": adx_val,
            "market_structure": structure
        }
