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
