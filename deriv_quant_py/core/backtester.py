import pandas as pd
import pandas_ta as ta
import plotly.express as px
import plotly.graph_objects as go
from deriv_quant_py.core.connection import DerivClient
from deriv_quant_py.strategies.triple_confluence import TripleConfluenceStrategy
from deriv_quant_py.config import Config
import asyncio

class Backtester:
    def __init__(self, client: DerivClient):
        self.client = client

    async def fetch_history(self, symbol, count=5000):
        req = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": 60
        }
        res = await self.client.send_request(req)
        if 'candles' in res:
            df = pd.DataFrame(res['candles'])
            df['close'] = df['close'].astype(float)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
            return df
        return pd.DataFrame()

    def run_grid_search(self, df):
        """
        Runs a grid search on RSI and EMA parameters.
        Returns a DataFrame of results.
        """
        results = []

        rsi_periods = [7, 14, 21]
        ema_periods = [50, 100, 200]

        # Simple backtest loop
        # We need to instantiate strategy with different configs
        # But Strategy class reads from Config singleton.
        # Refactor Strategy to accept params? Yes, or context.
        # For now, we'll manually calculate signals here to avoid global config messing up.

        for rsi_p in rsi_periods:
            for ema_p in ema_periods:
                # Calc indicators
                df['EMA'] = ta.ema(df['close'], length=ema_p)
                df['RSI'] = ta.rsi(df['close'], length=rsi_p)

                # We also need Patterns (independent of params)
                # But patterns are heavy, calc once?
                # Let's just simulate the "Trend + Momentum" part for speed,
                # as patterns are standard.
                # Or use the Strategy logic if we can mock Config.

                # Let's simplify: Count profitable signals based on Reversal Logic
                # BUY: Price > EMA and RSI < 30
                # SELL: Price < EMA and RSI > 70
                # Win = Price closes higher after 3 candles (3 mins)

                signals = 0
                wins = 0

                for i in range(max(rsi_p, ema_p), len(df) - 3):
                    price = df['close'].iloc[i]
                    ema_val = df['EMA'].iloc[i]
                    rsi_val = df['RSI'].iloc[i]

                    signal = None
                    if price > ema_val and rsi_val < 30:
                        signal = 'CALL'
                    elif price < ema_val and rsi_val > 70:
                        signal = 'PUT'

                    if signal:
                        signals += 1
                        # Check result 3 candles later
                        future_price = df['close'].iloc[i+3]
                        if signal == 'CALL' and future_price > price:
                            wins += 1
                        elif signal == 'PUT' and future_price < price:
                            wins += 1

                win_rate = (wins / signals * 100) if signals > 0 else 0
                results.append({
                    'RSI': rsi_p,
                    'EMA': ema_p,
                    'Signals': signals,
                    'WinRate': win_rate
                })

        return pd.DataFrame(results)

    def generate_heatmap(self, results_df):
        pivot = results_df.pivot(index='EMA', columns='RSI', values='WinRate')
        fig = px.imshow(pivot,
                        labels=dict(x="RSI Period", y="EMA Period", color="Win Rate (%)"),
                        title="Backtest Grid Search Results")
        return fig
