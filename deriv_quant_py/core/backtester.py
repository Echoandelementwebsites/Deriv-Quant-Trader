import pandas as pd
import pandas_ta as ta
import plotly.express as px
import plotly.graph_objects as go
from deriv_quant_py.core.connection import DerivClient
from deriv_quant_py.strategies.triple_confluence import TripleConfluenceStrategy
from deriv_quant_py.config import Config
from deriv_quant_py.database import init_db, StrategyParams
from deriv_quant_py.shared_state import state
import asyncio
import logging

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, client: DerivClient):
        self.client = client
        self.SessionLocal = init_db(Config.DB_PATH)

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

        for rsi_p in rsi_periods:
            for ema_p in ema_periods:
                # Calc indicators
                df['EMA'] = ta.ema(df['close'], length=ema_p)
                df['RSI'] = ta.rsi(df['close'], length=rsi_p)

                signals = 0
                wins = 0

                for i in range(max(rsi_p, ema_p), len(df) - 3):
                    price = df['close'].iloc[i]
                    ema_val = df['EMA'].iloc[i]
                    rsi_val = df['RSI'].iloc[i]

                    signal = None
                    # Reversal Logic:
                    # CALL: Price > EMA (Trend Up) + RSI < 30 (Dip)
                    # PUT: Price < EMA (Trend Down) + RSI > 70 (Spike)
                    if price > ema_val and rsi_val < 30:
                        signal = 'CALL'
                    elif price < ema_val and rsi_val > 70:
                        signal = 'PUT'

                    if signal:
                        signals += 1
                        # Check result 3 candles later (3 min expiration proxy)
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

    def determine_best_config(self, results_df, min_trades=5):
        """
        Finds the best configuration.
        Logic: Filter for min_trades, then max(WinRate).
        Fallback: max(WinRate) ignoring trades if none meet threshold.
        """
        if results_df.empty:
            return None

        # Filter
        qualified = results_df[results_df['Signals'] >= min_trades]

        if not qualified.empty:
            # Sort by WinRate DESC, then Signals DESC
            best = qualified.sort_values(by=['WinRate', 'Signals'], ascending=[False, False]).iloc[0]
        else:
            # Fallback
            best = results_df.sort_values(by=['WinRate', 'Signals'], ascending=[False, False]).iloc[0]

        return best.to_dict()

    def save_best_result(self, symbol, result):
        if not result:
            return

        session = self.SessionLocal()
        try:
            # Upsert
            existing = session.query(StrategyParams).filter_by(symbol=symbol).first()
            if not existing:
                existing = StrategyParams(symbol=symbol)
                session.add(existing)

            existing.rsi_period = int(result['RSI'])
            existing.ema_period = int(result['EMA'])
            existing.win_rate = float(result['WinRate'])
            existing.signal_count = int(result['Signals'])
            existing.last_updated = pd.Timestamp.utcnow()

            session.commit()
            logger.info(f"Saved best config for {symbol}: RSI={existing.rsi_period} EMA={existing.ema_period} WR={existing.win_rate:.1f}% ({existing.signal_count})")
        except Exception as e:
            logger.error(f"Error saving strategy params for {symbol}: {e}")
            session.rollback()
        finally:
            session.close()

    async def run_full_scan(self):
        """
        Iterates all symbols in Scanner, runs backtest, saves best result.
        Updates SharedState progress.
        """
        # Get all symbols
        scanner_data = state.get_scanner_data()
        all_symbols = []
        for cat, assets in scanner_data.items():
            for a in assets:
                all_symbols.append(a['symbol'])

        total = len(all_symbols)
        if total == 0:
            logger.warning("No symbols found in scanner for full scan.")
            state.update_scan_progress(0, 0, None, "complete")
            return

        logger.info(f"Starting Full System Scan on {total} symbols...")

        for i, symbol in enumerate(all_symbols):
            state.update_scan_progress(total, i + 1, symbol, "running")

            try:
                # 1. Fetch History
                df = await self.fetch_history(symbol, count=2000) # 2000 candles sufficient for test
                if df.empty:
                    continue

                # 2. Run Grid Search
                results = self.run_grid_search(df)

                # 3. Determine Best
                best = self.determine_best_config(results)

                # 4. Save
                self.save_best_result(symbol, best)

                # Sleep slightly to avoid rate limits
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        state.update_scan_progress(total, total, None, "complete")
        logger.info("Full System Scan Complete.")

    def generate_heatmap(self, results_df):
        pivot = results_df.pivot(index='EMA', columns='RSI', values='WinRate')
        fig = px.imshow(pivot,
                        labels=dict(x="RSI Period", y="EMA Period", color="Win Rate (%)"),
                        title="Backtest Grid Search Results")
        return fig
