import pandas as pd
import pandas_ta as ta
import plotly.express as px
from deriv_quant_py.core.connection import DerivClient
from deriv_quant_py.config import Config
from deriv_quant_py.database import init_db, StrategyParams
from deriv_quant_py.shared_state import state
from deriv_quant_py.utils.indicators import calculate_chop
import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, client: DerivClient):
        self.client = client
        self.SessionLocal = init_db(Config.DB_PATH)

    async def fetch_history_paginated(self, symbol, months=1):
        """
        Fetches historical data by paging backwards.
        Approx 1 month of 1m data ~ 43200 candles.
        Max count per request ~ 5000.
        """
        all_candles = []
        end_time = "latest"

        # 1 month approx 30 days * 24h * 60m = 43200
        # Let's target ~45000 candles
        target_count = 45000 * months
        batch_size = 5000

        fetched = 0
        while fetched < target_count:
            req = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": batch_size,
                "end": end_time,
                "start": 1,
                "style": "candles",
                "granularity": 60
            }

            res = await self.client.send_request(req)
            if 'error' in res:
                logger.error(f"Error fetching history for {symbol}: {res['error']['message']}")
                break

            if 'candles' in res:
                candles = res['candles']
                if not candles:
                    break

                # Parse
                df_batch = pd.DataFrame(candles)
                df_batch['close'] = df_batch['close'].astype(float)
                df_batch['open'] = df_batch['open'].astype(float)
                df_batch['high'] = df_batch['high'].astype(float)
                df_batch['low'] = df_batch['low'].astype(float)
                df_batch['epoch'] = pd.to_datetime(df_batch['epoch'], unit='s')

                # Prepend to list (since we are going backwards)
                all_candles = [df_batch] + all_candles

                fetched += len(candles)

                # Update end_time for next batch (oldest epoch of current batch)
                # API 'end' is exclusive? Or inclusive?
                # Usually we take the oldest epoch - 1s
                oldest_epoch = candles[0]['epoch'] # int timestamp
                end_time = oldest_epoch - 1

                # Safety break if we aren't getting data
                if len(candles) < 10:
                    break

                await asyncio.sleep(0.2) # Rate limit
            else:
                break

        if all_candles:
            final_df = pd.concat(all_candles, ignore_index=True)
            # Sort just in case
            final_df = final_df.sort_values('epoch').drop_duplicates('epoch').reset_index(drop=True)
            return final_df

        return pd.DataFrame()

    def calculate_expectancy(self, wins, losses, avg_win_payout=0.85):
        """
        Expectancy = (Win % * Avg Win Size) - (Loss % * Avg Loss Size)
        Assumes Stake = 1.0, Loss = 1.0
        """
        total = wins + losses
        if total == 0: return 0.0

        win_rate = wins / total
        loss_rate = losses / total

        # Expectancy per dollar risked
        ev = (win_rate * avg_win_payout) - (loss_rate * 1.0)
        return ev

    def _evaluate_strategy(self, df, rsi_p, ema_p, rsi_window, duration):
        """
        Vectorized evaluation of strategy on a dataframe.
        Returns (wins, losses, signals_count)
        """
        # Calculate Indicators
        # Using pandas_ta directly for speed
        df = df.copy() # Avoid SettingWithCopy

        # EMA
        df['EMA'] = ta.ema(df['close'], length=ema_p)

        # RSI
        rsi_series = ta.rsi(df['close'], length=rsi_p)
        df['RSI'] = rsi_series

        # Dynamic Bands
        rsi_roll = rsi_series.rolling(window=rsi_window)
        rsi_mean = rsi_roll.mean()
        rsi_std = rsi_roll.std()

        df['Dyn_Upper'] = rsi_mean + (2 * rsi_std)
        df['Dyn_Lower'] = rsi_mean - (2 * rsi_std)

        # CHOP
        df['CHOP'] = calculate_chop(df['high'], df['low'], df['close'], length=14)

        # Logic Vectors
        # 1. Chop Filter
        # Reject if CHOP < 38
        # Allow if CHOP >= 38
        valid_regime = df['CHOP'] >= 38

        # 2. Reversal Signals
        # CALL
        # Price > EMA (Trend)
        # RSI < Dyn_Lower (Trigger)
        # RSI < 45 (Safety)

        # PUT
        # Price < EMA
        # RSI > Dyn_Upper
        # RSI > 55

        call_signal = (
            valid_regime &
            (df['close'] > df['EMA']) &
            (df['RSI'] < df['Dyn_Lower']) &
            (df['RSI'] < 45)
        )

        put_signal = (
            valid_regime &
            (df['close'] < df['EMA']) &
            (df['RSI'] > df['Dyn_Upper']) &
            (df['RSI'] > 55)
        )

        # Evaluation
        # We need to look ahead 'duration' candles
        # Shift close price backward by duration to align 'future_close' with current row
        future_close = df['close'].shift(-duration)

        # Wins/Losses
        # Call Win: Future > Current
        call_wins = call_signal & (future_close > df['close'])
        call_losses = call_signal & (future_close <= df['close'])

        # Put Win: Future < Current
        put_wins = put_signal & (future_close < df['close'])
        put_losses = put_signal & (future_close >= df['close'])

        total_wins = call_wins.sum() + put_wins.sum()
        total_losses = call_losses.sum() + put_losses.sum()
        total_signals = call_signal.sum() + put_signal.sum()

        return total_wins, total_losses, total_signals

    def run_wfa_optimization(self, df):
        """
        Walk-Forward Analysis:
        1. Split Data (70% Train, 30% Verify)
        2. Optimize on Train
        3. Verify on Test
        """
        if len(df) < 500:
            return None

        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Search Space
        # Reduced space to keep runtime reasonable in sandbox
        rsi_periods = [7, 14]
        ema_periods = [100, 200]
        durations = [3, 5, 10, 15] # Minutes
        rsi_vol_windows = [50, 100]

        best_train_config = None
        best_train_ev = -100

        # TRAIN
        for rsi_p in rsi_periods:
            for ema_p in ema_periods:
                for win in rsi_vol_windows:
                    for dur in durations:
                        wins, losses, signals = self._evaluate_strategy(train_df, rsi_p, ema_p, win, dur)

                        if signals < 5: continue # Min sample size

                        ev = self.calculate_expectancy(wins, losses)

                        # Maximize Expectancy * log(signals) to favor frequent good trades?
                        # Or just raw Expectancy. Let's stick to Expectancy but require min trades.
                        if ev > best_train_ev:
                            best_train_ev = ev
                            best_train_config = {
                                'rsi_period': rsi_p,
                                'ema_period': ema_p,
                                'rsi_vol_window': win,
                                'optimal_duration': dur
                            }

        if not best_train_config:
            return None

        # VERIFY
        # Run best config on Test Data
        v_wins, v_losses, v_signals = self._evaluate_strategy(
            test_df,
            best_train_config['rsi_period'],
            best_train_config['ema_period'],
            best_train_config['rsi_vol_window'],
            best_train_config['optimal_duration']
        )

        v_ev = self.calculate_expectancy(v_wins, v_losses)
        v_win_rate = (v_wins / (v_wins + v_losses) * 100) if (v_wins + v_losses) > 0 else 0

        result = best_train_config.copy()
        result['WinRate'] = v_win_rate
        result['Signals'] = v_signals
        result['Expectancy'] = v_ev

        # Threshold: Expectancy > 0 implies profitable (with 85% payout)
        # A safer threshold might be > 0.05
        if v_ev > 0:
            return result
        else:
            # If verification failed (negative EV), do we discard?
            # Or return it but flag it?
            # User requirement: "Deploy: If Verification Expectancy > Threshold, save params".
            # If not, return None?
            return None

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

            existing.rsi_period = int(result['rsi_period'])
            existing.ema_period = int(result['ema_period'])
            existing.rsi_vol_window = int(result['rsi_vol_window'])
            existing.optimal_duration = int(result['optimal_duration'])

            existing.win_rate = float(result['WinRate'])
            existing.signal_count = int(result['Signals'])
            existing.last_updated = pd.Timestamp.utcnow()

            session.commit()
            logger.info(f"Saved optimized config for {symbol}: Dur={existing.optimal_duration}m EV={result.get('Expectancy',0):.2f}")
        except Exception as e:
            logger.error(f"Error saving strategy params for {symbol}: {e}")
            session.rollback()
        finally:
            session.close()

    async def run_full_scan(self):
        """
        Iterates all symbols in Scanner, runs WFA, saves best result.
        """
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

        logger.info(f"Starting WFA Scan on {total} symbols...")

        for i, symbol in enumerate(all_symbols):
            state.update_scan_progress(total, i + 1, symbol, "running")

            try:
                # 1. Fetch History (Paginated)
                df = await self.fetch_history_paginated(symbol, months=1)
                if df.empty:
                    continue

                # 2. Run WFA
                best = self.run_wfa_optimization(df)

                # 3. Save
                if best:
                    self.save_best_result(symbol, best)
                else:
                    logger.info(f"No profitable strategy found for {symbol} in WFA.")

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        state.update_scan_progress(total, total, None, "complete")
        logger.info("WFA Scan Complete.")

    def generate_heatmap(self, results_df):
        # Legacy support or update?
        # Since we don't output a grid search DF anymore (WFA internalizes it),
        # this might break if called.
        # For now, placeholder or removing usage in Dashboard.
        pass
