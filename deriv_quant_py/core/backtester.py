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
        Rolling Walk-Forward Analysis:
        Slide a training window over the data, test on the subsequent small window.
        Train: 3000 candles
        Test: 500 candles
        Step: 500 candles
        """
        TRAIN_SIZE = 3000
        TEST_SIZE = 500
        STEP_SIZE = 500

        if len(df) < TRAIN_SIZE + TEST_SIZE:
            logger.info(f"Not enough data for Rolling WFA. Needed {TRAIN_SIZE+TEST_SIZE}, got {len(df)}")
            return None

        # Search Space
        rsi_periods = [7, 14]
        ema_periods = [100, 200]
        durations = [3, 5, 10, 15] # Minutes
        rsi_vol_windows = [50, 100]

        # Aggregate Results
        # Key: (rsi, ema, vol, dur) -> {wins, losses, signals}
        agg_results = {}

        # Rolling Window Loop
        for start_idx in range(0, len(df) - TRAIN_SIZE - TEST_SIZE, STEP_SIZE):
            train_start = start_idx
            train_end = start_idx + TRAIN_SIZE
            test_start = train_end
            test_end = train_end + TEST_SIZE

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            # 1. Optimize on Train Window
            best_local_config = None
            best_local_ev = -100

            for rsi_p in rsi_periods:
                for ema_p in ema_periods:
                    for win in rsi_vol_windows:
                        for dur in durations:
                            wins, losses, signals = self._evaluate_strategy(train_df, rsi_p, ema_p, win, dur)
                            if signals < 3: continue # Lower threshold for shorter windows

                            ev = self.calculate_expectancy(wins, losses)
                            if ev > best_local_ev:
                                best_local_ev = ev
                                best_local_config = (rsi_p, ema_p, win, dur)

            # 2. Verify on Test Window
            if best_local_config:
                # Run the chosen config on the Test set
                rsi_p, ema_p, win, dur = best_local_config
                v_wins, v_losses, v_signals = self._evaluate_strategy(test_df, rsi_p, ema_p, win, dur)

                # Accumulate results for this configuration
                # Note: We are accumulating results for the "Best Local Config".
                # But different windows might pick different configs.
                # However, the goal of WFA is often to simulate "If I followed the strategy optimizer, what would be my result?"
                # So we sum the results of the *chosen* strategies.

                # But wait, usually we want to see which *parameter set* is robust?
                # Or do we want to simulate the PnL of the system that re-optimizes?
                # "Logic: Divide data into N windows. Train on Window i, Test on Window i+1. Sum the results."
                # This usually means summing the PnL of the "Trading System" (which adapts).
                # So we sum the v_wins/v_losses of whatever config was chosen for that window.

                key = "System_Performance"
                if key not in agg_results:
                    agg_results[key] = {'wins': 0, 'losses': 0, 'signals': 0, 'configs': []}

                agg_results[key]['wins'] += v_wins
                agg_results[key]['losses'] += v_losses
                agg_results[key]['signals'] += v_signals
                agg_results[key]['configs'].append(best_local_config)

        # Final Evaluation
        if "System_Performance" not in agg_results:
            return None

        res = agg_results["System_Performance"]
        total_wins = res['wins']
        total_losses = res['losses']

        final_ev = self.calculate_expectancy(total_wins, total_losses)
        final_win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0

        # What to return? Ideally the "Latest" optimal params so the live system can use them.
        # So we should return the params from the LAST window's optimization.
        # And the metrics are the historical performance of the WFA process.

        last_config = res['configs'][-1] if res['configs'] else None

        if not last_config:
            return None

        # If overall Expectancy is positive, we consider it a pass
        if final_ev > 0:
            return {
                'rsi_period': last_config[0],
                'ema_period': last_config[1],
                'rsi_vol_window': last_config[2],
                'optimal_duration': last_config[3],
                'WinRate': final_win_rate,
                'Signals': res['signals'],
                'Expectancy': final_ev
            }

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
