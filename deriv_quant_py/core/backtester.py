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
import json

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

    def _eval_reversal(self, df, params):
        # Params: rsi_period, ema_period, rsi_vol_window, duration
        rsi_p = params['rsi_period']
        ema_p = params['ema_period']
        rsi_window = params['rsi_vol_window']
        duration = params['duration']

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

        valid_regime = df['CHOP'] >= 38
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
        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _eval_trend(self, df, params):
        # Params: macd_fast, macd_slow, ema_period, duration
        # Default signal length 9
        fast = params['macd_fast']
        slow = params['macd_slow']
        ema_p = params['ema_period']
        duration = params['duration']

        macd_df = ta.macd(df['close'], fast=fast, slow=slow, signal=9)
        # Columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        # Names depend on params, so we select by index or robust naming
        if macd_df is None or macd_df.empty:
            return 0, 0, 0

        macd_line = macd_df.iloc[:, 0]
        signal_line = macd_df.iloc[:, 2]

        df['EMA'] = ta.ema(df['close'], length=ema_p)

        call_signal = (macd_line > signal_line) & (df['close'] > df['EMA'])
        put_signal = (macd_line < signal_line) & (df['close'] < df['EMA'])

        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _eval_breakout(self, df, params):
        # Params: bb_length, bb_std, duration
        length = params['bb_length']
        std = params['bb_std']
        duration = params['duration']

        bb = ta.bbands(df['close'], length=length, std=std)
        if bb is None or bb.empty:
             return 0, 0, 0

        # Columns: BBL, BBM, BBU, BBB, BBP
        lower = bb.iloc[:, 0]
        upper = bb.iloc[:, 2]
        # bandwidth = bb.iloc[:, 3] # Check width if needed (threshold)

        # Simple breakout: Close outside bands
        call_signal = df['close'] > upper
        put_signal = df['close'] < lower

        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _calc_win_loss(self, df, call_signal, put_signal, duration):
        future_close = df['close'].shift(-duration)

        call_wins = call_signal & (future_close > df['close'])
        call_losses = call_signal & (future_close <= df['close'])

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
        Iterate through Strategy Types -> Grids.
        """
        TRAIN_SIZE = 3000
        TEST_SIZE = 500
        STEP_SIZE = 500

        if len(df) < TRAIN_SIZE + TEST_SIZE:
            logger.info(f"Not enough data for Rolling WFA. Needed {TRAIN_SIZE+TEST_SIZE}, got {len(df)}")
            return None

        # Strategy Grids
        strategies = {
            'REVERSAL': {
                'rsi_period': [7, 14],
                'ema_period': [100, 200],
                'rsi_vol_window': [50, 100],
                'duration': [3, 5]
            },
            'TREND': {
                'macd_fast': [12],
                'macd_slow': [26],
                'ema_period': [50, 100],
                'duration': [5, 10, 15]
            },
            'BREAKOUT': {
                'bb_length': [20],
                'bb_std': [2.0, 2.5],
                'duration': [3, 5]
            }
        }

        # Generate parameter combinations helper
        import itertools
        def get_combinations(grid):
            keys = grid.keys()
            values = grid.values()
            return [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Aggregate Results: best strategy type & params
        # We need to pick the best strategy type GLOBALLY across the WFA or per window?
        # Usually WFA simulates a system that can switch.
        # But here the user wants to find "The best strategy for this asset".
        # So we should probably optimize each strategy type independently via WFA,
        # then compare their final System Performance.

        best_system_result = None
        best_system_ev = -100

        for strat_type, grid in strategies.items():
            param_combos = get_combinations(grid)

            # Run WFA for this strategy type
            agg_results = {'wins': 0, 'losses': 0, 'signals': 0, 'configs': []}

            for start_idx in range(0, len(df) - TRAIN_SIZE - TEST_SIZE, STEP_SIZE):
                train_start = start_idx
                train_end = start_idx + TRAIN_SIZE
                test_start = train_end
                test_end = train_end + TEST_SIZE

                train_df = df.iloc[train_start:train_end]
                test_df = df.iloc[test_start:test_end]

                # 1. Optimize on Train
                best_local_config = None
                best_local_ev = -100

                for params in param_combos:
                    if strat_type == 'REVERSAL':
                        wins, losses, signals = self._eval_reversal(train_df, params)
                    elif strat_type == 'TREND':
                        wins, losses, signals = self._eval_trend(train_df, params)
                    elif strat_type == 'BREAKOUT':
                        wins, losses, signals = self._eval_breakout(train_df, params)

                    if signals < 3: continue

                    ev = self.calculate_expectancy(wins, losses)
                    if ev > best_local_ev:
                        best_local_ev = ev
                        best_local_config = params

                # 2. Verify on Test
                if best_local_config:
                    if strat_type == 'REVERSAL':
                        v_wins, v_losses, v_signals = self._eval_reversal(test_df, best_local_config)
                    elif strat_type == 'TREND':
                        v_wins, v_losses, v_signals = self._eval_trend(test_df, best_local_config)
                    elif strat_type == 'BREAKOUT':
                        v_wins, v_losses, v_signals = self._eval_breakout(test_df, best_local_config)

                    agg_results['wins'] += v_wins
                    agg_results['losses'] += v_losses
                    agg_results['signals'] += v_signals
                    agg_results['configs'].append(best_local_config)

            # Evaluate System Performance for this Strategy Type
            total_wins = agg_results['wins']
            total_losses = agg_results['losses']
            system_ev = self.calculate_expectancy(total_wins, total_losses)

            if system_ev > best_system_ev and len(agg_results['configs']) > 0:
                best_system_ev = system_ev

                final_win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
                last_config = agg_results['configs'][-1]

                best_system_result = {
                    'strategy_type': strat_type,
                    'config': last_config,
                    'WinRate': final_win_rate,
                    'Signals': agg_results['signals'],
                    'Expectancy': system_ev,
                    'optimal_duration': last_config['duration'] # Extract specifically for top-level access
                }

        return best_system_result

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

            # Save core metrics
            existing.win_rate = float(result['WinRate'])
            existing.signal_count = int(result['Signals'])
            existing.last_updated = pd.Timestamp.utcnow()
            existing.optimal_duration = int(result['optimal_duration'])

            # Save new fields
            existing.strategy_type = result['strategy_type']
            existing.config_json = json.dumps(result['config'])

            # Legacy fields (for backward compatibility if possible, or just set defaults)
            # Some UI components might still read rsi_period/ema_period directly.
            # We map REVERSAL params if available.
            config = result['config']
            if result['strategy_type'] == 'REVERSAL':
                existing.rsi_period = int(config.get('rsi_period', 14))
                existing.ema_period = int(config.get('ema_period', 200))
                existing.rsi_vol_window = int(config.get('rsi_vol_window', 100))
            else:
                # For Trend/Breakout, maybe set to 0 or keep last value to indicate "Not used"
                # Or set to defaults to avoid NULLs if code expects integers
                existing.rsi_period = 0
                existing.ema_period = int(config.get('ema_period', 0)) # Trend uses ema_period
                existing.rsi_vol_window = 0

            session.commit()
            logger.info(f"Saved optimized config for {symbol} ({existing.strategy_type}): Dur={existing.optimal_duration}m EV={result.get('Expectancy',0):.2f}")
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
