import pandas as pd
import pandas_ta as ta
import plotly.express as px
from deriv_quant_py.core.connection import DerivClient
from deriv_quant_py.config import Config
from deriv_quant_py.database import init_db, StrategyParams, BacktestResult
from deriv_quant_py.shared_state import state
from deriv_quant_py.utils.indicators import calculate_chop
from deriv_quant_py.core.metrics import MetricsEngine
from deriv_quant_py.core.optimization_worker import run_optimization_task
from deriv_quant_py.core.scan_manager import ScanManager
from deriv_quant_py.core import strategies_lib
import asyncio
import logging
import numpy as np
import json
import itertools
from itertools import combinations
import importlib.util
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, client: DerivClient):
        self.client = client
        self.SessionLocal = init_db(Config.DB_PATH)
        self.executor = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))
        self.scan_manager = ScanManager()

        # Default Strategy Configuration Grid
        default_durations = [1, 2, 3, 5, 10, 15]
        default_st_mult = [2.0, 3.0]

        self.strategies = {
            'SUPERTREND': {
                'length': [10, 14], 'multiplier': default_st_mult, 'adx_threshold': [20, 25, 30], 'trend_ema': [100, 200], 'duration': default_durations
            },
            'TREND_HEIKIN_ASHI': {
                'ema_period': [50, 100, 200], 'adx_threshold': [20, 25], 'rsi_max': [70, 75, 80], 'duration': default_durations
            },
            'BB_REVERSAL': {
                'bb_length': [20], 'bb_std': [2.0, 2.5], 'rsi_period': [14], 'stoch_oversold': [15, 20], 'stoch_overbought': [80, 85], 'duration': default_durations
            },
            'BREAKOUT': {
                'rsi_entry_bull': [50, 55], 'rsi_entry_bear': [45, 50], 'duration': default_durations
            },
            'ICHIMOKU': {
                'tenkan': [9], 'kijun': [26], 'senkou_b': [52], 'duration': default_durations
            },
            'EMA_CROSS': {
                'ema_fast': [9, 20], 'ema_slow': [50, 100, 200], 'duration': default_durations
            },
            'PARABOLIC_SAR': {
                'af': [0.01, 0.02, 0.03], 'max_af': [0.2], 'adx_threshold': [20, 25], 'duration': default_durations
            },
            'EMA_PULLBACK': {
                'ema_trend': [200], 'ema_pullback': [20, 50], 'rsi_limit': [55, 60, 65], 'duration': default_durations
            },
            'MTF_TREND': {
                'mtf_ema': [1000, 2000, 3000], 'local_ema': [50, 100], 'duration': default_durations
            },
            'STREAK_EXHAUSTION': {
                'streak_length': [5, 7, 9], 'rsi_threshold': [80, 85, 90], 'duration': default_durations
            },
            'VOL_SQUEEZE': {
                'squeeze_lookback': [20, 30, 50], 'bb_length': [20], 'bb_std': [2.0], 'duration': default_durations
            },
            'AI_GENERATED': {
                'duration': default_durations,
                'symbol': []
            }
        }

    async def fetch_history_paginated(self, symbol, months=1):
        """
        Fetches historical data by paging backwards.
        Circuit Breaker: Retry 3 times, then fail safely.
        """
        target_count = 45000 * months
        batch_size = 5000
        fetched = 0
        end_time = "latest"
        all_candles = []

        # Circuit Breaker / Retry State
        retries = 0
        max_retries = 3

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

            try:
                res = await self.client.send_request(req)

                # Check API Error
                if 'error' in res:
                    logger.error(f"API Error fetching history for {symbol}: {res['error']['message']}")
                    retries += 1
                    if retries > max_retries:
                        self._log_failed_asset(symbol, f"API Error: {res['error']['message']}")
                        return pd.DataFrame()
                    await asyncio.sleep(1 * (2 ** retries)) # Exponential Backoff
                    continue

                if 'candles' in res:
                    candles = res['candles']
                    if not candles:
                        break

                    df_batch = pd.DataFrame(candles)
                    df_batch['close'] = df_batch['close'].astype(float)
                    df_batch['open'] = df_batch['open'].astype(float)
                    df_batch['high'] = df_batch['high'].astype(float)
                    df_batch['low'] = df_batch['low'].astype(float)
                    df_batch['epoch'] = pd.to_datetime(df_batch['epoch'], unit='s')

                    all_candles = [df_batch] + all_candles
                    fetched += len(candles)
                    oldest_epoch = candles[0]['epoch']
                    end_time = oldest_epoch - 1

                    retries = 0 # Reset retries on success

                    if len(candles) < 10:
                        break

                    await asyncio.sleep(0.2)
                else:
                    break

            except Exception as e:
                logger.error(f"Exception fetching history for {symbol}: {e}")
                retries += 1
                if retries > max_retries:
                    self._log_failed_asset(symbol, f"Exception: {str(e)}")
                    return pd.DataFrame()
                await asyncio.sleep(1 * (2 ** retries))

        if all_candles:
            final_df = pd.concat(all_candles, ignore_index=True)
            final_df = final_df.sort_values('epoch').drop_duplicates('epoch').reset_index(drop=True)
            return final_df

        return pd.DataFrame()

    def _log_failed_asset(self, symbol, reason):
        try:
            with open("failed_assets.log", "a") as f:
                f.write(f"{datetime.utcnow()} - {symbol} - {reason}\n")
        except Exception:
            pass

    def dispatch_signal(self, strat_type, df, params):
        """
        Delegates signal generation to the shared strategies library.
        Restores local functionality for quick checks/live usage.
        """
        return strategies_lib.dispatch_signal(strat_type, df, params)

    def calculate_advanced_metrics(self, df, call_signal, put_signal, duration, symbol=""):
        return MetricsEngine.calculate_performance(df, call_signal, put_signal, duration, symbol)

    def get_strategy_candidates(self, symbol):
        """Returns list of strategy types to test based on asset class."""
        candidates = []
        if any(x in symbol for x in ['R_', '1HZ']):
            candidates = ['MTF_TREND', 'SUPERTREND', 'TREND_HEIKIN_ASHI', 'BREAKOUT', 'STREAK_EXHAUSTION', 'VOL_SQUEEZE']
        elif symbol.startswith('frx') or symbol.startswith('OTC_'):
            candidates = ['EMA_PULLBACK', 'BB_REVERSAL', 'STREAK_EXHAUSTION']
        elif any(x in symbol for x in ['stp', 'JD']):
             candidates = ['PARABOLIC_SAR', 'SUPERTREND', 'VOL_SQUEEZE']
        elif any(x in symbol for x in ['CRASH', 'BOOM', 'RDBULL', 'RDBEAR']):
             candidates = ['SUPERTREND', 'TREND_HEIKIN_ASHI', 'PARABOLIC_SAR']
        else:
             candidates = ['MTF_TREND', 'SUPERTREND', 'TREND_HEIKIN_ASHI', 'BREAKOUT']

        # Check for AI File
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ai_strat_name = f"{symbol}_ai"
        module_path = os.path.join(base_dir, "strategies", "generated", f"{ai_strat_name}.py")

        if os.path.exists(module_path):
            candidates.append(ai_strat_name)
            if ai_strat_name not in self.strategies:
                 self.strategies[ai_strat_name] = {'duration': self.strategies.get('AI_GENERATED', {}).get('duration', [1, 2, 3])}

        return candidates

    def save_heatmap_entry(self, symbol, strategy_type, expectancy, win_rate, signal_count):
        """Saves a single backtest result window to the database for heatmap visualization."""
        session = self.SessionLocal()
        try:
            entry = BacktestResult(
                symbol=symbol,
                strategy_type=strategy_type,
                win_rate=float(win_rate),
                expectancy=float(expectancy),
                signal_count=int(signal_count),
                timestamp=datetime.utcnow()
            )
            session.add(entry)
            session.commit()
        except Exception as e:
            logger.error(f"Error saving heatmap entry for {symbol} - {strategy_type}: {e}")
            session.rollback()
        finally:
            session.close()

    async def run_wfa_optimization(self, df, symbol=""):
        """
        Async wrapper that runs the CPU-bound optimization in a separate process.
        """
        # 1. Prepare Grids
        if '1HZ' in symbol: durations = [1, 2]
        else: durations = [1, 2, 3, 5, 10, 15]

        if 'JD' in symbol: st_multipliers = [2.0, 3.0, 4.0]
        else: st_multipliers = [2.0, 3.0]

        for s_name, s_grid in self.strategies.items():
            if 'duration' in s_grid:
                s_grid['duration'] = durations
            if s_name == 'SUPERTREND' and 'multiplier' in s_grid:
                s_grid['multiplier'] = st_multipliers

        # Legacy AI fallback logic (can be cleaned up but harmless to keep for now)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ai_path = os.path.join(base_dir, "strategies", "generated", f"{symbol}_ai.py")
        if os.path.exists(ai_path):
            self.strategies['AI_GENERATED']['symbol'] = [symbol]

        candidates_types = self.get_strategy_candidates(symbol)
        if os.path.exists(ai_path) and 'AI_GENERATED' not in candidates_types:
             candidates_types.append('AI_GENERATED')

        # 2. Run in Executor
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor,
                run_optimization_task,
                df,
                self.strategies,
                candidates_types,
                symbol
            )
        except Exception as e:
            logger.error(f"Optimization task failed for {symbol}: {e}")
            return None

        # 3. Handle Results
        if result:
            best = result.get('best_result')
            heatmap = result.get('heatmap_data', [])

            # Batch Save Heatmap (Optimization: doing individual saves logic for now to match interface)
            for entry in heatmap:
                self.save_heatmap_entry(**entry)

            return best

        return None

    def calculate_final_metrics_from_pnl(self, pnl_series, symbol):
        # Kept for compatibility if used elsewhere, though worker has its own copy
        # Determine Payout
        payout = 0.94 if ('R_' in symbol or '1HZ' in symbol) else 0.85

        wins = (pnl_series > 0).sum()
        losses = (pnl_series < 0).sum()
        total = wins + losses
        if total == 0: return None

        win_rate = wins / total
        ev = (win_rate * payout) - ((losses/total) * 1.0)

        if ev > 0:
            kelly = ((payout * win_rate) - (1 - win_rate)) / payout
            kelly = max(0, kelly)
        else:
            kelly = 0.0

        equity_curve = (1 + (pnl_series * 0.02)).cumprod()
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min()

        return {
            'ev': ev,
            'kelly': kelly,
            'max_drawdown': max_dd,
            'win_rate': win_rate * 100,
            'signals': int(total)
        }

    def save_best_result(self, symbol, result):
        if not result:
            return

        session = self.SessionLocal()
        try:
            existing = session.query(StrategyParams).filter_by(symbol=symbol).first()
            if not existing:
                existing = StrategyParams(symbol=symbol)
                session.add(existing)

            existing.win_rate = float(result['WinRate'])
            existing.signal_count = int(result['Signals'])
            existing.last_updated = pd.Timestamp.utcnow()
            existing.optimal_duration = int(result['optimal_duration'])
            existing.strategy_type = result['strategy_type']
            existing.config_json = json.dumps(result['config'])
            existing.expectancy = float(result['Expectancy'])
            existing.kelly = float(result['Kelly'])
            existing.max_drawdown = float(result['MaxDD'])

            # Legacy mapping
            config = result['config']
            existing.rsi_period = int(config.get('rsi_period', 0))
            existing.ema_period = int(config.get('ema_period', 0))
            existing.rsi_vol_window = int(config.get('rsi_vol_window', 0))

            session.commit()
            logger.info(f"Saved optimized config for {symbol} ({existing.strategy_type}): EV={result['Expectancy']:.2f} Kelly={result['Kelly']:.1f}%")
        except Exception as e:
            logger.error(f"Error saving strategy params for {symbol}: {e}")
            session.rollback()
        finally:
            session.close()

    async def run_full_scan(self, resume=False):
        scanner_data = state.get_scanner_data()
        all_symbols = []
        for cat, assets in scanner_data.items():
            for a in assets:
                all_symbols.append(a['symbol'])

        # Filter using ScanManager
        pending_symbols = self.scan_manager.get_pending_symbols(all_symbols, resume=resume)
        total = len(all_symbols)

        if not pending_symbols:
            logger.info("No pending symbols to scan.")
            state.update_scan_progress(total, total, None, "complete")
            return

        logger.info(f"Starting WFA Scan on {len(pending_symbols)} symbols (Resume={resume})...")

        # Explicit garbage collection before start
        import gc
        gc.collect()

        processed_count = total - len(pending_symbols)

        for i, symbol in enumerate(pending_symbols):
            processed_count += 1
            state.update_scan_progress(total, processed_count, symbol, "running")
            self.scan_manager.update_status(symbol, "RUNNING")

            try:
                df = await self.fetch_history_paginated(symbol, months=1)
                if df.empty:
                    self.scan_manager.mark_failed(symbol)
                    continue

                best = await self.run_wfa_optimization(df, symbol=symbol)

                if best:
                    self.save_best_result(symbol, best)
                    self.scan_manager.mark_completed(symbol)
                else:
                    logger.info(f"No profitable strategy found for {symbol} in WFA.")
                    # Mark as completed (scanned but found nothing) to avoid infinite retry loop
                    self.scan_manager.mark_completed(symbol)

                # Cleanup memory
                del df
                if i % 5 == 0:
                    gc.collect()

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                self.scan_manager.mark_failed(symbol)

        state.update_scan_progress(total, total, None, "complete")
        logger.info("WFA Scan Complete.")

    def generate_heatmap(self, results_df):
        pass
