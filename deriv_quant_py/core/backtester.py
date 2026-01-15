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
import itertools

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
        # Legacy: Dynamic RSI + CHOP
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

    def _eval_trend_ha(self, df, params):
        # Trend Strategy (Heikin Ashi + ADX + EMA)
        df = df.copy() # Avoid SettingWithCopyWarning
        # Params: ema_period, duration
        ema_p = int(params.get('ema_period', 50))
        duration = params['duration']

        # 1. EMA Filter
        df['EMA'] = ta.ema(df['close'], length=ema_p)

        # 2. ADX Filter (Fixed length 14, Thresh 25)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is None or adx_df.empty:
            return 0, 0, 0
        # Column names: ADX_14, DMP_14, DMN_14
        df['ADX'] = adx_df.iloc[:, 0]

        # NEW: ADX Slope (Rising)
        df['ADX_Prev'] = df['ADX'].shift(1)
        adx_rising = df['ADX'] > df['ADX_Prev']

        # NEW: RSI Check (Avoid Exhaustion)
        df['RSI'] = ta.rsi(df['close'], length=14)

        # 3. Heikin Ashi
        ha_df = ta.ha(df['open'], df['high'], df['low'], df['close'])
        if ha_df is None or ha_df.empty:
            return 0, 0, 0
        # Columns: HA_open, HA_high, HA_low, HA_close

        # Logic:
        # Long: HA Green (Close > Open) AND Flat Bottom (Low == Open) AND ADX > 25 AND Price > EMA
        # Short: HA Red (Close < Open) AND Flat Top (High == Open) AND ADX > 25 AND Price < EMA

        ha_green = ha_df['HA_close'] > ha_df['HA_open']
        ha_red = ha_df['HA_close'] < ha_df['HA_open']

        # Flat Bottom: Low equals Open (within tiny float tolerance or exact?)
        # Using np.isclose for safety.
        ha_flat_bottom = np.isclose(ha_df['HA_low'], ha_df['HA_open'])
        ha_flat_top = np.isclose(ha_df['HA_high'], ha_df['HA_open'])

        strong_trend = (df['ADX'] > 25) & adx_rising

        call_signal = (
            ha_green &
            ha_flat_bottom &
            strong_trend &
            (df['close'] > df['EMA']) &
            (df['RSI'] < 75)
        )

        put_signal = (
            ha_red &
            ha_flat_top &
            strong_trend &
            (df['close'] < df['EMA']) &
            (df['RSI'] > 25)
        )

        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _eval_breakout(self, df, params):
        # Breakout Strategy (Volatility Squeeze)
        df = df.copy() # Avoid SettingWithCopyWarning
        # Params: duration (Fixed BB/KC params)
        duration = params['duration']

        # Fixed Parameters
        bb_length = 20
        bb_std = 2.0
        kc_length = 20
        kc_scalar = 1.5

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=bb_length, std=bb_std)
        if bb is None or bb.empty:
             return 0, 0, 0
        # BBL, BBM, BBU
        bb_lower = bb.iloc[:, 0]
        bb_upper = bb.iloc[:, 2]

        # Keltner Channels
        kc = ta.kc(df['high'], df['low'], df['close'], length=kc_length, scalar=kc_scalar)
        if kc is None or kc.empty:
             return 0, 0, 0
        # KCL, KCB, KCU (Lower, Basis, Upper)
        kc_lower = kc.iloc[:, 0]
        kc_upper = kc.iloc[:, 2]

        # Squeeze: BB inside KC
        is_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        prev_squeeze = is_squeeze.shift(1).fillna(False)

        # NEW: RSI Momentum
        rsi = ta.rsi(df['close'], length=14)

        # Breakout Signals
        # Long: Breakout + RSI > 55 (Strong Momentum)
        call_signal = prev_squeeze & (df['close'] > bb_upper) & (rsi > 55)

        # Short: Breakout + RSI < 45 (Strong Momentum)
        put_signal = prev_squeeze & (df['close'] < bb_lower) & (rsi < 45)

        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _eval_supertrend(self, df, params):
        # Supertrend (ATR Trailing Stop)
        df = df.copy()
        # Params: length, multiplier, duration
        length = params['length']
        multiplier = params['multiplier']
        duration = params['duration']

        # Supertrend
        st = ta.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
        if st is None or st.empty:
            return 0, 0, 0

        # Identify Supertrend Line Column (usually first column)
        st_line = st.iloc[:, 0]

        # ADX Filter > 20
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is None or adx_df.empty:
            return 0, 0, 0
        adx = adx_df.iloc[:, 0]

        # Logic:
        # Long: Close > Supertrend Line AND ADX > 20
        # Short: Close < Supertrend Line AND ADX > 20

        trend_filter = adx > 20

        # NEW: Major Trend Filter (EMA 200)
        # Even if not optimizing EMA, we calculate a fixed long-term EMA
        ema_200 = ta.ema(df['close'], length=200)
        # Handle beginning of data where EMA is NaN
        ema_200 = ema_200.fillna(df['close'])

        call_signal = (df['close'] > st_line) & trend_filter & (df['close'] > ema_200)
        put_signal = (df['close'] < st_line) & trend_filter & (df['close'] < ema_200)

        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _eval_bb_reversal(self, df, params):
        # BB Mean Reversion
        df = df.copy()
        # Params: bb_length, bb_std, rsi_period, duration
        bb_length = params['bb_length']
        bb_std = params['bb_std']
        rsi_period = params['rsi_period']
        duration = params['duration']

        # BB
        bb = ta.bbands(df['close'], length=bb_length, std=bb_std)
        if bb is None or bb.empty: return 0, 0, 0
        lower = bb.iloc[:, 0]
        upper = bb.iloc[:, 2]

        # RSI
        rsi = ta.rsi(df['close'], length=rsi_period)
        if rsi is None or rsi.empty: return 0, 0, 0

        # ADX Filter < 25 (Not trending)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is None or adx_df.empty: return 0, 0, 0
        adx = adx_df.iloc[:, 0]

        # NEW: Stochastic Oscillator (14, 3, 3)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        if stoch is None or stoch.empty: return 0, 0, 0
        # Columns: STOCHk_14_3_3, STOCHd_14_3_3
        stoch_k = stoch.iloc[:, 0]

        range_filter = adx < 25

        # Logic
        # Buy: Close < Lower & RSI < 30 & Stoch_K < 20
        call_signal = (df['close'] < lower) & (rsi < 30) & (stoch_k < 20) & range_filter

        # Sell: Close > Upper & RSI > 70 & Stoch_K > 80
        put_signal = (df['close'] > upper) & (rsi > 70) & (stoch_k > 80) & range_filter

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

    def get_strategy_candidates(self, symbol):
        """Returns list of strategy types to test based on asset class."""

        # 1. Forex & OTC
        # UPDATED: Added TREND_HEIKIN_ASHI and BREAKOUT.
        # Markets trend too often to rely solely on Mean Reversion.
        if symbol.startswith('frx') or symbol.startswith('OTC_'):
            return ['BB_REVERSAL', 'TREND_HEIKIN_ASHI', 'BREAKOUT']

        # 2. Step, Jump, Reset, Synthetics (Trend Focus)
        # symbol.startswith('R_') includes R_10, R_100 etc.
        # symbol.startswith('1HZ') includes 1HZ10V etc.
        elif (any(symbol.startswith(x) for x in ['stp', 'JD', 'RDBULL', 'RDBEAR', 'R_', '1HZ', 'CRASH', 'BOOM'])):
            return ['SUPERTREND', 'TREND_HEIKIN_ASHI', 'BREAKOUT']

        # Default (Try everything for others)
        else:
            return ['SUPERTREND', 'TREND_HEIKIN_ASHI', 'BB_REVERSAL']

    def run_wfa_optimization(self, df, symbol=""):
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

        # Determine Payout Ratio
        avg_win_payout = 0.85
        if 'R_' in symbol or '1HZ' in symbol:
            avg_win_payout = 0.94

        # Define Duration Lists
        # Constraint: Force durations to be small for 1HZ assets: [1, 2]
        if '1HZ' in symbol:
            durations = [1, 2]
        else:
            durations = [1, 2, 3, 5, 10, 15]

        # Define Multiplier Lists for Supertrend
        # Constraint: Ensure 4.0 is tested for JD assets
        if 'JD' in symbol:
            st_multipliers = [2.0, 3.0, 4.0]
        else:
            st_multipliers = [2.0, 3.0] # Default

        # Strategy Grids
        strategies = {
            'BB_REVERSAL': {
                'bb_length': [20],
                'bb_std': [2.0, 2.5],
                'rsi_period': [7, 14],
                'duration': durations
            },
            'SUPERTREND': {
                'length': [7, 10, 14],
                'multiplier': st_multipliers,
                'duration': durations
            },
            'TREND_HEIKIN_ASHI': {
                'ema_period': [50, 100, 200],
                'duration': durations
            },
            'BREAKOUT': {
                'duration': durations
            },
            # Legacy REVERSAL (kept if needed, or we can omit it if BB_REVERSAL supersedes it)
            # Keeping it out of the main loop for now unless requested to fallback.
        }

        candidates = self.get_strategy_candidates(symbol)

        # Generate parameter combinations helper
        def get_combinations(grid):
            keys = grid.keys()
            values = grid.values()
            return [dict(zip(keys, v)) for v in itertools.product(*values)]

        best_system_result = None
        best_system_ev = -100

        for strat_type in candidates:
            if strat_type not in strategies:
                continue

            grid = strategies[strat_type]
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
                    wins, losses, signals = 0, 0, 0

                    if strat_type == 'BB_REVERSAL':
                        wins, losses, signals = self._eval_bb_reversal(train_df, params)
                    elif strat_type == 'SUPERTREND':
                        wins, losses, signals = self._eval_supertrend(train_df, params)
                    elif strat_type == 'TREND_HEIKIN_ASHI':
                        wins, losses, signals = self._eval_trend_ha(train_df, params)
                    elif strat_type == 'BREAKOUT':
                        wins, losses, signals = self._eval_breakout(train_df, params)

                    if signals < 3: continue

                    ev = self.calculate_expectancy(wins, losses, avg_win_payout=avg_win_payout)
                    if ev > best_local_ev:
                        best_local_ev = ev
                        best_local_config = params

                # 2. Verify on Test
                if best_local_config:
                    v_wins, v_losses, v_signals = 0, 0, 0

                    if strat_type == 'BB_REVERSAL':
                        v_wins, v_losses, v_signals = self._eval_bb_reversal(test_df, best_local_config)
                    elif strat_type == 'SUPERTREND':
                        v_wins, v_losses, v_signals = self._eval_supertrend(test_df, best_local_config)
                    elif strat_type == 'TREND_HEIKIN_ASHI':
                        v_wins, v_losses, v_signals = self._eval_trend_ha(test_df, best_local_config)
                    elif strat_type == 'BREAKOUT':
                        v_wins, v_losses, v_signals = self._eval_breakout(test_df, best_local_config)

                    agg_results['wins'] += v_wins
                    agg_results['losses'] += v_losses
                    agg_results['signals'] += v_signals
                    agg_results['configs'].append(best_local_config)

            # Evaluate System Performance for this Strategy Type
            total_wins = agg_results['wins']
            total_losses = agg_results['losses']
            system_ev = self.calculate_expectancy(total_wins, total_losses, avg_win_payout=avg_win_payout)

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
            # We map params if available.
            config = result['config']

            # Map common fields if they exist, else 0
            existing.rsi_period = int(config.get('rsi_period', 0))
            existing.ema_period = int(config.get('ema_period', 0))
            # rsi_vol_window is from legacy Reversal, not used in new strategies
            existing.rsi_vol_window = int(config.get('rsi_vol_window', 0))

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
                best = self.run_wfa_optimization(df, symbol=symbol)

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
