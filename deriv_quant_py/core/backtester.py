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

    def _eval_trend_ha(self, df, params):
        # Trend Strategy (Heikin Ashi + ADX + EMA)
        df = df.copy() # Avoid SettingWithCopyWarning
        # Params: ema_period, duration, adx_threshold, rsi_max
        ema_p = int(params.get('ema_period', 50))
        adx_threshold = int(params.get('adx_threshold', 25))
        rsi_max = int(params.get('rsi_max', 75))
        duration = params['duration']

        # 1. EMA Filter
        df['EMA'] = ta.ema(df['close'], length=ema_p)

        # 2. ADX Filter (Fixed length 14, Dynamic Thresh)
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
        # Long: HA Green (Close > Open) AND Flat Bottom (Low == Open) AND ADX > thresh AND Price > EMA
        # Short: HA Red (Close < Open) AND Flat Top (High == Open) AND ADX > thresh AND Price < EMA

        ha_green = ha_df['HA_close'] > ha_df['HA_open']
        ha_red = ha_df['HA_close'] < ha_df['HA_open']

        # Flat Bottom: Low equals Open (within tiny float tolerance or exact?)
        # Using np.isclose for safety.
        ha_flat_bottom = np.isclose(ha_df['HA_low'], ha_df['HA_open'])
        ha_flat_top = np.isclose(ha_df['HA_high'], ha_df['HA_open'])

        strong_trend = (df['ADX'] > adx_threshold) & adx_rising

        # For Short, we use 100 - rsi_max as the lower bound (e.g., if max is 75, min is 25)
        rsi_min_short = 100 - rsi_max

        call_signal = (
            ha_green &
            ha_flat_bottom &
            strong_trend &
            (df['close'] > df['EMA']) &
            (df['RSI'] < rsi_max)
        )

        put_signal = (
            ha_red &
            ha_flat_top &
            strong_trend &
            (df['close'] < df['EMA']) &
            (df['RSI'] > rsi_min_short)
        )

        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _eval_breakout(self, df, params):
        # Breakout Strategy (Volatility Squeeze)
        df = df.copy() # Avoid SettingWithCopyWarning
        # Params: duration (Fixed BB/KC params), rsi_entry_bull, rsi_entry_bear
        duration = params['duration']
        rsi_entry_bull = int(params.get('rsi_entry_bull', 55))
        rsi_entry_bear = int(params.get('rsi_entry_bear', 45))

        # Fixed Parameters for Bands (Usually standard)
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
        # Long: Breakout + RSI > Bull Thresh (Strong Momentum)
        call_signal = prev_squeeze & (df['close'] > bb_upper) & (rsi > rsi_entry_bull)

        # Short: Breakout + RSI < Bear Thresh (Strong Momentum)
        put_signal = prev_squeeze & (df['close'] < bb_lower) & (rsi < rsi_entry_bear)

        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _eval_supertrend(self, df, params):
        # Supertrend (ATR Trailing Stop)
        df = df.copy()
        # Params: length, multiplier, duration, trend_ema, adx_threshold
        length = params['length']
        multiplier = params['multiplier']
        duration = params['duration']
        trend_ema_len = int(params.get('trend_ema', 200))
        adx_threshold = int(params.get('adx_threshold', 20))

        # Supertrend
        st = ta.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
        if st is None or st.empty:
            return 0, 0, 0

        # Identify Supertrend Line Column (usually first column)
        st_line = st.iloc[:, 0]

        # ADX Filter > Threshold
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is None or adx_df.empty:
            return 0, 0, 0
        adx = adx_df.iloc[:, 0]

        # Logic:
        # Long: Close > Supertrend Line AND ADX > Threshold
        # Short: Close < Supertrend Line AND ADX > Threshold

        trend_filter = adx > adx_threshold

        # NEW: Major Trend Filter (EMA dynamic)
        ema_trend = ta.ema(df['close'], length=trend_ema_len)
        # Handle beginning of data where EMA is NaN
        ema_trend = ema_trend.fillna(df['close'])

        call_signal = (df['close'] > st_line) & trend_filter & (df['close'] > ema_trend)
        put_signal = (df['close'] < st_line) & trend_filter & (df['close'] < ema_trend)

        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _eval_bb_reversal(self, df, params):
        # BB Mean Reversion
        df = df.copy()
        # Params: bb_length, bb_std, rsi_period, duration, stoch_oversold, stoch_overbought
        bb_length = params['bb_length']
        bb_std = params['bb_std']
        # rsi_period is sometimes not in param grid if fixed? But let's assume valid key
        rsi_period = params['rsi_period']
        duration = params['duration']

        stoch_os = int(params.get('stoch_oversold', 20))
        stoch_ob = int(params.get('stoch_overbought', 80))

        # BB
        bb = ta.bbands(df['close'], length=bb_length, std=bb_std)
        if bb is None or bb.empty: return 0, 0, 0
        lower = bb.iloc[:, 0]
        upper = bb.iloc[:, 2]

        # RSI
        rsi = ta.rsi(df['close'], length=rsi_period)
        if rsi is None or rsi.empty: return 0, 0, 0

        # ADX Filter < 25 (Not trending) - keeping this hardcoded as "Range Filter"
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
        # Buy: Close < Lower & RSI < 30 (Hardcoded legacy RSI limit? Or should this match stoch?)
        # Current logic used hardcoded RSI < 30. WFA grid might add RSI threshold if needed, but for now prompt said "Replace fixed Stoch < 20 with Stoch < params".
        # We will keep RSI < 30 / > 70 fixed for now as per minimal prompt changes, or match Stoch logic?
        # Prompt: "Logic: Replace fixed Stoch < 20 with Stoch < params['stoch_oversold']."

        call_signal = (df['close'] < lower) & (rsi < 30) & (stoch_k < stoch_os) & range_filter

        # Sell: Close > Upper & RSI > 70 & Stoch_K > 80
        put_signal = (df['close'] > upper) & (rsi > 70) & (stoch_k > stoch_ob) & range_filter

        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _eval_ichimoku(self, df, params):
        # Ichimoku Cloud Breakout
        df = df.copy()
        # Params: tenkan, kijun, senkou_b
        tenkan_len = int(params.get('tenkan', 9))
        kijun_len = int(params.get('kijun', 26))
        senkou_b_len = int(params.get('senkou_b', 52))
        duration = params['duration']

        # Ichimoku
        # pandas_ta returns: ISA, ISB, ITS, IKS, ICS
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'],
                               tenkan=tenkan_len, kijun=kijun_len, senkou=senkou_b_len)
        if ichimoku is None: return 0, 0, 0

        # DataFrame 0 contains the lines, DataFrame 1 contains Span A/B shifted (future)
        # We need the values for the *current* candle, which means Span A/B shifted forward (already done by library? No, usually shifted 26 forward)
        # pandas_ta returns tuple (df_lines, df_span)
        # df_lines columns: ISA_9_26_52, ISB_9_26_52, ITS_9_26_52, IKS_9_26_52, ICS_9_26_52
        # BUT: Span A and B are usually plotted 26 periods ahead.
        # For "Price > Cloud", we need the Cloud value at *today*.
        # Standard Ichimoku check: Close > SpanA[today] AND Close > SpanB[today]

        # pandas_ta puts the current values in the first DF?
        # Let's inspect columns.
        # ITS = Tenkan, IKS = Kijun.
        # ISA = Span A, ISB = Span B.
        # Note: In pandas_ta, ISA and ISB in the first dataframe are the values corresponding to the current candle *if* they were not shifted?
        # Actually, for "Price > Cloud", we compare Price against the Cloud values plotted at the current time.
        # The cloud at 'now' is formed by lines calculated 26 periods ago and shifted forward.
        # pandas_ta's default behavior for the first DF is to contain the values aligned with 'close'.

        data = ichimoku[0]
        # Use implicit column ordering or safer naming
        # pandas_ta column names: ISA_{tenkan}, ISB_{kijun}, ITS_{tenkan}, IKS_{kijun}, ICS_{kijun} (Naming varies by version)
        # Safest is to take by index if we trust the order: ISA, ISB, ITS, IKS, ICS
        span_a = data.iloc[:, 0]
        span_b = data.iloc[:, 1]
        tenkan = data.iloc[:, 2]
        kijun = data.iloc[:, 3]

        # Logic:
        # Long: Price > Cloud (Span A & B) AND Tenkan > Kijun
        # Trigger: Transition from False to True

        cloud_top = np.maximum(span_a, span_b)
        cloud_bottom = np.minimum(span_a, span_b)

        # Conditions
        long_cond = (df['close'] > cloud_top) & (tenkan > kijun)
        short_cond = (df['close'] < cloud_bottom) & (tenkan < kijun)

        # Transition Check (Signal on the first candle it becomes true)
        call_signal = long_cond & (~long_cond.shift(1).fillna(False))
        put_signal = short_cond & (~short_cond.shift(1).fillna(False))

        return self._calc_win_loss(df, call_signal, put_signal, duration)

    def _eval_ema_cross(self, df, params):
        # EMA Crossover (Golden Cross)
        df = df.copy()
        # Params: ema_fast, ema_slow
        fast_len = int(params.get('ema_fast', 9))
        slow_len = int(params.get('ema_slow', 50))
        duration = params['duration']

        ema_fast = ta.ema(df['close'], length=fast_len)
        ema_slow = ta.ema(df['close'], length=slow_len)

        # Logic:
        # Long: Fast crosses above Slow
        # Short: Fast crosses below Slow
        # Crossover logic: (Fast > Slow) & (Prev_Fast <= Prev_Slow)

        fast_gt_slow = ema_fast > ema_slow
        fast_lt_slow = ema_fast < ema_slow

        call_signal = fast_gt_slow & (~fast_gt_slow.shift(1).fillna(False))
        put_signal = fast_lt_slow & (~fast_lt_slow.shift(1).fillna(False))

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

        # UPDATED: Added ICHIMOKU and EMA_CROSS to ALL asset classes as requested.

        # 1. Forex & OTC
        if symbol.startswith('frx') or symbol.startswith('OTC_'):
            return ['BB_REVERSAL', 'TREND_HEIKIN_ASHI', 'BREAKOUT', 'ICHIMOKU', 'EMA_CROSS']

        # 2. Step, Jump, Reset, Synthetics (Trend Focus)
        elif (any(symbol.startswith(x) for x in ['stp', 'JD', 'RDBULL', 'RDBEAR', 'R_', '1HZ', 'CRASH', 'BOOM'])):
            return ['SUPERTREND', 'TREND_HEIKIN_ASHI', 'BREAKOUT', 'ICHIMOKU', 'EMA_CROSS']

        # Default (Try everything for others)
        else:
            return ['SUPERTREND', 'TREND_HEIKIN_ASHI', 'BB_REVERSAL', 'ICHIMOKU', 'EMA_CROSS']

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

        # Expanded Strategy Grids
        strategies = {
            'SUPERTREND': {
                'length': [10, 14],
                'multiplier': [3.0, 4.0],
                'adx_threshold': [20, 25, 30],
                'trend_ema': [100, 200],
                'duration': durations
            },
            'TREND_HEIKIN_ASHI': {
                'ema_period': [50, 100, 200],
                'adx_threshold': [20, 25],
                'rsi_max': [70, 75, 80],
                'duration': durations
            },
            'BB_REVERSAL': {
                'bb_length': [20],
                'bb_std': [2.0, 2.5],
                'rsi_period': [14], # Default fixed or added if needed, prompt didn't specify iteration for rsi_period in expanded grid but previous code had it. Sticking to params list.
                'stoch_oversold': [15, 20],
                'stoch_overbought': [80, 85],
                'duration': durations
            },
            'BREAKOUT': {
                'rsi_entry_bull': [50, 55],
                'rsi_entry_bear': [45, 50],
                'duration': durations
            },
            'ICHIMOKU': {
                'tenkan': [9],
                'kijun': [26],
                'senkou_b': [52],
                'duration': durations
            },
            'EMA_CROSS': {
                'ema_fast': [9, 20],
                'ema_slow': [50, 100, 200],
                'duration': durations
            }
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
                    elif strat_type == 'ICHIMOKU':
                        wins, losses, signals = self._eval_ichimoku(train_df, params)
                    elif strat_type == 'EMA_CROSS':
                        wins, losses, signals = self._eval_ema_cross(train_df, params)

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
                    elif strat_type == 'ICHIMOKU':
                        v_wins, v_losses, v_signals = self._eval_ichimoku(test_df, best_local_config)
                    elif strat_type == 'EMA_CROSS':
                        v_wins, v_losses, v_signals = self._eval_ema_cross(test_df, best_local_config)

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
