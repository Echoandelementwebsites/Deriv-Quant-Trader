import pandas as pd
import numpy as np
import itertools
from itertools import combinations
import logging
import gc
from deriv_quant_py.core import strategies_lib
from deriv_quant_py.core.metrics import MetricsEngine

logger = logging.getLogger(__name__)

def calculate_final_metrics_from_pnl(pnl_series, symbol):
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

    # Max DD
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

def get_combinations(grid):
    keys = grid.keys()
    values = grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def run_optimization_task(df, strategies_grid, candidates_types, symbol):
    """
    Executes the Walk-Forward Analysis (WFA) in a separate process.
    Returns a dictionary with 'best_result' and 'heatmap_data'.
    """
    TRAIN_SIZE = 3000
    TEST_SIZE = 500
    STEP_SIZE = 500

    if len(df) < TRAIN_SIZE + TEST_SIZE:
        return None

    heatmap_data = [] # List of dicts to return for DB saving
    accumulated_pnl = pd.Series(0.0, index=df.index)
    last_winner_config = None
    last_winner_type = None

    # Iterate Windows
    for start_idx in range(0, len(df) - TRAIN_SIZE - TEST_SIZE, STEP_SIZE):
        train_start = start_idx
        train_end = start_idx + TRAIN_SIZE
        test_start = train_end
        test_end = train_end + TEST_SIZE

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        # Phase 1: Tournament
        tournament_results = []

        for strat_type in candidates_types:
            if strat_type not in strategies_grid: continue
            grid = strategies_grid[strat_type]

            best_local_ev = -100
            best_local_res = None # (config, call_sig, put_sig)

            for params in get_combinations(grid):
                # Dispatch Signal Logic
                try:
                    call, put = strategies_lib.dispatch_signal(strat_type, train_df, params)
                except Exception as e:
                    # Catch errors safely in worker
                    continue

                duration = params.get('duration', 1)
                metrics = MetricsEngine.calculate_performance(train_df, call, put, duration, symbol)

                if metrics and metrics['signals'] >= 5:
                    if metrics['ev'] > best_local_ev:
                        best_local_ev = metrics['ev']
                        best_local_res = (params, call, put, metrics)

            if best_local_res:
                # Capture Heatmap Data for this Window & Strategy Type
                heatmap_data.append({
                    'symbol': symbol,
                    'strategy_type': strat_type,
                    'expectancy': float(best_local_ev),
                    'win_rate': float(best_local_res[3]['win_rate']), # Metrics has win_rate * 100
                    'signal_count': int(best_local_res[3]['signals'])
                })

                tournament_results.append({
                    'type': strat_type,
                    'config': best_local_res[0],
                    'ev': best_local_ev,
                    'call': best_local_res[1],
                    'put': best_local_res[2]
                })

        # Phase 2: Selection & Team-Up
        if not tournament_results:
            continue

        tournament_results.sort(key=lambda x: x['ev'], reverse=True)

        unique_top_candidates = []
        seen_types = set()
        for res in tournament_results:
            if res['type'] not in seen_types:
                unique_top_candidates.append(res)
                seen_types.add(res['type'])
            if len(unique_top_candidates) >= 3: break

        if not unique_top_candidates: continue
        top_3 = unique_top_candidates

        best_window_strategy = top_3[0]

        # Ensemble Logic
        for pair in combinations(top_3, 2):
            s1 = pair[0]
            s2 = pair[1]
            ens_call = s1['call'] & s2['call']
            ens_put = s1['put'] & s2['put']
            duration = s1['config'].get('duration', 1) # Default to 1 if missing

            metrics = MetricsEngine.calculate_performance(train_df, ens_call, ens_put, duration, symbol)
            if metrics and metrics['signals'] > 5:
                if metrics['ev'] > best_window_strategy['ev']:
                    best_window_strategy = {
                        'type': 'ENSEMBLE',
                        'config': {
                            'mode': 'ENSEMBLE',
                            'members': [
                                {**s1['config'], 'strategy_type': s1['type']},
                                {**s2['config'], 'strategy_type': s2['type']}
                            ],
                            'duration': duration
                        },
                        'ev': metrics['ev']
                    }

        # Phase 3: Test on Out-of-Sample
        win_type = best_window_strategy['type']
        win_config = best_window_strategy['config']

        test_call = pd.Series(False, index=test_df.index)
        test_put = pd.Series(False, index=test_df.index)

        try:
            if win_type == 'ENSEMBLE':
                members = win_config['members']
                m1_cfg = members[0]
                m1_type = m1_cfg['strategy_type']
                c1, p1 = strategies_lib.dispatch_signal(m1_type, test_df, m1_cfg)

                m2_cfg = members[1]
                m2_type = m2_cfg['strategy_type']
                c2, p2 = strategies_lib.dispatch_signal(m2_type, test_df, m2_cfg)

                test_call = c1 & c2
                test_put = p1 & p2
            else:
                test_call, test_put = strategies_lib.dispatch_signal(win_type, test_df, win_config)

            duration = win_config.get('duration', 1)
            test_metrics = MetricsEngine.calculate_performance(test_df, test_call, test_put, duration, symbol)

            if test_metrics:
                accumulated_pnl = accumulated_pnl.add(test_metrics['pnl_series'], fill_value=0)

            last_winner_config = win_config
            last_winner_type = win_type

        except Exception as e:
            # If testing fails, just skip accumulation for this window
            pass

        # Explicit cleanup inside loop
        del train_df, test_df, tournament_results, top_3
        # gc.collect() # Calling GC too often kills perf. Maybe every few steps?
        # But we are in a separate process, so memory is freed when process dies?
        # No, ProcessPoolExecutor reuses processes. So we MUST clean up.
        if (start_idx / STEP_SIZE) % 5 == 0:
            gc.collect()

    # Final Calculation
    final_res = calculate_final_metrics_from_pnl(accumulated_pnl, symbol)

    if final_res and last_winner_config:
        return {
            'best_result': {
                'strategy_type': last_winner_type,
                'config': last_winner_config,
                'WinRate': final_res['win_rate'],
                'Signals': final_res['signals'],
                'Expectancy': final_res['ev'],
                'Kelly': final_res['kelly'],
                'MaxDD': final_res['max_drawdown'],
                'optimal_duration': last_winner_config.get('duration', 1)
            },
            'heatmap_data': heatmap_data
        }

    return None
