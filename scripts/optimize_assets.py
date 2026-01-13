import os
import sys
import pandas as pd
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deriv_quant_py.database import Trade, SignalLog, StrategyParams, init_db
from deriv_quant_py.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AssetOptimizer")

def optimize_assets():
    """
    Offline analysis script to optimize asset parameters based on historical trade/signal data.
    """
    logger.info("Starting Asset Optimization...")

    Session = init_db(Config.DB_PATH)
    session = Session()

    try:
        # Load Data
        # In a real scenario, we would join Trades and SignalLogs
        # For now, let's look at Trades to identify "Blacklist" candidates (High Loss Rate assets)

        trades_query = session.query(Trade).statement
        trades_df = pd.read_sql(trades_query, session.bind)

        if trades_df.empty:
            logger.warning("No trade history found. Skipping optimization.")
            return

        # 1. Analyze Performance by Symbol
        # Group by Symbol
        stats = trades_df.groupby('symbol').agg(
            total_trades=('id', 'count'),
            wins=('status', lambda x: (x == 'WON').sum()),
            losses=('status', lambda x: (x == 'LOST').sum())
        )

        stats['win_rate'] = stats['wins'] / stats['total_trades']

        logger.info(f"Asset Performance:\n{stats}")

        # 2. Blacklist Logic
        # If Win Rate < 40% over > 10 trades, mark as dangerous?
        # Or, as per prompt: "Generate a Blacklist (e.g., Do not trade EURUSD between 21:00-23:00)"
        # Analysis by Hour
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['hour'] = trades_df['entry_time'].dt.hour

        hourly_stats = trades_df.groupby(['symbol', 'hour']).agg(
            trades=('id', 'count'),
            win_rate=('status', lambda x: (x == 'WON').mean())
        ).reset_index()

        # Identify bad hours (Win Rate < 40%, Min 5 trades)
        bad_hours = hourly_stats[
            (hourly_stats['win_rate'] < 0.4) &
            (hourly_stats['trades'] >= 5)
        ]

        if not bad_hours.empty:
            logger.info(f"Identified Low Performance Hours:\n{bad_hours}")

            # Update StrategyParams with this metadata
            for _, row in bad_hours.iterrows():
                symbol = row['symbol']
                hour = row['hour']

                sp = session.query(StrategyParams).filter_by(symbol=symbol).first()
                if sp:
                    # Append to details JSON
                    import json
                    try:
                        # Ensure 'details' column exists on the object
                        if hasattr(sp, 'details'):
                            details = json.loads(sp.details) if sp.details else {}
                            blacklist = details.get('blacklist_hours', [])
                            if hour not in blacklist:
                                blacklist.append(int(hour))
                                details['blacklist_hours'] = blacklist
                                sp.details = json.dumps(details)
                                logger.info(f"Updated {symbol} blacklist hours: {blacklist}")
                        else:
                            logger.warning(f"Schema mismatch: 'details' column missing on StrategyParams for {symbol}.")
                    except Exception as e:
                        logger.error(f"Error updating details for {symbol}: {e}")

            session.commit()
        else:
            logger.info("No significant negative patterns found yet.")

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
    finally:
        session.close()
        logger.info("Optimization Complete.")

if __name__ == "__main__":
    optimize_assets()
