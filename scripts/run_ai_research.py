import os
import sys
import pandas as pd
import asyncio
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deriv_quant_py.config import Config
from deriv_quant_py.core.connection import DerivClient
from deriv_quant_py.core.backtester import Backtester
from deriv_quant_py.core.market_physics import MarketPhysics
from deriv_quant_py.core.ai_researcher import AIResearcher
from deriv_quant_py.shared_state import state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI_Researcher")

async def run_research():
    logger.info("Starting AI Research & Strategy Generation...")

    # Check API Key
    if not Config.DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY not found in Config/Env. Aborting.")
        return

    # Initialize Components
    client = DerivClient()
    await client.connect()
    if not client.connected:
        logger.error("Failed to connect to Deriv API.")
        return

    backtester = Backtester(client)
    researcher = AIResearcher(api_key=Config.DEEPSEEK_API_KEY)

    # Get Symbols from Scanner State (or default list if empty)
    # We can try to fetch active symbols from API or use a hardcoded list for safety
    # Let's use a broad list of popular assets
    target_symbols = [
        "R_100", "R_75", "R_50", "R_25", "R_10",
        "1HZ100V", "1HZ75V", "1HZ50V", "1HZ25V", "1HZ10V",
        "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD",
        "CRASH_1000", "BOOM_1000", "CRASH_500", "BOOM_500"
    ]

    # Try to get from scanner state if populated (might be empty if not running)
    scanner_data = state.get_scanner_data()
    if scanner_data:
        symbols_from_scanner = []
        for cat, assets in scanner_data.items():
            for a in assets:
                symbols_from_scanner.append(a['symbol'])
        if symbols_from_scanner:
            target_symbols = symbols_from_scanner
            logger.info(f"Loaded {len(target_symbols)} symbols from Scanner State.")

    for symbol in target_symbols:
        logger.info(f"Processing {symbol}...")
        try:
            # 1. Fetch History (1 month is enough for DNA)
            df = await backtester.fetch_history_paginated(symbol, months=1)
            if df.empty or len(df) < 1000:
                logger.warning(f"Insufficient data for {symbol}. Skipping.")
                continue

            # 2. Market Physics Analysis (DNA)
            dna = MarketPhysics.get_asset_dna(df)
            logger.info(f"DNA for {symbol}: Hurst={dna['hurst']:.2f}, Noise={dna['noise_index']:.2f}, Regime={dna['regime']}")

            # 3. AI Strategy Generation
            success = researcher.generate_strategy(symbol, dna)
            if success:
                logger.info(f"Successfully generated AI strategy for {symbol}.")
            else:
                logger.error(f"Failed to generate strategy for {symbol}.")

            # Rate limit respect
            await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    logger.info("AI Research Run Complete.")

    # Close connection
    # Client doesn't have explicit close method in snippet, but we can let script exit

if __name__ == "__main__":
    try:
        asyncio.run(run_research())
    except KeyboardInterrupt:
        logger.info("Research interrupted.")
