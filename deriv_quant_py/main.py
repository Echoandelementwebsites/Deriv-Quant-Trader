import asyncio
import threading
import logging
import time
from deriv_quant_py.config import Config
from deriv_quant_py.core.connection import DerivClient
from deriv_quant_py.core.scanner import MarketScanner
from deriv_quant_py.core.engine import TradingEngine
from deriv_quant_py.core.executor import TradeExecutor
from deriv_quant_py.core.backtester import Backtester
from deriv_quant_py.core.market_physics import MarketPhysics
from deriv_quant_py.core.ai_researcher import AIResearcher
from deriv_quant_py.database import init_db
from deriv_quant_py.dashboard.app import app
from deriv_quant_py.shared_state import state

# Logging Setup
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL, "INFO"))
logger = logging.getLogger("Main")

# Global Event Loop for Backend
backend_loop = asyncio.new_event_loop()

def run_backend():
    """Runs the Asyncio Backend in a separate thread."""
    asyncio.set_event_loop(backend_loop)

    # Init Components
    db_session_factory = init_db(Config.DB_PATH)
    db_session = db_session_factory()

    client = DerivClient(Config.WEBSOCKET_URL, Config.DERIV_TOKEN)
    scanner = MarketScanner(client)
    executor = TradeExecutor(client, db_session)
    engine = TradingEngine(client, db_session, executor)
    backtester = Backtester(client)

    async def run_ai_research_task():
        logger.info("Starting AI Research Task...")

        # Check API Key
        if not Config.DEEPSEEK_API_KEY:
            logger.error("DEEPSEEK_API_KEY not found. Aborting AI Research.")
            state.update_scan_progress(100, 100, "Error: Missing API Key", status="complete")
            return

        try:
            researcher = AIResearcher(api_key=Config.DEEPSEEK_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize AIResearcher: {e}")
            state.update_scan_progress(100, 100, "Error: Init Failed", status="complete")
            return

        # Target Symbols
        target_symbols = [
            "R_100", "R_75", "R_50", "R_25", "R_10",
            "1HZ100V", "1HZ75V", "1HZ50V", "1HZ25V", "1HZ10V",
            "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD",
            "CRASH_1000", "BOOM_1000", "CRASH_500", "BOOM_500"
        ]

        # Use Scanner Data if available
        scanner_data = state.get_scanner_data()
        if scanner_data:
            symbols_from_scanner = []
            for cat, assets in scanner_data.items():
                for a in assets:
                    symbols_from_scanner.append(a['symbol'])
            if symbols_from_scanner:
                target_symbols = symbols_from_scanner

        total = len(target_symbols)

        for i, symbol in enumerate(target_symbols):
            state.update_scan_progress(total, i + 1, symbol, status="running")
            logger.info(f"AI Researching {symbol}...")

            try:
                # 1. Fetch History
                df = await backtester.fetch_history_paginated(symbol, months=1)
                if df.empty or len(df) < 1000:
                    logger.warning(f"Insufficient data for {symbol}. Skipping.")
                    continue

                # 2. DNA Analysis
                dna = MarketPhysics.get_asset_dna(df)

                # 3. Generate Strategy
                success = researcher.generate_strategy(symbol, dna)
                if success:
                    logger.info(f"Generated strategy for {symbol}")

                # Rate limit
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error researching {symbol}: {e}")

        state.update_scan_progress(total, total, "Complete", status="complete")
        logger.info("AI Research Task Completed.")

    # Workflow
    async def main_workflow():
        await client.connect()

        # 1. Scan Market
        assets = await scanner.scan_market()
        state.update_scanner(assets)

        # 2. Pick symbols to watch (e.g., Synthetics)
        # For demo, let's watch R_100 and R_10 if available
        watchlist = []
        if 'Synthetics' in assets:
            watchlist = [a['symbol'] for a in assets['Synthetics'] if a['symbol'] in ['R_100', 'R_10']]

        if not watchlist:
            watchlist = ['R_100'] # Fallback

        logger.info(f"Starting Engine on: {watchlist}")

        # 3. Start Engine
        engine.start(watchlist)

        # 4. Start Trade Monitor
        await executor.monitor_trades()

        # Keep alive Loop (Poll for Backtest Requests)
        while True:
            # Check for backtest request
            bt_req = state.get_backtest_request()
            if bt_req:
                if bt_req == "FULL_SCAN":
                    logger.info("Starting Full System Scan...")
                    # Run in background to not block loop?
                    # Since this is the main loop, we can await it, but it blocks engine updates for seconds/minutes.
                    # Ideally create_task, but run_full_scan is async.
                    asyncio.create_task(backtester.run_full_scan(resume=False))
                elif bt_req == "FULL_SCAN_RESUME":
                     logger.info("Resuming System Scan...")
                     asyncio.create_task(backtester.run_full_scan(resume=True))
                elif bt_req == "AI_RESEARCH_START":
                     asyncio.create_task(run_ai_research_task())
                else:
                    logger.info(f"Running WFA for {bt_req}...")
                    df = await backtester.fetch_history_paginated(bt_req, months=1)
                    if not df.empty:
                        # Single WFA run returns a single dict (best config) or None
                        best = backtester.run_wfa_optimization(df)
                        if best:
                            # Frontend expects a list for the table/chart.
                            # We can wrap the single result in a list.
                            state.set_backtest_result([best])
                        else:
                            state.set_backtest_result([])
                    else:
                        state.set_backtest_result([])

            await asyncio.sleep(1)

    backend_loop.run_until_complete(main_workflow())

if __name__ == "__main__":
    # Start Backend Thread
    t = threading.Thread(target=run_backend, daemon=True)
    t.start()

    # Start Dash Server (Main Thread)
    # Using debug=False to avoid reloader spawning double threads
    logger.info("Starting Dashboard on http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
