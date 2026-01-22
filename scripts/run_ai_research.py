import os
import sys
import asyncio
import logging
import argparse

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deriv_quant_py.config import Config
from deriv_quant_py.core.connection import DerivClient
from deriv_quant_py.core.backtester import Backtester
from deriv_quant_py.core.ai_researcher import AIResearcher, run_research_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Script_Runner")

async def main(resume):
    if not Config.DEEPSEEK_API_KEY:
        logger.error("No API Key.")
        return

    # 1. Setup Components
    client = DerivClient(Config.WEBSOCKET_URL, Config.DERIV_TOKEN)
    await client.connect()

    backtester = Backtester(client)
    researcher = AIResearcher(api_key=Config.DEEPSEEK_API_KEY)

    # 2. Run Shared Logic
    await run_research_session(client, backtester, researcher, resume=resume)

    # 3. Cleanup
    if client.ws: await client.ws.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip existing strategies")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.resume))
    except KeyboardInterrupt:
        pass
