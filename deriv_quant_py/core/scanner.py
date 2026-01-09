import logging
import asyncio
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class MarketScanner:
    def __init__(self, client):
        self.client = client
        self.assets = {
            'Forex': [],
            'Synthetics': [],
            'Crypto': [],
            'Stocks': [],
            'Commodities': []
        }

    async def scan_market(self):
        """Fetches active symbols and categorizes them."""
        logger.info("Scanning market for active symbols...")
        req = {
            "active_symbols": "brief",
            "product_type": "basic"
        }
        try:
            res = await self.client.send_request(req)
            if "active_symbols" in res:
                self._process_symbols(res["active_symbols"])
                logger.info(f"Market Scan Complete. Found {sum(len(v) for v in self.assets.values())} assets.")
                return self.assets
            else:
                logger.error(f"Failed to fetch active symbols: {res.get('error')}")
                return {}
        except Exception as e:
            logger.error(f"Market Scan Error: {e}")
            return {}

    def _process_symbols(self, symbols: List[Dict[str, Any]]):
        # Reset categories
        for k in self.assets:
            self.assets[k] = []

        for asset in symbols:
            # Filter Logic (reproducing JS logic)
            # 1. Must be 'callput' (since we requested product_type=basic, this is implied usually, but check market)
            # The JS code checked contract_category === 'callput' OR product_type === 'basic'
            # We already requested product_type='basic'.

            # Map to category
            market = asset.get('market')
            submarket = asset.get('submarket')

            category = None
            if market == 'forex':
                category = 'Forex'
            elif market == 'synthetic_index':
                category = 'Synthetics'
            elif market == 'cryptocurrency':
                category = 'Crypto'
            elif market in ['indices', 'stocks']:
                category = 'Stocks'
            elif market == 'commodities':
                category = 'Commodities'

            if category:
                clean_asset = {
                    'symbol': asset['symbol'],
                    'name': asset['display_name'],
                    'market': market,
                    'submarket': submarket,
                    'pip': asset.get('pip'),
                    # API might return min_duration in different formats, but we'll store raw for now
                    # or assume '1m' default as per JS
                }
                self.assets[category].append(clean_asset)
