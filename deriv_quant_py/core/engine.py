import asyncio
import logging
from collections import deque
from deriv_quant_py.config import Config
from deriv_quant_py.core.connection import DerivClient
from deriv_quant_py.strategies.triple_confluence import TripleConfluenceStrategy
from deriv_quant_py.database import SignalLog
from deriv_quant_py.shared_state import state
from sqlalchemy.orm import Session
import json

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, client: DerivClient, db_session: Session, executor):
        self.client = client
        self.db = db_session
        self.executor = executor
        self.strategy = TripleConfluenceStrategy()
        # Store candles: {symbol: deque([candles], maxlen=300)}
        self.candles = {}
        self.running = False
        self.watched_symbols = []

    def start(self, symbols):
        self.watched_symbols = symbols
        self.running = True
        # Subscribe to ticks/candles
        for symbol in symbols:
            self.candles[symbol] = deque(maxlen=Config.EMA_PERIOD + 20)
            asyncio.create_task(self._subscribe_candles(symbol))
            asyncio.create_task(self._subscribe_ticks(symbol))

    async def _subscribe_ticks(self, symbol):
        """Subscribes to ticks for spread calculation."""
        req = {"ticks": symbol}

        async def callback(data):
            # STRICT FILTER: Ensure this message belongs to THIS symbol
            if data.get('msg_type') == 'tick':
                tick = data.get('tick')
                if tick.get('symbol') != symbol:
                    return

                # tick object: {ask, bid, quote, ...}
                ask = tick.get('ask')
                bid = tick.get('bid')
                if ask and bid:
                    spread = ask - bid
                    state.update_spread(symbol, spread)

        await self.client.subscribe(req, callback)

    async def _subscribe_candles(self, symbol):
        # Deriv API: subscribe to history + events
        # "ticks_history" with subscribe=1 returns history then updates
        req = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": Config.EMA_PERIOD + 20,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": 60 # 1 minute candles default
        }

        # We need a callback specific to this symbol
        async def callback(data):
            msg_type = data.get('msg_type')

            # STRICT FILTER
            # For 'candles' (history response), check echo_req
            if msg_type == 'candles':
                echo_req = data.get('echo_req', {})
                if echo_req.get('ticks_history') != symbol:
                    return

                # Initial history
                clist = data.get('candles')
                self.candles[symbol].extend(clist)
                pass

            elif msg_type == 'ohlc':
                # Update
                ohlc = data.get('ohlc')
                if ohlc.get('symbol') != symbol:
                    return

                # OHLC packet: {open, high, low, close, open_time, epoch}
                # Check if it's a new candle or update to current
                last_epoch = self.candles[symbol][-1]['epoch'] if self.candles[symbol] else 0
                current_epoch = ohlc.get('open_time')

                if current_epoch == last_epoch:
                    # Update last candle
                    self.candles[symbol][-1] = self._map_ohlc(ohlc)
                elif current_epoch > last_epoch:
                    # New candle formed, the previous one closed.
                    # Run strategy on the COMPLETED history (up to last_epoch)
                    await self._process_signal(symbol)

                    # Append new
                    self.candles[symbol].append(self._map_ohlc(ohlc))

        await self.client.subscribe(req, callback)

    def _map_ohlc(self, ohlc):
        return {
            'open': ohlc['open'],
            'high': ohlc['high'],
            'low': ohlc['low'],
            'close': ohlc['close'],
            'epoch': ohlc['open_time'] # standardize key
        }

    async def _process_signal(self, symbol):
        if not self.running: return

        data = list(self.candles[symbol])

        # Update shared state with latest history (for Chart)
        state.update_history(symbol, data)

        result = self.strategy.analyze(data)

        # Update shared state with latest price
        if result:
             state.update_candle(symbol, result['price'])

        # Check Spread Filter (e.g., if spread > 0.1% of price? Or simple absolute check?)
        # Let's use a simple config threshold or relative.
        # User requested: "Spread Filter: Do not trade if spread is too wide".
        # We'll check if spread is available and reasonably low.
        spread = state.get_spread(symbol)
        price = result['price'] if result else 0

        # Simple heuristic: If spread > 0.05% of price, it's wide.
        if spread > 0 and price > 0:
            if (spread / price) > 0.0005:
                # logger.debug(f"Spread too high for {symbol}: {spread}")
                return # Skip signal

        if result and result['signal']:
            logger.info(f"SIGNAL DETECTED: {symbol} {result['signal']} - {result['reason']}")

            # Log to DB
            log = SignalLog(
                symbol=symbol,
                signal=result['signal'],
                reason=result['reason'],
                price=result['price'],
                indicators=json.dumps(result['analysis'])
            )
            self.db.add(log)
            self.db.commit()

            # Execute Trade (Check master switch first)
            if state.is_trading_active():
                # Find symbol info from shared state
                scanner_data = state.get_scanner_data()
                symbol_info = {}
                # Flatten
                for cat, assets in scanner_data.items():
                    for a in assets:
                        if a['symbol'] == symbol:
                            symbol_info = a
                            break
                    if symbol_info: break

                await self.executor.execute_trade(
                    signal_data={**result, 'symbol': symbol},
                    symbol_info=symbol_info
                )
            else:
                logger.info(f"Signal ignored (Trading Inactive): {symbol}")
