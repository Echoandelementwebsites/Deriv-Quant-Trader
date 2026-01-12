import logging
import asyncio
import json
from datetime import datetime
from deriv_quant_py.config import Config
from deriv_quant_py.core.connection import DerivClient
from deriv_quant_py.database import Trade
from deriv_quant_py.shared_state import state
from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self, client: DerivClient, db_session: Session):
        self.client = client
        self.db = db_session
        self.daily_loss = 0.0
        self.risk_multiplier = Config.RISK_MULTIPLIER
        self.base_stake = 10.0 # Default, should come from config or UI

        # Initialize Active Trades in SharedState
        try:
            open_trades = self.db.query(Trade).filter_by(status='OPEN').all()
            for t in open_trades:
                state.add_active_trade(t.symbol)
            logger.info(f"Initialized active trades: {[t.symbol for t in open_trades]}")
        except Exception as e:
            logger.error(f"Error initializing active trades: {e}")

    async def execute_trade(self, signal_data: dict, symbol_info: dict):
        """
        Executes a trade based on signal.
        signal_data: {signal, reason, price, ...}
        symbol_info: {min_duration, max_duration, ...}
        """
        symbol = signal_data['symbol']
        direction = signal_data['signal']

        # 1. Check Risk
        if not Config.TRADING_ACTIVE:
            logger.info("Trading Disabled (Master Switch). Skipping.")
            return

        if self._check_daily_loss_limit():
            logger.warning("Daily Loss Limit Reached. Skipping.")
            return

        # 2. Calculate Stake (Martingale)
        stake = self._calculate_stake(symbol)

        # 3. Calculate Duration & Validate
        # Default 3m (3 * timeframe 1m)
        duration = 3
        duration_unit = 'm'

        if symbol_info:
            min_d = symbol_info.get('min_duration', '1m')
            # max_d = symbol_info.get('max_duration', '365d') # Not checking max for now

            # Simple parsing: assumes format "123s" or "123m"
            if min_d.endswith('m'):
                min_minutes = int(min_d[:-1])
                if duration < min_minutes and duration_unit == 'm':
                    logger.warning(f"Duration {duration}m < Min {min_d}. Adjusting.")
                    duration = min_minutes
            elif min_d.endswith('s'):
                min_seconds = int(min_d[:-1])
                if duration * 60 < min_seconds and duration_unit == 'm':
                    logger.warning(f"Duration {duration}m < Min {min_d}. Adjusting.")
                    # Round up to minutes or switch unit? Keeping minutes for now.
                    import math
                    duration = math.ceil(min_seconds / 60)

        # 4. Construct Proposal
        contract_type = "CALL" if direction == "CALL" else "PUT"

        req = {
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol
        }

        try:
            # Get Proposal
            res = await self.client.send_request(req)
            if 'error' in res:
                logger.error(f"Proposal Error: {res['error']['message']}")
                return

            prop_id = res['proposal']['id']

            # Buy
            buy_req = {
                "buy": prop_id,
                "price": res['proposal']['ask_price']
            }

            buy_res = await self.client.send_request(buy_req)
            if 'error' in buy_res:
                 logger.error(f"Buy Error: {buy_res['error']['message']}")
            else:
                logger.info(f"Trade Executed: {buy_res['buy']['contract_id']} | {direction} | ${stake}")
                # Log to DB
                self._log_trade(buy_res['buy'], symbol, direction, stake)
                state.add_active_trade(symbol)

        except Exception as e:
            logger.error(f"Execution Exception: {e}")

    def _calculate_stake(self, symbol):
        # Martingale: Look at last closed trade for this symbol
        last_trade = self.db.query(Trade).filter(
            Trade.symbol == symbol,
            Trade.status.in_(['WON', 'LOST'])
        ).order_by(Trade.entry_time.desc()).first()

        if last_trade and last_trade.status == 'LOST':
            return last_trade.stake * self.risk_multiplier

        return self.base_stake

    def _check_daily_loss_limit(self):
        # Sum profit of trades today
        today = datetime.utcnow().date()
        trades = self.db.query(Trade).filter(
            func.date(Trade.entry_time) == today,
            Trade.status != 'OPEN'
        ).all()

        total_pnl = sum(t.profit for t in trades)
        return total_pnl < -Config.DAILY_LOSS_LIMIT

    def _log_trade(self, buy_data, symbol, direction, stake):
        trade = Trade(
            symbol=symbol,
            signal_type=direction,
            contract_id=str(buy_data['contract_id']),
            stake=stake,
            status='OPEN',
            details=json.dumps(buy_data)
        )
        self.db.add(trade)
        self.db.commit()

    async def monitor_trades(self):
        """Subscribes to updates for open contracts to track Win/Loss."""
        # Subscribe to all open contracts
        req = {
            "proposal_open_contract": 1,
            "subscribe": 1
        }

        async def callback(data):
            if data.get('msg_type') == 'proposal_open_contract':
                contract = data.get('proposal_open_contract')
                contract_id = str(contract.get('contract_id'))

                if contract.get('is_sold'):
                    # Trade Finished
                    profit = float(contract.get('profit', 0.0))
                    status = "WON" if profit > 0 else "LOST"

                    # Update DB
                    try:
                        trade = self.db.query(Trade).filter_by(contract_id=contract_id).first()
                        if trade and trade.status == 'OPEN':
                            trade.status = status
                            trade.profit = profit
                            trade.exit_time = datetime.utcnow()
                            self.db.commit()
                            logger.info(f"Trade Closed: {contract_id} | {status} | ${profit}")
                            state.remove_active_trade(trade.symbol)
                    except Exception as e:
                        logger.error(f"Error updating trade {contract_id}: {e}")
                        self.db.rollback()

        await self.client.subscribe(req, callback)
