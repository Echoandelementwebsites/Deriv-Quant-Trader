from threading import Lock

class SharedState:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SharedState, cls).__new__(cls)
                cls._instance._init_data()
            return cls._instance

    def _init_data(self):
        self.scanner_data = {} # {Category: [Assets]}
        self.latest_candles = {} # {Symbol: ClosePrice}
        self.candles_history = {} # {Symbol: [Candles]}
        self.current_spreads = {} # {Symbol: Spread}
        self.active_trade_symbols = set() # {Symbol}
        self.ui_visible_symbols = set() # {Symbol}
        self.backtest_request = None # {symbol: str}
        self.backtest_result = None # DataFrame or Dict
        self.scan_progress = {
            "total": 0,
            "current": 0,
            "current_symbol": None,
            "status": "idle" # idle, running, complete
        }
        self.system_status = {
            "trading_active": False,
            "risk_percentage": 1.0,
            "daily_loss_limit": 50.0
        }
        self.connection_status = False
        self.balance = 0.0

    def add_active_trade(self, symbol):
        with self._lock:
            self.active_trade_symbols.add(symbol)

    def remove_active_trade(self, symbol):
        with self._lock:
            self.active_trade_symbols.discard(symbol)

    def get_active_trades(self):
        with self._lock:
            return self.active_trade_symbols.copy()

    def set_ui_visible_symbols(self, symbols: list):
        with self._lock:
            self.ui_visible_symbols = set(symbols)

    def get_ui_visible_symbols(self):
        with self._lock:
            return self.ui_visible_symbols.copy()

    def update_scanner(self, data):
        with self._lock:
            self.scanner_data = data

    def update_spread(self, symbol, spread):
        with self._lock:
            self.current_spreads[symbol] = spread

    def get_spread(self, symbol):
        with self._lock:
            return self.current_spreads.get(symbol, 0.0)

    def update_candle(self, symbol, price):
        with self._lock:
            self.latest_candles[symbol] = price

    def update_history(self, symbol, candles):
        with self._lock:
            self.candles_history[symbol] = list(candles) # copy

    def get_history(self, symbol):
        with self._lock:
            return self.candles_history.get(symbol, [])

    def set_backtest_request(self, symbol):
        with self._lock:
            self.backtest_request = symbol
            self.backtest_result = None

    def get_backtest_request(self):
        with self._lock:
            req = self.backtest_request
            self.backtest_request = None
            return req

    def set_backtest_result(self, result):
        with self._lock:
            self.backtest_result = result

    def get_backtest_result(self):
        with self._lock:
            return self.backtest_result

    def update_scan_progress(self, total, current, symbol, status="running"):
        with self._lock:
            self.scan_progress = {
                "total": total,
                "current": current,
                "current_symbol": symbol,
                "status": status
            }

    def get_scan_progress(self):
        with self._lock:
            return self.scan_progress.copy()

    def set_trading_active(self, active: bool):
        with self._lock:
            self.system_status["trading_active"] = active

    def set_risk_settings(self, percentage, limit):
        with self._lock:
            self.system_status["risk_percentage"] = float(percentage)
            self.system_status["daily_loss_limit"] = float(limit)

    def get_scanner_data(self):
        with self._lock:
            return self.scanner_data.copy()

    def update_balance(self, balance):
        with self._lock:
            self.balance = float(balance)

    def get_balance(self):
        with self._lock:
            return self.balance

    def is_trading_active(self):
        with self._lock:
            return self.system_status["trading_active"]

    def get_risk_settings(self):
        with self._lock:
            return self.system_status.copy()

state = SharedState()
