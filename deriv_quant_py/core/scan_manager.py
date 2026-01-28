import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ScanManager:
    def __init__(self, state_file="scan_state.json"):
        self.state_file = state_file
        self.state = self.load_state()

    def load_state(self):
        if not os.path.exists(self.state_file):
            return {}
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading scan state: {e}")
            return {}

    def save_state(self):
        try:
            # Atomic write (write to temp then rename)
            temp_file = self.state_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.state, f, indent=4)
            os.replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Error saving scan state: {e}")

    def update_status(self, symbol, status):
        """
        Updates the status of a symbol.
        Status: 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'SKIPPED'
        """
        self.state[symbol] = {
            'status': status,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.save_state()

    def get_status(self, symbol):
        return self.state.get(symbol, {}).get('status', 'PENDING')

    def get_pending_symbols(self, all_symbols, resume=True):
        """
        Returns symbols that need to be scanned.
        If resume=False, returns all_symbols (and resets state).
        If resume=True, returns only PENDING or FAILED (or unlisted) symbols.
        """
        if not resume:
            # Reset state for these symbols
            for s in all_symbols:
                self.update_status(s, 'PENDING')
            return all_symbols

        pending = []
        for s in all_symbols:
            status = self.get_status(s)
            # We re-scan if it's PENDING, RUNNING (crashed mid-run), or FAILED.
            # We SKIP only if COMPLETED.
            if status not in ['COMPLETED']:
                pending.append(s)

        return pending

    def mark_completed(self, symbol):
        self.update_status(symbol, 'COMPLETED')

    def mark_failed(self, symbol):
        self.update_status(symbol, 'FAILED')
