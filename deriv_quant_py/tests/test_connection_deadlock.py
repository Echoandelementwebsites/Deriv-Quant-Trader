import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from deriv_quant_py.core.connection import DerivClient
import json

class TestConnectionDeadlock(unittest.IsolatedAsyncioTestCase):
    async def test_connect_structure_fix(self):
        """
        Verifies that with the fix, the read loop is active during authorization
        and connect() returns successfully without timeout.
        """
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        # Setup an iterator for mock_ws to simulate incoming messages
        # We need it to yield a response to authorize

        auth_response = {
            "req_id": 1,
            "authorize": {"email": "test@example.com"}
        }

        # The mock websocket needs to behave like an async iterator
        async def async_iter():
            # Wait a tiny bit to simulate network delay
            await asyncio.sleep(0.01)
            yield json.dumps(auth_response)
            # Keep open for a bit then close?
            await asyncio.sleep(1)

        mock_ws.__aiter__.side_effect = async_iter

        mock_connect_ctx = AsyncMock()
        mock_connect_ctx.__aenter__.return_value = mock_ws

        with patch('websockets.connect', return_value=mock_connect_ctx):
            client = DerivClient("wss://fake", "fake_token")

            # We wrap in wait_for to ensure it doesn't hang.
            # If the fix works, this should return almost immediately (after auth).
            try:
                await asyncio.wait_for(client.connect(), timeout=1.0)
            except asyncio.TimeoutError:
                self.fail("client.connect() timed out! Deadlock might still be present.")

            # Additional assertions
            self.assertTrue(client._connected_event.is_set(), "Connected event should be set")
            # Note: is_connected might be True or False depending on race with read loop closing,
            # but since read loop sleeps for 1s, it should be True here.
            self.assertTrue(client.is_connected, "Client should be marked as connected")

            # Clean up the background task to avoid "Task was destroyed but it is pending" warning
            if client._reconnection_task:
                client._reconnection_task.cancel()
                try:
                    await client._reconnection_task
                except asyncio.CancelledError:
                    pass
