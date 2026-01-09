import asyncio
import websockets
import json
import logging
import time
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)

class DerivClient:
    def __init__(self, websocket_url: str, token: str):
        self.url = websocket_url
        self.token = token
        self.ws = None
        self.is_connected = False
        self.req_id_counter = 1
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.subscriptions: Dict[str, Callable] = {} # subscription_id -> callback
        self.msg_handlers: list[Callable] = [] # General message handlers

    async def connect(self):
        """Establishes the WebSocket connection and handles reconnection."""
        while True:
            try:
                logger.info(f"Connecting to {self.url}...")
                async with websockets.connect(self.url) as ws:
                    self.ws = ws
                    self.is_connected = True
                    logger.info("Connected to Deriv API.")

                    # Authenticate immediately
                    if self.token:
                        await self.authorize()

                    # Start Ping Loop
                    ping_task = asyncio.create_task(self._ping_loop())

                    # Message Loop
                    try:
                        async for message in ws:
                            await self._handle_message(message)
                    except websockets.ConnectionClosed as e:
                        logger.warning(f"Connection closed: {e}")
                    finally:
                        self.is_connected = False
                        ping_task.cancel()

            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.is_connected = False
                await asyncio.sleep(5) # Backoff before reconnect

    async def _ping_loop(self):
        """Sends a ping every 20 seconds to keep connection alive."""
        while self.is_connected:
            try:
                await asyncio.sleep(20)
                await self.send_request({"ping": 1})
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ping error: {e}")
                break

    async def authorize(self):
        """Authorizes the connection."""
        res = await self.send_request({"authorize": self.token})
        if "error" in res:
            logger.error(f"Authorization failed: {res['error']['message']}")
        else:
            logger.info(f"Authorized as {res['authorize']['email']}")

    async def send_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Sends a request and awaits the response (matched by req_id)."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Deriv API")

        req_id = self.req_id_counter
        self.req_id_counter += 1
        req["req_id"] = req_id

        future = asyncio.get_running_loop().create_future()
        self.pending_requests[req_id] = future

        await self.ws.send(json.dumps(req))

        try:
            # Wait for response with timeout
            return await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError:
            del self.pending_requests[req_id]
            raise TimeoutError(f"Request {req_id} timed out")

    async def subscribe(self, req: Dict[str, Any], callback: Callable):
        """Sends a subscription request and registers a callback."""
        req["subscribe"] = 1
        # We don't await the response here in the same way, but we could.
        # For simplicity, we send and let the generic handler route it?
        # Better: Send request, get 'subscription' field in response, map ID to callback.

        # NOTE: Deriv API returns the initial response with `req_id`, then updates with `subscription`.
        # However, the updates don't have `req_id`. They have `msg_type` or specific keys.
        # We need a way to route specific stream updates.

        # Strategy: Map stream type (e.g., 'tick', 'proposal_open_contract') to callback?
        # Or just append to general handlers and let the handler filter?
        # Let's use general handlers for now, but specialized routing is better.

        self.msg_handlers.append(callback)
        await self.ws.send(json.dumps(req))


    async def _handle_message(self, message: str):
        data = json.loads(message)

        # 1. Handle Request Responses
        req_id = data.get("req_id")
        if req_id in self.pending_requests:
            if not self.pending_requests[req_id].done():
                self.pending_requests[req_id].set_result(data)
            del self.pending_requests[req_id]

        # 2. General Handlers (Subscriptions)
        for handler in self.msg_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")

    def add_handler(self, callback: Callable):
        self.msg_handlers.append(callback)
