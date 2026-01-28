import asyncio
import websockets
import json
import logging
import time
from typing import Optional, Dict, Any, Callable
from deriv_quant_py.shared_state import state

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

        self._connected_event = asyncio.Event()
        self._reconnection_task: Optional[asyncio.Task] = None
        self.last_msg_time = time.time()

    async def connect(self):
        """
        Starts the connection maintenance task and waits for the initial
        connection and authorization to complete.
        """
        if self._reconnection_task is None:
            self._reconnection_task = asyncio.create_task(self._maintain_connection())

        logger.info("Waiting for initial connection and authorization...")
        await self._connected_event.wait()
        logger.info("Initial connection established.")

    async def _maintain_connection(self):
        """Establishes the WebSocket connection and handles reconnection."""
        while True:
            try:
                logger.info(f"Connecting to {self.url}...")
                async with websockets.connect(self.url) as ws:
                    self.ws = ws
                    self.is_connected = True
                    self.last_msg_time = time.time() # Reset on new connection
                    logger.info("Connected to Deriv API.")

                    # Start Message Loop (CRITICAL: Must start before authorize)
                    read_task = asyncio.create_task(self._read_loop())

                    # Start Ping Loop
                    ping_task = asyncio.create_task(self._ping_loop())

                    # Start Heartbeat Monitor
                    heartbeat_task = asyncio.create_task(self._monitor_heartbeat())

                    try:
                        # Authenticate
                        if self.token:
                            await self.authorize()
                            await self.send_request({"balance": 1, "subscribe": 1})

                        # Signal that we are connected and (attempted to) authorize
                        self._connected_event.set()

                        # Wait for the read loop to finish (connection closed)
                        await read_task

                    except Exception as e:
                        logger.error(f"Error during connection session: {e}")
                        # Ensure we cancel the read loop if it's still running (though unlikely if ws closed)
                        if not read_task.done():
                            read_task.cancel()
                        raise # Re-raise to trigger reconnection in outer loop

                    finally:
                        self.is_connected = False
                        self._connected_event.clear()
                        ping_task.cancel()
                        heartbeat_task.cancel()
                        if not read_task.done():
                             read_task.cancel()

            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.is_connected = False
                await asyncio.sleep(5) # Backoff before reconnect

    async def _read_loop(self):
        """Reads messages from the websocket."""
        try:
            async for message in self.ws:
                await self._handle_message(message)
        except websockets.ConnectionClosed:
            logger.warning("Websocket connection closed.")
        except Exception as e:
            logger.error(f"Error in read loop: {e}")

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

    async def _monitor_heartbeat(self):
        """Monitors for incoming messages. Force reconnect if dead."""
        while self.is_connected:
            try:
                await asyncio.sleep(10)
                if time.time() - self.last_msg_time > 30:
                    logger.warning("Heartbeat Timeout! Last message > 30s ago. Reconnecting...")
                    if self.ws:
                        await self.ws.close()
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
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
        self.msg_handlers.append(callback)
        await self.ws.send(json.dumps(req))


    async def _handle_message(self, message: str):
        self.last_msg_time = time.time()
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
            return

        if not data:
            return

        # 1. Handle Request Responses
        req_id = data.get("req_id")
        if req_id and req_id in self.pending_requests:
            if not self.pending_requests[req_id].done():
                self.pending_requests[req_id].set_result(data)
            del self.pending_requests[req_id]

        # 2. Balance Update
        if data.get('msg_type') == 'balance':
            balance = data.get('balance', {}).get('balance')
            if balance is not None:
                state.update_balance(balance)

        # 3. General Handlers (Subscriptions)
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
