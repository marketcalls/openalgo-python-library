# -*- coding: utf-8 -*-
"""
OpenAlgo WebSocket API Documentation - Feed Methods
    https://docs.openalgo.in
"""

import json
import threading
import time
from typing import List, Dict, Any, Callable, Optional, Tuple
import websocket
from .base import BaseAPI

class FeedAPI(BaseAPI):
    """
    Market data feed API methods for OpenAlgo using WebSockets.
    Inherits from the BaseAPI class.
    """

    def __init__(self, api_key, host="http://127.0.0.1:5000", version="v1", ws_port=8765, ws_url=None, verbose=False, auto_reconnect=True):
        """
        Initialize the FeedAPI object with API key and optionally a host URL, API version, and WebSocket details.

        Attributes:
        - api_key (str): User's API key.
        - host (str): Base URL for the API endpoints. Defaults to localhost.
        - version (str): API version. Defaults to "v1".
        - ws_port (int): WebSocket server port. Defaults to 8765.
        - ws_url (str, optional): Custom WebSocket URL. If provided, this overrides host and ws_port settings.
        - verbose (int): Logging verbosity level. Defaults to False.
            - 0 or False: Silent mode (errors only)
            - 1 or True: Basic info (connection, auth, subscription status)
            - 2: Full debug (all market data updates)
        - auto_reconnect (bool): If True (default), the SDK transparently
            reconnects after a drop, re-authenticates, and replays all active
            subscriptions with exponential backoff. Existing strategies will
            survive network blips and broker session refreshes without any
            extra code. Set to False to keep the previous manual-reconnect
            behaviour.
        """
        super().__init__(api_key, host, version)

        # Verbosity control
        self.verbose = int(verbose) if verbose is not False else 0
        
        # WebSocket configuration
        self.ws_port = ws_port
        
        # Use custom WebSocket URL if provided
        if ws_url:
            self.ws_url = ws_url
        else:
            # Extract host without protocol for WebSocket
            if host.startswith("http://"):
                self.ws_host = host[7:]
            elif host.startswith("https://"):
                self.ws_host = host[8:]
            else:
                self.ws_host = host
                
            # Remove any path component and port if present
            self.ws_host = self.ws_host.split('/')[0].split(':')[0]
            
            # Create default WebSocket URL
            self.ws_url = f"ws://{self.ws_host}:{self.ws_port}"
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.ws_thread = None
        
        # Message management
        self.message_queue = []
        self.lock = threading.Lock()
        
        # Data storage
        self.ltp_data = {}  # Structure: {'EXCHANGE:SYMBOL': {'price': price, 'timestamp': timestamp}}
        self.quotes_data = {}  # Structure: {'EXCHANGE:SYMBOL': {'open': open, 'high': high, 'low': low, 'close': close, 'ltp': ltp, 'timestamp': timestamp}}
        self.depth_data = {}  # Structure: {'EXCHANGE:SYMBOL': {'ltp': ltp, 'timestamp': timestamp, 'depth': {'buy': [...], 'sell': [...]}}}
        
        # Callback registry
        self.ltp_callback = None
        self.quote_callback = None
        self.quotes_callback = None
        self.depth_callback = None

        # Auto-reconnect state (private, additive — does not change any
        # existing public attribute or behaviour).
        self.auto_reconnect = bool(auto_reconnect)
        self._shutting_down = False
        self._reconnect_thread = None
        self._reconnect_lock = threading.Lock()
        # Active subscription registry keyed by mode -> {(exchange, symbol): instrument_dict}
        # Replayed verbatim on reconnect.
        self._active_subs: Dict[int, Dict[Tuple[str, str], Dict[str, Any]]] = {1: {}, 2: {}, 3: {}}

    def _log(self, level: int, category: str, message: str) -> None:
        """
        Internal logging method with verbosity control.

        Args:
            level (int): Required verbosity level (1=basic, 2=debug)
            category (str): Log category for alignment (WS, AUTH, SUB, LTP, QUOTE, DEPTH, ERROR)
            message (str): The message to log
        """
        if self.verbose >= level:
            # Fixed width category for alignment
            cat_width = 6
            formatted_cat = f"[{category}]".ljust(cat_width + 2)
            print(f"{formatted_cat} {message}")

    def connect(self) -> bool:
        """
        Connect to the WebSocket server and authenticate.

        Returns:
            bool: True if connection and authentication are successful, False otherwise.
        """
        # User-initiated connect resets the shutdown flag so auto-reconnect
        # is armed. (disconnect() sets _shutting_down=True to suppress it.)
        self._shutting_down = False
        return self._do_connect()

    def _do_connect(self) -> bool:
        """
        Internal: open one WebSocket connection, start the read thread, and
        wait for authentication. Used by both connect() and the reconnect
        loop. Public surface (self.ws, self.ws_thread, self.connected,
        self.authenticated) is unchanged.
        """
        try:
            def on_message(ws, message):
                self._process_message(message)

            def on_error(ws, error):
                self._log(1, "ERROR", f"WebSocket error: {error}")

            def on_open(ws):
                self._log(1, "WS", f"Connected to {self.ws_url}")
                self.connected = True
                self._authenticate()

            def on_close(ws, close_status_code, close_reason):
                self._log(1, "WS", f"Disconnected from {self.ws_url}")
                self.connected = False
                self.authenticated = False
                # Schedule auto-reconnect unless the user called disconnect()
                # or auto_reconnect is disabled. Any registered subscriptions
                # will be replayed once the new connection authenticates.
                if self.auto_reconnect and not self._shutting_down:
                    self._schedule_reconnect()

            # Initialize WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_open=on_open,
                on_close=on_close
            )

            # Start WebSocket connection in a separate thread.
            # ping_interval sends a WebSocket ping every 20s so the OS-level TCP
            # connection cannot become a silent "zombie" (broker stops streaming
            # ticks but the socket appears OPEN with no error). ping_timeout
            # closes and triggers on_close if the pong is not received within
            # 10s, allowing the caller to detect and reconnect.
            self.ws_thread = threading.Thread(
                target=self.ws.run_forever,
                kwargs={"ping_interval": 20, "ping_timeout": 10},
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()

            # Wait for connection to establish
            timeout = 5
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)

            if not self.connected:
                self._log(1, "ERROR", "Failed to connect to WebSocket server")
                return False

            # Wait for authentication to complete
            timeout = 5
            start_time = time.time()
            while not self.authenticated and time.time() - start_time < timeout and self.connected:
                time.sleep(0.1)

            return self.authenticated

        except Exception as e:
            self._log(1, "ERROR", f"Error connecting to WebSocket: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        # Mark a graceful shutdown so on_close does not trigger auto-reconnect.
        self._shutting_down = True
        if self.ws:
            self.ws.close()
            # Wait for websocket to close
            timeout = 2
            start_time = time.time()
            while self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            self.ws = None
            self.connected = False
            self.authenticated = False

    def _schedule_reconnect(self) -> None:
        """
        Spawn the reconnect thread if one is not already running. Called from
        on_close. No-op when auto_reconnect is disabled or the user has
        called disconnect().
        """
        with self._reconnect_lock:
            if self._shutting_down or not self.auto_reconnect:
                return
            if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
                return
            self._reconnect_thread = threading.Thread(
                target=self._reconnect_loop,
                name="openalgo-ws-reconnect",
                daemon=True,
            )
            self._reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        """
        Reconnect with exponential backoff, then replay every active
        subscription. Stops when the user calls disconnect() or a connection
        is fully re-authenticated.
        """
        # Exponential backoff schedule (seconds), capped at 60s.
        backoffs = [1, 2, 5, 10, 30, 60]
        attempt = 0
        while not self._shutting_down and not self.authenticated:
            delay = backoffs[min(attempt, len(backoffs) - 1)]
            self._log(1, "WS", f"Reconnect attempt {attempt + 1} in {delay}s...")
            # Sleep in small slices so a disconnect() is responsive.
            slept = 0.0
            while slept < delay and not self._shutting_down:
                time.sleep(0.2)
                slept += 0.2
            if self._shutting_down:
                return
            try:
                ok = self._do_connect()
                if ok:
                    self._log(1, "WS", "Reconnected and re-authenticated.")
                    self._replay_subscriptions()
                    return
            except Exception as e:
                self._log(1, "ERROR", f"Reconnect attempt {attempt + 1} failed: {e}")
            attempt += 1

    def _replay_subscriptions(self) -> None:
        """
        Re-issue every previously active subscription as a single bulk message
        per mode. Called once after a successful reconnect+authenticate. The
        existing self.ltp_callback / self.quote_callback / self.depth_callback
        references are preserved across reconnects, so callbacks resume
        firing without any extra plumbing.
        """
        for mode, store in self._active_subs.items():
            if not store:
                continue
            instruments = list(store.values())
            mode_name = {1: "LTP", 2: "Quote", 3: "Depth"}.get(mode, str(mode))
            self._log(1, "SUB", f"Replaying {len(instruments)} {mode_name} subscription(s) after reconnect")
            self._send_subscribe_msg(mode, instruments)

    def _send_subscribe_msg(self, mode: int, instruments: List[Dict[str, Any]], depth: int = 5) -> bool:
        """
        Send a single bulk subscribe message containing the symbols array.
        The server has supported `data["symbols"]` since the unified
        WebSocket proxy was introduced; using the bulk path replaces the
        previous per-symbol loop with a 100ms forced sleep. Returns True if
        the frame was sent, False on transport failure.
        """
        if not self.ws:
            return False
        msg = {
            "action": "subscribe",
            "symbols": instruments,
            "mode": mode,
            "depth": depth,
        }
        try:
            self.ws.send(json.dumps(msg))
            return True
        except Exception as e:
            self._log(1, "ERROR", f"Error sending bulk subscribe: {e}")
            return False

    def _send_unsubscribe_msg(self, mode: int, instruments: List[Dict[str, Any]]) -> bool:
        """
        Send a single bulk unsubscribe message. Each entry carries its own
        mode field as required by the server contract (server.py:1187).
        """
        if not self.ws:
            return False
        msg = {
            "action": "unsubscribe",
            "symbols": [
                {"symbol": i["symbol"], "exchange": i["exchange"], "mode": mode}
                for i in instruments
            ],
        }
        try:
            self.ws.send(json.dumps(msg))
            return True
        except Exception as e:
            self._log(1, "ERROR", f"Error sending bulk unsubscribe: {e}")
            return False

    def _authenticate(self) -> None:
        """Authenticate with the WebSocket server using the API key."""
        if not self.connected:
            self._log(1, "ERROR", "Cannot authenticate: not connected")
            return

        auth_msg = {
            "action": "authenticate",
            "api_key": self.api_key
        }

        self._log(1, "AUTH", f"Authenticating with API key: {self.api_key[:8]}...{self.api_key[-8:]}")
        self.ws.send(json.dumps(auth_msg))

    def _process_message(self, message_str: str) -> None:
        """
        Process incoming WebSocket messages.
        
        Args:
            message_str (str): The message string received from the WebSocket.
        """
        try:
            message = json.loads(message_str)
            
            # Handle authentication response
            if message.get("type") == "auth":
                if message.get("status") == "success":
                    self.authenticated = True
                    broker = message.get("broker", "unknown")
                    user_id = message.get("user_id", "unknown")
                    self._log(1, "AUTH", f"Success | Broker: {broker} | User: {user_id}")
                    self._log(2, "AUTH", f"Full response: {message}")
                else:
                    self._log(1, "ERROR", f"Authentication failed: {message.get('message', 'Unknown error')}")
                return

            # Handle subscription response
            if message.get("type") == "subscribe":
                subs = message.get("subscriptions", [])
                for sub in subs:
                    sym = sub.get("symbol", "?")
                    exch = sub.get("exchange", "?")
                    status = sub.get("status", "?")
                    mode = sub.get("mode", 0)
                    mode_name = {1: "LTP", 2: "Quote", 3: "Depth"}.get(mode, "Unknown")
                    self._log(1, "SUB", f"{exch}:{sym} | Mode: {mode_name} | Status: {status}")
                self._log(2, "SUB", f"Full response: {message}")
                return
                
            # Handle market data
            if message.get("type") == "market_data":
                exchange = message.get("exchange")
                symbol = message.get("symbol")
                if exchange and symbol:
                    mode = message.get("mode")
                    market_data = message.get("data", {})
                    
                    # Handle LTP data (mode 1)
                    if mode == 1 and "ltp" in market_data:
                        with self.lock:
                            # Get LTP and timestamp from the message
                            ltp = market_data.get("ltp")
                            timestamp = market_data.get("timestamp", int(time.time() * 1000))
                            
                            # Store both price and timestamp with format 'EXCHANGE:SYMBOL'
                            symbol_key = f"{exchange}:{symbol}"
                            self.ltp_data[symbol_key] = {
                                'price': ltp,
                                'timestamp': timestamp
                            }

                            self._log(2, "LTP", f"{symbol_key:<20} | LTP: {ltp}")
                        
                        # Invoke callback if set
                        if self.ltp_callback:
                            try:
                                # Create a clean market data update without redundant fields
                                clean_data = {
                                    'type': 'market_data',
                                    'symbol': symbol,
                                    'exchange': exchange,
                                    'mode': mode,
                                    'data': {
                                        'ltp': ltp,
                                        'timestamp': timestamp
                                    }
                                }
                                # Include LTT if available in original data
                                if 'ltt' in market_data:
                                    clean_data['data']['ltt'] = market_data['ltt']
                                # Check if ltt is in the nested data structure (which seems to be the case)
                                elif 'ltt' in market_data.get('data', {}):
                                    clean_data['data']['ltt'] = market_data['data']['ltt']
                                    
                                # Pass the cleaned message to callback
                                self.ltp_callback(clean_data)
                            except Exception as e:
                                self._log(1, "ERROR", f"LTP callback error: {str(e)}")                 
                    # Handle Quotes data (mode 2)
                    elif mode == 2:
                        with self.lock:
                            # Extract quote data fields
                            quote_data = {
                                'open': market_data.get("open", 0),
                                'high': market_data.get("high", 0),
                                'low': market_data.get("low", 0),
                                'close': market_data.get("close", 0),
                                'ltp': market_data.get("ltp", 0),
                                'volume': market_data.get("volume", 0),
                                'last_trade_quantity': market_data.get("last_trade_quantity", 0),
                                'avg_trade_price': market_data.get("avg_trade_price", 0),
                                'change': market_data.get("change", 0),
                                'change_percent': market_data.get("change_percent", 0),
                                'timestamp': market_data.get("timestamp", int(time.time() * 1000))
                            }
                            
                            # Store quote data with format 'EXCHANGE:SYMBOL'
                            symbol_key = f"{exchange}:{symbol}"
                            self.quotes_data[symbol_key] = quote_data

                            self._log(2, "QUOTE", f"{symbol_key:<20} | O: {quote_data['open']:<10} H: {quote_data['high']:<10} L: {quote_data['low']:<10} C: {quote_data['close']:<10} LTP: {quote_data['ltp']}")
                        
                        # Invoke callback if set
                        if self.quote_callback:
                            try:
                                # Create a clean market data update without redundant fields
                                clean_data = {
                                    'type': 'market_data',
                                    'symbol': symbol,
                                    'exchange': exchange,
                                    'mode': mode,
                                    'data': quote_data.copy()
                                }
                                # Pass the cleaned message to callback
                                self.quote_callback(clean_data)
                            except Exception as e:
                                self._log(1, "ERROR", f"Quote callback error: {str(e)}")                 
                    # Handle Market Depth data (mode 3)
                    elif mode == 3 and "depth" in market_data:
                        with self.lock:
                            # Extract depth data
                            depth_data = {
                                'ltp': market_data.get("ltp", 0),
                                'timestamp': market_data.get("timestamp", int(time.time() * 1000)),
                                'depth': market_data.get("depth", {"buy": [], "sell": []})
                            }
                            
                            # Store depth data with format 'EXCHANGE:SYMBOL'
                            symbol_key = f"{exchange}:{symbol}"
                            self.depth_data[symbol_key] = depth_data

                            # Log depth data
                            buy_depth = depth_data.get('depth', {}).get('buy', [])
                            sell_depth = depth_data.get('depth', {}).get('sell', [])

                            self._log(2, "DEPTH", f"{symbol_key:<20} | LTP: {depth_data.get('ltp')}")

                            if self.verbose >= 2:
                                # Print buy depth summary
                                print(f"         {'BUY':<30} | {'SELL':<30}")
                                print(f"         {'Price':<10} {'Qty':<10} {'Orders':<8} | {'Price':<10} {'Qty':<10} {'Orders':<8}")
                                print("         " + "-" * 62)
                                max_levels = max(len(buy_depth), len(sell_depth), 1)
                                for i in range(max_levels):
                                    buy_lvl = buy_depth[i] if i < len(buy_depth) else {}
                                    sell_lvl = sell_depth[i] if i < len(sell_depth) else {}
                                    bp = buy_lvl.get('price', '-')
                                    bq = buy_lvl.get('quantity', '-')
                                    bo = buy_lvl.get('orders', '-')
                                    sp = sell_lvl.get('price', '-')
                                    sq = sell_lvl.get('quantity', '-')
                                    so = sell_lvl.get('orders', '-')
                                    print(f"         {str(bp):<10} {str(bq):<10} {str(bo):<8} | {str(sp):<10} {str(sq):<10} {str(so):<8}")
                        
                        # Invoke callback if set
                        if self.depth_callback:
                            try:
                                # Create a clean market data update
                                clean_data = {
                                    'type': 'market_data',
                                    'symbol': symbol,
                                    'exchange': exchange,
                                    'mode': mode,
                                    'data': depth_data.copy()
                                }
                                # Pass the cleaned message to callback
                                self.depth_callback(clean_data)
                            except Exception as e:
                                self._log(1, "ERROR", f"Depth callback error: {str(e)}")
                        
        except json.JSONDecodeError:
            self._log(1, "ERROR", f"Invalid JSON message: {message_str[:100]}...")
        except Exception as e:
            self._log(1, "ERROR", f"Error handling message: {e}")

    def subscribe_ltp(self, instruments: List[Dict[str, Any]], on_data_received: Optional[Callable] = None) -> bool:
        """
        Subscribe to LTP updates for instruments.

        Args:
            instruments: List of instrument dictionaries with keys:
                - exchange (str): Exchange code (e.g., 'NSE', 'BSE', 'NFO')
                - symbol (str): Trading symbol
                - exchange_token (str, optional): Exchange token for the instrument
            on_data_received: Callback function for data updates

        Returns:
            bool: True if subscription successful, False otherwise
        """
        if not self.connected:
            self._log(1, "ERROR", "Not connected to WebSocket server")
            return False

        if not self.authenticated:
            self._log(1, "ERROR", "Not authenticated with WebSocket server")
            return False

        # Set callback if provided
        if on_data_received:
            self.ltp_callback = on_data_received

        # Validate inputs and register in the active-subscription store
        # (used for replay across reconnects). Behaviour preserved: invalid
        # instruments are logged and skipped just like the previous loop.
        valid: List[Dict[str, Any]] = []
        for instrument in instruments:
            exchange = instrument.get("exchange")
            symbol = instrument.get("symbol")
            exchange_token = instrument.get("exchange_token")

            # If only exchange_token is provided, we need to map it to a symbol
            if not symbol and exchange_token:
                symbol = exchange_token

            if not exchange or not symbol:
                self._log(1, "ERROR", f"Invalid instrument: {instrument}")
                continue

            entry = {"symbol": symbol, "exchange": exchange}
            valid.append(entry)
            self._active_subs[1][(exchange, symbol)] = entry
            self._log(1, "SUB", f"Subscribing {exchange}:{symbol} LTP...")

        if not valid:
            return False

        # Single bulk message instead of N×100ms loop. The unified WebSocket
        # proxy (server.py) accepts `data["symbols"]` as an array; using it
        # makes a 200-symbol option-chain subscribe near-instant instead of
        # ~20 seconds.
        return self._send_subscribe_msg(mode=1, instruments=valid)

    def unsubscribe_ltp(self, instruments: List[Dict[str, Any]]) -> bool:
        """
        Unsubscribe from LTP updates for instruments.

        Args:
            instruments: List of instrument dictionaries with keys:
                - exchange (str): Exchange code (e.g., 'NSE', 'BSE', 'NFO')
                - symbol (str): Trading symbol
                - exchange_token (str, optional): Exchange token for the instrument

        Returns:
            bool: True if unsubscription successful, False otherwise
        """
        if not self.connected or not self.authenticated:
            return False

        valid: List[Dict[str, Any]] = []
        for instrument in instruments:
            exchange = instrument.get("exchange")
            symbol = instrument.get("symbol")
            exchange_token = instrument.get("exchange_token")

            if not symbol and exchange_token:
                symbol = exchange_token

            if not exchange or not symbol:
                self._log(1, "ERROR", f"Invalid instrument: {instrument}")
                continue

            self._log(1, "UNSUB", f"Unsubscribing {exchange}:{symbol} LTP")
            valid.append({"symbol": symbol, "exchange": exchange})

            # Drop from active-subscription store so reconnect does not replay.
            self._active_subs[1].pop((exchange, symbol), None)

            # Clean up cached LTP data for this symbol (preserved behaviour).
            with self.lock:
                symbol_key = f"{exchange}:{symbol}"
                if symbol_key in self.ltp_data:
                    del self.ltp_data[symbol_key]

        if not valid:
            return False

        return self._send_unsubscribe_msg(mode=1, instruments=valid)

    def subscribe_quote(self, instruments: List[Dict[str, Any]], on_data_received: Optional[Callable] = None) -> bool:
        """
        Subscribe to Quote updates for instruments.

        Args:
            instruments: List of instrument dictionaries with keys:
                - exchange (str): Exchange code (e.g., 'NSE', 'BSE', 'NFO')
                - symbol (str): Trading symbol
                - exchange_token (str, optional): Exchange token for the instrument
            on_data_received: Callback function for data updates

        Returns:
            bool: True if subscription request sent successfully
        """
        if not self.connected:
            self._log(1, "ERROR", "Not connected to WebSocket server")
            return False

        if not self.authenticated:
            self._log(1, "ERROR", "Not authenticated with WebSocket server")
            return False

        if on_data_received:
            self.quote_callback = on_data_received

        valid: List[Dict[str, Any]] = []
        for instrument in instruments:
            exchange = instrument.get("exchange")
            symbol = instrument.get("symbol")
            exchange_token = instrument.get("exchange_token")

            if not symbol and exchange_token:
                symbol = exchange_token

            if not exchange or not symbol:
                self._log(1, "ERROR", f"Invalid instrument: {instrument}")
                continue

            entry = {"symbol": symbol, "exchange": exchange}
            valid.append(entry)
            self._active_subs[2][(exchange, symbol)] = entry
            self._log(1, "SUB", f"Subscribing {exchange}:{symbol} Quote...")

        if not valid:
            return False

        return self._send_subscribe_msg(mode=2, instruments=valid)

    def unsubscribe_quote(self, instruments: List[Dict[str, Any]]) -> bool:
        """
        Unsubscribe from Quote updates for instruments.

        Args:
            instruments: List of instrument dictionaries with keys:
                - exchange (str): Exchange code (e.g., 'NSE', 'BSE', 'NFO')
                - symbol (str): Trading symbol
                - exchange_token (str, optional): Exchange token for the instrument

        Returns:
            bool: True if unsubscription successful, False otherwise
        """
        if not self.connected or not self.authenticated:
            return False

        valid: List[Dict[str, Any]] = []
        for instrument in instruments:
            exchange = instrument.get("exchange")
            symbol = instrument.get("symbol")
            exchange_token = instrument.get("exchange_token")

            if not symbol and exchange_token:
                symbol = exchange_token

            if not exchange or not symbol:
                self._log(1, "ERROR", f"Invalid instrument: {instrument}")
                continue

            self._log(1, "UNSUB", f"Unsubscribing {exchange}:{symbol} Quote")
            valid.append({"symbol": symbol, "exchange": exchange})

            self._active_subs[2].pop((exchange, symbol), None)

            with self.lock:
                symbol_key = f"{exchange}:{symbol}"
                if symbol_key in self.quotes_data:
                    del self.quotes_data[symbol_key]

        if not valid:
            return False

        return self._send_unsubscribe_msg(mode=2, instruments=valid)

    def subscribe_depth(self, instruments: List[Dict[str, Any]], on_data_received: Optional[Callable] = None) -> bool:
        """
        Subscribe to Market Depth updates for instruments.

        Args:
            instruments: List of instrument dictionaries with keys:
                - exchange (str): Exchange code (e.g., 'NSE', 'BSE', 'NFO')
                - symbol (str): Trading symbol
                - exchange_token (str, optional): Exchange token for the instrument
            on_data_received: Callback function for data updates

        Returns:
            bool: True if subscription request sent successfully
        """
        if not self.connected:
            self._log(1, "ERROR", "Not connected to WebSocket server")
            return False

        if not self.authenticated:
            self._log(1, "ERROR", "Not authenticated with WebSocket server")
            return False

        if on_data_received:
            self.depth_callback = on_data_received

        valid: List[Dict[str, Any]] = []
        for instrument in instruments:
            exchange = instrument.get("exchange")
            symbol = instrument.get("symbol")
            exchange_token = instrument.get("exchange_token")

            if not symbol and exchange_token:
                symbol = exchange_token

            if not exchange or not symbol:
                self._log(1, "ERROR", f"Invalid instrument: {instrument}")
                continue

            entry = {"symbol": symbol, "exchange": exchange}
            valid.append(entry)
            self._active_subs[3][(exchange, symbol)] = entry
            self._log(1, "SUB", f"Subscribing {exchange}:{symbol} Depth...")

        if not valid:
            return False

        return self._send_subscribe_msg(mode=3, instruments=valid)
    
    def unsubscribe_depth(self, instruments: List[Dict[str, Any]]) -> bool:
        """
        Unsubscribe from Market Depth updates for instruments.

        Args:
            instruments: List of instrument dictionaries with keys:
                - exchange (str): Exchange code (e.g., 'NSE', 'BSE', 'NFO')
                - symbol (str): Trading symbol
                - exchange_token (str, optional): Exchange token for the instrument

        Returns:
            bool: True if unsubscription successful, False otherwise
        """
        if not self.connected or not self.authenticated:
            return False

        valid: List[Dict[str, Any]] = []
        for instrument in instruments:
            exchange = instrument.get("exchange")
            symbol = instrument.get("symbol")
            exchange_token = instrument.get("exchange_token")

            if not symbol and exchange_token:
                symbol = exchange_token

            if not exchange or not symbol:
                self._log(1, "ERROR", f"Invalid instrument: {instrument}")
                continue

            self._log(1, "UNSUB", f"Unsubscribing {exchange}:{symbol} Depth")
            valid.append({"symbol": symbol, "exchange": exchange})

            self._active_subs[3].pop((exchange, symbol), None)

            with self.lock:
                symbol_key = f"{exchange}:{symbol}"
                if symbol_key in self.depth_data:
                    del self.depth_data[symbol_key]

        if not valid:
            return False

        return self._send_unsubscribe_msg(mode=3, instruments=valid)

    def get_ltp(self, exchange: str = None, symbol: str = None) -> Dict[str, Any]:
        """
        Get the latest LTP data in nested format.
        
        Args:
            exchange (str, optional): Filter by exchange
            symbol (str, optional): Filter by symbol (requires exchange to be specified)
            
        Returns:
            dict: Dictionary with LTP data in nested format:
                {"ltp": {"EXCHANGE": {"SYMBOL": {"timestamp": timestamp, "ltp": price}}}}
        """
        with self.lock:
            # Create nested format response
            result = {"ltp": {}}
            
            # Process each item in the data structure
            for symbol_key, data in self.ltp_data.items():
                # Extract exchange and symbol from the key (format: "EXCHANGE:SYMBOL")
                if ":" in symbol_key:
                    parts = symbol_key.split(":")
                    ex = parts[0]  # Exchange
                    sym = parts[1]  # Symbol
                    
                    # Filter by exchange if specified
                    if exchange and ex != exchange:
                        continue
                        
                    # Filter by symbol if specified
                    if symbol and sym != symbol:
                        continue
                    
                    # Initialize exchange dict if not exists
                    if ex not in result["ltp"]:
                        result["ltp"][ex] = {}
                    
                    # Add data to the nested structure
                    result["ltp"][ex][sym] = {
                        "timestamp": data['timestamp'],
                        "ltp": data['price']
                    }
            
            return result
            
    def get_quotes(self, exchange: str = None, symbol: str = None) -> Dict[str, Any]:
        """
        Get the latest Quote data in nested format.
        
        Args:
            exchange (str, optional): Filter by exchange
            symbol (str, optional): Filter by symbol (requires exchange to be specified)
            
        Returns:
            dict: Dictionary with Quote data in nested format:
                {"quote": {"EXCHANGE": {"SYMBOL": {
                    "timestamp": timestamp,
                    "open": open,
                    "high": high,
                    "low": low,
                    "close": close,
                    "ltp": ltp,
                    "volume": volume,
                    "last_trade_quantity": last_trade_quantity,
                    "avg_trade_price": avg_trade_price,
                    "change": change,
                    "change_percent": change_percent
                }}}}
        """
        with self.lock:
            # Create nested format response
            result = {"quote": {}}
            
            # Process each item in the data structure
            for symbol_key, data in self.quotes_data.items():
                # Extract exchange and symbol from the key (format: "EXCHANGE:SYMBOL")
                if ":" in symbol_key:
                    parts = symbol_key.split(":")
                    ex = parts[0]  # Exchange
                    sym = parts[1]  # Symbol
                    
                    # Filter by exchange if specified
                    if exchange and ex != exchange:
                        continue
                        
                    # Filter by symbol if specified
                    if symbol and sym != symbol:
                        continue
                    
                    # Initialize exchange dict if not exists
                    if ex not in result["quote"]:
                        result["quote"][ex] = {}
                    
                    # Add data to the nested structure
                    result["quote"][ex][sym] = {
                        "timestamp": data['timestamp'],
                        "open": data['open'],
                        "high": data['high'],
                        "low": data['low'],
                        "close": data['close'],
                        "ltp": data['ltp'],
                        "volume": data.get('volume', 0),
                        "last_trade_quantity": data.get('last_trade_quantity', 0),
                        "avg_trade_price": data.get('avg_trade_price', 0),
                        "change": data.get('change', 0),
                        "change_percent": data.get('change_percent', 0)
                    }
            
            return result
            
    def get_depth(self, exchange: str = None, symbol: str = None) -> Dict[str, Any]:
        """
        Get the latest Market Depth data in nested format.
        
        Args:
            exchange (str, optional): Filter by exchange
            symbol (str, optional): Filter by symbol (requires exchange to be specified)
            
        Returns:
            dict: Dictionary with Market Depth data in nested format:
                {"depth": {"EXCHANGE": {"SYMBOL": {
                    "timestamp": timestamp,
                    "ltp": ltp,
                    "buyBook": {
                        "1": {"price": price, "qty": quantity, "orders": orders},
                        # Additional levels...
                    },
                    "sellBook": {
                        "1": {"price": price, "qty": quantity, "orders": orders},
                        # Additional levels...
                    }
                }}}}
        """
        with self.lock:
            # Create nested format response
            result = {"depth": {}}
            
            # Process each item in the data structure
            for symbol_key, data in self.depth_data.items():
                # Extract exchange and symbol from the key (format: "EXCHANGE:SYMBOL")
                if ":" in symbol_key:
                    parts = symbol_key.split(":")
                    ex = parts[0]  # Exchange
                    sym = parts[1]  # Symbol
                    
                    # Filter by exchange if specified
                    if exchange and ex != exchange:
                        continue
                        
                    # Filter by symbol if specified
                    if symbol and sym != symbol:
                        continue
                    
                    # Initialize exchange dict if not exists
                    if ex not in result["depth"]:
                        result["depth"][ex] = {}
                    
                    # Initialize the symbol structure
                    result["depth"][ex][sym] = {
                        "timestamp": data.get('timestamp', int(time.time() * 1000)),
                        "ltp": data.get('ltp', 0),
                        "buyBook": {},
                        "sellBook": {}
                    }
                    
                    # Process buy depth book
                    buy_depth = data.get('depth', {}).get('buy', [])
                    for i, level in enumerate(buy_depth):
                        level_num = str(i + 1)
                        result["depth"][ex][sym]["buyBook"][level_num] = {
                            "price": float(level.get('price') or 0),
                            "qty": int(level.get('quantity') or 0),
                            "orders": int(level.get('orders') or 0)
                        }

                    # If there are fewer than 5 levels, add empty levels to complete the structure
                    for i in range(len(buy_depth), 5):
                        level_num = str(i + 1)
                        result["depth"][ex][sym]["buyBook"][level_num] = {
                            "price": 0.0,
                            "qty": 0,
                            "orders": 0
                        }

                    # Process sell depth book
                    sell_depth = data.get('depth', {}).get('sell', [])
                    for i, level in enumerate(sell_depth):
                        level_num = str(i + 1)
                        result["depth"][ex][sym]["sellBook"][level_num] = {
                            "price": float(level.get('price') or 0),
                            "qty": int(level.get('quantity') or 0),
                            "orders": int(level.get('orders') or 0)
                        }

                    # If there are fewer than 5 levels, add empty levels to complete the structure
                    for i in range(len(sell_depth), 5):
                        level_num = str(i + 1)
                        result["depth"][ex][sym]["sellBook"][level_num] = {
                            "price": 0.0,
                            "qty": 0,
                            "orders": 0
                        }
            
            return result
