# -*- coding: utf-8 -*-
"""
OpenAlgo WebSocket API Documentation - Feed Methods
    https://docs.openalgo.in
"""

import json
import threading
import time
from typing import List, Dict, Any, Callable, Iterator, Optional, Tuple
import websocket
from .base import BaseAPI
from . import _feed_ticks as ticks
from ._feed_errors import ErrorReporter, FeedServerError, server_rejection
from ._feed_hooks import HookRegistry
from ._feed_replay import PendingReplay, ReplayTracker
from ._feed_state import MODES, Link, Phase, Timing
from ._feed_url import DEFAULT_WS_PORT, derive_ws_url

class FeedAPI(BaseAPI):
    """
    Market data feed API methods for OpenAlgo using WebSockets.
    Inherits from the BaseAPI class.
    """

    def __init__(self, api_key, host="http://127.0.0.1:5000", version="v1", ws_port=None, ws_url=None, verbose=False, auto_reconnect=True):
        """
        Initialize the FeedAPI object with API key and optionally a host URL, API version, and WebSocket details.

        Attributes:
        - api_key (str): User's API key.
        - host (str): Base URL for the API endpoints. Defaults to localhost.
        - version (str): API version. Defaults to "v1".
        - ws_port (int, optional): WebSocket server port. Defaults to 8765, which is
            also what self.ws_port reads when it is not passed.
        - ws_url (str, optional): Custom WebSocket URL. If provided, this overrides host and ws_port settings.
            Without it the WebSocket URL is derived from host:
            - http://<host>, or a host without a scheme (local or LAN):
              ws://<host>:<ws_port>
            - https://<host> (a server hosted on a domain): wss://<host>/ws,
              where the OpenAlgo installers serve the feed
            - https://<host> with ws_port passed: ws://<host>:<ws_port>
            The port and the path of host belong to the REST API and are not used.
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
        self._init_feed(host, ws_port, ws_url, verbose, auto_reconnect)

    def _init_feed(self, host, ws_port, ws_url, verbose, auto_reconnect) -> None:
        """
        Create the WebSocket state of the client. FeedAPI.__init__ and
        openalgo.api.__init__ (which sets up BaseAPI itself) both call it, so
        the two cannot drift apart.
        """
        # Verbosity control
        self.verbose = int(verbose) if verbose is not False else 0
        
        # WebSocket configuration
        self.ws_port = DEFAULT_WS_PORT if ws_port is None else ws_port
        
        # Use custom WebSocket URL if provided
        if ws_url:
            self.ws_url = ws_url
        else:
            self.ws_host, self.ws_url = derive_ws_url(host, ws_port)
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
        self.order_callback = None

        # Order-update storage: most-recent-first list, capped.
        self.order_data = []
        self._orders_subscribed = False

        # Auto-reconnect state (private).
        self.auto_reconnect = bool(auto_reconnect)
        self._shutting_down = False
        self._reconnect_thread = None
        self._reconnect_lock = threading.Lock()
        # Active subscription registry keyed by mode -> {(exchange, symbol): instrument_dict}
        # Replayed verbatim on reconnect.
        self._active_subs: Dict[int, Dict[Tuple[str, str], Dict[str, Any]]] = {1: {}, 2: {}, 3: {}}

        # Connection state (private). _state_lock guards _link, ws, ws_thread
        # and the requested subscriptions, and is held while their frames are
        # sent, so a request cannot overtake a replay. It is never held while
        # user code runs. connected, authenticated and _shutting_down are
        # derived from _link (see _set_link).
        self._state_lock = threading.RLock()
        self._link = Link()
        self._transport_error: Optional[BaseException] = None
        self._replay: Optional[PendingReplay] = None
        # Waiting times. This is the one seam for tests: they replace it with
        # a Timing that holds shorter values.
        self._timing = Timing()
        hook_events = ("connect", "disconnect", "reconnect", "error")
        self._hooks = HookRegistry(hook_events, self._hook_failed)
        self._errors = ErrorReporter(self._hooks, lambda: self._timing.error_log)

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

    def on_connect(self, callback: Callable[[Dict[str, Any]], None]) -> Callable:
        """
        Register a hook for "the feed is usable": it fires once for every
        connection, the automatic reconnects included, when the server has
        accepted the API key and the requested subscriptions have been sent.

        Receives one dict: {"event": "connect", "seq": <connection number>,
        "url": <WebSocket URL>, "reconnect": <True when this connection
        restores a session that was lost>, "broker": ..., "user_id": ...}.

        Thread and order, the same for every hook: hooks are delivered one at
        a time, in the order things happened. A hook normally runs on the
        WebSocket thread of its connection (self.ws_thread), or on the thread
        that called disconnect(). While one hook runs, whatever happens next
        is queued and delivered afterwards by that same thread. No SDK lock is
        held, so a hook may call any feed method. It must return quickly: the
        WebSocket thread processes no ticks while it runs.

        A connect or reconnect listener is not called for a connection that
        has already gone again when its turn comes. "seq" grows with every
        connection attempt, so a listener can also drop an event that is older
        than one it has seen.

        Returns the callback, so this can be used as a decorator.
        """
        return self._hooks.add("connect", callback)

    def on_disconnect(self, callback: Callable[[Dict[str, Any]], None]) -> Callable:
        """
        Register a hook for "the feed is down": it fires once for every
        connection that had been accepted by the server, whether the socket
        dropped or the application called disconnect(). An attempt that never
        got that far (a failed reconnect attempt, for example) fires nothing.

        Receives one dict: {"event": "disconnect", "seq": <connection number>,
        "url": ..., "code": <WebSocket close code, or None when there was no
        close frame>, "reason": <close reason, else the transport error text>,
        "error": <the transport exception when the socket was lost, None for a
        clean close>, "requested": <True when disconnect() caused it>,
        "will_reconnect": <True when the SDK reconnects by itself>}.

        Thread and order: see on_connect. self.connected is already False.
        Returns the callback, so this can be used as a decorator.
        """
        return self._hooks.add("disconnect", callback)

    def on_reconnect(self, callback: Callable[[Dict[str, Any]], None]) -> Callable:
        """
        Register a hook for "the session is restored": after a connection was
        lost, a new one is up and the server has answered the replay of the
        requested subscriptions (or 10 seconds have passed without an answer).
        It follows the on_connect hooks of that connection. Use it to
        resynchronise state: the server does not replay the ticks and order
        updates that were missed. The first connection of a client is never a
        reconnect.

        Receives the on_connect dict with "event": "reconnect", plus
        "subscriptions": what the server accepted, in the format of
        get_subscriptions(). What it refused or did not answer goes to the
        on_error hooks.

        Thread and order: see on_connect. When the server did not answer in
        time, the hook runs on a timer thread of the SDK instead.
        Returns the callback, so this can be used as a decorator.
        """
        return self._hooks.add("reconnect", callback)

    def on_error(self, callback: Callable[[BaseException, Dict[str, Any]], None]) -> Callable:
        """
        Register a hook that receives every error the feed catches. The feed
        stays up and keeps processing messages in every case.

        Called as callback(error, context). context["kind"] says what failed:
        - "callback": a data callback raised. context["callback"] is "ltp",
          "quote", "depth" or "order"; context["message"] is the server
          message that was being delivered.
        - "hook": a connection hook raised. context["callback"] is its
          registration method ("on_connect", ...); context["message"] is the
          dict it received.
        - "frame": the SDK could not process a frame; context["message"] is
          the raw text.
        - "authentication": the server refused the API key.
        - "server": the server rejected something else. For these two, error
          is a FeedServerError (with .code and .message), context["code"] is
          the server's code and context["message"] its frame. A subscribe or
          unsubscribe reply adds context["rejected"], the entries that were
          not accepted; a replay the server did not answer adds
          context["unanswered"].
        Every context carries "seq", the connection number.

        Without an error hook the same errors are logged at ERROR level on
        the "openalgo.feed" logger. Only there, a failing tick callback is
        logged once per 10 seconds and exception type, with a count after it.

        Thread and order: see on_connect.
        Returns the callback, so this can be used as a decorator.
        """
        return self._hooks.add("error", callback)

    def remove_hook(self, callback: Callable) -> bool:
        """
        Remove a callback registered with on_connect, on_disconnect,
        on_reconnect or on_error. Safe to call from any thread.

        Returns:
            bool: True if the callback was registered, False otherwise.
        """
        return self._hooks.remove(callback)

    def _report(self, error: BaseException, line: str, **context: Any) -> None:
        """Print the verbose line of an error, then hand it to the error hooks or the log."""
        self._log(1, "ERROR", line)
        self._errors.report(error, seq=self._link.seq, **context)

    def _hook_failed(self, event: str, error: BaseException, args: Tuple[Any, ...]) -> None:
        """A hook raised. The hook registry calls this on the delivering thread."""
        context = {"kind": "hook", "callback": f"on_{event}", "message": args[-1]}
        if event == "error":
            # An error hook that fails is not handed to the error hooks again.
            self._errors.log(error, context)
        else:
            self._report(error, f"on_{event} hook error: {error}", **context)

    def _set_link(self, phase: Phase, **changes: Any) -> None:
        """
        Change the connection state: the phase, and seq or restoring when
        given. Every change goes through here, with _state_lock held, and the
        flags that existing code reads follow from it.
        """
        self._link = self._link._replace(phase=phase, **changes)
        self.connected = self._link.connected
        self.authenticated = self._link.live
        self._shutting_down = phase is Phase.STOPPED

    def connect(self) -> bool:
        """
        Connect to the WebSocket server and authenticate.

        When the feed is already connected and authenticated this returns
        True without opening a second socket.

        Returns:
            bool: True if connection and authentication are successful, False otherwise.
        """
        # User-initiated connect resets the shutdown flag so auto-reconnect
        # is armed. (disconnect() sets _shutting_down=True to suppress it.)
        with self._state_lock:
            if self._link.phase is Phase.STOPPED:
                self._set_link(Phase.IDLE)
        return self._do_connect()

    def _do_connect(self) -> bool:
        """
        Internal: open one WebSocket connection, start the read thread, and
        wait for authentication. Used by both connect() and the reconnect
        loop. Public surface (self.ws, self.ws_thread, self.connected,
        self.authenticated) is unchanged.
        """
        try:
            with self._state_lock:
                phase = self._link.phase
                if phase is Phase.LIVE:
                    return True
                if phase is Phase.STOPPED:
                    # disconnect() ran after this attempt was decided: stay down.
                    return False
                # An attempt that is still under way is replaced.
                replaced = None if phase is Phase.IDLE else self.ws

                # Initialize WebSocket connection
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_open=self._on_ws_open,
                    on_close=self._on_ws_close
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
                thread = self.ws_thread
                self._transport_error = None
                self._set_link(Phase.CONNECTING, seq=self._link.seq + 1)

            thread.start()
            if replaced is not None:
                self._close_unused(replaced)

            # Wait for connection to establish
            timeout = self._timing.connect
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                if self._shutting_down:
                    break  # disconnect() was called: there is nothing left to wait for
                time.sleep(self._timing.poll)

            if not self.connected:
                self._log(
                    1, "ERROR",
                    f"Failed to connect to WebSocket server at {self.ws_url}. "
                    "If the feed is served at another URL, pass ws_url= to the client.",
                )
                return False

            # Wait for authentication to complete
            timeout = self._timing.authenticate
            start_time = time.time()
            while not self.authenticated and time.time() - start_time < timeout and self.connected:
                time.sleep(self._timing.poll)

            return self.authenticated

        except Exception as e:
            self._log(1, "ERROR", f"Error connecting to WebSocket: {e}")
            return False

    def _close_unused(self, ws: Any) -> None:
        """Close a socket that was replaced or abandoned. Its later events are ignored."""
        try:
            ws.close()
        except Exception as e:
            self._log(2, "WS", f"Ignored an error while closing an unused socket: {e}")

    def _on_ws_open(self, ws: Any) -> None:
        """websocket-client on_open. Runs on the WebSocket thread of ws."""
        with self._state_lock:
            current = ws is self.ws and self._link.phase is Phase.CONNECTING
            if current:
                self._set_link(Phase.OPEN)
        if not current:
            # Opened after it was replaced, or after disconnect(): drop it.
            self._close_unused(ws)
            return
        self._log(1, "WS", f"Connected to {self.ws_url}")
        self._authenticate()

    def _on_ws_message(self, ws: Any, message: str) -> None:
        """websocket-client on_message. Runs on the WebSocket thread of ws."""
        if ws is self.ws:
            self._process_message(message)

    def _on_ws_error(self, ws: Any, error: Any) -> None:
        """websocket-client on_error. Runs on the WebSocket thread of ws."""
        if ws is self.ws:
            self._transport_error = error
        self._log(1, "ERROR", f"WebSocket error: {error}")

    def _on_ws_close(self, ws: Any, close_status_code: Any, close_reason: Any) -> None:
        """websocket-client on_close. Runs on the WebSocket thread of ws."""
        self._log(1, "WS", f"Disconnected from {self.ws_url}")
        with self._state_lock:
            was = self._link
            if ws is not self.ws or was.phase in (Phase.IDLE, Phase.STOPPED):
                # Replaced by a newer attempt, or disconnect() already reported it.
                return
            # Schedule auto-reconnect unless auto_reconnect is disabled.
            # (After disconnect() the socket is no longer the current one.)
            retry = bool(self.auto_reconnect)
            was_live = was.phase is Phase.LIVE
            self._set_link(Phase.IDLE, restoring=retry and (was_live or was.restoring))
            self._replay = None
            if not retry:
                # Nothing restores this session. What was requested in it is
                # not sent again by a later connect(), as in 2.0.5.
                self._clear_requests()
            if was_live:
                # websocket-client reports a close frame through on_error as
                # well: with a close code there was no transport failure.
                error = self._transport_error if close_status_code is None else None
                reason = close_reason or (str(error) if error is not None else None)
                self._post_disconnect(was.seq, close_status_code, reason, error, retry)
        if retry:
            self._schedule_reconnect()
        self._hooks.drain()

    def _post_disconnect(
        self, seq: int, code: Any, reason: Any, error: Any, will_reconnect: bool
    ) -> None:
        """Queue the disconnect hook. The caller holds _state_lock and drains afterwards."""
        self._hooks.post("disconnect", {
            "event": "disconnect",
            "seq": seq,
            "url": self.ws_url,
            "code": code,
            "reason": reason,
            "error": error,
            # Only disconnect() leaves the feed stopped, and it never reconnects.
            "requested": self._link.phase is Phase.STOPPED,
            "will_reconnect": will_reconnect,
        })

    def _on_authenticated(self, message: Dict[str, Any]) -> None:
        """
        The server accepted the API key. Runs on the WebSocket thread: mark
        the connection live, send the requested subscriptions and queue the
        connect hook, and the reconnect hook when a session is being restored.
        """
        with self._state_lock:
            on_current_socket = threading.current_thread() is self.ws_thread
            if not on_current_socket or self._link.phase is not Phase.OPEN:
                return
            restoring = self._link.restoring
            self._set_link(Phase.LIVE, restoring=False)
            live = self._link

            def still_live() -> bool:
                return self._link is live

            # The requested subscriptions are sent whenever a connection
            # authenticates. This happens under the lock that every request
            # takes, so a request can neither be lost nor overtaken.
            requested = self.get_subscriptions()
            self._replay_subscriptions()
            payload = {
                "event": "connect",
                "seq": live.seq,
                "url": self.ws_url,
                "reconnect": restoring,
                "broker": message.get("broker"),
                "user_id": message.get("user_id"),
            }
            self._hooks.post("connect", payload, still_current=still_live)
            if restoring:
                tracker = ReplayTracker(requested)
                replay = PendingReplay(tracker, dict(payload, event="reconnect"), still_live)
                self._replay = replay
                if not tracker.done:
                    # Do not wait for ever for a server that does not answer.
                    deadline = threading.Timer(self._timing.replay_ack, self._settle_replay, (replay,))
                    deadline.daemon = True
                    deadline.start()
        self._settle_replay()

    def _settle_replay(self, expired: Optional[PendingReplay] = None) -> None:
        """
        Queue the reconnect hook once the server has answered the whole
        replay, or when the wait for its answers is over (expired is then the
        replay that was waited for), and deliver the queued hooks.
        """
        unanswered = None
        with self._state_lock:
            replay = self._replay
            if replay is not None and (replay.tracker.done or replay is expired):
                self._replay = None
                if not replay.tracker.done:
                    unanswered = replay.tracker.unanswered()
                self._hooks.post(
                    "reconnect",
                    dict(replay.payload, subscriptions=replay.tracker.accepted),
                    still_current=replay.still_live,
                )
        if unanswered is not None:
            error = FeedServerError("REPLAY_NOT_ACKNOWLEDGED", "the server did not answer")
            self._report(
                error, f"Server error: {error}",
                kind="server", callback=None, code=error.code, message=None, unanswered=unanswered,
            )
        self._hooks.drain()

    def disconnect(self) -> None:
        """
        Disconnect from the WebSocket server.

        Stops the automatic reconnect and forgets the requested subscriptions:
        after a later connect() nothing is subscribed until the application
        subscribes again.
        """
        with self._state_lock:
            ws, thread = self.ws, self.ws_thread
            was = self._link
            # Mark a graceful shutdown so on_close does not trigger auto-reconnect.
            self._set_link(Phase.STOPPED, restoring=False)
            self.ws = None
            self._replay = None
            self._clear_requests()
            if was.phase is Phase.LIVE:
                self._post_disconnect(was.seq, None, "disconnect() called", None, False)
        self._hooks.drain()
        if ws:
            ws.close()
            # Wait for websocket to close, if it was open (as before, there is
            # no wait for an attempt that never opened). Called on the socket's
            # own thread, from a hook or a data callback, there is nothing to
            # wait for.
            was_open = was.phase in (Phase.OPEN, Phase.LIVE)
            if was_open and thread is not threading.current_thread():
                thread.join(self._timing.disconnect)

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
        Reconnect with exponential backoff. Stops when the user calls
        disconnect() or a connection is fully re-authenticated; the requested
        subscriptions are replayed as soon as it is (see _on_authenticated).
        """
        # Exponential backoff schedule (seconds), capped at 60s.
        backoffs = self._timing.backoff
        attempt = 0
        while not self._reconnect_finished():
            delay = backoffs[min(attempt, len(backoffs) - 1)]
            self._log(1, "WS", f"Reconnect attempt {attempt + 1} in {delay}s...")
            # Sleep in small slices so a disconnect() is responsive.
            slept = 0.0
            while slept < delay and not self._shutting_down:
                time.sleep(self._timing.backoff_slice)
                slept += self._timing.backoff_slice
            try:
                # Refused, without opening a socket, if disconnect() ran meanwhile.
                ok = self._do_connect()
                if ok:
                    self._log(1, "WS", "Reconnected and re-authenticated.")
            except Exception as e:
                self._log(1, "ERROR", f"Reconnect attempt {attempt + 1} failed: {e}")
            attempt += 1

    def _reconnect_finished(self) -> bool:
        """
        True when the reconnect loop has nothing left to do: the feed is live
        again, or it is no longer wanted. The reconnect thread then exits.

        The decision is taken, and the thread slot released, under the lock
        _schedule_reconnect takes. A drop that lands while the loop finishes
        therefore either makes it go round again or starts a new thread.
        """
        with self._reconnect_lock:
            finished = self._link.phase in (Phase.LIVE, Phase.STOPPED) or not self.auto_reconnect
            if finished and self._reconnect_thread is threading.current_thread():
                self._reconnect_thread = None
            return finished

    def _clear_requests(self) -> None:
        """Forget the requested subscriptions. The caller holds _state_lock."""
        for store in self._active_subs.values():
            store.clear()
        self._orders_subscribed = False

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
            mode_name = MODES[mode].label
            self._log(1, "SUB", f"Replaying {len(instruments)} {mode_name} subscription(s) after reconnect")
            self._send_subscribe_msg(mode, instruments)

        # Replay the account-level order-update subscription too.
        if self._orders_subscribed and self.ws:
            self._log(1, "SUB", "Replaying order-update subscription after reconnect")
            try:
                self.ws.send(json.dumps({"action": "subscribe_orders"}))
            except Exception as e:
                self._log(1, "ERROR", f"Error replaying order subscription: {e}")

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

        # No part of the key is printed: verbose output ends up in logs and screenshots.
        self._log(1, "AUTH", "Authenticating with the API key...")
        self.ws.send(json.dumps(auth_msg))

    def _process_message(self, message_str: str) -> None:
        """
        Process incoming WebSocket messages.

        Args:
            message_str (str): The message string received from the WebSocket.
        """
        try:
            message = json.loads(message_str)
            self._surface_rejection(message)
            kind = message.get("type")
            if kind == "auth":
                self._on_auth_reply(message)
            elif kind in ("subscribe_orders", "unsubscribe_orders"):
                self._on_orders_reply(message)
            elif kind == "order_update":
                self._on_order_update(message)
            elif kind == "subscribe":
                self._on_subscribe_reply(message)
            elif kind == "market_data":
                self._on_market_data(message)
        except json.JSONDecodeError as e:
            line = f"Invalid JSON message: {message_str[:100]}..."
            self._report(e, line, kind="frame", callback=None, message=message_str)
        except Exception as e:
            line = f"Error handling message: {e}"
            self._report(e, line, kind="frame", callback=None, message=message_str)

    def _surface_rejection(self, message: Any) -> None:
        """Report a frame in which the server rejects something. Nothing else is done about it."""
        rejection = server_rejection(message)
        if rejection is None:
            return
        kind, error, details = rejection
        if kind == "authentication":
            line = f"Authentication failed: {error.message}"
        else:
            line = f"Server error: {error}"
        self._report(
            error, line, kind=kind, callback=None, code=error.code, message=message, **details
        )

    def _deliver(
        self, label: str, callback: Callable, payload: Dict[str, Any], message: Dict[str, Any]
    ) -> None:
        """
        Hand payload to a user data callback. An exception it raises is
        reported with the server message it came from, and never stops the feed.
        """
        try:
            callback(payload)
        except Exception as e:
            line = f"{label} callback error: {str(e)}"
            self._report(e, line, kind="callback", callback=label.lower(), message=message)

    def _on_auth_reply(self, message: Dict[str, Any]) -> None:
        """Handle authentication response (a refusal was reported by _surface_rejection)."""
        if message.get("status") == "success":
            broker = message.get("broker", "unknown")
            user_id = message.get("user_id", "unknown")
            self._log(1, "AUTH", f"Success | Broker: {broker} | User: {user_id}")
            self._log(2, "AUTH", f"Full response: {message}")
            self._on_authenticated(message)

    def _on_orders_reply(self, message: Dict[str, Any]) -> None:
        """Handle order-update subscription responses."""
        self._log(1, "SUB", f"Order updates | {message.get('type')} | Status: {message.get('status', '?')}")
        self._log(2, "SUB", f"Full response: {message}")
        if message.get("type") == "subscribe_orders":
            self._note_replay_reply(ReplayTracker.orders_reply, message)

    def _on_subscribe_reply(self, message: Dict[str, Any]) -> None:
        """Handle subscription response."""
        subs = message.get("subscriptions", [])
        for sub in subs:
            sym = sub.get("symbol", "?")
            exch = sub.get("exchange", "?")
            status = sub.get("status", "?")
            mode = sub.get("mode", 0)
            # The server names the mode ("LTP"); older ones number it.
            mode_name = MODES[mode].label if mode in MODES else mode or "Unknown"
            self._log(1, "SUB", f"{exch}:{sym} | Mode: {mode_name} | Status: {status}")
        self._log(2, "SUB", f"Full response: {message}")
        self._note_replay_reply(ReplayTracker.subscribe_reply, message)

    def _note_replay_reply(
        self, take_in: Callable[[ReplayTracker, Dict[str, Any]], None], message: Dict[str, Any]
    ) -> None:
        """Let the replay that waits for replies, if there is one, take this reply in."""
        with self._state_lock:
            if self._replay is not None:
                take_in(self._replay.tracker, message)
        self._settle_replay()

    def _on_order_update(self, message: Dict[str, Any]) -> None:
        """Handle real-time order updates (account-level, no symbol subscription)."""
        with self.lock:
            self.order_data.insert(0, message)
            del self.order_data[500:]  # cap retained history

        self._log(
            2,
            "ORDER",
            f"{message.get('symbol', '?'):<20} | {message.get('order_status', '?'):<10} "
            f"| Qty: {message.get('filled_quantity', 0)}/{message.get('quantity', 0)} "
            f"| Avg: {message.get('average_price', 0)} | ID: {message.get('orderid', '?')}",
        )

        if self.order_callback:
            self._deliver("Order", self.order_callback, message, message)

    def _on_market_data(self, message: Dict[str, Any]) -> None:
        """Handle market data: store the latest values, then invoke the callback of the mode."""
        if not message.get("exchange") or not message.get("symbol"):
            return
        mode = message.get("mode")
        market_data = message.get("data", {})
        # Store with format 'EXCHANGE:SYMBOL'
        symbol_key = f"{message['exchange']}:{message['symbol']}"
        # Handle LTP data (mode 1)
        if mode == 1 and "ltp" in market_data:
            update = ticks.ltp_update(market_data)
            with self.lock:
                self.ltp_data[symbol_key] = {
                    'price': update["ltp"],
                    'timestamp': update["timestamp"]
                }
                self._log(2, "LTP", f"{symbol_key:<20} | LTP: {update['ltp']}")
            callback = self.ltp_callback
        # Handle Quotes data (mode 2)
        elif mode == 2:
            update = ticks.quote_update(market_data)
            with self.lock:
                self.quotes_data[symbol_key] = update
                self._log(2, "QUOTE", f"{symbol_key:<20} | O: {update['open']:<10} H: {update['high']:<10} L: {update['low']:<10} C: {update['close']:<10} LTP: {update['ltp']}")
            callback = self.quote_callback
        # Handle Market Depth data (mode 3)
        elif mode == 3 and "depth" in market_data:
            update = ticks.depth_update(market_data)
            with self.lock:
                self.depth_data[symbol_key] = update
                self._log(2, "DEPTH", f"{symbol_key:<20} | LTP: {update.get('ltp')}")
                if self.verbose >= 2:
                    print("\n".join(ticks.depth_ladder(update)))
            callback = self.depth_callback
        else:
            return

        # Invoke callback if set, with a clean copy of what was stored
        if callback:
            clean_data = ticks.callback_message(message, update.copy())
            self._deliver(MODES[mode].label, callback, clean_data, message)

    def _valid_instruments(self, instruments: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """
        Yield a {"symbol", "exchange"} entry for each instrument that has
        both. The others are logged and skipped.
        """
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

            yield {"symbol": symbol, "exchange": exchange}

    def _log_not_ready(self) -> None:
        """Print why a subscribe frame cannot be sent now: no socket, or not authenticated."""
        if not self.connected:
            self._log(1, "ERROR", "Not connected to WebSocket server")
        else:
            self._log(1, "ERROR", "Not authenticated with WebSocket server")

    def _subscribe(self, mode: int, instruments: List[Dict[str, Any]], on_data_received: Optional[Callable]) -> bool:
        """
        Shared implementation of subscribe_ltp / subscribe_quote /
        subscribe_depth (mode 1 / 2 / 3).

        The request is recorded first, then the frame is sent if the feed is
        live. Both happen under _state_lock, which the replay also takes, so
        a request can neither be lost nor be overtaken by a replay.
        """
        names = MODES[mode]
        with self._state_lock:
            live = self._link.phase is Phase.LIVE
            if not live:
                self._log_not_ready()

            # Set callback if provided
            if on_data_received:
                setattr(self, names.callback_attr, on_data_received)

            # Register in the active-subscription store (used for replay
            # across reconnects).
            valid: List[Dict[str, Any]] = []
            for entry in self._valid_instruments(instruments):
                exchange, symbol = entry["exchange"], entry["symbol"]
                valid.append(entry)
                self._active_subs[mode][(exchange, symbol)] = entry
                if live:
                    self._log(1, "SUB", f"Subscribing {exchange}:{symbol} {names.label}...")

            if not live or not valid:
                return False

            # Single bulk message for all instruments. The unified WebSocket
            # proxy (server.py) accepts `data["symbols"]` as an array; using it
            # makes a 200-symbol option-chain subscribe near-instant instead of
            # ~20 seconds.
            return self._send_subscribe_msg(mode=mode, instruments=valid)

    def _unsubscribe(self, mode: int, instruments: List[Dict[str, Any]]) -> bool:
        """
        Shared implementation of unsubscribe_ltp / unsubscribe_quote /
        unsubscribe_depth (mode 1 / 2 / 3). See _subscribe for the locking.
        """
        names = MODES[mode]
        with self._state_lock:
            valid: List[Dict[str, Any]] = []
            for entry in self._valid_instruments(instruments):
                exchange, symbol = entry["exchange"], entry["symbol"]
                self._log(1, "UNSUB", f"Unsubscribing {exchange}:{symbol} {names.label}")
                valid.append(entry)

                # Drop from active-subscription store so reconnect does not replay.
                self._active_subs[mode].pop((exchange, symbol), None)

                # Clean up cached data for this symbol.
                with self.lock:
                    getattr(self, names.cache_attr).pop(f"{exchange}:{symbol}", None)

            if not valid or self._link.phase is not Phase.LIVE:
                return False

            return self._send_unsubscribe_msg(mode=mode, instruments=valid)

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
            bool: True if the subscription frame was sent, False otherwise
                (for example while the socket is down). The requested set is
                updated either way: it is what the SDK sends whenever the
                connection authenticates, so also after a reconnect.
        """
        return self._subscribe(1, instruments, on_data_received)

    def unsubscribe_ltp(self, instruments: List[Dict[str, Any]]) -> bool:
        """
        Unsubscribe from LTP updates for instruments.

        Args:
            instruments: List of instrument dictionaries with keys:
                - exchange (str): Exchange code (e.g., 'NSE', 'BSE', 'NFO')
                - symbol (str): Trading symbol
                - exchange_token (str, optional): Exchange token for the instrument

        Returns:
            bool: True if the unsubscription frame was sent, False otherwise
                (for example while the socket is down). The requested set is
                updated either way, so the instruments are not replayed.
        """
        return self._unsubscribe(1, instruments)

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
            bool: True if the subscription frame was sent, False otherwise
                (for example while the socket is down). The requested set is
                updated either way: it is what the SDK sends whenever the
                connection authenticates, so also after a reconnect.
        """
        return self._subscribe(2, instruments, on_data_received)

    def unsubscribe_quote(self, instruments: List[Dict[str, Any]]) -> bool:
        """
        Unsubscribe from Quote updates for instruments.

        Args:
            instruments: List of instrument dictionaries with keys:
                - exchange (str): Exchange code (e.g., 'NSE', 'BSE', 'NFO')
                - symbol (str): Trading symbol
                - exchange_token (str, optional): Exchange token for the instrument

        Returns:
            bool: True if the unsubscription frame was sent, False otherwise
                (for example while the socket is down). The requested set is
                updated either way, so the instruments are not replayed.
        """
        return self._unsubscribe(2, instruments)

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
            bool: True if the subscription frame was sent, False otherwise
                (for example while the socket is down). The requested set is
                updated either way: it is what the SDK sends whenever the
                connection authenticates, so also after a reconnect.
        """
        return self._subscribe(3, instruments, on_data_received)

    def unsubscribe_depth(self, instruments: List[Dict[str, Any]]) -> bool:
        """
        Unsubscribe from Market Depth updates for instruments.

        Args:
            instruments: List of instrument dictionaries with keys:
                - exchange (str): Exchange code (e.g., 'NSE', 'BSE', 'NFO')
                - symbol (str): Trading symbol
                - exchange_token (str, optional): Exchange token for the instrument

        Returns:
            bool: True if the unsubscription frame was sent, False otherwise
                (for example while the socket is down). The requested set is
                updated either way, so the instruments are not replayed.
        """
        return self._unsubscribe(3, instruments)

    def subscribe_orders(self, on_order_update: Optional[Callable] = None) -> bool:
        """
        Subscribe to real-time order updates for the authenticated account.

        Order updates cover asynchronous status changes pushed by the broker
        (or the sandbox engine in analyze mode) after an order is placed:
        fills, partial fills, rejections, cancellations. No symbols are
        involved; this is an account-level stream.

        Args:
            on_order_update: Callback invoked with each order_update message:
                {"type": "order_update", "orderid": ..., "symbol": ...,
                 "exchange": ..., "action": ..., "quantity": ...,
                 "pricetype": ..., "product": ..., "order_status": ...,
                 "filled_quantity": ..., "pending_quantity": ...,
                 "average_price": ..., "rejection_reason": ..., "broker": ...}

        Returns:
            bool: True if the subscription frame was sent, False otherwise
                (for example while the socket is down). The request is
                recorded either way and sent whenever the connection
                authenticates, exactly like a market-data subscription.
        """
        with self._state_lock:
            live = self._link.phase is Phase.LIVE
            if not live:
                self._log_not_ready()

            if on_order_update:
                self.order_callback = on_order_update

            self._orders_subscribed = True
            if not live:
                return False

            self._log(1, "SUB", "Subscribing to order updates...")
            try:
                self.ws.send(json.dumps({"action": "subscribe_orders"}))
                return True
            except Exception as e:
                self._log(1, "ERROR", f"Error sending order subscription: {e}")
                return False

    def unsubscribe_orders(self) -> bool:
        """
        Unsubscribe from real-time order updates.

        Returns:
            bool: True if the unsubscription frame was sent, False otherwise
                (for example while the socket is down). The request is
                recorded either way, so the stream is not replayed.
        """
        with self._state_lock:
            self._orders_subscribed = False
            if self._link.phase is not Phase.LIVE:
                return False

            self._log(1, "UNSUB", "Unsubscribing from order updates")
            try:
                self.ws.send(json.dumps({"action": "unsubscribe_orders"}))
                return True
            except Exception as e:
                self._log(1, "ERROR", f"Error sending order unsubscription: {e}")
                return False

    def get_subscriptions(self) -> Dict[str, Any]:
        """
        Get the requested subscriptions: what subscribe and unsubscribe calls
        have asked for so far, whether or not their frames could be sent. It
        is what the SDK sends whenever the connection authenticates, and
        disconnect() empties it. Safe to call from any thread.

        Returns:
            dict: {"ltp": [{"symbol": ..., "exchange": ...}, ...],
                   "quote": [...], "depth": [...], "orders": <bool>}
        """
        with self._state_lock:
            result: Dict[str, Any] = {
                MODES[mode].key: [dict(entry) for entry in store.values()]
                for mode, store in self._active_subs.items()
            }
            result["orders"] = bool(self._orders_subscribed)
            return result

    def get_orders(self) -> Dict[str, Any]:
        """
        Get the order updates received so far (most recent first, capped).

        Returns:
            dict: {"orders": [ {...order_update message...}, ... ]}
        """
        with self.lock:
            return {"orders": list(self.order_data)}

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
                    "change_percent": change_percent,
                    "ltt": last_trade_time,
                    "total_buy_quantity": total_buy_quantity,
                    "total_sell_quantity": total_sell_quantity,
                    "oi": open_interest
                }}}}
                The server sends the average traded price as average_price and
                the last traded quantity as last_quantity; they are read into
                avg_trade_price and last_trade_quantity. A field the server did
                not send is 0. A quote callback receives the same fields.
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
                        "change_percent": data.get('change_percent', 0),
                        "ltt": data.get('ltt', 0),
                        "total_buy_quantity": data.get('total_buy_quantity', 0),
                        "total_sell_quantity": data.get('total_sell_quantity', 0),
                        "oi": data.get('oi', 0)
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
