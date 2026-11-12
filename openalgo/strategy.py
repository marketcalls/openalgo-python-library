# -*- coding: utf-8 -*-
"""
OpenAlgo Strategy Webhook Module
    https://docs.openalgo.in

Posts to the public webhook at ``/strategy/webhook/<token>``, the surface
charting-platform alerts and any other alert sender reach. It is not under ``/api/v1`` and
takes no API key: the token in the URL is the whole of the credential, so treat
it the way you would treat a password.

The token is shown exactly once, in the browser, when the strategy is created or
when the token is rotated. No endpoint returns it - only its SHA-256 digest is
stored - so if it is lost, rotate it on the strategy page and copy the new one.

Two vocabularies, and each strategy kind refuses the other's:

- batch: ``start`` (which requires ``mode``) and ``stop``. A multi-leg spread
  entered and exited as a unit.
- signal: ``long_entry``, ``long_exit``, ``short_entry`` and ``short_exit``. One
  alert moves one leg. There is no start and no mode: the first signal after the
  platform session boundary opens the run, and the mode comes from the
  strategy's own live opt-in.
"""

from typing import Any, Dict, Optional

import httpx

#: Actions a batch strategy accepts.
BATCH_ACTIONS = ("start", "stop")

#: Actions a signal strategy accepts.
SIGNAL_ACTIONS = ("long_entry", "long_exit", "short_entry", "short_exit")

#: Run modes for a batch start. Exact and case-sensitive on the server.
RUN_MODES = ("live", "sandbox")


class Strategy:
    """Send signals to an OpenAlgo strategy through its public webhook."""

    def __init__(self, host_url: str, webhook_id: Optional[str] = None, *,
                 webhook_token: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize the webhook sender.

        Args:
            host_url (str): OpenAlgo server URL, e.g. "http://127.0.0.1:5000".
            webhook_id (str, optional): The strategy's webhook token, the
                ``oaws_...`` value shown once in the browser. Named ``webhook_id``
                for backwards compatibility with earlier releases.
            webhook_token (str, optional): The same value under its accurate
                name. Wins over ``webhook_id`` when both are given.
            timeout (float): Request timeout in seconds. Defaults to 30.0.

        Raises:
            ValueError: If no token is given under either name.
        """
        token = webhook_token if webhook_token is not None else webhook_id
        if not token:
            raise ValueError(
                "A webhook token is required. Pass webhook_token='oaws_...' - the "
                "value shown once in the browser when the strategy was created or "
                "its token rotated."
            )
        self._host_url = host_url.rstrip('/')
        self._webhook_id = token
        self._webhook_url = None
        self.timeout = timeout
        # Reuse one keep-alive connection for repeated webhook posts instead
        # of opening a fresh socket per signal.
        self._client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=120.0,
            ),
        )

    def close(self):
        """Close the underlying HTTP client and release pooled connections."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        # The token is the whole credential, so it is never rendered.
        return f"Strategy(host_url={self._host_url!r}, webhook_token=<redacted>)"

    @property
    def webhook_url(self) -> str:
        """
        Cached property for the webhook URL to avoid reconstructing it every time.

        The URL embeds the token, so it is a credential in its own right. Do not
        log it or paste it anywhere it will be recorded.
        """
        if self._webhook_url is None:
            self._webhook_url = f"{self._host_url}/strategy/webhook/{self._webhook_id}"
        return self._webhook_url

    def send(self, payload: Dict[str, Any]) -> dict:
        """
        Post a raw payload to the webhook and return the parsed response.

        Every documented outcome is returned rather than raised, including the
        rejections, because the ``result`` label is the contract: a 409
        ``rejected_cooling_off`` and a 200 ``rejected_dedupe`` both need reading
        rather than a traceback. Only a transport failure or a non-JSON answer
        produces a locally-built error dict.

        Args:
            payload (dict): The JSON body to send.

        Returns:
            dict: The server's response with ``code`` attached when the HTTP
            status was not 200. Carries ``status``, ``result``, ``message`` and,
            where applicable, ``strategy_id``, ``run_id``, ``stop_pending`` and
            ``exits``.
        """
        try:
            response = self._client.post(self.webhook_url, json=payload)
        except httpx.TimeoutException:
            return {
                'status': 'error',
                'message': 'Request timed out. The server took too long to respond.',
                'error_type': 'timeout_error'
            }
        except httpx.ConnectError:
            return {
                'status': 'error',
                'message': 'Failed to connect to the server. Please check if the server is running.',
                'error_type': 'connection_error'
            }
        except httpx.HTTPError as e:
            return {
                'status': 'error',
                'message': f'HTTP error occurred: {str(e)}',
                'error_type': 'http_error'
            }

        try:
            data = response.json()
        except ValueError:
            return {
                'status': 'error',
                'message': f'HTTP {response.status_code}: {response.text}',
                'raw_response': response.text,
                'code': response.status_code,
                'error_type': 'http_error'
            }

        if not isinstance(data, dict):
            return {
                'status': 'error',
                'message': 'Unexpected JSON response from server',
                'raw_response': data,
                'code': response.status_code,
                'error_type': 'json_error'
            }

        if response.status_code != 200:
            data.setdefault('status', 'error')
            data['code'] = response.status_code
        return data

    # ------------------------------------------------------------------
    # Batch strategies
    # ------------------------------------------------------------------

    def start(self, mode: str) -> dict:
        """
        Start a batch strategy: enter every leg.

        Args:
            mode (str): "live" or "sandbox". Required, exact and case-sensitive
                on the server. There is no default: the one a hurried reader
                would reach for is the one that places real orders.

        Returns:
            dict: ``{"status", "result", "message", "strategy_id", "run_id"}``.

        Notes:
            - Live is opt-in per strategy. ``mode="live"`` on a strategy that has
              not enabled live trading answers 403 ``rejected_live_disabled``.
            - Two identical starts within 60 seconds are one signal
              (``rejected_dedupe``, reported as a success), because senders retry
              deliveries they believe failed.
            - A strategy that stopped within the last 30 seconds refuses a start
              with 409 ``rejected_cooling_off``, so a misconfigured pair of alerts
              cannot oscillate.
            - ``start`` against a signal strategy is ``rejected_invalid_action``.
        """
        return self.send({"action": "start", "mode": mode})

    def stop(self) -> dict:
        """
        Stop a batch strategy: exit every owned position at market.

        Returns:
            dict: ``{"status", "result", "message", "strategy_id", "run_id",
            "stop_pending", "exits"}``.

        Notes:
            - "Strategy stop accepted" means the durable request reached the
              engine, not that the broker is flat. ``stop_pending: true`` keeps
              the run current, subscribed and managed until exit fills confirm
              every owner is flat.
            - A stop is never blocked by the cooling-off window, and a stop
              against an already-stopped strategy does not arm it.
        """
        return self.send({"action": "stop"})

    # ------------------------------------------------------------------
    # Signal strategies
    # ------------------------------------------------------------------

    def signal(self, action: str, *, leg_id: Optional[int] = None,
               symbol: Optional[str] = None, exchange: Optional[str] = None) -> dict:
        """
        Send a signal-strategy action, naming the leg by id or by symbol.

        Args:
            action (str): long_entry, long_exit, short_entry or short_exit.
            leg_id (int, optional): Which leg the signal targets. Wins over
                symbol/exchange when both are given.
            symbol (str, optional): Leg symbol, used when leg_id is absent.
            exchange (str, optional): Exchange for ``symbol``.

        Returns:
            dict: ``{"status", "result", "message", "strategy_id", "run_id"}``.

        Notes:
            - A signal that does nothing is a success with a note, not a failure:
              ``Signal accepted (already_long)``. The notes are ``already_long``,
              ``already_short``, ``no_matching_position``, ``outside_entry_window``
              and ``outside_trading_window``. Reporting a no-op as a failure
              invites a retry, and a retry on an order path is how one alert
              becomes two positions.
            - Being refused is different. A signal blocked by the strategy's
              direction, or naming a leg that does not exist, answers 400
              ``rejected_invalid_action`` with the engine's own message.
            - An opposite entry squares the existing side first, then opens.
            - Signal actions skip the dedupe and cooling-off windows: they are
              idempotent by meaning, and a 60 second window would suppress a
              genuine long, short, long sequence.
            - A signal leg on a derivatives exchange must name an exact listed
              contract. A base symbol plus an expiry rank is refused, not guessed.
        """
        payload: Dict[str, Any] = {"action": action}
        if leg_id is not None:
            payload["leg_id"] = int(leg_id)
        if symbol is not None:
            payload["symbol"] = symbol
        if exchange is not None:
            payload["exchange"] = exchange
        return self.send(payload)

    def long_entry(self, *, leg_id: Optional[int] = None, symbol: Optional[str] = None,
                   exchange: Optional[str] = None) -> dict:
        """Open or reverse into a long on one leg. See :meth:`signal`."""
        return self.signal("long_entry", leg_id=leg_id, symbol=symbol, exchange=exchange)

    def long_exit(self, *, leg_id: Optional[int] = None, symbol: Optional[str] = None,
                  exchange: Optional[str] = None) -> dict:
        """Close a long on one leg. See :meth:`signal`."""
        return self.signal("long_exit", leg_id=leg_id, symbol=symbol, exchange=exchange)

    def short_entry(self, *, leg_id: Optional[int] = None, symbol: Optional[str] = None,
                    exchange: Optional[str] = None) -> dict:
        """Open or reverse into a short on one leg. See :meth:`signal`."""
        return self.signal("short_entry", leg_id=leg_id, symbol=symbol, exchange=exchange)

    def short_exit(self, *, leg_id: Optional[int] = None, symbol: Optional[str] = None,
                   exchange: Optional[str] = None) -> dict:
        """Close a short on one leg. See :meth:`signal`."""
        return self.signal("short_exit", leg_id=leg_id, symbol=symbol, exchange=exchange)

    # ------------------------------------------------------------------
    # Removed
    # ------------------------------------------------------------------

    def strategyorder(self, symbol: str, action: str, position_size: Optional[int] = None) -> dict:
        """
        Removed. The webhook it posted to no longer exists.

        The old surface took ``{"symbol", "action": "BUY"/"SELL", "position_size"}``
        and read the LONG_ONLY / SHORT_ONLY / BOTH mode from the strategy's own
        configuration. The strategy module replaced it with two explicit
        vocabularies, so there is no mapping this method could make on its own:
        ``BUY`` is ``long_entry`` on a flat leg and ``short_exit`` on a short one,
        and guessing which would be guessing at an order.

        Use :meth:`start` and :meth:`stop` for a batch strategy, or
        :meth:`long_entry`, :meth:`long_exit`, :meth:`short_entry` and
        :meth:`short_exit` for a signal strategy.
        """
        raise NotImplementedError(
            "Strategy.strategyorder() was removed: the webhook it posted to no longer "
            "exists. Use start(mode)/stop() for a batch strategy, or long_entry()/"
            "long_exit()/short_entry()/short_exit() for a signal strategy."
        )
