"""The mode guard of a live session (08-live-trading section 3; D21, D30).

OpenAlgo routes every order by its analyzer switch at the moment of the call, and that
switch is the only control of sandbox or live: the engine never sets it and nothing in
code states an expected mode (D21). The guard reads the switch when the session starts
and adopts that mode, then watches for a change in three places: the `mode` of every
order response and order update, an `analyzerstatus` read every `poll_seconds`, and a
fresh read just before each entry. The server checks no expected mode on an order (D30),
so a change between two reads is found on the next response, update or read.

A change locks the guard for the rest of the session: the order path sends nothing more,
one MODE_BREACH alert names the orders whose book is in doubt, and the user starts the
session again in the mode they want. Positions and stops opened in one mode do not exist
in the other book, so a session is never carried across a change.

Semi-auto (the Action Center mode, set per user on the API key page of OpenAlgo's web
interface) answers a placed order with a pending order id: nothing reaches the broker
until someone approves it there, in sandbox mode as well as in live mode. In live mode it
also refuses every modify and cancel. No call made with an API key reports it (only the
logged-in API key page shows it, and the analyzer toggle, which the engine never calls,
refuses while it is on), so `start` cannot refuse it; the first order response that
carries a pending order id, or a semi-auto refusal, locks the guard.

Not thread-safe: the session loop owns the guard, as it owns the engine.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol

from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError, LiveError, ModeChangedError
from ._guard_breach import ModeBreach, SeenIn, breach_text
from ._types import LOGGER_NAME, AlertKind, AlertsLike, Mode, Now, OrderUpdate, mode_of

#: Seconds between two `analyzerstatus` reads (08-live-trading section 3).
DEFAULT_POLL_SECONDS = 5.0
#: Consecutive failed `analyzerstatus` reads after which an alert is sent (the guard's
#: brief, Phase 2 step 4): one failure is a blip, three mean the switch is unwatched.
FAILED_READS_BEFORE_ALERT = 3

# The fields OpenAlgo puts on an order it queued in the Action Center instead of sending
# it (services/order_router_service.py, queue_order), and on every order it placed.
_PENDING_ORDER_ID_FIELD = "pending_order_id"
_SEMI_AUTO_MODE_FIELD = "semi_auto"
_ORDER_ID_FIELD = "orderid"
_SUCCESS = "success"
# Lower case, part of the message of the 403 that refuses a modify or cancel while
# semi-auto is on in live mode ("Modify order operation is not allowed in Semi-Auto
# mode. ...": services/modify_order_service.py and services/cancel_order_service.py,
# the semi-auto check of their API-key path). Matched in the server's body and in the
# SDK's "HTTP 403: <body>" wrapper alike.
_SEMI_AUTO_REFUSAL = "is not allowed in semi-auto mode"

logger = logging.getLogger(LOGGER_NAME)


class _StatusClient(Protocol):
    """What the guard needs of an `openalgo.api` client: its analyzer status read."""

    def analyzerstatus(self) -> Mapping[str, Any]:
        """The server's answer to `/api/v1/analyzer`."""


class ModeGuard:
    """The mode guard of one live session (implements `GuardLike`).

    `client` is an `openalgo.api` client; only `analyzerstatus()` is called. `now` gives
    the present in UTC nanoseconds, `alerts` (optional) receives the alerts, and
    `poll_seconds` is the least time between two polled reads.

    Raises ConfigError when `poll_seconds` is not a positive number of seconds.
    """

    def __init__(
        self,
        client: _StatusClient,
        *,
        now: Now,
        alerts: AlertsLike | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        if not (math.isfinite(poll_seconds) and poll_seconds > 0):
            raise ConfigError(
                f"poll_seconds is the time between two analyzerstatus reads and must be a "
                f"positive number of seconds, got {poll_seconds!r}"
            )
        self._client = client
        self._now = now
        self._alerts = alerts
        self._poll_ns = round(poll_seconds * NANOS_PER_SECOND)
        self._session_mode: Mode | None = None
        self._next_read_ns = 0
        self._failed_reads = 0
        self._breach: ModeBreach | None = None
        # Orders this session sent, the ones whose own response or update showed the
        # session's mode, and (ordered, as dicts) the ones sent and the intents left in
        # doubt since the switch was last read in the session's mode that nothing has
        # confirmed yet.
        self._own_orders: set[str] = set()
        self._confirmed_orders: set[str] = set()
        self._unconfirmed_orders: dict[str, None] = {}
        self._intents_in_doubt: dict[str, None] = {}

    @property
    def locked(self) -> bool:
        """True once the mode changed or semi-auto appeared: nothing more is sent."""
        return self._breach is not None

    @property
    def breach(self) -> ModeBreach | None:
        """Why the guard locked, or None while it is open."""
        return self._breach

    @property
    def failed_polls(self) -> int:
        """How many `analyzerstatus` reads in a row have failed (0 after a good one)."""
        return self._failed_reads

    def start(self) -> Mode:
        """Read OpenAlgo's analyzer switch and adopt its mode for the session.

        Returns the mode. Raises LiveError when OpenAlgo does not report its mode (it
        is not running, cannot be reached, or rejects the API key) or when the guard
        has started already.
        """
        if self._session_mode is not None:
            raise LiveError("the mode guard has started already: a session starts it once")
        read_ns = self._now()
        try:
            mode = _switch_mode(self._client.analyzerstatus())
        except LiveError as error:
            raise LiveError(
                f"{error}. Check that OpenAlgo is running and the API key is valid, then "
                f"start the session again."
            ) from error
        self._session_mode = mode
        self._next_read_ns = read_ns + self._poll_ns
        logger.info("OpenAlgo's analyzer switch is in %s mode: the session runs in it", mode.name)
        return mode

    def banner(self, broker: str) -> str:
        """The first line of the start banner, for the session's mode: "OpenAlgo is in
        SANDBOX mode: orders are simulated" or "OpenAlgo is in LIVE mode: orders go to
        <broker>".

        Raises LiveError before `start`, ConfigError for a LIVE session without a broker.
        """
        if self._started() is Mode.SANDBOX:
            return "OpenAlgo is in SANDBOX mode: orders are simulated"
        if not broker.strip():
            raise ConfigError(
                "the LIVE banner names the broker the orders go to: pass the broker "
                "OpenAlgo is logged in to"
            )
        return f"OpenAlgo is in LIVE mode: orders go to {broker}"

    def poll(self) -> None:
        """Read the switch when `poll_seconds` have passed since the last read, and lock
        on a change. Called often by the session loop; reads nothing once locked.

        A failed read is not a mode change: it is counted (`failed_polls`) and the third
        in a row sends an alert. Raises LiveError before `start`.
        """
        self._started()
        if self._breach is not None or self._now() < self._next_read_ns:
            return
        try:
            self._read_switch(SeenIn.POLL)
        except LiveError:
            # Counted, logged and alerted by _read_switch; the next poll tries again.
            return

    def before_entry(self) -> None:
        """Read the switch just before an entry is sent.

        Raises ModeChangedError when the guard is locked or this read locks it, and
        LiveError (the entry is not sent, the session goes on) when OpenAlgo does not
        answer, since the mode the entry would go to is then unconfirmed. Raises
        LiveError before `start`.
        """
        self._started()
        if self._breach is None:
            try:
                self._read_switch(SeenIn.BEFORE_ENTRY)
            except LiveError as error:
                raise LiveError(
                    f"{error}: the entry is not sent, because the mode it would go to is "
                    f"not confirmed. Check that OpenAlgo is running."
                ) from error
        if self._breach is not None:
            raise ModeChangedError(breach_text(self._breach))

    def check_response(self, response: Mapping[str, Any]) -> None:
        """Check the mode of an order response; lock on a change or on semi-auto.

        Pass the responses of the calls the analyzer switch routes (place, modify and
        cancel orders, order status, the order, trade and position books, funds), a
        non-200 answer as the server's own body: a market-data response carries no mode
        in either. A response marked "analyze" was handled in the sandbox; a success
        without it went to the broker. An error without it says nothing (the SDK's
        transport errors, the rate limiter and the key check answer that way in both
        modes) and is left to the next read. A pending order id or a semi-auto refusal
        locks the guard as semi-auto. Raises LiveError before `start`.
        """
        session_mode = self._started()
        if self._breach is not None:
            return
        if _semi_auto_seen(response):
            pending = response.get(_PENDING_ORDER_ID_FIELD)
            self._lock(None, SeenIn.RESPONSE, None if pending is None else str(pending))
            return
        seen = mode_of(response)
        if seen is Mode.LIVE and response.get("status") != _SUCCESS:
            logger.debug("an error response without a mode says nothing of the switch")
            return
        order_id = str(response.get(_ORDER_ID_FIELD) or "")
        if seen is session_mode:
            if order_id:
                self._confirm_order(order_id)
            return
        if order_id:
            self._own_orders.add(order_id)
            self._unconfirmed_orders[order_id] = None
        self._lock(seen, SeenIn.RESPONSE)

    def check_update(self, update: OrderUpdate) -> None:
        """Check the mode of an order update; lock when an order of this session is
        reported in the other mode.

        An update of an order this session did not send (a manual order at the broker, a
        run of the server's strategy module in the other mode) says nothing of the
        switch and is not judged; an update that arrives before its order is noted is
        covered by that order's response. Raises LiveError before `start`.
        """
        session_mode = self._started()
        if self._breach is not None:
            return
        if update.order_id not in self._own_orders:
            logger.debug(
                "order update of %s not judged: not an order of this session", update.order_id
            )
            return
        if update.mode is session_mode:
            self._confirm_order(update.order_id)
            return
        self._confirmed_orders.discard(update.order_id)
        self._unconfirmed_orders[update.order_id] = None
        self._lock(update.mode, SeenIn.UPDATE)

    def note_order_sent(self, order_id: str) -> None:
        """Record an order the session sent (its ID as OpenAlgo answered it), so that a
        breach names every order whose book is in doubt. An order noted after the lock
        is added to `breach` and logged as an error; no second alert is sent.

        Raises ConfigError for an empty ID and LiveError before `start`.
        """
        self._started()
        if not order_id:
            raise ConfigError("note_order_sent takes the order ID OpenAlgo answered with")
        self._own_orders.add(order_id)
        if self._breach is None:
            if order_id not in self._confirmed_orders:
                self._unconfirmed_orders[order_id] = None
            return
        if order_id in self._confirmed_orders or order_id in self._breach.orders_after:
            return
        self._breach = replace(self._breach, orders_after=(*self._breach.orders_after, order_id))
        logger.error("order %s was sent after the mode guard locked", order_id)

    def note_order_in_doubt(self, intent_id: str) -> None:
        """Record an intent whose order may have reached the server with no order ID to
        show for it (a timeout, an HTTP 502, a lost connection), so that a breach names
        it. It does not lock the guard. An intent noted after the lock is added to
        `breach`; no second alert is sent.

        Raises ConfigError for an empty ID and LiveError before `start`.
        """
        self._started()
        if not intent_id:
            raise ConfigError("note_order_in_doubt takes the intent ID of the journal")
        if self._breach is None:
            self._intents_in_doubt[intent_id] = None
        elif intent_id not in self._breach.intents_in_doubt:
            in_doubt = (*self._breach.intents_in_doubt, intent_id)
            self._breach = replace(self._breach, intents_in_doubt=in_doubt)
            logger.error("intent %s was left in doubt after the mode guard locked", intent_id)

    def _started(self) -> Mode:
        """The session's mode. Raises LiveError before `start`."""
        if self._session_mode is None:
            raise LiveError("the mode guard has not read OpenAlgo's mode yet: call start() first")
        return self._session_mode

    def _confirm_order(self, order_id: str) -> None:
        """An order's own response or update showed the session's mode."""
        self._own_orders.add(order_id)
        self._confirmed_orders.add(order_id)
        self._unconfirmed_orders.pop(order_id, None)

    def _read_switch(self, seen_in: SeenIn) -> None:
        """Read the switch, schedule the next poll from now, and lock on a change.

        Raises LiveError when the read failed, after counting it.
        """
        read_ns = self._now()
        self._next_read_ns = read_ns + self._poll_ns
        try:
            mode = _switch_mode(self._client.analyzerstatus())
        except LiveError as error:
            self._count_failed_read(error)
            raise
        if self._failed_reads:
            logger.info("OpenAlgo answered analyzerstatus again")
        self._failed_reads = 0
        if mode is self._session_mode:
            self._unconfirmed_orders.clear()
            self._intents_in_doubt.clear()
        else:
            self._lock(mode, seen_in, seen_ns=read_ns)

    def _count_failed_read(self, error: LiveError) -> None:
        self._failed_reads += 1
        logger.warning("analyzerstatus read failed (%d in a row): %s", self._failed_reads, error)
        if self._failed_reads == FAILED_READS_BEFORE_ALERT:
            self._alert(
                AlertKind.ERROR,
                f"OpenAlgo has not answered analyzerstatus {self._failed_reads} times in a "
                f"row ({error}), so a change of its analyzer switch is found only on the "
                f"next order response. Check that OpenAlgo is running.",
            )

    def _lock(
        self,
        seen_mode: Mode | None,
        seen_in: SeenIn,
        pending_order_id: str | None = None,
        *,
        seen_ns: int | None = None,
    ) -> None:
        self._breach = ModeBreach(
            session_mode=self._started(),
            seen_mode=seen_mode,
            seen_ns=self._now() if seen_ns is None else seen_ns,
            seen_in=seen_in,
            orders_after=tuple(self._unconfirmed_orders),
            pending_order_id=pending_order_id,
            intents_in_doubt=tuple(self._intents_in_doubt),
        )
        text = breach_text(self._breach)
        logger.error(text)
        self._alert(AlertKind.MODE_BREACH, text)

    def _alert(self, kind: AlertKind, text: str) -> None:
        if self._alerts is not None:
            self._alerts.send(kind, text)


def _switch_mode(response: Mapping[str, Any]) -> Mode:
    """The mode an `analyzerstatus` answer reports: `data.analyze_mode`, a boolean
    (services/analyzer_service.py). Raises LiveError for an error or any other answer."""
    if response.get("status") != _SUCCESS:
        reason = response.get("message") or "an error without a message"
        raise LiveError(f"OpenAlgo did not report its mode (analyzerstatus: {reason})")
    data = response.get("data")
    analyze_mode = data.get("analyze_mode") if isinstance(data, Mapping) else None
    if not isinstance(analyze_mode, bool):
        raise LiveError(
            "OpenAlgo did not report its mode (analyzerstatus answered without data.analyze_mode)"
        )
    return Mode.SANDBOX if analyze_mode else Mode.LIVE


def _semi_auto_seen(response: Mapping[str, Any]) -> bool:
    """True for an answer of semi-auto mode: an order queued in the Action Center, or a
    modify or cancel the server refused because semi-auto is on."""
    if _PENDING_ORDER_ID_FIELD in response or response.get("mode") == _SEMI_AUTO_MODE_FIELD:
        return True
    message = response.get("message")
    return (
        response.get("status") != _SUCCESS
        and isinstance(message, str)
        and _SEMI_AUTO_REFUSAL in message.lower()
    )


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "FAILED_READS_BEFORE_ALERT",
    "ModeBreach",
    "ModeGuard",
    "SeenIn",
]
