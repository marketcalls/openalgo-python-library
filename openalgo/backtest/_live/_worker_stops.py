"""The journal's stop of each leg, kept by the order worker on journal handle W
(16-session-design 4.1): one exit owner per leg, so `set_stop` refuses a second stop of
a leg on the day and `clear_stop` forgets it (`_journal_stops`).

- A STOP intent's terminal news clears its leg's stop when the journal names that order,
  before the news is posted, so a leg's second trade of the day, a re-placed stop, the
  protect after failed exits and a restart's re-placed stop each find the leg clear.
- A STOP intent that binds (an accepted placement, or BOUND by resolution) is recorded.
  When the journal names another stop of the leg: one the order path or the journal
  knows terminal is cleared first; else one `check_order` read decides: terminal,
  cleared; still working, it is an own stop the kernel does not know (two stops on a leg
  could reverse the position), so it is cancelled as an orphan with a RECONCILE_MISMATCH
  alert and the new stop is recorded once the orphan's terminal news comes; unreadable,
  the new stop is not recorded and an ERROR alert says so.

A refusal of `set_stop` is never fatal: it is logged and alerted. Only a `JournalError`
for a file failure (SQLite's own error chained) is raised, which stops the worker (3.5).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager

from ..errors import JournalError, LiveError, ModeChangedError
from ._order_answers import Outcome
from ._types import LOGGER_NAME, AlertKind, AlertsLike, OrderIntent, OrderUpdate
from .journal import Journal
from .orders import OrderPath

logger = logging.getLogger(LOGGER_NAME)

#: A leg: (strategy, symbol, exchange), as the journal keys its stops.
Leg = tuple[str, str, str]


def leg_of(intent: OrderIntent) -> Leg:
    """The leg an intent's order trades."""
    return intent.strategy, intent.symbol, intent.exchange


class StopBook:
    """The journal's stops of the worker's legs (module docstring). `journal` is handle W,
    `path` the order path, `alerts` the session's; `calling` marks the worker busy for a
    call that may reach the server, and `deliver` posts an order's news (which comes back
    to `ended` for a STOP's terminal news)."""

    def __init__(
        self,
        journal: Journal,
        path: OrderPath,
        *,
        alerts: AlertsLike,
        calling: Callable[[], AbstractContextManager[None]],
        deliver: Callable[[OrderUpdate], None],
    ) -> None:
        self._journal = journal
        self._path = path
        self._alerts = alerts
        self._calling = calling
        self._deliver = deliver
        # An orphan stop being cancelled, and the stop to record once it has ended.
        self._after: dict[str, tuple[Leg, str]] = {}

    def intent_of(self, order_id: str) -> OrderIntent:
        """The intent of an order of the session. Raises LiveError for one it did not
        send (the order path hands over news of the session's own orders only)."""
        intent = self._journal.intent_of(order_id)
        if intent is None:
            raise LiveError(f"order {order_id} has news but no intent in the journal")
        return intent

    def ended(self, intent: OrderIntent, order_id: str) -> bool:
        """A STOP intent's order ended: clear its leg's stop when the journal names it,
        and record a stop that waited for it; True when it was cleared."""
        leg = leg_of(intent)
        cleared = self._journal.stop_of(*leg) == order_id
        if cleared:
            self._journal.clear_stop(*leg)
        waiting = self._after.pop(order_id, None)
        if waiting is not None:
            self._set(*waiting)
        return cleared

    def bound(self, intent: OrderIntent, order_id: str) -> None:
        """A STOP intent was bound to `order_id`: record it as its leg's stop (module
        docstring)."""
        leg = leg_of(intent)
        held = self._journal.stop_of(*leg)
        if held is None or held == order_id:
            self._set(leg, order_id)
            return
        known = self._path.last_known(held) or self._journal.last_update(held)
        if known is not None and known.status.terminal:
            self._journal.clear_stop(*leg)
            self._set(leg, order_id)
            return
        try:
            with self._calling():
                check = self._path.check_order(held)
        except ModeChangedError:
            return
        except LiveError as error:
            self._alert(AlertKind.ERROR, f"{self._not_recorded(leg, order_id, held)}: {error}")
            return
        if check.news is not None:
            self._deliver(check.news)
        if check.status is None:
            why = f"its row cannot be read ({check.reason})"
            self._alert(AlertKind.ERROR, f"{self._not_recorded(leg, order_id, held)}: {why}")
        elif check.status.terminal:
            if self._journal.stop_of(*leg) == held:
                self._journal.clear_stop(*leg)
            self._set(leg, order_id)
        else:
            self._orphan(leg, order_id, held)

    def _orphan(self, leg: Leg, order_id: str, held: str) -> None:
        strategy, symbol, exchange = leg
        self._after[held] = (leg, order_id)
        self._alert(
            AlertKind.RECONCILE_MISMATCH,
            f"{strategy} on {exchange}:{symbol} has stop {held} working at the broker "
            f"besides its new stop {order_id}: {held} is cancelled, since two stops could "
            "reverse the position",
        )
        try:
            with self._calling():
                outcome = self._path.cancel(held)
        except LiveError as error:
            outcome = None
            reason = str(error)
        else:
            reason = outcome.reason
        if outcome is None or outcome.kind is not Outcome.ACCEPTED:
            self._alert(
                AlertKind.ERROR,
                f"stop {held} of {strategy} on {exchange}:{symbol} could not be cancelled "
                f"({reason}): cancel it by hand in OpenAlgo",
            )

    def _set(self, leg: Leg, order_id: str) -> None:
        try:
            self._journal.set_stop(*leg, order_id)
        except JournalError as error:
            if isinstance(error.__cause__, sqlite3.Error):
                raise
            self._alert(AlertKind.ERROR, f"{self._not_recorded(leg, order_id, '')}: {error}")

    def _not_recorded(self, leg: Leg, order_id: str, held: str) -> str:
        strategy, symbol, exchange = leg
        besides = f" (the journal names stop {held})" if held else ""
        return f"stop {order_id} of {strategy} on {exchange}:{symbol} is not recorded{besides}"

    def _alert(self, kind: AlertKind, text: str) -> None:
        logger.error(text)
        self._alerts.send(kind, text)


__all__ = ["Leg", "StopBook", "leg_of"]
