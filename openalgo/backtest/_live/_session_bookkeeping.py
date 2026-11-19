"""A live strategy's bookkeeping beside its kernel session (16-session-design 1.4, 2.5 F and
6.2; units 8 and 13): its entry locks, each kept in its records for the result, and its
`self.state` as journal L saves it.

`Bookkeeping` is a part of `_session_trader.Trader`, which holds the kernel session
(`engine`), the records (`records`, a `_session_records.StrategyRecords`), whether the
kernel session finished (`finished`), the session's clock (`_now`) and the last state
saved (`_state_text`, `_state_warned`).

Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .._engine import call
from ..errors import JournalError
from ._journal_rows import state_json
from ._session_records import StrategyRecords
from ._types import LOGGER_NAME, Now

logger = logging.getLogger(LOGGER_NAME)


class Bookkeeping:
    """The entry locks and the saved state of a `_session_trader.Trader` (module
    docstring)."""

    name: str
    part: Any
    finished: bool
    records: StrategyRecords
    _now: Now
    _state_text: str | None
    _state_warned: bool

    @property
    def engine(self) -> Any:
        """The strategy's kernel session."""
        return self.part.engine

    def lock(self, reason: str) -> bool:
        """Lock the strategy's entries for `reason`; True when they were not locked for
        it (False once the kernel session finished: it takes no entry)."""
        locked = not self.finished and bool(call(self.engine.lock_entries, reason))
        if locked:
            self.records.locked(reason, self._now())
        return locked

    def unlock(self, reason: str) -> bool:
        """Lift the entry lock of `reason`; True when it was set."""
        lifted = not self.finished and bool(call(self.engine.unlock_entries, reason))
        if lifted:
            self.records.unlocked(reason, self._now())
        return lifted

    def state_changed(self) -> dict[str, Any] | None:
        """The strategy's `self.state` when it changed since the last call, else None. A
        state the journal cannot keep (a value JSON does not hold) is warned about once
        and not saved: the session goes on, and a restart would not have it."""
        text = self._state_json()
        if text is None or text == self._state_text:
            return None
        self._state_text = text
        return json.loads(text)

    def _state_json(self) -> str | None:
        try:
            return state_json(self.part.strategy.state, "self.state", self.name)
        except JournalError as error:
            if not self._state_warned:
                self._state_warned = True
                logger.warning("%s: %s; it is not saved", self.name, error)
            return None


__all__ = ["Bookkeeping"]
