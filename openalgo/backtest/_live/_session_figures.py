"""The day's figures of a live session across its restarts (16-session-design 1.4 and 2.5 F;
unit 13): what each start saves to journal L as it runs, and how the close adds the
earlier starts of a restarted day to its own, counting nothing twice.

Each start saves its running figures with the state they describe, at every save of that
state, so a restart adds exactly what the state it restores does not hold:

- **Per strategy**, in its saved state (`FIGURES`): the round trips closed today, and the
  gross P&L and modelled charges of its account. The account is in the saved state
  itself (the kernel restores the cash with its realized P&L and its charges), so a
  restored engine's gross P&L and charges are the day's, and the saved ones are never
  added to them. Its round trips are not (a restored engine's result begins at the
  restart), so the saved count is carried and the restored engine's closed trades are
  added to it. A fill applied after the save closes its round trip again in the restored
  engine, which the saved count did not hold yet: it is counted once. A strategy an
  earlier start saved and this start does not run adds its saved figures whole.
- **For the session**, under `SESSION_KEY`: the orders refused, the reconciliation
  mismatches, the seconds positions had no confirmed stop and the alerts not delivered,
  as counted at the session's last save. A restart adds them to its own counts.

A count made after the last save before a crash is lost (an order refused in the seconds
before it, say); fills and round trips are not, since the restored engine applies them
again. Amounts are rupees, times seconds.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

from .._engine import call
from ..errors import LiveError

#: The key of a strategy's saved state that holds its figures (`_session_saved`).
FIGURES = "figures"
#: The name journal L keeps the session's own figures under, beside the strategies'
#: states: with its parentheses, no strategy class can have it.
SESSION_KEY = "(session)"


@dataclass(frozen=True)
class StrategyFigures:
    """A strategy's figures of the day (module docstring): round trips closed, gross P&L
    and modelled charges, rupees."""

    closed_trades: int = 0
    gross_pnl: float = 0.0
    charges: float = 0.0

    def __add__(self, other: StrategyFigures) -> StrategyFigures:
        return StrategyFigures(*(a + b for a, b in zip(_values(self), _values(other), strict=True)))


@dataclass(frozen=True)
class SessionFigures:
    """The session's own counts of the day (module docstring): orders refused,
    reconciliation mismatches, seconds a position had no confirmed stop at the broker, and
    alerts not delivered."""

    rejected_orders: int = 0
    reconcile_mismatches: int = 0
    unprotected_seconds: float = 0.0
    undelivered_alerts: int = 0

    def __add__(self, other: SessionFigures) -> SessionFigures:
        return SessionFigures(*(a + b for a, b in zip(_values(self), _values(other), strict=True)))


class DayFigures:
    """What the earlier starts of the day left (module docstring): `carried`, each
    running strategy's saved figures; `absent`, the saved figures of the strategies this
    start does not run, together; `session`, the session's saved counts."""

    def __init__(
        self,
        carried: Mapping[str, StrategyFigures],
        absent: StrategyFigures,
        session: SessionFigures,
    ) -> None:
        self.carried = dict(carried)
        self.absent = absent
        self.session = session

    @classmethod
    def at_start(cls, saved: Mapping[str, Mapping[str, Any]], running: Sequence[str]) -> DayFigures:
        """The figures `saved` holds (journal L's saved states of the day, by name, the
        session's under `SESSION_KEY`) for a start that runs the strategies `running`.
        Raises LiveError for figures that cannot be read, naming them: the day's summary
        is never guessed."""
        carried, absent = {}, StrategyFigures()
        for name, state in saved.items():
            if name == SESSION_KEY:
                continue
            figures = _read(StrategyFigures, state.get(FIGURES), name)
            if name in running:
                carried[name] = figures
            else:
                absent = absent + figures
        session = _read(SessionFigures, saved.get(SESSION_KEY), SESSION_KEY)
        return cls(carried, absent, session)

    def strategy(self, trader: Any) -> dict[str, Any]:
        """The running figures of `trader`'s strategy now, to save with its state: the
        round trips carried and closed since, its account's gross P&L and charges."""
        _equity, _cash, realized, unrealized, costs, _margin = call(trader.engine.account)
        closed = self._carried(trader.name).closed_trades + trader.records.closed_trades
        return asdict(StrategyFigures(closed, realized + unrealized, costs))

    def session_counts(self, session: Any) -> dict[str, Any]:
        """The session's running counts now, carried ones included, to save."""
        return asdict(self.session + _own(session, session.stops.windows.seconds(session.now())))

    def day(
        self,
        session: Any,
        outputs: Mapping[str, tuple[Mapping[str, Any], tuple[float, float]]],
        unprotected_s: float,
    ) -> tuple[StrategyFigures, SessionFigures]:
        """The day's figures at the close (module docstring): `outputs` are each running
        strategy's `finish()` tables and (gross P&L, charges) of its account, by name;
        `unprotected_s` this start's seconds unprotected."""
        total = self.absent
        for name, (output, (gross, charges)) in outputs.items():
            closed = self._carried(name).closed_trades + len(output["trades"]["trade_id"])
            total = total + StrategyFigures(closed, gross, charges)
        return total, self.session + _own(session, unprotected_s)

    def _carried(self, name: str) -> StrategyFigures:
        return self.carried.get(name, StrategyFigures())


def _own(session: Any, unprotected_s: float) -> SessionFigures:
    """This start's own counts so far, with `unprotected_s` seconds unprotected."""
    return SessionFigures(
        session.rejected, session.mismatches, unprotected_s, session.parts.sender.undelivered
    )


def _values(figures: Any) -> tuple[Any, ...]:
    return tuple(getattr(figures, field.name) for field in fields(figures))


def _read(kind: type, record: Any, name: str) -> Any:
    """`kind` from a saved record (None: nothing saved, zeros). Raises LiveError for a
    record whose fields are not finite numbers of at least zero, a count not whole."""
    if record is None:
        return kind()
    if not isinstance(record, Mapping):
        raise _unreadable(name, record)
    values = []
    for field in fields(kind):
        value = record.get(field.name, 0)
        counted = field.type in ("int", int)
        number = isinstance(value, numbers.Real) and not isinstance(value, bool)
        if not number or not math.isfinite(value) or (counted and value != int(value)):
            raise _unreadable(name, record)
        if counted and value < 0:
            raise _unreadable(name, record)
        values.append(int(value) if counted else float(value))
    return kind(*values)


def _unreadable(name: str, record: Any) -> LiveError:
    return LiveError(
        f"the day's figures journal L saved for {name} cannot be read ({record!r}): the "
        "day's summary would be wrong. Check the account in OpenAlgo, then start under "
        "another account name (account=)"
    )


__all__ = ["FIGURES", "SESSION_KEY", "DayFigures", "SessionFigures", "StrategyFigures"]
