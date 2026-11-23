"""The day's figures of a live session across its restarts (16-session-design 1.4 and 2.5 F;
unit 13): what each start saves to journal L as it runs, and how the close adds the
earlier starts of a restarted day to its own, counting nothing twice.

Each start saves its running figures with the state they describe, at every save of that
state, so a restart adds exactly what the state it restores does not hold:

- **Per strategy**, in its saved state (`FIGURES`): the round trips closed today, the
  gross P&L and modelled charges of its account, and the slippage of its MARKET fills
  and of its stops' fills against the backtest's (`_fill_comparison`) with the counts of
  those compared and not compared; beside them its fills' rows of the comparison
  (`_fill_comparison.COMPARED`, carried as the slippage is: `rows`). The account is in
  the saved state itself (the kernel restores the cash with its realized P&L and its
  charges), so a restored engine's gross P&L and charges are the day's, and the saved
  ones are never added to them. Its round trips and its slippage are not (a restored
  engine's result begins at the restart, and a start compares its own fills only), so
  the saved ones are carried and the restored start's own are added to them. A fill
  applied after the save is applied again in the restored engine, which the saved
  figures did not hold yet: it is counted once. A strategy an
  earlier start saved and this start does not run adds its saved figures whole, and is
  named on the summary (`Earlier`) with the position its last save held, if any: the day
  is not flat while one is held. Its entries of the day, as its kernel counted them at
  its last save, still count toward the account's trade cap (`earlier_entries`;
  18-multi-strategy 2.5a rule 8).
- **For the session**, under `SESSION_KEY`: the orders refused, the reconciliation
  mismatches, the seconds positions had no confirmed stop and the alerts not delivered,
  as counted at the session's last save. A restart adds them to its own counts.

The close saves both once more, after its final fills and before the kernel sessions
stop, with the mismatches it names (`_session_close`), so a clean stop loses nothing a
restart of the day needs. A count made after the last save before a crash is lost (an
order refused in the seconds before it, say); fills and round trips are not, since the
restored engine applies them again. Amounts are rupees, times seconds, quantities signed
units.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from .._engine import call
from ..errors import LiveError
from ._fill_comparison import DayRows
from ._plan_catch_up import INSTRUMENT
from ._session_slippage import Slippage
from ._venue_requests import LIVE_PRODUCT

#: The key of a strategy's saved state that holds its figures (`_session_saved`).
FIGURES = "figures"
#: The name journal L keeps the session's own figures under, beside the strategies'
#: states: with its parentheses, no strategy class can have it.
SESSION_KEY = "(session)"


@dataclass(frozen=True)
class StrategyFigures:
    """A strategy's figures of the day (module docstring): round trips closed, gross P&L,
    modelled charges and the slippage of its MARKET fills against the backtest's, rupees,
    and how many of those fills were not compared and compared; the same of its stops'
    fills. A save of an earlier version holds no slippage: it reads as none."""

    closed_trades: int = 0
    gross_pnl: float = 0.0
    charges: float = 0.0
    slippage: float = 0.0
    slippage_not_compared: int = 0
    slippage_compared: int = 0
    stop_slippage: float = 0.0
    stop_slippage_not_compared: int = 0
    stop_slippage_compared: int = 0

    def __add__(self, other: StrategyFigures) -> StrategyFigures:
        return StrategyFigures(*(a + b for a, b in zip(_values(self), _values(other), strict=True)))

    @property
    def summary_slippage(self) -> float | None:
        """The slippage the day summary shows: rupees, or None when MARKET fills were made
        and not one of them could be compared (there is no figure to show, only the
        count)."""
        if self.slippage_not_compared and not self.slippage_compared:
            return None
        return self.slippage

    @property
    def summary_stop_slippage(self) -> float | None:
        """The stops' slippage the day summary shows: rupees, or None when no stop's fill
        was compared (there is no figure to show)."""
        return self.stop_slippage if self.stop_slippage_compared else None


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


@dataclass(frozen=True)
class Earlier:
    """A strategy an earlier start of the day ran and this start does not (module
    docstring): its name and, as its last save held it, its instrument and signed units
    (0: no position)."""

    name: str
    symbol: str = ""
    exchange: str = ""
    units: int = 0

    def words(self) -> str:
        """The strategy as the summary's Strategies line names it."""
        if not self.units:
            return self.name
        return (
            f"{self.name} ({self.units} units of {self.symbol} {self.exchange} open at its "
            "last save)"
        )


class DayFigures:
    """What the earlier starts of the day left (module docstring): `carried`, each
    running strategy's saved figures; `absent`, the saved figures of the strategies this
    start does not run, together, `earlier`, those strategies, and `earlier_entries`, the
    entries of the day their kernels had counted; `session`, the session's saved counts;
    `rows`, the fill comparison's saved rows."""

    def __init__(
        self,
        carried: Mapping[str, StrategyFigures],
        absent: StrategyFigures,
        session: SessionFigures,
        earlier: Sequence[Earlier] = (),
        earlier_entries: int = 0,
        rows: DayRows | None = None,
    ) -> None:
        self.carried = dict(carried)
        self.absent = absent
        self.session = session
        self.earlier = tuple(earlier)
        self.earlier_entries = earlier_entries
        self.rows = DayRows({}, []) if rows is None else rows

    @classmethod
    def at_start(cls, saved: Mapping[str, Mapping[str, Any]], running: Sequence[str]) -> DayFigures:
        """The figures `saved` holds (journal L's saved states of the day, by name, the
        session's under `SESSION_KEY`) for a start that runs the strategies `running`.
        Raises LiveError for figures, or the position of a strategy it does not run, that
        cannot be read, naming them: the day's summary is never guessed."""
        carried, absent, earlier, entries = {}, StrategyFigures(), [], 0
        for name, state in saved.items():
            if name == SESSION_KEY:
                continue
            figures = _read(StrategyFigures, state.get(FIGURES), name)
            if name in running:
                carried[name] = figures
            else:
                absent = absent + figures
                earlier.append(_earlier(name, state))
                entries += _entries(name, state)
        session = _read(SessionFigures, saved.get(SESSION_KEY), SESSION_KEY)
        rows = DayRows.at_start(saved, running)
        return cls(carried, absent, session, earlier, entries, rows)

    @property
    def earlier_open(self) -> bool:
        """True when a strategy this start does not run held a position at its last
        save."""
        return any(strategy.units for strategy in self.earlier)

    def strategy(self, trader: Any, slipped: Slippage) -> dict[str, Any]:
        """The running figures of `trader`'s strategy now, to save with its state: the
        round trips and the slippage carried and its own since (`slipped`, this start's),
        its account's gross P&L and charges."""
        _equity, _cash, realized, unrealized, costs, _margin = call(trader.engine.account)
        traded = (trader.records.closed_trades, realized + unrealized, costs)
        return asdict(self._with_carried(trader.name, traded, slipped))

    def session_counts(
        self, session: Any, *, unprotected_s: float | None = None, named: int = 0
    ) -> dict[str, Any]:
        """The session's running counts now, carried ones included, to save: with
        `unprotected_s` this start's seconds unprotected (None: as its windows count them
        now) and `named` the mismatches the close names (`_session_close`)."""
        if unprotected_s is None:
            unprotected_s = session.stops.windows.seconds(session.now())
        named_mismatches = SessionFigures(reconcile_mismatches=named)
        return asdict(self.session + _own(session, unprotected_s) + named_mismatches)

    def day(
        self,
        session: Any,
        outputs: Mapping[str, tuple[Mapping[str, Any], tuple[float, float]]],
        unprotected_s: float,
        *,
        slippages: Mapping[str, Slippage],
    ) -> tuple[StrategyFigures, SessionFigures]:
        """The day's figures at the close (module docstring): `outputs` are each running
        strategy's `finish()` tables and (gross P&L, charges) of its account, by name;
        `unprotected_s` this start's seconds unprotected; `slippages` each running
        strategy's slippage in this start, by name, measured before its kernel session
        finished (`_fill_comparison.slippage_of`; a name not given had no fill). The day's
        slippages are rounded to the paisa."""
        total = self.absent
        for name, (output, (gross, charges)) in outputs.items():
            traded = (len(output["trades"]["trade_id"]), gross, charges)
            total = total + self._with_carried(name, traded, slippages.get(name, Slippage()))
        total = replace(
            total,
            slippage=round(total.slippage, 2),
            stop_slippage=round(total.stop_slippage, 2),
        )
        return total, self.session + _own(session, unprotected_s)

    def _carried(self, name: str) -> StrategyFigures:
        return self.carried.get(name, StrategyFigures())

    def _with_carried(
        self, name: str, traded: tuple[int, float, float], own: Slippage
    ) -> StrategyFigures:
        """Strategy `name`'s figures: `traded`, this start's (round trips closed, its
        account's gross P&L and charges, which hold the carried ones already), with the
        carried round trips and slippage added, and `own`, this start's slippage."""
        carried = self._carried(name)
        closed, gross, charges = traded
        return StrategyFigures(
            carried.closed_trades + closed,
            gross,
            charges,
            round(carried.slippage + own.rupees, 2),
            carried.slippage_not_compared + own.not_compared,
            carried.slippage_compared + own.compared,
            round(carried.stop_slippage + own.stop_rupees, 2),
            carried.stop_slippage_not_compared + own.stop_not_compared,
            carried.stop_slippage_compared + own.stop_compared,
        )


def notes_charges(charges: float, outputs: Mapping[str, Mapping[str, Any]]) -> float:
    """Rupees the strategies' contract notes charged their fills, what their results
    charge: `charges`, the fills' modelled charges, each rounded to the paisa, with the
    rounding adjustment of each settled note in `outputs` (each kernel session's
    `finish()` tables, by strategy, read after its stop settled its notes) added. It covers
    the day only at the day's first start, whose results hold the day's fills whole."""
    adjustments = 0.0
    for output in outputs.values():
        notes = output["charges"]["notes"]
        settled = zip(notes["adjustment"], notes["settled"], strict=True)
        adjustments += sum(float(amount) for amount, posted in settled if posted)
    return round(charges + adjustments, 2)


def _own(session: Any, unprotected_s: float) -> SessionFigures:
    """This start's own counts so far, with `unprotected_s` seconds unprotected."""
    return SessionFigures(
        session.rejected, session.mismatches, unprotected_s, session.parts.sender.undelivered
    )


def _values(figures: Any) -> tuple[Any, ...]:
    return tuple(getattr(figures, field.name) for field in fields(figures))


def _earlier(name: str, state: Mapping[str, Any]) -> Earlier:
    """Strategy `name`'s `Earlier` from its saved state: the instrument and the MIS
    position its kernel session's live state held (none when it saved none). Raises
    LiveError for a live state that cannot be read."""
    live_state = state.get("live_state") or {}
    try:
        units = sum(
            float(position["quantity_units"])
            for position in live_state.get("positions", [])
            if position.get("instrument") == INSTRUMENT and position.get("product") == LIVE_PRODUCT
        )
        instruments = live_state.get("instruments") or [{}]
        symbol = str(instruments[INSTRUMENT].get("symbol", ""))
        exchange = str(instruments[INSTRUMENT].get("exchange", ""))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        raise _unreadable(name, "its saved position") from None
    return Earlier(name, symbol, exchange, round(units))


def _entries(name: str, state: Mapping[str, Any]) -> int:
    """The entries of the day strategy `name`'s kernel had counted at its last save (its
    live state's "trades": those filled and those still working); 0 when it saved none.
    Raises LiveError for a count that cannot be read."""
    try:
        trades = ((state.get("live_state") or {}).get("risk") or {}).get("trades")
        if trades is None:
            return 0
        return int(trades["filled"]) + len(trades["waiting"])
    except (AttributeError, KeyError, TypeError, ValueError):
        raise _unreadable(name, "its saved count of entries") from None


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
    """The error of a saved `record` of strategy `name` (or of the session) that cannot be
    read; a string names it instead of showing it."""
    shown = record if isinstance(record, str) else repr(record)
    return LiveError(
        f"the day's figures journal L saved for {name} cannot be read ({shown}): the "
        "day's summary would be wrong. Check the account in OpenAlgo, then start under "
        "another account name (account=)"
    )


__all__ = [
    "FIGURES",
    "SESSION_KEY",
    "DayFigures",
    "Earlier",
    "SessionFigures",
    "StrategyFigures",
    "notes_charges",
]
