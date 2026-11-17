"""The kill switch's position-book reads (16-session-design 5.4; units 11 and 12): which read
counts for a fired strategy, and how often the kill switch asks for one. Split from
`_kill_switch`, which acts on what these give.

**Which read counts.** A strategy's state is its position (units and tranches) and its
working orders (`look`). A read counts for it only when the read began at or after its
state, as the kill switch last saw it, began: the fire, a fill, its cancels ending, its exit
filling. Each read is judged once (`judged`): the next judgement takes a read begun at
least `KILL_READ_EVERY_S` after it, so reads in a row are that far apart, and a new state
starts a new run of them. Any reader's read counts: the stops supervisor's for a held
cancel, and the reconciler's, whose every position book the kill switch is given
(`KillSwitch.positions`).

**Asking for reads** (`ReadPace`). At most one at a time: the next is asked for once a read
begun at or after the last ask has come and `KILL_READ_EVERY_S` have passed since that
ask. A read the worker never answers is asked for again after `KILL_READ_LOST_AFTER_S`.

Times are UTC nanoseconds, quantities units. Used by the loop only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._engine import call
from .._layout import NANOS_PER_SECOND
from ._plan_catch_up import INSTRUMENT
from ._session_trader import Trader
from ._venue_requests import LIVE_PRODUCT

#: Seconds between two asks for a position-book read, and at least between the starts of
#: two reads judged in a row: the stops supervisor's pace for a held cancel
#: (16-session-design 4.11).
KILL_READ_EVERY_S = 2.0
#: Seconds between two reads for a strategy whose disagreement was alerted: the user has
#: been told, and a book that stays apart is not read every few seconds all day.
KILL_READ_BACKOFF_S = 30.0
#: Seconds after which a read asked for and never answered counts as lost and is asked
#: for again: well past the orders client's own wait, the SDK's 120 s (`clients`), and a
#: call ahead of it on the worker. The worker answers every read (or the session stops),
#: so this only keeps a lost answer from silencing the kill switch.
KILL_READ_LOST_AFTER_S = 300.0

_NS = NANOS_PER_SECOND

#: A position as the kill switch tells one from another: its signed units and the
#: (number, opening time) of each open tranche. Numbers start again with each position,
#: so a position closed and opened again is told by its tranches' opening times.
Held = tuple[float, tuple[tuple[int, int], ...]]
#: A strategy's state as the kill switch watches it: its position and its working orders.
State = tuple[Held, tuple[int, ...]]


@dataclass
class Kill:
    """One fired strategy: why and when; its `state` as last seen and the instant from
    which a read counts for it (`since_ns`); the broker's units in the read that counts
    and is not judged yet, and when that read began (None for none); the position its last
    exit request was for; the reads in a row that disagreed about its state and the
    broker's figure in the last of them; the instant before which no read is asked for it
    after its alert; the position its last disagreement alert was about; whether its flat
    state was reported."""

    reason: str
    fired_ns: int
    state: State | None = None
    since_ns: int = 0
    broker: int | None = None
    read_ns: int | None = None
    exit_for: Held | None = None
    disagreements: int = 0
    noted: int | None = None
    quiet_until_ns: int = 0
    alerted_for: Held | None = None
    reported: bool = False


@dataclass
class ReadPace:
    """The kill switch's asks for a position-book read (module docstring): when it last
    asked, and whether a read begun since has come."""

    asked_ns: int | None = None
    answered: bool = True

    def due(self, now_ns: int) -> bool:
        """True when another read may be asked for at `now_ns`."""
        if self.asked_ns is None:
            return True
        wait_s = KILL_READ_EVERY_S if self.answered else KILL_READ_LOST_AFTER_S
        return now_ns >= self.asked_ns + seconds_ns(wait_s)

    def ask(self, now_ns: int) -> None:
        """A read was asked for at `now_ns`."""
        self.asked_ns, self.answered = now_ns, False

    def came(self, read_ns: int) -> None:
        """A read begun at `read_ns` came: it answers an ask made at or before then."""
        if self.asked_ns is not None and read_ns >= self.asked_ns:
            self.answered = True


def look(trader: Trader, kill: Kill, now_ns: int) -> State:
    """The strategy's state at `now_ns`, noted: a new one restarts what counts as its read
    and its run of disagreeing reads, and is looked at without the back-off of an earlier
    alert."""
    state = (held(trader), tuple(sorted(trader.working())))
    if state != kill.state:
        kill.state, kill.since_ns = state, now_ns
        kill.broker = kill.read_ns = None
        kill.disagreements = kill.quiet_until_ns = 0
    return state


def judged(kill: Kill, read_ns: int) -> None:
    """The read that counts for `kill`, begun at `read_ns`, is judged: the next one to
    count begins at least `KILL_READ_EVERY_S` after it, so a book that lags a fill by less
    catches up between two reads in a row."""
    kill.since_ns, kill.read_ns = read_ns + seconds_ns(KILL_READ_EVERY_S), None


def held(trader: Trader) -> Held:
    """The strategy's position (`Held`): flat with no tranche when it never opened."""
    position = call(trader.engine.position, INSTRUMENT, LIVE_PRODUCT)
    if position is None:
        return 0.0, ()
    tranches = tuple(
        (int(number), int(opened_at)) for number, _, _, opened_at in position["tranches"]
    )
    return float(position["quantity_units"]), tranches


def seconds_ns(seconds: float) -> int:
    """`seconds` as nanoseconds."""
    return round(seconds * _NS)


__all__ = [
    "KILL_READ_BACKOFF_S",
    "KILL_READ_EVERY_S",
    "KILL_READ_LOST_AFTER_S",
    "Held",
    "Kill",
    "ReadPace",
    "State",
    "held",
    "judged",
    "look",
    "seconds_ns",
]
