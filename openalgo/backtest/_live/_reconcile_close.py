"""What the close says of a leg whose difference with OpenAlgo's books no later read will
confirm (16-session-design 5.4 "As built", rounds 7 to 9; unit 12).

The reconciler keeps the position difference a leg's last comparison saw for the first
time (`_reconcile_findings.Sighting`: the engine's signed units as compared and
OpenAlgo's). Once the close has printed its last lines no later read confirms it, so the
close reports it (`differing`). The engine's units may have moved since that comparison:
an adoption of other orders' trades needs a second comparison, which ends the sighting,
so they moved only by the fills of the engine's own orders, and each of those was a fill
at the broker that moved OpenAlgo's books by the same units. OpenAlgo's figure is carried
by them (`Differing.broker_units`): a stop of 10 that fills after a read that showed 6
against the engine's 10 leaves -4 there and 0 in the engine. The words name the read's
figures, the fills since and what they leave, so no figure is called the engine's once
the engine holds other units (`Differing.words`, `Differing.summary`).

A leg OpenAlgo's books show flat by then (`Differing.flat_at_openalgo`) is no open
position: there is nothing to close by hand, and whatever of it still works at the
broker (its stop for the engine's units, say) is the operator's to cancel
(`NOTHING_TO_CLOSE`).

A leg no comparison judged after a fill of its own orders landed during a read
(`Reconciler.unjudged`, round 9) is never called flat or agreeing unqualified: the close
names the engine's units, and the carried figure when a sighting exists, as not confirmed
(`NOT_CONFIRMED`), and one alert per such leg asks the operator to check OpenAlgo's position
book (`not_confirmed_alert`). The shared alert kinds have no warning: it is a
RECONCILE_MISMATCH alert (`NOT_CONFIRMED_ALERT`), which the alerts log at WARNING level,
apart from the close's ERROR alerts.

Quantities are signed whole units. Used by the close only (`_session_close`,
`_stops_after`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._reconcile_findings import Sighting
from ._session_trader import Trader
from ._types import AlertKind

#: How the close names a difference no later read confirms, with what it did about it.
DIFFERS = (
    "OpenAlgo's books showed {broker} against the engine's {engine} at {read}{since}, and "
    "the session ended before they agreed: nothing was sent for the difference"
)
#: The read a sighting came from: the last one, or, once a later read could not judge the
#: leg (`NOT_CONFIRMED`), the last one to judge it.
LAST_READ = "the last read"
LAST_JUDGING_READ = "the last read to judge it"
#: How it names the engine's own fills after that read, and what they leave.
SINCE = (
    "; the engine's own fills ({fills}) came after it, which would leave {broker} in "
    "OpenAlgo's books against the engine's {engine}"
)
#: How the DAILY_SUMMARY names OpenAlgo's figure carried by those fills.
SINCE_SUMMARY = " by {read} and the engine's own fills since"
#: How the close qualifies a leg no read judged after a fill of its own orders landed
#: during a read (round 9).
NOT_CONFIRMED = (
    "not confirmed, since a fill of its own orders landed during a read of the books and no "
    "later read judged the position"
)
#: What the alert of such a leg asks of the operator, and its kind (module docstring).
CHECK_THE_POSITION_BOOK = "check OpenAlgo's position book"
NOT_CONFIRMED_ALERT = AlertKind.RECONCILE_MISMATCH
#: What the alert of a leg OpenAlgo's books show flat asks of the operator.
NOTHING_TO_CLOSE = (
    "no position is left to close by hand: check the order book in OpenAlgo, cancelling "
    "what still works"
)


@dataclass(frozen=True)
class Differing:
    """A leg's difference with OpenAlgo's books at the close (module docstring): `seen`,
    as its last comparison saw it, `engine_units`, the engine's signed units now, and
    `unjudged`, True when no read judged the leg after a fill of its own orders landed
    during a read (`NOT_CONFIRMED`)."""

    seen: Sighting
    engine_units: int
    unjudged: bool = False

    @property
    def fills_since(self) -> int:
        """The signed units of the engine's own fills since the comparison (a sale takes
        units away)."""
        return self.engine_units - self.seen.engine_units

    @property
    def broker_units(self) -> int:
        """OpenAlgo's signed units by the comparison and the engine's own fills since."""
        return self.seen.broker_units + self.fills_since

    @property
    def flat_at_openalgo(self) -> bool:
        """True when OpenAlgo's books hold no position of the leg by then."""
        return self.broker_units == 0

    @property
    def read(self) -> str:
        """The read the sighting came from, in words (`LAST_READ`)."""
        return LAST_JUDGING_READ if self.unjudged else LAST_READ

    def words(self) -> str:
        """The difference in a note's and an alert's words (`DIFFERS`): the read's
        figures, then, when the engine's own fills came after it, those fills and the
        figures they leave; last, when no read judged the leg after a fill, that it is not
        confirmed."""
        since = ""
        if self.fills_since:
            since = SINCE.format(
                fills=_fills(self.fills_since),
                broker=_held(self.broker_units),
                engine=self.engine_units,
            )
        words = DIFFERS.format(
            broker=_held(self.seen.broker_units),
            engine=self.seen.engine_units,
            read=self.read,
            since=since,
        )
        return f"{words}; {NOT_CONFIRMED}" if self.unjudged else words

    def figures(self) -> str:
        """OpenAlgo's figure now, and how it is known when the engine's own fills came
        after the read."""
        how = SINCE_SUMMARY.format(read=self.read) if self.fills_since else ""
        if self.unjudged and not how:
            how = f" by {self.read}"
        return f"{_held(self.broker_units)} in OpenAlgo's books{how}"

    def summary(self, trader: Trader) -> str:
        """`trader`'s leg in the DAILY_SUMMARY's words: OpenAlgo's figure now (`figures`),
        the engine's now, and whether it is confirmed."""
        words = (
            f"{trader.name} {trader.symbol} {trader.exchange} {self.figures()}, the engine's "
            f"{self.engine_units}"
        )
        return f"{words}, {NOT_CONFIRMED}" if self.unjudged else words


def differing(reconcile: Any, trader: Trader) -> Differing | None:
    """`trader`'s difference with OpenAlgo's books that no later read will confirm, once the
    close has printed its last lines (`Reconciler.unconfirmed`), with the engine's units
    now; else None. `reconcile` is the session's `reconcile.Reconciler`."""
    seen = reconcile.unconfirmed(trader.name)
    if seen is None:
        return None
    return Differing(seen, round(trader.open_units()), reconcile.unjudged(trader.name))


def not_confirmed_summary(trader: Trader, engine_units: int) -> str:
    """A leg no read judged after a fill of its own orders landed during a read, with no
    difference seen before it, in the DAILY_SUMMARY's words: the engine's signed units,
    not confirmed (`NOT_CONFIRMED`)."""
    return (
        f"{trader.name} {trader.symbol} {trader.exchange} the engine's {engine_units} units, "
        f"{NOT_CONFIRMED}"
    )


def not_confirmed_alert(trader: Trader, engine_units: int, differs: Differing | None) -> str:
    """The alert of a leg no read judged after a fill of its own orders landed during a
    read (module docstring): the engine's signed units, OpenAlgo's figure when a difference
    was seen before (`differs`), and the position book to check."""
    figures = f"the engine holds {_held(engine_units)}"
    if differs is not None:
        figures += f" against {differs.figures()}"
    return (
        f"{trader.name} {trader.symbol} {trader.exchange}: {figures}, {NOT_CONFIRMED}: "
        f"{CHECK_THE_POSITION_BOOK}"
    )


def _held(units: int) -> str:
    """Signed units held, in words: "no position" for none."""
    return "no position" if units == 0 else f"{units} units"


def _fills(units: int) -> str:
    """The engine's own fills of `units` (signed) in words."""
    return f"{abs(units)} units {'bought' if units > 0 else 'sold'}"


__all__ = [
    "CHECK_THE_POSITION_BOOK",
    "DIFFERS",
    "LAST_JUDGING_READ",
    "LAST_READ",
    "NOTHING_TO_CLOSE",
    "NOT_CONFIRMED",
    "NOT_CONFIRMED_ALERT",
    "SINCE",
    "SINCE_SUMMARY",
    "Differing",
    "differing",
    "not_confirmed_alert",
    "not_confirmed_summary",
]
