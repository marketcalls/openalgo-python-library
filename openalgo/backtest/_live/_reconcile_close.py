"""What the close says of a leg whose difference with OpenAlgo's books no later read will
confirm (16-session-design 5.4 "As built", rounds 7 and 8; unit 12).

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

Quantities are signed whole units. Used by the close only (`_session_close`,
`_stops_after`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._reconcile_findings import Sighting
from ._session_trader import Trader

#: How the close names a difference no later read confirms, with what it did about it.
DIFFERS = (
    "OpenAlgo's books showed {broker} against the engine's {engine} at the last read"
    "{since}, and the session ended before they agreed: nothing was sent for the difference"
)
#: How it names the engine's own fills after that read, and what they leave.
SINCE = (
    "; the engine's own fills ({fills}) came after it, which would leave {broker} in "
    "OpenAlgo's books against the engine's {engine}"
)
#: How the DAILY_SUMMARY names OpenAlgo's figure carried by those fills.
SINCE_SUMMARY = " by the last read and the engine's own fills since"
#: What the alert of a leg OpenAlgo's books show flat asks of the operator.
NOTHING_TO_CLOSE = (
    "no position is left to close by hand: check the order book in OpenAlgo, cancelling "
    "what still works"
)


@dataclass(frozen=True)
class Differing:
    """A leg's difference with OpenAlgo's books at the close (module docstring): `seen`,
    as its last comparison saw it, and `engine_units`, the engine's signed units now."""

    seen: Sighting
    engine_units: int

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

    def words(self) -> str:
        """The difference in a note's and an alert's words (`DIFFERS`): the read's
        figures, then, when the engine's own fills came after it, those fills and the
        figures they leave."""
        since = ""
        if self.fills_since:
            since = SINCE.format(
                fills=_fills(self.fills_since),
                broker=_held(self.broker_units),
                engine=self.engine_units,
            )
        return DIFFERS.format(
            broker=_held(self.seen.broker_units), engine=self.seen.engine_units, since=since
        )

    def summary(self, trader: Trader) -> str:
        """`trader`'s leg in the DAILY_SUMMARY's words: OpenAlgo's figure now, how it is
        known when the engine's own fills came after the read, and the engine's now."""
        how = SINCE_SUMMARY if self.fills_since else ""
        return (
            f"{trader.name} {trader.symbol} {trader.exchange} {_held(self.broker_units)} in "
            f"OpenAlgo's books{how}, the engine's {self.engine_units}"
        )


def differing(reconcile: Any, trader: Trader) -> Differing | None:
    """`trader`'s difference with OpenAlgo's books that no later read will confirm, once the
    close has printed its last lines (`Reconciler.unconfirmed`), with the engine's units
    now; else None. `reconcile` is the session's `reconcile.Reconciler`."""
    seen = reconcile.unconfirmed(trader.name)
    return None if seen is None else Differing(seen, round(trader.open_units()))


def _held(units: int) -> str:
    """Signed units held, in words: "no position" for none."""
    return "no position" if units == 0 else f"{units} units"


def _fills(units: int) -> str:
    """The engine's own fills of `units` (signed) in words."""
    return f"{abs(units)} units {'bought' if units > 0 else 'sold'}"


__all__ = ["DIFFERS", "NOTHING_TO_CLOSE", "SINCE", "SINCE_SUMMARY", "Differing", "differing"]
