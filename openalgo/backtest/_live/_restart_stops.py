"""The alert of a leg a restart resumes without its stop (16-session-design 3.1 step 10;
unit 12), split from `_restart_books` so each module keeps one job: an ERROR alert for each
leg that resumes with a position whose saved stop does not work in the order book, saying
what the restart does about it by how the start judged the leg against the broker's books
(`missing_stop_words`): its stop is placed again at the end of step 11, unless the books
show the broker holding fewer units that no trade explains, or the other side (the exit
guard then holds a stop for the journal's units), or other orders closed the position.

Quantities are signed units. Run by the main thread at the start.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ._group_rules import and_join
from ._reconcile_rules import Kind
from ._types import AlertKind

if TYPE_CHECKING:
    from ._restart_books import LegStart


def alert_missing_stops(spec: Any, legs: Mapping[str, LegStart], journal: Any, alerts: Any) -> None:
    """An ERROR alert for each leg that resumes with a position whose saved stop does not
    work in the order book (module docstring)."""
    for each in spec.strategies:
        leg = legs[each.name]
        stop = None if leg.saved is None else leg.saved.stop
        if leg.journal_units == 0 or stop is None:
            continue
        order_id = stop.order_id or journal.order_id_of(stop.intent.intent_id) or ""
        row = leg.books.row(order_id) if order_id else None
        if row is not None and row.working:
            continue
        alerts.send(
            AlertKind.ERROR,
            f"{each.name}: {leg.journal_units} units of {each.symbol} {each.exchange} have no "
            f"working stop at the broker (stop {order_id or 'not sent'}): "
            f"{missing_stop_words(leg, each.name)}",
        )


def missing_stop_words(leg: LegStart, name: str) -> str:
    """What the restart does about strategy `name`'s resumed leg's missing stop, by how
    the start judged the leg against the broker's books (module docstring); a member of a
    shared instrument's words name the group and the sum it was judged by, and a member
    an explained difference leaves flat (its share, `_reconcile_allocate`) is closed."""
    shown = f"OpenAlgo's books show {leg.broker_units} units"
    if leg.group is not None:
        shown += f" for {and_join(leg.group.members)} together, who hold {leg.group.units}"
    share = None if leg.group is None else leg.group.share(name)
    if leg.kind is Kind.EXPLAINED_FLAT or (share is not None and share.left == 0):
        return (
            "OpenAlgo's books show the position closed by other orders, which the engine "
            "takes as its own, so no stop is placed"
        )
    if leg.kind is Kind.UNEXPLAINED:
        return (
            f"{shown}, which no trade explains, so no stop is placed for the journal's units "
            "while they disagree: check the account in OpenAlgo"
        )
    if leg.kind is Kind.OTHER_SIDE:
        return (
            f"{shown}, the other side, so no stop is placed: close the position by hand in OpenAlgo"
        )
    return "it is placed again once the engine is restored"


__all__ = ["alert_missing_stops", "missing_stop_words"]
