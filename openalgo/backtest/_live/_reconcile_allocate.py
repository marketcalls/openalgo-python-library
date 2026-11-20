"""The allocation of a difference other orders' trades explain on an instrument several
strategies of the session share (18-multi-strategy 4.6 and 4.7; built in unit M6 for the
comparisons of a session and of a restart).

A broker nets the members of an instrument group into one position, so a sale or a buy by
hand that the trade book explains (`Kind.EXPLAINED_SHARED`: the broker holds fewer units
on the group's side than the members' sum, or none) is taken from the group, never from
one member alone, by one fixed rule (`allocate`):

1. An external sale closes long members, an external buy short members.
2. Among those, the member whose current position opened most recently takes first, up to
   its units; then the next. Ties go by call order. A member whose position's opening is
   not known (it opened after a restart's saved state) counts as the most recent.
3. Units are whole: every member's units and every external trade are lot multiples on
   F&O, so the shares of whole members stay multiples.

The newest of the other orders' trades go to the first share, a share's price being the
trades it takes averaged by units, as a leg adopts the newest first
(`_reconcile_rules.adoption_price`).

Each member's share is then adopted as a leg's difference is (16-session-design 5.3 rows 1
and 2, `_reconcile_act.Corrections`): `apply_external_fill` of the share at its price,
with which the kernel fits the member's broker-held stop to what is left, cut down, or
cancelled when nothing is left. The kernel keeps a leg's stop on its leg's units, so this
is the cut of kept stops of 4.7 for a difference the members absorb: no stop is left
covering more than its member holds, so the stops together never cover more than the
broker holds. With the broker on the other side of the group (`Kind.OTHER_SIDE`) nothing
is adopted and every member's stop is cancelled, a leg's rule. Every member's entries are
locked, whether it took a share or not, and its RECONCILE_MISMATCH alert names the
allocation (`allocation_words`).

Quantities are signed units (long above 0), prices rupees, times UTC nanoseconds. Pure.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._group_rules import GroupFigure, Share, and_join
from ._reconcile_rules import Trade

#: A member as the allocation sees it: its strategy name, its signed units, and when its
#: current position opened (UTC nanoseconds, None when not known); members in call order.
Holder = tuple[str, int, int | None]


def allocate(
    holders: Sequence[Holder], difference: int, external: Sequence[Trade]
) -> tuple[Share, ...]:
    """The shares of `difference` (the broker's net less the members' sum, signed) among
    `holders` (call order) by the module docstring's rule, in the order they take, then
    the members it takes nothing from, in call order. `external` are the other orders'
    trades of the instrument, oldest first, which price each share."""
    closing = [(at, holder) for at, holder in enumerate(holders) if holder[1] * difference < 0]
    # The unknown opening first (the most recent), then the latest, then call order.
    closing.sort(key=lambda entry: (entry[1][2] is not None, -(entry[1][2] or 0), entry[0]))
    side = "SELL" if difference < 0 else "BUY"
    trades = _newest_first(external, side)
    left, shares = abs(difference), []
    for _, (name, units, _opened) in closing:
        take = min(abs(units), left)
        left -= take
        price = None
        if take:
            # Trades the members adopted before may leave fewer units on the side than the
            # difference: a share past them is priced as a leg's adoption is.
            price = _price(trades, take) or _price(_newest_first(external, side), take)
        shares.append(Share(name, units, take if difference > 0 else -take, price))
    taking = {share.name for share in shares}
    shares += [Share(name, units, 0) for name, units, _opened in holders if name not in taking]
    return tuple(shares)


def allocation_words(figure: GroupFigure, name: str) -> str:
    """What the allocation of `figure` does, in the words of strategy `name`'s
    RECONCILE_MISMATCH alert: the units other orders sold or bought, the share each member
    takes and what it leaves it, the most recently opened first; that `name` keeps its
    units when it takes none; and that each share is adopted with its stop cut."""
    taken = [share for share in figure.shares if share.taken]
    units = sum(abs(share.taken) for share in taken)
    verb = "sold" if taken and taken[0].taken < 0 else "bought"
    parts = and_join([f"{s.name} {abs(s.taken)} ({_left(s.left)})" for s in taken])
    words = (
        f"{units} units {verb} by other orders are shared out, the most recently opened "
        f"position first: taken from {parts}"
    )
    own = figure.share(name)
    if own is not None and not own.taken:
        words += f"; {name} keeps its {abs(own.units)} units"
    return words + (
        "; each takes its share as its own, its stop cut down to what it is left with, or cancelled"
    )


def _newest_first(external: Sequence[Trade], side: str) -> list[list[float]]:
    """The trades of `external` on `side`, newest first, as [units, price] the shares
    take units from."""
    return [[trade.units, trade.price] for trade in reversed(external) if trade.side == side]


def _price(trades: list[list[float]], units: int) -> float | None:
    """The price of `units` taken from `trades` ([units, price], newest first), which
    lose them: averaged by units over what they hold; None when they hold none."""
    taken, value = 0.0, 0.0
    for trade in trades:
        part = min(trade[0], units - taken)
        trade[0] -= part
        taken, value = taken + part, value + part * trade[1]
        if taken == units:
            break
    return value / taken if taken else None


def _left(units: int) -> str:
    """What a share leaves its member with, in words."""
    return "flat" if units == 0 else f"{abs(units)} left"


__all__ = ["Holder", "allocate", "allocation_words"]
