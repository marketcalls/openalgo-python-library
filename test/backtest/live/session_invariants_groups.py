"""The session invariants of instrument groups (18-multi-strategy 5.1; unit M6), checked over
every session test with the others (`session_invariants`), at every change of an order at
the fake while a run is open (`OrderChange.members`). A strategy alone on its instrument is
a group of one: each holds for it as for a group.

- **I4, the sum rule.** At every change, the fake's net position of an instrument in its
  book equals the sum of the members' kernel units, plus the units of the session's own
  orders the fake has filled and their fill ledgers have not given a kernel yet (all of
  them for an order of the open run not bound yet; none of an earlier run's order no
  venue map holds now, which the restored kernels hold), plus the units of other
  clients' orders the fake has filled that the members have not adopted. So no fill
  reaches a kernel twice, or a kernel whose order it is not, and no kernel's position
  strays from the fills its own orders made.
- **I5, isolation.** No request of a member is caused by another member's news (the
  design's cause of each request, 18-multi-strategy 5.1), checked where a cause can pass
  from one member to another: (1) every order word a kernel takes is of an order its own
  venue map binds and the fake records under its strategy's name; (2) at every change no
  broker order is bound in two members' venue maps, and every order a member's venue map
  binds is recorded at the fake under that member's name; (3) no session call makes a
  member's kernel act (an exit, a cancel, a protect) while another member's order word
  is applied (`Run.strayed`, noted by `session_hooks`). A member's requests come from its
  own kernel, which then takes only its own orders' words, and the session never acts on
  one member for another's word; the kill switch, the exit guard and the group's
  corrections act from the books' reads, never from an order word.
- **I6, one side.** At no change do two members of a live group hold positions on
  opposite sides.

Quantities are signed units.
"""

from __future__ import annotations

from collections.abc import Iterable

from session_trace import OrderChange, Run

Instrument = tuple[object, str, str]


def sum_rule(changes: Iterable[OrderChange], mine: frozenset[str]) -> None:
    """I4 (module docstring) over `changes` (in order), `mine` the session's clients."""
    own: dict[str, OrderChange] = {}
    others: dict[Instrument, int] = {}
    for change in changes:
        if change.client in mine:
            own[change.order_id] = change
        elif change.fill_units:
            signed = change.fill_units if change.action == "BUY" else -change.fill_units
            others[change.instrument] = others.get(change.instrument, 0) + signed
        if change.members is None:
            continue
        held = sum(member.units for member in change.members)
        waiting = _not_given(change, own.values())
        adopted = sum(member.adopted for member in change.members)
        expected = held + waiting + others.get(change.instrument, 0) - adopted
        assert change.position == expected, (
            f"18-multi-strategy I4: the fake holds {change.position} {change.symbol} after "
            f"{change}, against the members' {held}, {waiting} own units filled and not yet "
            f"given, and {others.get(change.instrument, 0) - adopted} of other orders not "
            f"adopted ({change.members})"
        )


def one_side(changes: Iterable[OrderChange]) -> None:
    """I6 (module docstring)."""
    for change in changes:
        if change.members is None:
            continue
        sides = {member.units > 0 for member in change.members if member.units}
        assert len(sides) <= 1, (
            f"18-multi-strategy I6: members of {change.symbol} hold opposite sides after "
            f"{change} ({change.members})"
        )


def isolation(run: Run, changes: Iterable[OrderChange], mine: frozenset[str]) -> None:
    """I5 (module docstring) of `run`, over `changes` (in order), `mine` the session's
    clients."""
    assert not run.strayed, f"18-multi-strategy I5: {run.strayed}"
    changes = list(changes)
    tags = {change.order_id: change.strategy for change in changes if change.client in mine}
    for change in changes:
        owners: dict[str, list[str]] = {}
        for member in change.members or ():
            for order_id, _ in member.given:
                owners.setdefault(order_id, []).append(member.strategy)
        for order_id, names in owners.items():
            assert len(names) == 1, (
                f"18-multi-strategy I5: order {order_id} is bound to both "
                f"{' and '.join(names)} after {change}"
            )
            tag = tags.get(order_id, names[0])
            assert tag == names[0], (
                f"18-multi-strategy I5: {names[0]}'s venue map binds order {order_id}, "
                f"which the fake records for {tag}, after {change}"
            )


def _not_given(change: OrderChange, own: Iterable[OrderChange]) -> int:
    """The signed units of the session's `own` orders (each as it last stood) of
    `change`'s instrument the fake has filled and no kernel has been given, as the
    change's members show their ledgers (module docstring)."""
    given = {order_id: units for m in change.members or () for order_id, units in m.given}
    waiting = 0
    for order in own:
        if order.instrument != change.instrument:
            continue
        if order.order_id in given:
            missing = order.filled - given[order.order_id]
        elif order.client in change.run_clients:
            missing = order.filled
        else:
            missing = 0
        waiting += missing if order.action == "BUY" else -missing
    return waiting


__all__ = ["isolation", "one_side", "sum_rule"]
