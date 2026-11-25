"""A strategy that continues a position from an earlier day, judged at the start against the
broker's books (20-positional-live 4.2 step 5, 4.3, 7.3; the lead's rulings N5a-1 and
N5a-2), split from `_restart_books` so each module keeps one job.

- **Its journal position** (`carried_units`) is what its day-close record carries
  (`_carry_start`), when nothing was saved of it today: its units of its instrument and
  product, or none for a contract whose last trading day has passed (the exchange settled
  it; the restore records its exit at its last close).
- **Its judgement** (`carried_verdict`). The broker holding its units resumes it. A
  broker position below it by the units of a carried GTT that the start's read of the GTT
  book found fired or gone (its `IN_FLIGHT` finding among what the order worker posted
  during the start's reads, `gone_moves`) resumes it too: the GTT follower decides whether
  it fired (3.3; on a missed day, 4.3), and the leg is in flight for the reconciler
  meanwhile. Any other difference is a reconciliation mismatch (7.3): never a refusal of
  the start and never an exit, the leg resumes with its entries locked from the start and
  the exit guard open at the broker's units (`Reconciler.differs_at_start`). The trade
  book covers only today, so a difference made on an earlier day is never explained by
  its trades.

Quantities are signed whole units. Run by the main thread at the start.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._carry_start import CarryAtStart, expired_instruments, gtt_seqs
from ._gtt_types import GttEvent, GttNews, GttOutcome
from ._plan_strategies import StrategySpec
from ._reconcile_rules import Kind, RestartKind
from .inbox import Envelope


def carried_units(carry: CarryAtStart | None, spec: StrategySpec) -> int | None:
    """The journal position of strategy `spec` that a day-close record of an earlier day
    carries (module docstring), or None when it carries none."""
    carried = None if carry is None else carry.of(spec.name)
    if carry is None or carried is None:
        return None
    if spec.name in expired_instruments(carry, {spec.name: spec.instruments}):
        return 0
    leg = (spec.symbol, spec.exchange, spec.product)
    held = [p.units for p in carried.positions if (p.symbol, p.exchange, p.product) == leg]
    return round(sum(held))


def gone_moves(
    carry: CarryAtStart | None, name: str, deferred: Iterable[Envelope]
) -> frozenset[int]:
    """The signed moves of strategy `name`'s leg that its carried GTTs make by firing, of
    those the start's read of the GTT book found fired or gone (`deferred`, what the order
    worker posted during the start's reads: module docstring)."""
    carried = None if carry is None else carry.of(name)
    if carried is None:
        return frozenset()
    news = [envelope.message.news for envelope in deferred if _is_news(envelope)]
    gone = {
        found.seq
        for found in news
        if isinstance(found, GttOutcome) and found.event is GttEvent.IN_FLIGHT
    }
    seqs = gtt_seqs(carried.record)
    moves = set()
    for gtt in carried.record.carried.get("gtts", []):
        if seqs.get(int(gtt["gtt"])) in gone:
            units = round(float(gtt["terms"]["units"]))
            moves.add(units if gtt["terms"]["side"] == "BUY" else -units)
    return frozenset(moves)


def carried_verdict(
    journal_units: int, broker_units: int, judged: tuple[RestartKind, Kind], moves: frozenset[int]
) -> tuple[RestartKind, Kind]:
    """How a carried leg resumes (module docstring): `judged` is what the restart rules
    found of its `journal_units` against the `broker_units`, `moves` its carried GTTs'
    moves by firing that the start's read found fired or gone (`gone_moves`)."""
    verdict, kind = judged
    if verdict is RestartKind.RESUME:
        return judged
    if broker_units - journal_units in moves:
        return RestartKind.RESUME, Kind.AGREE
    if verdict is RestartKind.UNEXPLAINED_FLAT:
        return RestartKind.DIFFERS, kind
    return judged


def _is_news(envelope: Envelope) -> bool:
    return isinstance(envelope.message, GttNews)


__all__ = ["carried_units", "carried_verdict", "gone_moves"]
