"""The session invariants checked over each run's event stream and the fake OpenAlgo's log
(`session_invariants` 7 to 9; 16-session-design 7.1, unit 14).

7. **Every Modify and Cancel answered.** A Modify got exactly one of `apply_modify`,
   `apply_modify_refused` or its order's terminal news (none in a run that crashed before
   its close), a modify the exit guard still holds at the close included; a Cancel its
   order's terminal news, `apply_cancel_refused`, or the run's end.
8. **Nothing after a guard lock.** No order call and no book read by the run's clients in
   the fake's log after the entry it held when the run's mode guard began to lock (by the
   log's order, so a call at the very instant of the lock, or while it alerts, is one).
9. **Priced fills.** Every fill the kernel took, and every fill of other orders it adopted,
   has a positive price; a fill not flagged estimated is at a price the broker's fills of
   that order hold.

Prices are rupees, times UTC nanoseconds.
"""

from __future__ import annotations

from fake_openalgo import FakeOpenAlgo
from fake_openalgo.log import CALL
from session_trace import Run

from openalgo.backtest._live._venue_requests import ModifyRequest, request_fields

#: The order stream's terminal words.
NEWS_ENDED = frozenset({"complete", "cancelled", "rejected"})
#: The calls that send an order or read a book (8).
ORDER_CALLS = frozenset({"placeorder", "modifyorder", "cancelorder"})
BOOK_READS = frozenset({"orderstatus", "orderbook", "tradebook", "positionbook", "funds"})
#: A price the broker's fills hold, within half a paisa.
PRICE_TOLERANCE = 0.005


def modifies_and_cancels_answered(run: Run) -> None:
    """7 (module docstring)."""
    events = run.events
    closed = any(e["kind"] == "closed" for e in events)
    ended = closed or run.outcome is None
    for at, request in enumerate(events):
        kind = request.get("type") if request["kind"] == "request" else None
        if kind not in ("ModifyRequest", "CancelRequest"):
            continue
        later = events[at + 1 :]
        answers = {
            e["applied"]
            for e in later
            if e["kind"] in ("sent", "not_sent", "settled")
            and e.get("request_id") == request["request_id"]
        }
        news = _ended(later, request["kernel_order"], request["strategy"])
        if kind == "ModifyRequest":
            applied = answers & {"modified", "modify_refused"}
            assert len(applied) <= 1, f"7.1: modify {request} answered twice: {applied}"
            assert applied or news or not closed, (
                f"7.1: modify {request} got no apply_modify, no refusal and no terminal news"
            )
        else:
            assert news or "cancel_refused" in answers or ended, (
                f"7.1: cancel {request} got no terminal news and no refusal"
            )
    if not closed:
        return
    # A modify the exit guard held and never handed over has no request line: it is
    # answered only by its order's terminal news.
    for held in run.session.reconcile.guard.left():
        if not isinstance(held, ModifyRequest):
            continue
        fields = request_fields(held)
        news = _ended(events, fields["kernel_order"], fields["strategy"])
        assert news, f"7.1: modify {fields}, held by the exit guard, got no answer"


def nothing_after_a_guard_lock(run: Run, fake: FakeOpenAlgo, end: int | None) -> None:
    """8: the calls the fake logged after the entry it held as the lock began, to the
    run's end."""
    if run.locked_at is None:
        return
    late = [
        (entry.what, entry.detail.get("client"), entry.ts_ns)
        for entry in fake.log.entries[run.locked_at : end]
        if entry.kind == CALL and entry.detail.get("client") in run.clients
        and entry.what in ORDER_CALLS | BOOK_READS
    ]  # fmt: skip
    assert late == [], f"7.1: calls after the guard lock (log entry {run.locked_at}): {late}"


def priced_fills(run: Run, fake: FakeOpenAlgo) -> None:
    """9 (module docstring)."""
    for adopted in (e for e in run.events if e["kind"] == "adopted"):
        assert adopted["price"] > 0, f"7.1: the kernel adopted a fill at no price: {adopted}"
    fills = [e for e in run.events if e["kind"] == "fill"]
    broker: dict[str, list[float]] = {}
    for fill in fake.fills(run.mode):
        broker.setdefault(fill.order_id, []).append(fill.price)
    for fill in fills:
        assert fill["price"] > 0, f"7.1: the kernel took a fill at no price: {fill}"
        prices = broker.get(fill["order_id"])
        if fill.get("estimated") or not prices:
            continue
        low, high = min(prices) - PRICE_TOLERANCE, max(prices) + PRICE_TOLERANCE
        assert low <= fill["price"] <= high, (
            f"7.1: fill {fill} is not flagged estimated, but the broker filled the order at "
            f"{prices}"
        )


def _ended(events: list[dict], kernel_order: int, strategy: str) -> bool:
    """Whether `events` hold terminal news of `strategy`'s kernel order `kernel_order`."""
    return any(
        e["kind"] == "news" and e.get("kernel_order") == kernel_order
        and e.get("status") in NEWS_ENDED and e.get("strategy") == strategy
        for e in events
    )  # fmt: skip


__all__ = [
    "BOOK_READS",
    "NEWS_ENDED",
    "ORDER_CALLS",
    "modifies_and_cancels_answered",
    "nothing_after_a_guard_lock",
    "priced_fills",
]
