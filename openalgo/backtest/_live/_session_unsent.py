"""The placements a live strategy's kernel makes that the loop takes and never sends
(16-session-design 2.5 E and 6.6; unit 8; 2.4.0 item P2-3 a, review round 1). `to_send`
gives `Trader.commands` the rest.

- **Withdrawn.** A placement the kernel cancelled in the batch that made it (the safe
  stop's cancel of an entry decided in the same loop step) never reached the broker: the
  kernel is told it was cancelled (`apply_cancel`).
- **Refused while entries are locked.** The kernel judges its entry gates (`venue_gates`,
  `live_lock` among them) when an order is submitted. A reversal (`buy(...,
  reverse=True)` beyond a short, or the sale beyond a long) is an entry the kernel holds
  until its position's stop has been cancelled (`venue_owner`), and it then places the
  order without judging the gates again. A lock set meanwhile, because the feed died,
  the books differed, an exit failed or the account lost its limit, would not stop the
  part that opens the other side, which the lock is there to stop. So the placement of
  an entry or add taken while any reason locks the strategy's entries is refused: the
  kernel is told the venue rejected it with `live_lock`, the code a submission gets under
  the lock. A reversal is refused whole, as at submission under a lock: its closing part
  is not sent either, since the kernel cannot cut an order it has placed down to the
  units that close the position. The position keeps its leg, and the session places the
  leg's stop again (`_stops_bare`). Stops and exits always pass.
- **Refused as a resting reversal** (2.4.0 item P3-2). A reversal that rests (`reverse=
  True` on a LIMIT, SL or SL-M order) works at the broker beside the position's stop, both
  on the closing side, and both can fill: a short of N with its stop buying N and a LIMIT
  buy of 2N resting ends long 2N, not long N, and an engine-held target firing (a MARKET
  exit) while it rests overfills the same way. So a live session never sends one: its
  placement is refused in the loop step of the decision that made it, before anything
  reaches the broker, and the kernel is told the venue rejected it (`venue`, since no
  kernel check makes this refusal; the strategy sees it in `on_order` before its next
  decision). The strategy's context notes each such order as the kernel takes it
  (`_series_context.LiveRunContext.resting_reversals`), since the kernel's order no
  longer says `reverse`. The session's warnings say, once per strategy, to send a
  reversal as a MARKET order or to exit and enter separately. The backtest is unchanged.

  The 2.4.0 ruling asked for the refusal at submission; it comes at the handover
  instead, after `on_bar` has returned. The kernel takes no venue result while it is
  paused at a bar close, where the strategy decides, and has no reject code of its own
  for this refusal. So inside `on_bar` the order reads Submitted, the decisions file lists
  it so (as it lists an order the broker refuses later), and until the decision ends the
  kernel's entry gates count it as working. The log names it as it names a rejection at
  submission, with its side, units and reject code.

Each is logged and written to the event stream ("withdrawn", "refused_locked",
"refused_resting_reversal"). Used by the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from .._engine import call
from ._order_state import ENTRY_PURPOSES
from ._session_decisions import units_text
from ._types import LOGGER_NAME
from ._venue_requests import ROLE_PURPOSES

#: The reject code of an entry or add while entries are locked (the kernel's `live_lock`).
LOCKED_CODE = "live_lock"
#: The reject code of a resting reversal the session refuses: the venue's, since no kernel
#: check makes this refusal (module docstring).
RESTING_REVERSAL_CODE = "venue"
#: The session's warning of a strategy whose resting reversal it refused (module docstring).
RESTING_REVERSAL_WORDS = (
    "a resting reversal (reverse=True on a LIMIT, SL or SL-M order) was refused and not "
    "sent: at the broker it would work beside the position's stop, both closing the "
    "position, and both could fill, leaving a larger position than the one meant. A live "
    "session takes a reversal as a MARKET order only: submit the reversal as a MARKET "
    "order, or exit and enter separately"
)

logger = logging.getLogger(LOGGER_NAME)


def to_send(
    engine: Any,
    strategy: str,
    commands: Sequence[dict[str, Any]],
    *,
    event: Callable[..., None],
    reversals: set[int],
    warn: Callable[[str], None],
) -> list[dict[str, Any]]:
    """`commands` (`engine.take_commands()`'s dicts, of `strategy`'s kernel session
    `engine`) without those of the orders withdrawn or refused (module docstring), each
    such order ended in the kernel; `event` writes the event stream, `reversals` holds the
    kernel orders of the strategy's resting reversals not refused yet (each refused one,
    and one that ended before its placement came, taken out), and `warn` keeps a warning
    of the session."""
    kept = _withdraw_cancelled(engine, strategy, commands, event)
    kept = _refuse_resting_reversals(engine, strategy, kept, reversals, event, warn)
    return _refuse_locked(engine, strategy, kept, event)


def _withdraw_cancelled(
    engine: Any, strategy: str, commands: Sequence[dict[str, Any]], event: Callable[..., None]
) -> list[dict[str, Any]]:
    placed = {int(c["order_id"]) for c in commands if c.get("command") == "place"}
    cancelled = {int(c["order_id"]) for c in commands if c.get("command") == "cancel"}
    withdrawn = placed & cancelled
    for kernel_order in sorted(withdrawn):
        call(engine.apply_cancel, kernel_order)
        logger.info("%s: order o%d was cancelled before it was sent", strategy, kernel_order)
        event("withdrawn", strategy=strategy, kernel_order=kernel_order)
    return [c for c in commands if int(c["order_id"]) not in withdrawn]


def _refuse_resting_reversals(
    engine: Any,
    strategy: str,
    commands: list[dict[str, Any]],
    reversals: set[int],
    event: Callable[..., None],
    warn: Callable[[str], None],
) -> list[dict[str, Any]]:
    if not reversals:
        return commands
    placed = {int(c["order_id"]) for c in commands if c.get("command") == "place"}
    refused = sorted(placed & reversals)
    for kernel_order in refused:
        call(engine.apply_reject, kernel_order, RESTING_REVERSAL_CODE)
        order = call(engine.order, kernel_order)
        logger.info(
            "%s: order o%d rejected before it was sent: %s %s units, %s (a resting reversal, %s)",
            strategy,
            kernel_order,
            order["side"],
            units_text(order["quantity_units"]),
            RESTING_REVERSAL_CODE,
            order["order_type"],
        )
        event("refused_resting_reversal", strategy=strategy, kernel_order=kernel_order)
        warn(f"{strategy}: {RESTING_REVERSAL_WORDS}")
    # Refused, or ended before its placement came: no placement of it will come.
    reversals.intersection_update(call(engine.working_orders))
    return [c for c in commands if int(c["order_id"]) not in refused]


def _refuse_locked(
    engine: Any, strategy: str, commands: list[dict[str, Any]], event: Callable[..., None]
) -> list[dict[str, Any]]:
    opening = sorted({int(c["order_id"]) for c in commands if _opens(c)})
    # Read only for an entry or add: most batches hold none, and every loop step takes one.
    locks = sorted(call(engine.entry_locks)) if opening else []
    if not locks:
        return commands
    for kernel_order in opening:
        call(engine.apply_reject, kernel_order, LOCKED_CODE)
        logger.warning(
            "%s: order o%d opens a position while entries are locked (%s): refused, not sent",
            strategy,
            kernel_order,
            ", ".join(locks),
        )
        event("refused_locked", strategy=strategy, kernel_order=kernel_order, locks=locks)
    return [c for c in commands if int(c["order_id"]) not in opening]


def _opens(command: dict[str, Any]) -> bool:
    """True for the placement of an entry or add."""
    role = command.get("role")
    return command.get("command") == "place" and ROLE_PURPOSES.get(role) in ENTRY_PURPOSES


__all__ = ["LOCKED_CODE", "RESTING_REVERSAL_CODE", "RESTING_REVERSAL_WORDS", "to_send"]
