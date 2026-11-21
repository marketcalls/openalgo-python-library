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

Each is logged and written to the event stream ("withdrawn", "refused_locked"). Used by
the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from .._engine import call
from ._order_state import ENTRY_PURPOSES
from ._types import LOGGER_NAME
from ._venue_requests import ROLE_PURPOSES

#: The reject code of an entry or add while entries are locked (the kernel's `live_lock`).
LOCKED_CODE = "live_lock"

logger = logging.getLogger(LOGGER_NAME)


def to_send(
    engine: Any, strategy: str, commands: Sequence[dict[str, Any]], *, event: Callable[..., None]
) -> list[dict[str, Any]]:
    """`commands` (`engine.take_commands()`'s dicts, of `strategy`'s kernel session
    `engine`) without those of the orders withdrawn or refused (module docstring), each
    such order ended in the kernel; `event` writes the event stream."""
    kept = _withdraw_cancelled(engine, strategy, commands, event)
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


__all__ = ["LOCKED_CODE", "to_send"]
