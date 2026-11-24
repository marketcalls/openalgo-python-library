"""A live strategy's orders and position as the loop reads and changes them (16-session-
design 2.5 E, 2.7, 3.3; unit 8), split from `_session_trader` so each module keeps one
job.

- **Commands** (`commands`): the kernel's commands as the order worker's requests. A
  placement the kernel cancelled before the loop took it (the safe stop's cancel of an
  entry decided in the same loop step) is withdrawn: it is never sent, and the kernel is
  told it was cancelled (`apply_cancel`). An entry or add placed while the strategy's
  entries are locked (a reversal the kernel held for its stop's cancel) is refused, never
  sent (`_session_unsent`); one that still waits for the order worker when they lock is
  withdrawn (`_venue_withdraw`), and the worker refuses one it takes meanwhile
  (`_locked_entries`). A resting reversal is refused, never sent, and so is a resting
  order against the open position that its closing room cannot take (`_session_unsent`).
- **Position and working orders**: the kernel's position in units, its working orders,
  those that open or add and those that close, and how an alert names an order.
- **Exits and cancels** the session asks of the kernel.

`TraderOrders` is a part of `_session_trader.Trader`, which holds the kernel session
(`engine`) and the strategy's part of the plan (`part`, its run context among it), the
venue map (`venue`), the strategy's name and product, whether the kernel session finished
(`finished`), the event stream's writer (`_event`), the session's warnings (`_warn`), and
the risk exits' reasons and the position's units at each placement (`reasons`,
`units_at_place`). Times are UTC nanoseconds, quantities units. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._engine import call
from ..errors import LiveError
from ._plan_catch_up import INSTRUMENT
from ._session_unsent import to_send
from ._venue_requests import OrderRequest
from ._worker_queue import WithdrawPlace
from .venue_map import VenueMap


class TraderOrders:
    """The orders and the position of a `_session_trader.Trader` (module docstring)."""

    name: str
    #: The product the strategy trades ("MIS", "CNC" or "NRML"): its kernel position's,
    #: and its leg's at the broker (`_broker_units`).
    product: str
    #: The strategy's kernel session (`_session_bookkeeping.Bookkeeping.engine`).
    engine: Any
    part: Any
    venue: VenueMap
    finished: bool
    reasons: dict[int, str]
    units_at_place: dict[int, int]
    _event: Callable[..., None]
    _warn: Callable[[str], None]

    def commands(self) -> list[OrderRequest | WithdrawPlace]:
        """The kernel's commands since the last call, as the order worker's requests (a
        placement the kernel cancelled before this call withdrawn: module docstring; one
        it cancelled after, or that waits once its entries lock, withdrawn by the worker:
        `WithdrawPlace`); the reason of each risk exit placed is kept (`reasons`), and the
        position's units at each placement (`units_at_place`)."""
        if self.finished:
            return []
        requests: list[OrderRequest | WithdrawPlace] = []
        commands = call(self.engine.take_commands)
        reversals = self.part.context.resting_reversals
        while commands:
            kept = to_send(
                self.engine,
                self.name,
                commands,
                event=self._event,
                reversals=reversals,
                warn=self._warn,
            )
            for command in kept:
                if command.get("command") != "place":
                    continue
                kernel_order = int(command["order_id"])
                self.units_at_place[kernel_order] = round(abs(self.open_units()))
                if command.get("reason"):
                    self.reasons[kernel_order] = str(command["reason"])
            requests += self.venue.requests(kept)
            # A withdrawal or a refusal ends an order in the kernel, which may free an exit
            # that waited for it: its commands go out in this same call.
            commands = call(self.engine.take_commands) if len(kept) < len(commands) else []
        return requests + self.venue.withdraw_locked()

    def open_units(self) -> float:
        """The kernel's position in the strategy's instrument and product, signed units (0
        flat)."""
        position = call(self.engine.position, INSTRUMENT, self.product)
        return 0.0 if position is None else float(position["quantity_units"])

    def working(self) -> list[int]:
        """The kernel orders that can still fill."""
        return list(call(self.engine.working_orders))

    def order_name(self, kernel_order: int) -> str:
        """How an alert names a kernel order: the broker's order ID once bound, else
        "o<n>" (also for an order the kernel made in this loop step, which the venue map
        has not taken yet)."""
        try:
            order_id = self.venue.order_id_of(kernel_order)
        # The venue map knows only the orders whose commands it has taken.
        except LiveError:
            order_id = ""
        return order_id or f"o{kernel_order}"

    def exit_orders(self) -> list[int]:
        """The working orders that close the position (reduce-only)."""
        entries = set(self.entry_orders())
        return [kernel_order for kernel_order in self.working() if kernel_order not in entries]

    @property
    def flat(self) -> bool:
        """True with no position and no working order."""
        return self.open_units() == 0 and not self.working()

    def entry_orders(self) -> list[int]:
        """The working orders that open or add to the position: those not reduce-only
        (the kernel classes a sale beyond what closes the position as an entry too)."""
        return [
            kernel_order
            for kernel_order in self.working()
            if not call(self.engine.order, kernel_order)["reduce_only"]
        ]

    def request_exit(self, reason: str) -> str | None:
        """Ask the kernel to exit the position for `reason` (one of its RISK_EXIT_REASONS);
        what it did (its EXIT_REQUESTS), or None once the kernel session finished."""
        if self.finished:
            return None
        return str(call(self.engine.request_exit, INSTRUMENT, self.product, reason))

    def cancel(self, kernel_order: int) -> None:
        """Ask the kernel to cancel a working order (nothing once it finished)."""
        if not self.finished:
            call(self.engine.cancel, kernel_order)


__all__ = ["TraderOrders"]
