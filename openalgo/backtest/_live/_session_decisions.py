"""What a live session writes about each strategy bar it reaches (2.4.0 item P2-2): one
line per bar in the day's decisions file (`SessionLogs.decision`, beside the event stream,
off it and off the console), and the log lines that say why a decision sent nothing.

- **A decision** (kind "decision"): the strategy, the decision's instant
  (`decided_at_ns`, and in IST), the bar `on_bar` was given (its open and close instants,
  prices and volume) and where its base bars came from (`source`: "ticks", "candle", or
  "ticks and candle" for a bar of several minutes holding both; a candle is a minute's
  bar only when the minute made no tick bar), the value of each `self.I` output by its
  label (`LiveRunContext.indicator_values`: null while not known), the position in
  units (signed, at the decision: its own orders have not filled
  yet), and what `on_bar` submitted: each kernel order (its id, side, units, type,
  status and reject code: a rejection at submission has its code) and each order request
  that made none (`skipped`, its code). A decision whose `on_bar` raised has its line too,
  with what it submitted before the exception and the exception (`error`).
- **A strategy bar not decided on** (kind "skipped"): the strategy, the decision's
  instant and why (`reason`): missing data, indicators not ready, trading not begun
  (before the first whole minute watched), the session stopping, or the strategy failed.
- **The day's log**: each kernel order the strategy's code submitted (in `on_bar`, a
  callback or a timer) that the kernel rejected, and each request that made no order,
  at INFO with the strategy, side, units and code; the first bar per strategy not
  decided on because its indicators are not ready at INFO, later ones at DEBUG.

Diagnostic only: nothing here changes a decision, an order or a figure. Four 1-minute
strategies write about 1,500 short lines a day, at the bar closes, never on a tick. Times
are UTC nanoseconds, prices rupees, quantities units. Used by the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .._calendar_convert import format_ist
from .._engine import call
from .._series import Bar
from ._plan_catch_up import StrategyPlan
from ._types import LOGGER_NAME

#: The kinds of the decisions file's lines.
DECISION = "decision"
SKIPPED = "skipped"
#: Why a strategy bar was not decided on (missing data is `_session_trader.SKIP_MISSING`).
NOT_READY = "indicators not ready"
NOT_TRADING = "trading not begun"
STOPPING = "session stopping"
FAILED = "strategy failed"

logger = logging.getLogger(LOGGER_NAME)


class DecisionNotes:
    """The decisions file's lines and the log lines of one strategy (module docstring):
    `name` is the strategy's, `part` its share of the plan, `write` the decisions file's
    writer (`SessionLogs.decision`, which never raises for a line it cannot write)."""

    def __init__(self, name: str, part: StrategyPlan, write: Callable[..., None]) -> None:
        self._name, self._part, self._write = name, part, write
        self._told_not_ready = self._told_unread = False

    def submitted(self) -> list[dict[str, Any]]:
        """What the strategy's code submitted since the last call, as a decision line
        lists it; each rejection at submission and each request that made no order
        logged at INFO."""
        listed: list[dict[str, Any]] = []
        for submission in self._part.context.take_submissions():
            if submission.order is None:
                logger.info(
                    "%s: an order request made no order: %s", self._name, submission.skipped
                )
                listed.append({"skipped": submission.skipped})
                continue
            order = call(self._part.engine.order, submission.order)
            listed.append(
                {
                    "order": f"o{submission.order}",
                    "side": order["side"],
                    "units": order["quantity_units"],
                    "order_type": order["order_type"],
                    "status": order["status"],
                    "reject_code": order["reject_code"],
                }
            )
            if order["reject_code"] is not None:
                logger.info(
                    "%s: order o%d rejected at submission: %s %s units, %s",
                    self._name,
                    submission.order,
                    order["side"],
                    units_text(order["quantity_units"]),
                    order["reject_code"],
                )
        return listed

    def decided(
        self,
        decided_at_ns: int,
        bar: Bar,
        submitted: list[dict[str, Any]],
        position_units: float,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Write the line of a decision made on `bar` (module docstring); `error`, the
        exception `on_bar` raised, adds its `error` field."""
        sources = set(self._part.series.sources(bar.ts_open, bar.ts_close))
        failed = {} if error is None else {"error": f"{type(error).__name__}: {error}"}
        self._write(
            DECISION,
            strategy=self._name,
            decided_at_ns=decided_at_ns,
            decided_at=format_ist(decided_at_ns),
            bar={
                "ts_open_ns": bar.ts_open,
                "ts_close_ns": bar.ts_close,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            },
            source=" and ".join(sorted(sources, reverse=True)) or None,
            indicators=self._indicators(),
            position_units=position_units,
            submitted=submitted,
            **failed,
        )

    def _indicators(self) -> dict[str, Any] | str:
        """The `self.I` values for a decision line, or what kept them from being read: an
        indicator the strategy did not read at this decision is computed here, and a
        failure of it must not stop a session the strategy's own reads let go on. Logged
        at WARNING the first time, at DEBUG after."""
        try:
            return self._part.context.indicator_values()
        # Diagnostic only: any failure is named in the line and logged, never raised.
        except Exception as error:
            level = logging.DEBUG if self._told_unread else logging.WARNING
            self._told_unread = True
            logger.log(level, "%s: the indicator values could not be read: %s", self._name, error)
            return f"not read: {type(error).__name__}: {error}"

    def skipped(self, decided_at_ns: int, reason: str) -> None:
        """Write the line of a strategy bar not decided on, and why."""
        self._write(
            SKIPPED,
            strategy=self._name,
            decided_at_ns=decided_at_ns,
            decided_at=format_ist(decided_at_ns),
            reason=reason,
        )

    def not_ready(self, decided_at_ns: int) -> None:
        """A strategy bar not decided on because its indicators are not ready: logged (at
        INFO the first time, at DEBUG after) and written."""
        level = logging.DEBUG if self._told_not_ready else logging.INFO
        self._told_not_ready = True
        logger.log(level, "%s: indicators not ready at %s", self._name, format_ist(decided_at_ns))
        self.skipped(decided_at_ns, NOT_READY)

    def not_deciding(self, decided_at_ns: int, *, failed: bool, deciding: bool) -> None:
        """A strategy bar that closed while the strategy does not decide: written with why
        (the strategy `failed`, the session no longer `deciding`, or else trading has not
        begun)."""
        if failed:
            reason = FAILED
        elif not deciding:
            reason = STOPPING
        else:
            reason = NOT_TRADING
        self.skipped(decided_at_ns, reason)


def units_text(units: float) -> str:
    """`units` as the log writes them: a whole number exactly, with no exponent."""
    return str(int(units)) if float(units).is_integer() else str(units)


__all__ = [
    "DECISION",
    "FAILED",
    "NOT_READY",
    "NOT_TRADING",
    "SKIPPED",
    "STOPPING",
    "DecisionNotes",
    "units_text",
]
