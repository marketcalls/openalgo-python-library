"""Where the history before `start` meets the run's bars (decision D25; the history is
in `_warm_up_history`).

- **The join is checked** (`join_failure`): the bars a run keeps before `start` and the
  run's first session are one series to the data quality checks (`data.quality.
  check_bars`), with the corporate actions the run declares for the symbol. The run's
  own bars were checked when they were loaded and the history's range is the history's
  own, so what this finds is at the join or inside the history kept: a suspected
  unadjusted split between the last bar before `start` and the first after it, a price
  that is not a price. Any error of error severity leaves the run without history, and
  its messages say why.
- **The bar that straddles a datetime `start`** (`Opening`): with `start` at 10:00 a
  09:15 to 10:15 strategy bar opens before the run and closes in it. The kernel builds it
  from the base bars of the run, from 10:00; the history holds the morning's. The
  strategy sees the bar built from all of them, as a run that began earlier would: the
  first open and the highest high, lowest low and total volume of both parts, with the
  run's close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data._sessions import ist_days
from .data.quality import check_bars


@dataclass(frozen=True)
class Opening:
    """The part before `start` of the strategy bar that straddles a datetime `start`:
    its open (UTC nanoseconds), and its open, high and low in rupees and volume in
    units over the base bars before `start`."""

    ts_open: int
    open: float
    high: float
    low: float
    volume: float

    def rebuild(self, columns: dict[str, np.ndarray]) -> bool:
        """Make the first of the kernel's strategy bars `columns` (replaced by copies)
        the aggregate of both parts when it is this bar; True when it was."""
        opens = columns["ts_open"]
        if not len(opens) or int(opens[0]) != self.ts_open:
            return False
        for name in ("open", "high", "low", "volume"):
            columns[name] = np.array(columns[name], dtype=float)
        columns["open"][0] = self.open
        columns["high"][0] = max(self.high, float(columns["high"][0]))
        columns["low"][0] = min(self.low, float(columns["low"][0]))
        columns["volume"][0] += self.volume
        return True


def opening_part(frame: pd.DataFrame, run_from: int) -> Opening | None:
    """The part before `run_from` (UTC nanoseconds) of the strategy bar of `frame` (the
    strategy bars the history's base bars make) that is cut by it; None without one."""
    if frame.empty:
        return None
    last = frame.iloc[-1]
    if not int(last["ts_open"]) < run_from < int(last["ts_close"]):
        return None
    values = (float(last[name]) for name in ("open", "high", "low", "volume"))
    return Opening(int(last["ts_open"]), *values)


def _first_session(run_base: pd.DataFrame) -> pd.DataFrame:
    """The run's base bars of its first IST day."""
    if run_base.empty:
        return run_base
    days = ist_days(run_base["ts_open"].to_numpy())
    return run_base[days == days[0]]


def join_failure(
    symbol: str,
    history: pd.DataFrame,
    run_base: pd.DataFrame,
    declared: Any,
    plan: Any,
    calendar: Any,
) -> str:
    """Why the base bars `history` cannot be joined to `run_base`, the run's base bars
    of `symbol` (module docstring); "" when they can. `declared` are the corporate
    actions the run declares for the symbol, `plan` the run's (`_settings.RunPlan`) and
    `calendar` its calendar."""
    if history.empty or run_base.empty:
        return ""
    joined = pd.concat([history, _first_session(run_base)], ignore_index=True)
    joined.attrs = {"symbol": symbol, "exchange": plan.exchange, "interval": plan.base}
    report = check_bars(joined, calendar=calendar, now_ns=plan.present, corporate_actions=declared)
    return "; ".join(issue.message for issue in report.errors)


__all__ = ["Opening", "join_failure", "opening_part"]
