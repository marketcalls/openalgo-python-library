"""Where the history before `start` meets the run's bars (decision D25; the history is
in `_warm_up_history`).

- **The join is checked** (`join`): the bars a run keeps before `start` and the run's
  first base bar are one series to the data quality checks (`data.quality.check_bars`),
  with the corporate actions the run declares for the symbol. Only those: the first
  decision is made at the close of the first strategy bar, so no bar after it may
  decide whether the history is kept (a check that read the whole first session would
  let the afternoon decide whether the morning trades). The run's own bars were checked
  when they were loaded and the history's range is the history's own, so what this
  finds is at the join or inside the history kept: a suspected unadjusted split between
  the last bar before `start` and the first after it, a price that is not a price. Any
  error of error severity leaves the run without history, and its messages say why.
- **A declared split or bonus inside the history** (or at the join), which the bars are
  not adjusted for, is adjusted for in the history, as a split-adjusted chart shows the
  bars before it: their prices divided by its factor and their volume multiplied by it,
  so the indicators at `start` are those of the adjusted series. The Assumptions line of
  the warm-up says so. The run's own bars, and an ex-date inside the run, stay as they
  are (`_run_corporate_actions`).
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

from .data.quality import check_bars
from .data.splits import UNADJUSTED, PriceGap

PRICES = ("open", "high", "low", "close")


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


@dataclass(frozen=True)
class Joined:
    """The history as joined to the run (module docstring): `failure` says why it cannot
    be ("" when it can); `base` is its base bars, adjusted for the declared splits and
    bonuses before the run; `adjusted` says what was adjusted, one clause per action."""

    failure: str
    base: pd.DataFrame
    adjusted: tuple[str, ...] = ()


def _adjusted_for(gap: PriceGap) -> str:
    """For example "adjusted for a split (2 shares for every 1) on 2024-12-17: prices
    before it divided by 2, volume multiplied by 2"."""
    return (
        f"adjusted for {gap.reading} on {gap.ex_date}: prices before it divided by "
        f"{gap.factor:g}, volume multiplied by {gap.factor:g}"
    )


def _back_adjusted(history: pd.DataFrame, gaps: list[PriceGap]) -> pd.DataFrame:
    """`history` with the bars before each gap's ex-date divided by its factor (prices)
    and multiplied by it (volume)."""
    if not gaps:
        return history
    opens = history["ts_open"].to_numpy()
    factor = np.ones(len(history))
    for gap in gaps:
        factor[opens < gap.ex_open_ns] *= gap.factor
    adjusted = history.copy()
    for name in PRICES:
        adjusted[name] = history[name].to_numpy(dtype=float) / factor
    adjusted["volume"] = history["volume"].to_numpy(dtype=float) * factor
    return adjusted


def join(
    symbol: str,
    history: pd.DataFrame,
    run_base: pd.DataFrame,
    declared: Any,
    plan: Any,
    calendar: Any,
) -> Joined:
    """The base bars `history` joined to `run_base`, the run's base bars of `symbol`
    (module docstring). `declared` are the corporate actions the run declares for the
    symbol, `plan` the run's (`_settings.RunPlan`) and `calendar` its calendar."""
    if history.empty or run_base.empty:
        return Joined("", history)
    joined = pd.concat([history, run_base.iloc[:1]], ignore_index=True)
    joined.attrs = {"symbol": symbol, "exchange": plan.exchange, "interval": plan.base}
    report = check_bars(joined, calendar=calendar, now_ns=plan.present, corporate_actions=declared)
    failure = "; ".join(issue.message for issue in report.errors)
    if failure:
        return Joined(failure, history)
    gaps = [gap for gap in report.price_gaps if gap.status == UNADJUSTED]
    return Joined("", _back_adjusted(history, gaps), tuple(_adjusted_for(gap) for gap in gaps))


__all__ = ["Joined", "Opening", "join", "opening_part"]
