"""`RunPlan`: every resolved setting of one run, in the units the kernel takes, and the
config dict of the native engine built from it. `_settings.resolve` makes one from the
documented defaults, the strategy's class attributes and the run arguments.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from ._calendar_convert import interval_nanos, to_day, to_seconds
from ._futures_margin import MarginSetting
from ._layout import parse_interval
from ._perturbation import perturbation

SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class RunPlan:
    """Every resolved setting of one run, in the units the kernel takes."""

    symbols: tuple[str, ...]
    exchange: str
    interval: str
    base: str
    product: str
    start: dt.date
    end: dt.date
    first_slice: int | None
    last_slice: int | None
    # The present, UTC nanoseconds (`now=`, the clock by default): the run and the data
    # load stop at it (`rust/oa_bt/src/engine/run_end.rs`).
    present: int
    capital: float
    fill: str
    quantity_mode: str
    costs: str | None
    slippage: tuple[str, float, int]
    risk: dict[str, Any]
    entry_from: str | None
    entry_until: str | None
    square_off: str | None
    leverage: float | None
    warnings: tuple[str, ...]
    # Sessions before expiry at which a future rolls, or None to settle at expiry.
    roll: int | None = 1
    # The run's own futures margin (`margin=`, `_futures_margin`), None for the default.
    margin: MarginSetting = None
    # Strategy bars a decided entry and a decided exit are held before they can fill
    # (`delay_entries=`, `delay_exits=`; the kernel's `engine::delay`).
    delay_entries: int = 0
    delay_exits: int = 0
    # Set by the robustness reruns only (`_robust`), never by a run argument: base bars by
    # which every strategy bar ends earlier (`engine::shifted_bars`), and the factors of
    # each trade's leg-rule distances (`engine::exit_scale`).
    bar_shift: int = 0
    exit_scales: tuple[float, ...] = ()

    def native_config(self, deliver_events: bool) -> dict[str, Any]:
        """The config dict of the native engine."""
        # Nanoseconds for an intraday interval, else its code ("D", or "W", "M", "Q" and
        # "Y", which the kernel refuses saying what to do instead).
        intraday = parse_interval(self.interval)[0] == "intraday"
        strategy = interval_nanos(self.interval) if intraday else self.interval
        minutes = {
            f"{name}_minute": None
            if value is None
            else to_seconds(value, what=name) // SECONDS_PER_MINUTE
            for name, value in (
                ("entry_from", self.entry_from),
                ("entry_until", self.entry_until),
                ("square_off", self.square_off),
            )
        }
        return {
            "first_day": to_day(self.start),
            "last_day": to_day(self.end),
            "first_slice": self.first_slice,
            "last_slice": self.last_slice,
            "present": self.present,
            "base_nanos": interval_nanos(self.base),
            "strategy_interval": strategy,
            "fill": self.fill,
            "quantity_mode": self.quantity_mode,
            "capital": self.capital,
            "costs": self.costs,
            "slippage": self.slippage,
            "risk": self.risk,
            "deliver_events": deliver_events,
            **minutes,
            **perturbation(self),
        }


__all__ = ["RunPlan"]
