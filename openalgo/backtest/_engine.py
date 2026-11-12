"""The native engine of one run: built from the resolved plan, and called with the
kernel's errors converted to the package's public types.

`NativeEngineInputError` (bad configuration, data or request) becomes ConfigError,
`NativeEngineError` (a call at the wrong time, a kernel defect) becomes BacktestError, and
NotImplementedError (a part not built yet) passes through with the kernel's message.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ._calendar_convert import native_attr, to_day
from .errors import BacktestError, ConfigError
from .reference import Instrument

# The kernel's instrument kind codes (oa_bt::model::InstrumentKind).
KIND_CODES = {"equity": "EQ", "index": "INDEX", "future": "FUT", "option": "OPT"}


def call(func: Callable[..., Any], *args: Any) -> Any:
    """Call into the native engine, converting its errors (see the module docstring)."""
    try:
        return func(*args)
    except native_attr("NativeEngineInputError") as exc:
        raise ConfigError(str(exc)) from None
    except native_attr("NativeEngineError") as exc:
        raise BacktestError(str(exc)) from None


def instrument_dict(instrument: Instrument) -> dict[str, Any]:
    """The binding's instrument dict. Raises ConfigError when the tick size is unknown."""
    if instrument.tick_size is None:
        raise ConfigError(
            f"{instrument.symbol}: the tick size depends on a price slab; pass an instrument "
            "with tick_size set"
        )
    last_day = instrument.last_trading_day or instrument.expiry
    return {
        "symbol": instrument.symbol,
        "exchange": instrument.exchange,
        "kind": KIND_CODES[instrument.kind],
        "lot_size": int(instrument.lot_size),
        "tick_size": float(instrument.tick_size),
        "multiplier": float(instrument.multiplier),
        "freeze_qty": instrument.freeze_qty,
        "last_trading_day": None if last_day is None else to_day(last_day),
    }


def bar_arrays(bars: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """The six contiguous arrays the binding reads from a canonical bars frame."""
    columns = ("ts_open", "open", "high", "low", "close", "volume")
    dtypes = (np.int64, *(np.float64,) * 5)
    return tuple(
        np.ascontiguousarray(bars[name].to_numpy(dtype=dtype))
        for name, dtype in zip(columns, dtypes, strict=True)
    )


def build_engine(
    native_calendar: Any,
    config: dict[str, Any],
    instruments: list[Instrument],
    frames: list[pd.DataFrame],
) -> Any:
    """A native Engine for one run. Raises ConfigError, BacktestError or
    NotImplementedError (see the module docstring)."""
    engine_class = native_attr("Engine")
    specs = [instrument_dict(instrument) for instrument in instruments]
    arrays = [bar_arrays(frame) for frame in frames]
    return call(engine_class, native_calendar, config, specs, arrays)


__all__ = ["KIND_CODES", "bar_arrays", "build_engine", "call", "instrument_dict"]
