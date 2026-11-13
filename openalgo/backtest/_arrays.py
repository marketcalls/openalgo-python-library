"""Array and function strategies: `bt.from_arrays` and `bt.from_signals` (10-simple-path
section 3, 14-strategy-api 4.3).

    def signals(df):                                  # the run's strategy bars
        f, s = ta.ema(df.close, 10), ta.ema(df.close, 30)
        return ta.crossover(f, s), ta.crossunder(f, s)

    result = bt.backtest(bt.from_signals(signals, "SBIN", "NSE", "5m", stop=bt.pct(1)), ...)
    result = bt.backtest(bt.from_arrays(entries, exits, symbol="SBIN", exchange="NSE",
                                        interval="5m", size=bt.size.qty(10)), ...)

Both return a `bt.Strategy` subclass whose `init` declares the arrays as conditions
(`entry_when(entries, **order_args)`, `exit_when(exits)`, and `short_when` and
`cover_when` for shorts and covers), so they run through the same Rust signal driver as
conditions and are read with the same level-triggered rules of 14-strategy-api 4.1
(SPEC-29): true while flat enters, true while in a position exits, `pyramiding` adds.

Arrays hold one boolean per strategy bar of the run, the value at that bar's close:
a NumPy array or list in bar order, or a pandas Series indexed by bar open time (IST,
or tz-aware). A `from_signals` function receives the strategy bars as a DataFrame
indexed by IST bar open (columns open, high, low, close, volume) and returns `(entries,
exits)` or `(entries, exits, shorts, covers)`, any of which may be None. It runs in
`init` on the strategy bars of the history before `start` (that of a 200-bar EMA,
`_warm_up_probe`) and of the run together, so its indicators are computed over the
history; the values of the history bars are left out, as no decision is made before
`start`.

Before the run the function is checked for look-ahead (03-architecture 5): it is run
again on the run's bars cut at fixed points, and it must return the same outputs there (as
many, None where it gave None) with every value up to a cut the same as on the whole
series. A function that reads later bars stops the run with ConfigError. The cuts are a
sample (CUT_FRACTIONS): a value that reads a later bar shows only at the bar just before
a cut, so a sparse signal read one bar ahead (a crossover shifted back a bar) passes
unless it fires there. The check catches look-ahead; it does not prove its absence.

The classes are built at run time, so no module defines them and a spawned worker
process cannot import one: `made_by` names the function that built a class (or a class
it derives from), and `bt.optimize` sweeps such a strategy in its own process.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._conditions import array_values, check_order_arguments, strategy_frame
from ._views import TimelineView
from .errors import ConfigError
from .strategy import Strategy

WHEN_METHODS = ("entry_when", "exit_when", "short_when", "cover_when")
OUTPUT_NAMES = ("entries", "exits", "shorts", "covers")
# Where the look-ahead check cuts the bars, as fractions of their number: fixed, so a
# run is reproducible. A cut needs at least MIN_CUT_BARS bars before it.
CUT_FRACTIONS = (0.25, 0.5, 0.75)
MIN_CUT_BARS = 2
# The class attribute naming the function that built a strategy class here (private, so
# `Strategy.parameters` does not take it for a parameter).
MADE_BY = "_made_by"
# The keywords of the one-line examples in from_arrays' errors.
FROM_ARRAYS_MARKET = 'symbol="SBIN", exchange="NSE", interval="5m"'


def made_by(strategy: type) -> str | None:
    """The function that built `strategy` or a class it derives from, "bt.from_arrays"
    or "bt.from_signals"; None for any other strategy class."""
    return getattr(strategy, MADE_BY, None)


def _class(
    maker: str,
    name: str,
    symbol: str,
    exchange: str,
    interval: str,
    settings: tuple[int, str | None],
    init: Callable[[Strategy], None],
) -> type[Strategy]:
    pyramiding, product = settings
    attributes = {
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "pyramiding": pyramiding,
        "product": product,
        "init": init,
        MADE_BY: maker,
        "__doc__": f"A condition strategy on {symbol} ({exchange}, {interval}) from {name}.",
    }
    return type(name, (Strategy,), attributes)


def _declare(strategy: Strategy, arrays: tuple[Any, ...], order_args: dict[str, Any]) -> None:
    """entry_when, exit_when, short_when, cover_when for the arrays that are given."""
    if arrays[0] is None and (len(arrays) < 3 or arrays[2] is None):
        raise ConfigError("give entries or shorts: a strategy with neither never trades")
    for method, values in zip(WHEN_METHODS, arrays, strict=False):
        if values is None:
            continue
        arguments = order_args if method in ("entry_when", "short_when") else {}
        getattr(strategy, method)(values, **arguments)


def _check_from_arrays_call(extra: tuple[Any, ...], market: dict[str, str | None]) -> None:
    """Raise ConfigError, with a one-line example, for values given by position after
    `exits` (every argument after it is a keyword) or a missing `symbol`, `exchange` or
    `interval`."""
    if extra:
        raise ConfigError(
            f"from_arrays takes entries and exits by position and got "
            f"{len(extra) + 2} values: pass symbol=, exchange=, interval=, shorts= and "
            "covers= by name, for example "
            "bt.from_arrays(entries, exits, shorts=shorts, covers=covers, "
            f"{FROM_ARRAYS_MARKET})"
        )
    missing = [name for name, value in market.items() if value is None]
    if missing:
        needs = ", ".join(f"{name}=" for name in missing)
        raise ConfigError(
            f"from_arrays needs {needs} (the strategy's symbol, exchange and interval, by "
            f"name), for example bt.from_arrays(entries, exits, {FROM_ARRAYS_MARKET})"
        )


def from_arrays(
    entries: Any,
    exits: Any = None,
    *extra: Any,
    symbol: str | None = None,
    exchange: str | None = None,
    interval: str | None = None,
    shorts: Any = None,
    covers: Any = None,
    pyramiding: int = 0,
    product: str | None = None,
    **order_args: Any,
) -> type[Strategy]:
    """A strategy from boolean arrays, one value per strategy bar (see the module
    docstring). `entries` and `exits` may be given by position; everything else is a
    keyword, and `symbol`, `exchange` and `interval` are required. `order_args` are
    those of `entry_when` (14-strategy-api 4.1) and apply to entries and shorts;
    `product` is the strategy's product (its default by interval when None). Raises
    ConfigError for a missing `symbol`, `exchange` or `interval`, for arrays given by
    position after `exits` (`extra`), and for order arguments 4.1 does not list; the
    arrays are checked against the run's strategy bars when the run starts."""
    _check_from_arrays_call(extra, {"symbol": symbol, "exchange": exchange, "interval": interval})
    check_order_arguments(order_args, "from_arrays")
    arrays = (entries, exits, shorts, covers)

    def init(self: Strategy) -> None:
        _declare(self, arrays, order_args)

    settings = (pyramiding, product)
    return _class("bt.from_arrays", "FromArrays", symbol, exchange, interval, settings, init)


def from_signals(
    func: Callable[..., Any],
    symbol: str,
    exchange: str,
    interval: str,
    *,
    pyramiding: int = 0,
    product: str | None = None,
    **order_args: Any,
) -> type[Strategy]:
    """A strategy from a function of the strategy bars returning `(entries, exits)` or
    `(entries, exits, shorts, covers)` (see the module docstring). The function runs
    once on the history before `start` and the run's bars together, plus once per cut
    of the look-ahead check; the cuts are a sample of three, so a function that reads a
    later bar only away from them passes. Raises
    ConfigError for order arguments 4.1 does not list; when the run starts, ConfigError
    for a function that returns anything else or looks ahead."""
    if not callable(func):
        raise ConfigError("from_signals takes a function of the strategy bars")
    check_order_arguments(order_args, "from_signals")

    def init(self: Strategy) -> None:
        frame = strategy_frame(self)
        arrays = _signals(func, frame)
        _check_causal(func, frame, arrays)
        _declare(self, _signal_views(self._ctx.store(None), frame, arrays), order_args)

    name = f"FromSignals_{getattr(func, '__name__', 'function')}"
    settings = (pyramiding, product)
    return _class("bt.from_signals", name, symbol, exchange, interval, settings, init)


def _signals(func: Callable[..., Any], frame: pd.DataFrame) -> tuple[Any, ...]:
    """The function's outputs on these bars, checked to be 2 or 4 of them."""
    outputs = func(frame)
    if not isinstance(outputs, tuple | list) or len(outputs) not in (2, 4):
        raise ConfigError(
            f"{getattr(func, '__name__', 'the function')} must return (entries, exits) or "
            "(entries, exits, shorts, covers)"
        )
    return tuple(outputs)


def _signal_views(store: Any, frame: pd.DataFrame, arrays: tuple[Any, ...]) -> tuple[Any, ...]:
    """The function's outputs as views of the strategy bars, history and run (None
    where it gave None)."""
    opens = frame.index.tz_convert("UTC").as_unit("ns").asi8
    return tuple(
        None if values is None else TimelineView(store, array_values(values, opens, name), name)
        for name, values in zip(OUTPUT_NAMES, arrays, strict=False)
    )


def _given(outputs: Sequence[Any]) -> str:
    """The outputs a function returned, by name, with None for one it left out."""
    names = [
        name if values is not None else "None"
        for name, values in zip(OUTPUT_NAMES, outputs, strict=False)
    ]
    return f"({', '.join(names)})"


def _check_causal(func: Callable[..., Any], frame: pd.DataFrame, arrays: tuple) -> None:
    """The truncation check of 03-architecture 5: on the bars cut at each point of
    CUT_FRACTIONS the function returns the same outputs (as many, and None where it gave
    None) and their values equal the full-run values up to the cut. Raises ConfigError
    naming the first bar that changed, or the outputs that did."""
    opens = frame.index.tz_convert("UTC").as_unit("ns").asi8
    full = [
        None if values is None else array_values(values, opens, name)
        for name, values in zip(OUTPUT_NAMES, arrays, strict=False)
    ]
    for fraction in CUT_FRACTIONS:
        count = int(len(frame) * fraction)
        if count < MIN_CUT_BARS:
            continue
        cut = _signals(func, frame.iloc[:count])
        if _given(cut) != _given(full):
            raise ConfigError(
                f"{getattr(func, '__name__', 'the function')} looks ahead: it returns "
                f"{_given(full)} on all the bars and {_given(cut)} on the first {count}. A "
                "signal may use only bars up to its own close."
            )
        for name, whole, values in zip(OUTPUT_NAMES, full, cut, strict=False):
            if whole is None:
                continue
            changed = np.flatnonzero(array_values(values, opens[:count], name) != whole[:count])
            if changed.size:
                when = frame.index[int(changed[0])].isoformat()
                raise ConfigError(
                    f"{getattr(func, '__name__', 'the function')} looks ahead: its {name} "
                    f"value at the bar opening {when} changes when the bars after the first "
                    f"{count} are removed. A signal may use only bars up to its own close."
                )


__all__ = ["from_arrays", "from_signals", "made_by"]
