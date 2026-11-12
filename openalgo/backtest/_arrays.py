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
indexed by IST bar open (columns open, high, low, close, volume) and returns
`(entries, exits)` or `(entries, exits, shorts, covers)`, any of which may be None.

Before the run the function is checked for look-ahead (03-architecture 5): it is run
again on the bars cut at fixed points, and every value up to a cut must be the same as
on the whole series. A function that reads later bars stops the run with ConfigError.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._conditions import check_order_arguments, run_bars
from ._expr_eval import StrategyBars, condition_values
from .errors import ConfigError
from .strategy import Strategy

WHEN_METHODS = ("entry_when", "exit_when", "short_when", "cover_when")
OUTPUT_NAMES = ("entries", "exits", "shorts", "covers")
# Where the look-ahead check cuts the bars, as fractions of their number: fixed, so a
# run is reproducible. A cut needs at least MIN_CUT_BARS bars before it.
CUT_FRACTIONS = (0.25, 0.5, 0.75)
MIN_CUT_BARS = 2


def _class(
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


def from_arrays(
    entries: Any,
    exits: Any = None,
    *,
    symbol: str,
    exchange: str,
    interval: str,
    shorts: Any = None,
    covers: Any = None,
    pyramiding: int = 0,
    product: str | None = None,
    **order_args: Any,
) -> type[Strategy]:
    """A strategy from boolean arrays, one value per strategy bar (see the module
    docstring). `order_args` are those of `entry_when` (14-strategy-api 4.1) and apply
    to entries and shorts; `product` is the strategy's product (its default by interval
    when None). Raises ConfigError for order arguments 4.1 does not list; the arrays are
    checked against the run's strategy bars when the run starts."""
    check_order_arguments(order_args, "from_arrays")
    arrays = (entries, exits, shorts, covers)

    def init(self: Strategy) -> None:
        _declare(self, arrays, order_args)

    return _class("FromArrays", symbol, exchange, interval, (pyramiding, product), init)


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
    once per run, plus once per cut of the look-ahead check. Raises ConfigError for
    order arguments 4.1 does not list; when the run starts, ConfigError for a function
    that returns anything else or looks ahead."""
    if not callable(func):
        raise ConfigError("from_signals takes a function of the strategy bars")
    check_order_arguments(order_args, "from_signals")

    def init(self: Strategy) -> None:
        bars = run_bars(self)
        arrays = _signals(func, bars)
        _check_causal(func, bars, arrays)
        _declare(self, arrays, order_args)

    name = f"FromSignals_{getattr(func, '__name__', 'function')}"
    return _class(name, symbol, exchange, interval, (pyramiding, product), init)


def _signals(func: Callable[..., Any], bars: StrategyBars) -> tuple[Any, ...]:
    """The function's outputs on these bars, checked to be 2 or 4 of them."""
    outputs = func(bars.frame())
    if not isinstance(outputs, tuple | list) or len(outputs) not in (2, 4):
        raise ConfigError(
            f"{getattr(func, '__name__', 'the function')} must return (entries, exits) or "
            "(entries, exits, shorts, covers)"
        )
    return tuple(outputs)


def _check_causal(func: Callable[..., Any], bars: StrategyBars, arrays: tuple) -> None:
    """The truncation check of 03-architecture 5: the outputs on the bars cut at each
    point of CUT_FRACTIONS equal the full-run outputs up to the cut. Raises ConfigError
    naming the first bar that changed."""
    full = [
        None if values is None else condition_values(values, bars, name)
        for name, values in zip(OUTPUT_NAMES, arrays, strict=False)
    ]
    for fraction in CUT_FRACTIONS:
        count = int(len(bars) * fraction)
        if count < MIN_CUT_BARS:
            continue
        part = bars.head(count)
        cut = _signals(func, part)
        for name, whole, values in zip(OUTPUT_NAMES, full, cut, strict=False):
            if whole is None or values is None:
                continue
            changed = np.flatnonzero(condition_values(values, part, name) != whole[:count])
            if changed.size:
                when = part.frame().index[int(changed[0])].isoformat()
                raise ConfigError(
                    f"{getattr(func, '__name__', 'the function')} looks ahead: its {name} "
                    f"at the bar opening {when} changes when the bars after the first "
                    f"{count} are removed (03-architecture 5). A signal may use only bars "
                    "up to its own close."
                )


__all__ = ["from_arrays", "from_signals"]
