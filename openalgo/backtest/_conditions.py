"""Condition strategies (14-strategy-api 4): `entry_when`, `exit_when`, `short_when` and
`cover_when`, declared in `init` and run entirely in Rust.

    class EmaCross(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"

        def init(self):
            fast, slow = self.I(ta.ema, self.close, 10), self.I(ta.ema, self.close, 30)
            self.entry_when(fast.crossed_above(slow) & bt.time_between("09:30", "14:30"),
                            stop=bt.pct(1))
            self.exit_when(fast.crossed_below(slow))

What a condition is (the grammar of 4.2): a view of booleans built from series views and
`self.I` indicators with `+ - * /`, comparisons, `&`, `|`, `~`, `.crossed_above`,
`.crossed_below`, `.shift(n)`, `.rolling_max(n)` and `.rolling_min(n)` (`_views`), a clock
condition (`bt.time_between`, `bt.weekday`, `_clock`), or an array with one boolean per
strategy bar: a NumPy array or list in bar order, or a pandas Series indexed by bar open
time (SPEC-29 reads every array input the same way). Views hold the values of the whole
run, each computed from bars up to its own close, so a condition at a bar never reads a
later one. A view of another interval of the same symbol (`self.data[s, "15m"]`) counts
at a strategy bar with the value of its latest bar closed by then. Anything else raises
ConfigError saying to use `on_bar`.

The run (`drive_conditions`, called in place of the `on_bar` driver):

1. each condition becomes a bool array over the strategy bars of the instrument, the
   bars the pauses deliver (`Engine.strategy_bars`);
2. the arrays and the order templates go to the Rust signal driver
   (`oa_bt::engine::signals`), which runs the whole grid with no Python call per bar and
   makes, at every strategy-bar close, the decision of the equivalent `on_bar` of 4.1;
3. `on_stop` runs and the run's result is collected as for any strategy.

Rules of 4.1: conditions are level-triggered; long entry before short when flat; exit
before adding when in a position; `pyramiding` (class attribute, default 0) adds; the
position is read in step 7 and a queued entry counts as not flat (SPEC-29). The order
arguments are `qty`, `size`, `lots`, `stop`, `target`, `targets`, `trail`, `breakeven`,
`after_first_target`, `product`, `leverage` and `tag`: a condition entry is always a
MARKET DAY order. ATR distances (`bt.atr`, `bt.trail_atr`) and `bt.size.atr_risk` take
the ATR of the deciding bar, as in `on_bar` (`_condition_atr`). A strategy uses `on_bar`
or conditions, never both; a condition strategy trades one symbol and has no per-event
callbacks, timers or sizing functions (they would call Python during the run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import _orders
from ._calendar_convert import IST
from ._clock import ClockCondition
from ._condition_atr import Slot, atr_inputs, atr_slots, template_atr
from ._engine import call
from ._messages import counted
from ._series import FIELDS, ist_time
from ._strategy_orders import order_leverage
from ._views import TimelineView
from .errors import ConfigError

KINDS = ("entry", "exit", "short", "cover")
ENTRY_SIDES = {"entry": "BUY", "short": "SELL"}
ORDER_ARGUMENTS = frozenset(
    {
        "qty",
        "size",
        "lots",
        "stop",
        "target",
        "targets",
        "trail",
        "breakeven",
        "after_first_target",
        "product",
        "leverage",
        "tag",
    }
)
# The grammar of the module docstring, as messages name it.
GRAMMAR = (
    "the condition grammar (series views and self.I indicators combined with arithmetic, "
    "comparisons, &, |, ~, crossed_above, crossed_below, shift, rolling_max and "
    "rolling_min, clock conditions, or one boolean per strategy bar)"
)
# The two evaluation paths of 03-architecture 5: `on_bar` (and timers and callbacks) in
# Python at every decision, or conditions run by the Rust signal driver.
EVENT_PATH = "event"
CONDITIONS_PATH = "conditions"
ARRAYS = (pd.Series, np.ndarray, list, tuple)


@dataclass(frozen=True)
class Declared:
    """One declared condition and, for an entry, the request its orders send and the
    ATRs that request names (`_condition_atr`)."""

    condition: Any
    order: dict[str, Any] | None
    atr: tuple[Slot, ...] = ()


def outside(what: str) -> ConfigError:
    """The error for a condition outside the grammar."""
    return ConfigError(f"{what} is outside {GRAMMAR}; write this logic in on_bar instead")


def check_order_arguments(order_args: dict[str, Any], what: str) -> None:
    """Refuse order arguments a condition entry does not take (14-strategy-api 4.1).
    Raises ConfigError naming them."""
    unknown = sorted(set(order_args) - ORDER_ARGUMENTS)
    if unknown:
        raise ConfigError(
            f"{what} does not take {unknown}: its order arguments are "
            f"{sorted(ORDER_ARGUMENTS)}. price, validity and reverse are absent on purpose "
            "(a condition entry is always a MARKET DAY order); use on_bar"
        )


def declare(strategy: Any, kind: str, condition: Any, order_args: dict[str, Any]) -> None:
    """Record `<kind>_when(condition, **order_args)` on a strategy instance.

    Raises ConfigError when the class also defines `on_bar` or a per-event callback,
    trades several symbols, declares the same kind twice, gives a condition outside the
    grammar, order arguments 4.1 does not list, or a sizing function."""
    what = f"{kind}_when"
    _check_form(type(strategy), what)
    declared = strategy.__dict__.setdefault("_conditions", {})
    if kind in declared:
        raise ConfigError(f"{what} is declared once; combine conditions with & and |")
    _check_condition(condition, what)
    if kind not in ENTRY_SIDES:
        declared[kind] = Declared(condition, None)
        return
    check_order_arguments(order_args, what)
    leverage = order_leverage(order_args.get("leverage"), type(strategy).leverage)
    arguments = {**order_args, "leverage": leverage}
    order = _entry_request(strategy._ctx, ENTRY_SIDES[kind], arguments, what)
    slots = tuple(atr_slots(arguments, order["quantity"]))
    declared[kind] = Declared(condition, order, slots)


def has_conditions(strategy: Any) -> bool:
    """True when the strategy declared at least one condition in `init`."""
    return bool(strategy.__dict__.get("_conditions"))


def evaluation_path(strategy: Any) -> str:
    """The evaluation path (03-architecture 5) a strategy instance runs on once its
    `init` has run: CONDITIONS_PATH when it declared conditions, else EVENT_PATH."""
    return CONDITIONS_PATH if has_conditions(strategy) else EVENT_PATH


def _check_form(cls: type, what: str) -> None:
    from .strategy import CALLBACKS, Strategy

    if cls.on_bar is not Strategy.on_bar:
        raise ConfigError(
            f"{cls.__name__} defines on_bar and calls {what}: a strategy uses on_bar or "
            "conditions, not both"
        )
    if cls.wants_events():
        raise ConfigError(
            f"{cls.__name__} calls {what} and defines {list(CALLBACKS)} callbacks: a "
            "condition strategy runs in Rust with no Python call per event; use on_bar"
        )
    if cls.symbol is None:
        raise ConfigError(
            f"{cls.__name__}: a condition strategy trades one symbol (set symbol); use "
            "on_bar for several symbols"
        )


def _check_condition(condition: Any, what: str) -> None:
    """Refuse, when it is declared, anything that cannot be a condition."""
    from ._indicators import LookaheadOutput

    if isinstance(condition, LookaheadOutput):
        condition._refuse()
    if isinstance(condition, TimelineView):
        if condition._values.dtype != np.bool_:
            raise ConfigError(
                f"{what} takes a condition (true or false); {condition.label} is a series "
                "of numbers: compare it, for example view > 0"
            )
        return
    if isinstance(condition, ClockCondition):
        return
    if isinstance(condition, bool | np.bool_):
        raise outside(f"{what}({condition}) (and, or, not, ==, a plain True or False)")
    if not isinstance(condition, ARRAYS):
        raise outside(f"a {type(condition).__name__} condition")


def _entry_request(ctx: Any, side: str, args: dict[str, Any], what: str) -> dict[str, Any]:
    """The request `buy(**args)` or `sell(**args)` sends, built once: the same dict as
    `OrderMethods._open` for a MARKET DAY order on the strategy's instrument, with NaN
    where an ATR goes (the driver puts in the deciding bar's). Raises ConfigError for a
    sizing function, which would call Python at each decision."""
    rules = _orders.leg_rules(
        args.get("stop"),
        args.get("target"),
        args.get("targets"),
        args.get("trail"),
        args.get("breakeven"),
        args.get("after_first_target"),
        atr_at=template_atr,
    )
    sizer = args.get("size")
    quantity = _orders.quantity(args.get("qty"), args.get("lots"), sizer, ctx.default_size)
    if quantity.get("function") is not None:
        raise ConfigError(
            f"{what}: a sizing function runs Python at each decision and a condition "
            "strategy runs in Rust; use a bt.size sizer, or on_bar"
        )
    return _orders.request(
        ctx.instrument_index(None),
        side,
        args.get("product") or ctx.product,
        quantity=quantity,
        validity=_orders.validity("day"),
        reverse=False,
        rules=rules,
        leverage=args.get("leverage"),
        tag=args.get("tag"),
        **_orders.order_shape(None),
    )


def strategy_frame(strategy: Any) -> pd.DataFrame:
    """Every strategy bar of the strategy's instrument over the run, as a DataFrame
    indexed by IST bar open (open, high, low, close in rupees, volume in units): what a
    `from_signals` function receives."""
    store = strategy._ctx.store(None)
    return pd.DataFrame({name: store.column(name) for name in FIELDS}, index=store.times())


def array_values(data: Any, opens: np.ndarray, what: str) -> np.ndarray:
    """An array input as one boolean per bar opening at `opens` (int64 UTC nanoseconds):
    a Series with a DatetimeIndex is matched by bar open time (a naive index is IST, extra
    times are ignored); anything else is read in bar order. Raises ConfigError for
    values that are not booleans, a wrong length or a bar the Series does not cover."""
    values = np.asarray(data.to_numpy() if isinstance(data, pd.Series) else data)
    if values.ndim != 1 or values.dtype != np.bool_:
        raise ConfigError(
            f"{what} takes one boolean per strategy bar; got {values.dtype} values of shape "
            f"{values.shape}"
        )
    if isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
        index = data.index if data.index.tz is not None else data.index.tz_localize(IST)
        times = pd.Index(index.tz_convert("UTC").as_unit("ns").asi8)
        if not times.is_unique:
            raise ConfigError(f"{what}: the Series has the same bar open time more than once")
        rows = times.get_indexer(opens)
        if (rows < 0).any():
            first = int(opens[int(np.argmax(rows < 0))])
            raise ConfigError(
                f"{what}: the Series has no value for the strategy bar opening at "
                f"{ist_time(first).isoformat()}; index it by the strategy-bar open times"
            )
        return values[rows]
    if len(values) != len(opens):
        raise ConfigError(
            f"{what} has {counted(len(values), 'value')} and the run has "
            f"{counted(len(opens), 'strategy bar')}: give one value per strategy bar, or a "
            "Series indexed by bar open time"
        )
    return values


def _view_values(view: TimelineView, store: Any, what: str) -> np.ndarray:
    """A view's values at the strategy bars: its own when it is a view of them; for
    another interval of the symbol, the value of its latest bar closed by each strategy
    bar's close (false before its first)."""
    own = view._store
    if own is store:
        return view._values
    if own.symbol != store.symbol:
        raise ConfigError(
            f"{what}: {view.label} is a series of {own.symbol}; a condition strategy reads "
            f"its own symbol {store.symbol}"
        )
    closed = np.searchsorted(own.column("ts_close"), store.column("ts_close"), side="right")
    if not len(view._values):
        return np.zeros(len(closed), dtype=bool)
    return np.where(closed > 0, view._values[np.maximum(closed - 1, 0)], False)


def condition_values(condition: Any, store: Any, what: str) -> np.ndarray:
    """A condition as a contiguous bool array, one value per bar of `store` (the strategy
    bars). Raises ConfigError for anything outside the grammar."""
    _check_condition(condition, what)
    if isinstance(condition, TimelineView):
        values = _view_values(condition, store, what)
    elif isinstance(condition, ClockCondition):
        values = condition.values(store.column("ts_close"))
    else:
        values = array_values(condition, store.column("ts_open"), what)
    return np.ascontiguousarray(values, dtype=bool)


def _pyramiding(cls: type) -> int:
    value = cls.pyramiding
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"pyramiding takes a whole number of adds, 0 or more, got {value!r}")
    return value


def signal_spec(strategy: Any) -> dict[str, Any]:
    """The spec dict the engine's `run_signals` reads: the condition arrays (None when
    not declared), the entry order templates and the ATRs they name, the product and
    `pyramiding`. Raises ConfigError when the causality registry does not clear the
    ATR."""
    declared: dict[str, Declared] = strategy.__dict__.get("_conditions", {})
    ctx = strategy._ctx
    store = ctx.store(None)
    spec: dict[str, Any] = {
        "instrument": ctx.instrument_index(None),
        "product": ctx.product,
        "bar_opens": np.ascontiguousarray(store.column("ts_open"), dtype=np.int64),
        "pyramiding": _pyramiding(type(strategy)),
    }
    for kind in KINDS:
        entry = declared.get(kind)
        what = f"{kind}_when"
        spec[kind] = None if entry is None else condition_values(entry.condition, store, what)
        if kind in ENTRY_SIDES:
            spec[f"{kind}_order"] = None if entry is None else entry.order
            spec[f"{kind}_atr"] = [] if entry is None else atr_inputs(list(entry.atr), store)
    return spec


def drive_conditions(strategy: Any, engine: Any) -> None:
    """Run a condition strategy to the end of the grid: evaluate its conditions once,
    hand them to the Rust signal driver, record the entries a sizer skipped as `on_bar`
    would, then call `on_stop`. Raises ConfigError for timers (they call Python during
    the run); errors of the evaluation and of the kernel propagate (ConfigError,
    BacktestError, NotImplementedError)."""
    strategy._ctx.begin_run()
    if call(engine.timer_counts):
        raise ConfigError(
            f"{type(strategy).__name__}: a condition strategy runs in Rust with no Python "
            "call during the run, and self.at or self.every would call Python; use on_bar"
        )
    ctx = strategy._ctx
    symbol = ctx.symbol_of(ctx.instrument_index(None))
    for decided_at, reason in call(engine.run_signals, signal_spec(strategy)):
        ctx.skipped.append((decided_at, symbol, reason))
    strategy.on_stop()


__all__ = [
    "CONDITIONS_PATH",
    "EVENT_PATH",
    "ORDER_ARGUMENTS",
    "array_values",
    "check_order_arguments",
    "condition_values",
    "declare",
    "drive_conditions",
    "evaluation_path",
    "has_conditions",
    "signal_spec",
    "strategy_frame",
]
