"""Condition strategies (14-strategy-api 4): `entry_when`, `exit_when`, `short_when` and
`cover_when`, declared in `init` and run entirely in Rust.

    class EmaCross(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"

        def init(self):
            self.entry_when(self.close.crossed_above(self.open), stop=bt.pct(1))
            self.exit_when(self.close.crossed_below(self.open))

The run (`drive_conditions`, called by `bt.backtest` in place of the `on_bar` driver):

1. the strategy bars of the instrument come from the engine (the same bars `on_bar`
   would see, one at a time);
2. each condition is evaluated once over all of them into a bool array (`_expr_eval`);
3. the arrays and the order templates go to the Rust signal driver
   (`oa_bt::engine::signals`), which runs the whole grid with no Python call per bar and
   makes, at every strategy-bar close, the decision of the equivalent `on_bar` of 4.1;
4. `on_stop` runs and `bt.backtest` collects the result as for any strategy.

Rules of 4.1: conditions are level-triggered; long entry before short when flat; exit
before adding when in a position; `pyramiding` (class attribute, default 0) adds; the
position is read in step 7 and a queued entry counts as not flat (SPEC-29). The order
arguments are `qty`, `size`, `lots`, `stop`, `target`, `targets`, `trail`, `breakeven`,
`after_first_target`, `product`, `leverage` and `tag`; a condition entry is always a
MARKET DAY order. A strategy uses `on_bar` or conditions, never both, and a condition
strategy trades one symbol and has no per-event callbacks (they would need Python per
event).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _orders
from ._engine import call
from ._expr import BOOL, Expr, typed
from ._expr_eval import StrategyBars, condition_values
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
NOT_BUILT = "is not implemented yet (Phase 1 work in progress)"


@dataclass(frozen=True)
class Declared:
    """One declared condition and, for an entry, the request its orders send."""

    condition: Expr
    order: dict[str, Any] | None


def check_order_arguments(order_args: dict[str, Any], what: str) -> None:
    """Refuse order arguments a condition entry does not take (14-strategy-api 4.1).
    Raises ConfigError naming them."""
    unknown = sorted(set(order_args) - ORDER_ARGUMENTS)
    if unknown:
        raise ConfigError(
            f"{what} does not take {unknown}: its order arguments are "
            f"{sorted(ORDER_ARGUMENTS)}. price, validity and reverse are absent on purpose "
            "(a condition entry is a MARKET DAY order, 14-strategy-api 4.1); use on_bar"
        )


def declare(strategy: Any, kind: str, condition: Any, order_args: dict[str, Any]) -> None:
    """Record `<kind>_when(condition, **order_args)` on a strategy instance.

    Raises ConfigError when the class also defines `on_bar` or a per-event callback,
    trades several symbols, declares the same kind twice, gives a condition outside the
    grammar or order arguments 4.1 does not list; NotImplementedError for an ATR
    distance or `bt.size.atr_risk` (resolved per decision, not built for conditions)."""
    what = f"{kind}_when"
    _check_form(type(strategy), what)
    declared = strategy.__dict__.setdefault("_conditions", {})
    if kind in declared:
        raise ConfigError(f"{what} is declared once; combine conditions with & and |")
    expr = typed(condition, BOOL, what)
    order = None
    if kind in ENTRY_SIDES:
        check_order_arguments(order_args, what)
        order = _entry_request(strategy._ctx, ENTRY_SIDES[kind], order_args)
    declared[kind] = Declared(expr, order)


def has_conditions(strategy: Any) -> bool:
    """True when the strategy declared at least one condition in `init`."""
    return bool(strategy.__dict__.get("_conditions"))


def _check_form(cls: type, what: str) -> None:
    from .strategy import CALLBACKS, Strategy

    if cls.on_bar is not Strategy.on_bar:
        raise ConfigError(
            f"{cls.__name__} defines on_bar and calls {what}: a strategy uses on_bar or "
            "conditions, not both (14-strategy-api 4.1)"
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


def _refuse_atr(period: int) -> float:
    raise NotImplementedError(
        f"ATR distances (bt.atr, bt.trail_atr) in condition order arguments {NOT_BUILT}: "
        "use points, pct, ticks or rupees, or on_bar"
    )


def _entry_request(ctx: Any, side: str, args: dict[str, Any]) -> dict[str, Any]:
    """The request `buy(**args)` or `sell(**args)` sends, built once: the same dict as
    `OrderMethods._open` for a MARKET DAY order on the strategy's instrument."""
    rules = _orders.leg_rules(
        args.get("stop"),
        args.get("target"),
        args.get("targets"),
        args.get("trail"),
        args.get("breakeven"),
        args.get("after_first_target"),
        atr_at=_refuse_atr,
    )
    quantity = _orders.quantity(
        args.get("qty"), args.get("lots"), args.get("size"), ctx.default_size
    )
    if quantity.get("sizer") == "atr_risk":
        raise NotImplementedError(f"bt.size.atr_risk in condition order arguments {NOT_BUILT}")
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


def run_bars(strategy: Any) -> StrategyBars:
    """Every strategy bar of the strategy's instrument over the run, from the engine:
    what its conditions are evaluated on."""
    ctx = strategy._ctx
    index = ctx.instrument_index(None)
    return StrategyBars.from_native(ctx.symbol_of(index), call(ctx.engine.strategy_bars, index))


def _pyramiding(cls: type) -> int:
    value = cls.pyramiding
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"pyramiding takes a whole number of adds, 0 or more, got {value!r}")
    return value


def signal_spec(strategy: Any, bars: StrategyBars) -> dict[str, Any]:
    """The spec dict the engine's `run_signals` reads: the condition arrays (None when
    not declared), the entry order templates, the product and `pyramiding`."""
    declared: dict[str, Declared] = strategy.__dict__.get("_conditions", {})
    ctx = strategy._ctx
    spec: dict[str, Any] = {
        "instrument": ctx.instrument_index(None),
        "product": ctx.product,
        "bar_opens": np.ascontiguousarray(bars.ts_open, dtype=np.int64),
        "pyramiding": _pyramiding(type(strategy)),
    }
    for kind in KINDS:
        entry = declared.get(kind)
        what = f"{kind}_when"
        spec[kind] = None if entry is None else condition_values(entry.condition, bars, what)
        if kind in ENTRY_SIDES:
            spec[f"{kind}_order"] = None if entry is None else entry.order
    return spec


def drive_conditions(strategy: Any, engine: Any, ctx: Any) -> None:
    """Run a condition strategy to the end of the grid: evaluate its conditions once,
    hand them to the Rust signal driver, then call `on_stop`. Errors of the evaluation
    and of the kernel propagate (ConfigError, BacktestError, NotImplementedError)."""
    spec = signal_spec(strategy, run_bars(strategy))
    call(engine.run_signals, spec)
    strategy.on_stop()


__all__ = [
    "ORDER_ARGUMENTS",
    "check_order_arguments",
    "declare",
    "drive_conditions",
    "has_conditions",
    "run_bars",
    "signal_spec",
]
