"""`Strategy`: the one class a user subclasses, the same in `bt.backtest` and `bt.live`
(14-strategy-api section 2).

    class EmaCross(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"
        fast = 10                                    # a parameter

        def on_bar(self, bar):
            if self.position.is_flat and bar.close > bar.open:
                self.buy(qty=10)
            elif self.position.is_long:
                self.exit()

Declaring what it trades: exactly one of `symbol`, `symbols`, `universe`, `underlying`,
plus `exchange` and `interval`. Futures in `symbols` are named by their underlying and
follow the same contract rule as `underlying` (lead ruling). Session and risk settings
may be class attributes (`max_daily_loss`, `entry_from`, `entry_until`, `square_off`,
`drawdown_halt`, `cooldown`): a class attribute wins over the run argument of the same
name (lead ruling). Parameters are any other class attributes with a number, bool or
string default, or `bt.param(default, values)`.

Hooks run in step 7 of the kernel slice: callbacks for the events since the previous
decision, then `on_bar`. Orders placed there are queued for the next base-bar open.
Timers (`self.at`, `self.every`) run in step 2 and their orders fill at that same open.
Indicators come from `openalgo.ta` through `self.I`. Parts not built yet raise
NotImplementedError naming the phase.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from ._series import Bar, InstrumentData, PositionView, SeriesView
from ._strategy_orders import NOT_BUILT, OrderMethods
from .errors import ConfigError
from .reference import Instrument
from .units import Param

# Class attributes with a meaning of their own; every other public one is a parameter.
DECLARATIONS = (
    "symbol",
    "symbols",
    "universe",
    "underlying",
    "exchange",
    "interval",
    "product",
    "leverage",
    "size",
    "pyramiding",
    "max_adds",
    "max_position",
    "min_spacing",
    "max_risk",
    "allow_averaging_down",
    "max_hold",
    "max_daily_loss",
    "entry_from",
    "entry_until",
    "square_off",
    "drawdown_halt",
    "cooldown",
    "combined_stop",
    "combined_target",
    "lock_profit",
    "trail_sl_to_entry",
    "state",
)
CALLBACKS = ("on_fill", "on_order", "on_position", "on_risk_event")


class Strategy(OrderMethods):
    """Base class of every strategy. See the module docstring."""

    symbol: ClassVar[str | None] = None
    symbols: ClassVar[tuple[str, ...] | None] = None
    universe: ClassVar[Any] = None
    underlying: ClassVar[str | None] = None
    exchange: ClassVar[str | None] = None
    interval: ClassVar[str | None] = None
    product: ClassVar[str | None] = None
    leverage: ClassVar[float | None] = None
    size: ClassVar[Any] = None
    pyramiding: ClassVar[int] = 0
    max_daily_loss: ClassVar[float | None] = None
    entry_from: ClassVar[str | None] = None
    entry_until: ClassVar[str | None] = None
    square_off: ClassVar[str | None] = None
    drawdown_halt: ClassVar[float | None] = None
    cooldown: ClassVar[int | None] = None

    def __init__(self, **parameters: Any) -> None:
        for name, default in self.parameters().items():
            setattr(self, name, parameters.pop(name, default))
        if parameters:
            raise ConfigError(f"unknown parameters {sorted(parameters)} for {type(self).__name__}")
        self.state: dict[str, Any] = {}
        self._ctx: Any = None

    # ----- declaration ---------------------------------------------------------------

    @classmethod
    def parameters(cls) -> dict[str, Any]:
        """The strategy's parameters and their defaults (`bt.param` resolved)."""
        found: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if name.startswith("_") or name in DECLARATIONS or callable(value):
                    continue
                if isinstance(value, Param):
                    found[name] = value.default
                elif isinstance(value, bool | int | float | str):
                    found[name] = value
        return found

    @classmethod
    def parameter_ranges(cls) -> dict[str, tuple[Any, ...]]:
        """The sweep values declared with `bt.param(default, values)`, by parameter."""
        found: dict[str, tuple[Any, ...]] = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if name in DECLARATIONS or name.startswith("_"):
                    continue
                if isinstance(value, Param) and value.values:
                    found[name] = value.values
                else:
                    found.pop(name, None)
        return found

    @classmethod
    def traded_symbols(cls) -> tuple[str, ...]:
        """What the class declares it trades. Raises ConfigError unless exactly one of
        symbol, symbols, universe, underlying is set; NotImplementedError for the forms
        of later phases."""
        declared = [n for n in ("symbol", "symbols", "universe", "underlying") if getattr(cls, n)]
        if len(declared) != 1:
            raise ConfigError(
                f"{cls.__name__}: set exactly one of symbol, symbols, universe, underlying"
            )
        if declared[0] == "symbol":
            return (cls.symbol,)
        if declared[0] == "symbols":
            return tuple(cls.symbols)
        raise NotImplementedError(f"strategies on a {declared[0]} {NOT_BUILT}")

    @classmethod
    def wants_events(cls) -> bool:
        """True when the class overrides a callback, so the engine must pause for events."""
        return any(getattr(cls, name) is not getattr(Strategy, name) for name in CALLBACKS)

    # ----- hooks ---------------------------------------------------------------------

    def init(self) -> None:
        """Called once before the first bar: register indicators, timers, conditions."""

    def on_bar(self, bar: Bar | dict[str, Bar]) -> None:
        """Called at each strategy-bar close: the bar, or a mapping by symbol."""

    def on_fill(self, fill: dict) -> None:
        """Called for every fill since the previous decision (step 7)."""

    def on_order(self, order: dict) -> None:
        """Called for every order event since the previous decision (step 7)."""

    def on_position(self, position: dict) -> None:
        """Called for every position event since the previous decision (step 7)."""

    def on_risk_event(self, event: dict) -> None:
        """Called when a risk rule fires or the trading state changes (step 7)."""

    def on_stop(self) -> None:
        """Called at the end of the run."""

    # ----- data ----------------------------------------------------------------------

    @property
    def data(self) -> _DataAccess:
        """`self.data[symbol]`: series views of the strategy bars; `self.data[symbol,
        "15m"]`: another interval, each bar visible once it has closed."""
        return _DataAccess(self._ctx)

    def _single(self) -> InstrumentData:
        if self.symbols is not None or self.universe is not None:
            raise AttributeError("with symbols or universe use self.data[symbol]")
        return InstrumentData(self._ctx.store(None))

    @property
    def open(self) -> SeriesView:
        return self._single().open

    @property
    def high(self) -> SeriesView:
        return self._single().high

    @property
    def low(self) -> SeriesView:
        return self._single().low

    @property
    def close(self) -> SeriesView:
        return self._single().close

    @property
    def volume(self) -> SeriesView:
        return self._single().volume

    def history(self, symbol: str | None, n: int) -> pd.DataFrame:
        """The last n revealed strategy bars of a symbol as a DataFrame indexed by IST
        bar open (prices in rupees, volume in units)."""
        return self._ctx.store(symbol).frame(last=n)

    def I(self, func: Any, *inputs: Any, **params: Any) -> Any:  # noqa: E743
        """Register an indicator from `openalgo.ta` (the name is fixed by the contract):
        `self.I(ta.ema, self.close, 20)`. Returns a view (`[-1]`, slices, `len()`,
        `.crossed_above`, `.crossed_below`, comparisons, arithmetic), or a tuple of views
        for a function with several outputs. Raises ConfigError for a function outside
        `openalgo.ta` and for an output that looks ahead (see `_indicators`)."""
        from ._indicators import register

        return register(self._ctx.run.checks, func, inputs, params)

    @property
    def position(self) -> PositionView:
        """The position of the single instrument."""
        return self._ctx.position(self._single_symbol())

    @property
    def positions(self) -> dict[str, PositionView]:
        """Positions by symbol."""
        return {symbol: self._ctx.position(symbol) for symbol in self._ctx.symbols}

    def _single_symbol(self) -> str:
        return self._ctx.symbol_of(self._ctx.instrument_index(None))

    @property
    def orders(self) -> list:
        """Handles of the working orders."""
        return self._ctx.working_orders()

    def instrument(self, symbol: str | None = None) -> Instrument:
        """Lot size, tick size, freeze quantity and expiry of a traded instrument."""
        return self._ctx.instruments[self._ctx.symbol_of(self._ctx.instrument_index(symbol))]

    @property
    def now(self):
        """The current decision instant in IST."""
        return self._ctx.now

    @property
    def equity(self) -> float:
        """Equity of the strategy's account, rupees."""
        return self._ctx.account()[0]

    @property
    def cash(self) -> float:
        """Free cash (equity not blocked as margin), rupees."""
        return self._ctx.account()[1]

    # ----- scheduling and conditions -----------------------------------------------

    def at(self, time_ist: str, fn: Any) -> None:
        """Call `fn()` at the first base-bar open at or after `time_ist` ("HH:MM", IST) in
        every session; its orders fill at that open. Register in `init`. Raises
        ConfigError for a bad time or a call after the run started."""
        self._ctx.timers.at(time_ist, fn)

    def every(self, period: str, fn: Any) -> None:
        """Call `fn()` for the close of the first session of each "week", "month",
        "quarter" or "year" (daily or longer strategies); its orders fill at the next
        session's open. Register in `init`. Raises ConfigError otherwise."""
        self._ctx.timers.every(period, fn)

    def entry_when(self, condition: Any, **order_args: Any) -> None:
        """Enter long while the condition holds (14-strategy-api 4.1)."""
        raise NotImplementedError(f"condition strategies (entry_when) {NOT_BUILT}")

    def exit_when(self, condition: Any) -> None:
        """Exit a long while the condition holds."""
        raise NotImplementedError(f"condition strategies (exit_when) {NOT_BUILT}")

    def short_when(self, condition: Any, **order_args: Any) -> None:
        """Enter short while the condition holds."""
        raise NotImplementedError(f"condition strategies (short_when) {NOT_BUILT}")

    def cover_when(self, condition: Any) -> None:
        """Cover a short while the condition holds."""
        raise NotImplementedError(f"condition strategies (cover_when) {NOT_BUILT}")

    def pair_long(self, *args: Any, **kwargs: Any) -> Any:
        """Buy leg 0 and sell leg 1 as one leg group."""
        raise NotImplementedError("pairs are planned for Phase 3")

    def pair_short(self, *args: Any, **kwargs: Any) -> Any:
        """Sell leg 0 and buy leg 1 as one leg group."""
        raise NotImplementedError("pairs are planned for Phase 3")


class _DataAccess:
    """`self.data`: indexing by symbol, or (symbol, interval)."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def __getitem__(self, key: str | tuple[str, str]) -> InstrumentData:
        if isinstance(key, tuple):
            symbol, interval = key
            return InstrumentData(self._ctx.store(symbol, interval))
        return InstrumentData(self._ctx.store(key))


__all__ = ["CALLBACKS", "DECLARATIONS", "Strategy"]
