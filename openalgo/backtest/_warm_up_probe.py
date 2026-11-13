"""What a strategy's indicators need before `start` (decision D25; the rule is in
`_warm_up`): from the run's own `init`, and for a sweep before any data is loaded.

A warm-up is a function of the indicator call alone, measured on the engine's reference
bars (`_warm_up_reference`), so the history a run needs follows from the indicators its
`init` registers (`history_needs`). A single run learns them from its own `init`, which
runs once, on the run's bars; the history is loaded and joined afterwards
(`_warm_up_attach`).

A sweep must load the history once, for the largest need of all its combinations, before
any combination runs (its worker processes load no data). So `init` of each combination
is first run on a probe context (`probe_largest`): every series it reads (`self.close`,
`self.data[symbol]`, `self.data[symbol, "W"]`) holds the reference bars, none of them
visible yet, as in the run's own `init`, and each `self.I` records its warm-up. Nothing
is traded and no data is loaded. Timers are accepted (the run checks them), positions
are flat, the account holds the capital, no order is working, and an order raises
BacktestError, as it does in the run's `init`. A sweep's `init` therefore runs twice for
each combination: once here, once in its run.

A `bt.from_signals` function is never run on the probe: its indicators are its own, out
of the engine's sight, so its series is given the history of `FUNCTION_LOOKBACK`, a
200-bar EMA of closes, the longest lookback in common use. The strategy does not wait
for it (it has no `self.I` indicator).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._arrays import made_by
from ._causality_checks import CheckOutcome, IndicatorChecks
from ._series import PositionView, SeriesStore
from ._timers import Timers
from ._warm_up import SeriesNeed, series_needs
from .errors import BacktestError, ConfigError

# The indicator whose warm-up a from_signals function's series is given: a 200-bar EMA
# of closes (see the module docstring).
FUNCTION_LOOKBACK = ("ema", "close", 200)
FROM_SIGNALS = "bt.from_signals"
ORDERS_IN_INIT = (
    "orders can only be placed, changed or cancelled while the engine is paused at a "
    "decision point (in on_bar, a timer or a callback), not in init"
)


@dataclass(frozen=True)
class ProbeSetup:
    """What `init` may read on the probe context: the names the strategy addresses (its
    symbols, or futures underlyings), the strategy interval, the instrument of each name
    (a futures underlying's first contract), the product, the default size and the
    capital in rupees."""

    names: tuple[str, ...]
    interval: str
    instruments: Mapping[str, Any]
    product: str
    default_size: Any
    capital: float


def probe_setup(strategy: type, plan: Any, instruments: list[Any], futures: Any) -> ProbeSetup:
    """The probe's setup for a run of `strategy` (`_settings.RunPlan`, its instruments
    and its `_futures.FuturesPlan`, None without one)."""
    if futures is None:
        names = tuple(plan.symbols)
        by_name = {instrument.symbol: instrument for instrument in instruments}
    else:
        names = tuple(series.name for series in futures.series)
        by_name = {series.name: series.contracts[0] for series in futures.series}
    return ProbeSetup(names, plan.interval, by_name, plan.product, strategy.size, plan.capital)


def _function_needs(
    strategy: type, name: str, interval: str, checks: IndicatorChecks
) -> tuple[SeriesNeed, ...]:
    """The need given a from_signals function's series (module docstring)."""
    from openalgo import ta

    function, field, length = FUNCTION_LOOKBACK

    def outputs(bars: Mapping[str, Any]) -> list[np.ndarray]:
        values = getattr(ta, function)(pd.Series(bars[field], index=bars["times"]), length)
        return [np.asarray(values, dtype=float)]

    label = f"{function}({field}, {length})"
    bars, _ = checks.warm_up(label, False, outputs)
    if bars is None:
        return ()
    return (SeriesNeed(name, interval, bars, f"the function of {strategy.__name__}, as {label}"),)


def history_needs(
    strategy: type,
    outcomes: Iterable[CheckOutcome],
    setup: ProbeSetup,
    checks: IndicatorChecks,
) -> tuple[SeriesNeed, ...]:
    """The bars each series needs before `start`: the warm-up of the `self.I`
    indicators of `outcomes`, or the lookback of a from_signals function."""
    if made_by(strategy) == FROM_SIGNALS:
        return _function_needs(strategy, setup.names[0], setup.interval, checks)
    return series_needs(outcomes)[0]


class _ReferenceStore(SeriesStore):
    """The reference bars as the series of one name and interval: no bar is visible in
    `init`, as in a run's, and none is revealed."""

    declaring = True

    def visible(self) -> int:
        return 0


class _ProbeChecks:
    """The checks recorder of the probe: each indicator's warm-up only (the truncation
    check runs on the run's bars), kept in registration order."""

    def __init__(self, shared: IndicatorChecks) -> None:
        self._shared = shared
        self.outcomes: list[CheckOutcome] = []

    def check(
        self,
        key: str,
        label: str,
        compute: Any,
        whole: Sequence[np.ndarray],
        anchored: bool,
        series: tuple[str, str] | None = None,
        reference: Any = None,
    ) -> CheckOutcome:
        warm_up, note = self._shared.warm_up(label, anchored, reference)
        outcome = CheckOutcome(label, None, warm_up, note, series=series, anchored=anchored)
        self.outcomes.append(outcome)
        return outcome


class _ProbeEngine:
    """Numbers the timers `init` registers; the run's engine checks them."""

    def __init__(self) -> None:
        self._count = 0

    def _next(self, _value: Any) -> int:
        self._count += 1
        return self._count - 1

    add_timer_at = _next
    add_timer_every = _next


@dataclass(frozen=True)
class _Run:
    """`RunContext.run` as `self.I` reads it."""

    checks: _ProbeChecks


class ProbeContext:
    """The run context of the probe (module docstring): what `init` reads of a
    `RunContext`, on the reference bars."""

    def __init__(self, setup: ProbeSetup, checks: IndicatorChecks) -> None:
        self._setup = setup
        self._columns = checks.reference_columns()
        self._stores: dict[tuple[str, str], _ReferenceStore] = {}
        self.run = _Run(_ProbeChecks(checks))
        self.names = setup.names
        self.symbols = list(setup.names)
        self.instruments = dict(setup.instruments)
        self.product = setup.product
        self.default_size = setup.default_size
        self.timers = Timers(_ProbeEngine())
        self.now = None

    def store(self, symbol: str | None, interval: str | None = None) -> _ReferenceStore:
        """The reference bars as the series of a name at an interval (the strategy's by
        default). Raises ConfigError for a name the strategy does not trade."""
        name = self.symbols[self.instrument_index(symbol)]
        interval = interval or self._setup.interval
        key = (name, interval)
        if key not in self._stores:
            shown = name if interval == self._setup.interval else f"{name}[{interval}]"
            self._stores[key] = _ReferenceStore(name, interval, self._columns, shown)
        return self._stores[key]

    def symbol_of(self, index: int) -> str:
        return self.symbols[index]

    def instrument_index(self, symbol: str | None) -> int:
        """As `RunContext.instrument_index`. Raises ConfigError for an unknown name, or
        for None with several."""
        if symbol is None:
            if len(self.symbols) != 1:
                raise ConfigError("this strategy trades several symbols: pass symbol=")
            return 0
        if symbol not in self.symbols:
            raise ConfigError(f"{symbol!r} is not traded by this strategy: {self.symbols}")
        return self.symbols.index(symbol)

    def position(self, symbol: str, product: str | None = None) -> PositionView:
        return PositionView.flat(symbol)

    def account(self) -> tuple[float, float, float, float, float, float]:
        capital = self._setup.capital
        return (capital, capital, 0.0, 0.0, 0.0, 0.0)

    def working_orders(self) -> list:
        return []

    def sizing_store(self, instrument: int) -> _ReferenceStore:
        return self.store(self.symbols[instrument])

    def _refuse(self, *args: Any) -> Any:
        raise BacktestError(ORDERS_IN_INIT)

    submit = submit_target = submit_rebalance = order = cancel = _refuse


def probe_largest(
    strategy: type, setup: ProbeSetup, parameter_sets: Sequence[Mapping[str, Any] | None]
) -> tuple[SeriesNeed, ...]:
    """The largest need of each series of `strategy` over the parameter sets (see the
    module docstring), in first-registration order. Raises what the strategy's `init`
    raises."""
    checks = IndicatorChecks("probe")
    largest: dict[tuple[str, str], SeriesNeed] = {}
    for params in parameter_sets:
        context = ProbeContext(setup, checks)
        if made_by(strategy) != FROM_SIGNALS:
            instance = strategy(**(params or {}))
            instance._ctx = context
            instance.init()
        for need in history_needs(strategy, context.run.checks.outcomes, setup, checks):
            held = largest.get((need.symbol, need.interval))
            if held is None or need.bars > held.bars:
                largest[(need.symbol, need.interval)] = need
    return tuple(largest.values())


def sweep_needs(
    strategy: type,
    plan: Any,
    instruments: list[Any],
    futures: Any,
    parameter_sets: Sequence[Mapping[str, Any] | None] | None,
) -> tuple[SeriesNeed, ...] | None:
    """The largest need of a sweep over its combinations' `parameter_sets`
    (`probe_largest`); None for a single run (no parameter sets), which learns its
    needs from its own `init`."""
    if parameter_sets is None:
        return None
    setup = probe_setup(strategy, plan, instruments, futures)
    return probe_largest(strategy, setup, parameter_sets)


__all__ = [
    "FUNCTION_LOOKBACK",
    "ORDERS_IN_INIT",
    "ProbeContext",
    "ProbeSetup",
    "history_needs",
    "probe_largest",
    "probe_setup",
    "sweep_needs",
]
