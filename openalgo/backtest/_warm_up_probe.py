"""What a strategy's indicators need before `start` (decision D25; the rule is in
`_warm_up`), learnt before any data is loaded.

A warm-up is a function of the indicator call alone, measured on the engine's reference
bars (`_warm_up_measure`), so the history a run needs follows from the indicators its
`init` registers (`history_needs`). To know them before the history is loaded, `init`
first runs on a probe context (`ProbeContext`): every series it reads (`self.close`,
`self.data[symbol]`, `self.data[symbol, "W"]`) holds the reference bars, none of them
visible yet, as in the run's own `init`, and each `self.I` records its warm-up. Nothing
is traded and no data is loaded. Timers are accepted (the run checks them), positions
are flat, the account holds the capital, no order is working, and an order raises
BacktestError, as it does in the run's `init`.

So `init` runs twice for every run: once on a probe instance here, then once in the run,
on the history and the run's bars together. A single run probes its own parameters, a
sweep every combination's (`probe_needs`, by `params_key`; the history is loaded once,
for the `largest` need of each series), and each run keeps what its own probe asked for,
so a combination of a sweep keeps exactly what its single run keeps.

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
from ._warm_up_measure import ReferenceCall
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
    found = checks.warm_up(label, None, ReferenceCall(outputs))
    if found.bars is None:
        return ()
    described = f"the function of {strategy.__name__}, as {label}"
    return (SeriesNeed(name, interval, found.bars, described, more=found.more),)


def history_needs(
    strategy: type,
    outcomes: Iterable[CheckOutcome],
    setup: ProbeSetup,
    checks: IndicatorChecks,
) -> tuple[SeriesNeed, ...]:
    """The needs of each series before `start`: those of the `self.I` indicators of
    `outcomes`, or the lookback of a from_signals function."""
    if made_by(strategy) == FROM_SIGNALS:
        return _function_needs(strategy, setup.names[0], setup.interval, checks)
    return series_needs(outcomes)[0]


class _ReferenceStore(SeriesStore):
    """The reference bars as the series of one name and interval: no bar is visible in
    `init`, as in a run's, and none is revealed. `reference` tells an indicator's errors
    where it was computed (`_indicators`)."""

    declaring = True
    reference = True

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
        anchor: str | None,
        series: tuple[str, str] | None = None,
        reference: ReferenceCall | None = None,
    ) -> CheckOutcome:
        """`_causality_checks.RunChecks.check` without the truncation check: the
        outcome of the indicator's warm-up, recorded."""
        found = self._shared.warm_up(label, anchor, reference)
        outcome = CheckOutcome(
            label, None, found.bars, found.note, series=series, anchor=anchor, more=found.more
        )
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
    `RunContext`, on the reference bars. `run.checks` records each indicator's warm-up;
    `names` and `symbols` are what the strategy addresses, `instruments` their
    instruments, `product` and `default_size` the run's, `timers` accepts timers, `now`
    is None (no decision yet) and `declaring` is True (it is `init`)."""

    declaring = True

    def __init__(self, setup: ProbeSetup, checks: IndicatorChecks) -> None:
        self._setup = setup
        self._checks = checks
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
        default), made on first use. Raises ConfigError for a name the strategy does not
        trade."""
        name = self.symbols[self.instrument_index(symbol)]
        interval = interval or self._setup.interval
        key = (name, interval)
        if key not in self._stores:
            shown = name if interval == self._setup.interval else f"{name}[{interval}]"
            columns = self._checks.reference_columns()
            self._stores[key] = _ReferenceStore(name, interval, columns, shown)
        return self._stores[key]

    def symbol_of(self, index: int) -> str:
        """The name the strategy addresses at registration index `index`."""
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
        """A flat position: nothing is traded on the probe."""
        return PositionView.flat(symbol)

    def account(self) -> tuple[float, float, float, float, float, float]:
        """(equity, free cash, realized, unrealized, costs, margin) in rupees: the
        capital, all of it free."""
        capital = self._setup.capital
        return (capital, capital, 0.0, 0.0, 0.0, 0.0)

    def working_orders(self) -> list:
        """No order works on the probe."""
        return []

    def sizing_store(self, instrument: int) -> _ReferenceStore:
        """The reference bars of the instrument at registration index `instrument`."""
        return self.store(self.symbols[instrument])

    def _refuse(self, *args: Any) -> Any:
        raise BacktestError(ORDERS_IN_INIT)

    submit = submit_target = submit_rebalance = order = cancel = _refuse


def combination_needs(
    strategy: type, setup: ProbeSetup, params: Mapping[str, Any] | None, checks: IndicatorChecks
) -> tuple[SeriesNeed, ...]:
    """The needs of `strategy` with the parameter values `params` (module docstring),
    measured with `checks`. Raises what the strategy's `init` raises."""
    context = ProbeContext(setup, checks)
    if made_by(strategy) != FROM_SIGNALS:
        instance = strategy(**(params or {}))
        instance._ctx = context
        instance.init()
    return history_needs(strategy, context.run.checks.outcomes, setup, checks)


def params_key(params: Mapping[str, Any] | None) -> tuple:
    """The key of a parameter set among the probed ones: its items, sorted (parameter
    values are numbers, booleans and strings, `Strategy.parameters`)."""
    return tuple(sorted((params or {}).items()))


def largest(found: Iterable[tuple[SeriesNeed, ...]]) -> tuple[SeriesNeed, ...]:
    """The largest need of each series and kind over the needs of several parameter sets,
    in first-registration order: the most bars, and each anchor period."""
    held: dict[tuple[str, str, str | None], SeriesNeed] = {}
    for needs in found:
        for need in needs:
            key = (need.symbol, need.interval, need.anchor)
            kept = held.get(key)
            if kept is None or (need.bars, need.more) > (kept.bars, kept.more):
                held[key] = need
    return tuple(held.values())


def probe_needs(
    strategy: type,
    setup: ProbeSetup,
    parameter_sets: Sequence[Mapping[str, Any] | None],
    checks: IndicatorChecks,
) -> dict[tuple, tuple[SeriesNeed, ...]]:
    """The needs of `strategy` with each of `parameter_sets` (one for a single run, every
    combination's for a sweep), by `params_key`. Raises what the strategy's `init`
    raises."""
    return {
        params_key(params): combination_needs(strategy, setup, params, checks)
        for params in parameter_sets
    }


__all__ = [
    "FUNCTION_LOOKBACK",
    "ORDERS_IN_INIT",
    "ProbeContext",
    "ProbeSetup",
    "combination_needs",
    "history_needs",
    "largest",
    "params_key",
    "probe_needs",
    "probe_setup",
]
