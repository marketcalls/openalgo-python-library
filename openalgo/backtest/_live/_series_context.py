"""The run context of a strategy in a live session (16-session-design 3.1 step 11, unit
6): the backtest's run context (`_context.RunContext`) over the live stores of
`series`, so the same strategy class reads its bars, places its orders and registers
its indicators through it. Part of `series`, which re-exports `LiveRunContext`.

Every store's clock is the decision instant of the kernel's latest pause, so a bar is
visible from the decision at or after its close, at the strategy interval and at any
other (`self.data[symbol, "15m"]`), as in a backtest. `self.I` is registered in `init`
only: the strategy's own rule (`Strategy.I` raises `IndicatorOutsideInitError` once the
context has left its declaration phase), which the live context keeps unchanged. A
history too short for an indicator refuses the start (`check_history`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .._context import RunContext, RunData
from .._series import SeriesStore
from ..errors import ConfigError
from ..reference import Instrument

if TYPE_CHECKING:
    from .series import LiveSeries, LiveStore


class LiveRunContext(RunContext):
    """The run context of one strategy in a live session (module docstring): the
    backtest's `RunContext` with live stores, one `series.LiveSeries` per traded symbol
    in `series` (by symbol).

    The other arguments are `RunContext`'s; `run.checks` records the indicators'
    causality checks and warm-ups (`_causality_checks.RunChecks`). Raises ConfigError
    when a traded symbol has no series, or a series is of another exchange than the run.
    """

    def __init__(
        self,
        engine: Any,
        instruments: list[Instrument],
        product: str,
        default_size: Any,
        run: RunData,
        series: Mapping[str, LiveSeries],
    ) -> None:
        for instrument in instruments:
            found = series.get(instrument.symbol)
            if found is None or found.exchange != run.exchange:
                raise ConfigError(
                    f"{instrument.symbol} is traded on {run.exchange} and needs a live "
                    "series of it on that exchange"
                )
        self._live = {instrument.symbol: series[instrument.symbol] for instrument in instruments}
        super().__init__(engine, instruments, product, default_size, run)

    def _decision(self) -> int | None:
        """The clock of the stores: the latest decision instant (UTC nanoseconds), None
        before the first. `RunContext.__init__` sets it after it has made the stores,
        which read it only when they are read."""
        return self.decided_at

    def _bar_store(self, symbol: str, number: int) -> LiveStore:
        return self._live[symbol].store(self.run.interval, self._decision)

    def store(self, symbol: str | None, interval: str | None = None) -> SeriesStore:
        """The bars of a symbol at the strategy interval, or at `interval`, made on first
        use from the same series. Raises ConfigError for an unknown symbol or an interval
        the base bars cannot give."""
        name = self.symbol_of(self.instrument_index(symbol))
        if interval is None or interval == self.run.interval:
            return self.stores[name]
        key = (name, interval)
        if key not in self.timeframes:
            labelled = f"{name}[{interval}]"
            store = self._live[name].store(interval, self._decision, name=labelled)
            self.timeframes[key] = self._marked(store)
        return self.timeframes[key]

    def check_history(self) -> None:
        """Refuse the start when an indicator `init` registered could not be computed at
        all on the bars it saw, the history (a series too short for it, `_indicators`): a
        backtest leaves its values unknown for the whole run, and they would never follow
        the session's bars. The session plan calls it right after `init` (16-session-
        design 3.1 step 11), so a start before the open, which has no catch-up pause, is
        refused before the session trades. Raises ConfigError naming each indicator and
        the bars it had."""
        short = [o for o in self.run.checks.outcomes if o.too_short is not None]
        if short:
            named = "; ".join(f"{o.label} on {o.too_short} bars" for o in short)
            raise ConfigError(
                f"the history before today is too short to compute {named}: a live session "
                "needs enough broker history for every indicator to compute at the start"
            )

    def begin_run(self) -> None:
        """End the declaration phase (`RunContext.begin_run`) after `check_history` once
        more, so a run whose start skipped the check is refused at its first pause rather
        than run with indicators that never follow its bars. Raises ConfigError as
        `check_history` does."""
        self.check_history()
        super().begin_run()


__all__ = ["LiveRunContext"]
