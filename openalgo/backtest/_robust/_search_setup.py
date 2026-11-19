"""Setting up the searching checks of a report (design 17, sections 2.1, 2.3, 2.16 and
2.17): the grid of the parameter ranges (`_grid`), the run's bars with the history of the
largest need of every combination they run (`_run_bars.bars_for`), a first rerun on them
that does the indicator checks once for the reruns after it, and a runner of their own
bars that shares the bar checks' worker processes (`_reruns.Reruns.sibling`).

What a searching check reads (`SearchRun`): the strategy and the run's capital, the
run's trade list and `Result` (None for `bt.walk_forward`, which runs no backtest of its
own), the objective's value on it, the grid (None without ranges), the run's own
`params=`, the bars and the runner (None while the report plans), the report's seed, the
run's session dates as the bars hold them (the windows of 2.1 and 2.3 cut them) and the
run's own `start` and `end`. A rerun of a combination changes the parameters that differ from the
run's (`Grid.changes`); one of a window changes `start` and `end` (`_windows`).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._warm_up_probe import params_key
from ._grid import Grid, cube, run_parameters, search_grid
from ._rerun_one import Known, rerun
from ._reruns import REAL, Family, Reruns
from ._run_bars import RunBars, bars_for, run_arguments
from ._trades import TradeRun
from ._windows import Window, window_arguments

#: The first day of the session-day numbering (`_ohlc.session_days`).
EPOCH = dt.date(1970, 1, 1)


@dataclass(frozen=True, eq=False)
class SearchRun:
    """The run as the searching checks read it (module docstring)."""

    strategy: type
    capital: float
    trades: TradeRun | None
    result: Any
    real: float
    grid: Grid | None
    given: Mapping[str, Any] | None
    bars: RunBars
    reruns: Reruns | None
    seed: int
    dates: tuple[dt.date, ...]
    start: Any
    end: Any

    def window(self, window: Window) -> tuple[tuple[str, Any], ...]:
        """The `start` and `end` of a rerun of `window` of the run's sessions."""
        return window_arguments(window, self.dates, self.start, self.end)

    def family(
        self,
        grid: Grid | None = None,
        position: int | None = None,
        window: Window | None = None,
        detail: bool = False,
    ) -> Family:
        """The rerun of combination `position` of `grid` (the run's own parameters for
        None) over `window` (the whole run for None), keeping its detail when asked."""
        params = () if grid is None or position is None else grid.changes(position)
        arguments = () if window is None else self.window(window)
        return Family(REAL, arguments=arguments, params=params, detail=detail)


def session_dates(bars: RunBars) -> tuple[dt.date, ...]:
    """The IST session dates of the run's bars, in order."""
    return tuple(EPOCH + dt.timedelta(days=session.day) for session in bars.sessions)


def parameter_sets(
    strategy: type, given: Mapping[str, Any] | None, grid: Grid | None, step: float | None
) -> list[dict[str, Any]]:
    """The `params=` of every combination the searching checks may run: the run's own,
    the grid's, and the neighbourhood of `step` when the parameter neighbourhood check
    runs with no grid (`step` None when it does not run); each once, in that order (a
    combination that sets the parameters a run would hold anyway is the same run).
    Raises ConfigError for a parameter value that cannot be told apart from another."""
    sets = [run_parameters(strategy, given)]
    if grid is not None:
        sets += [dict(combo) for combo in grid.combos]
    elif step is not None:
        found = cube(strategy, given, step)
        sets += [] if found is None else [dict(combo) for combo in found.combos]
    unique: dict[tuple, dict[str, Any]] = {}
    for params in sets:
        unique.setdefault(params_key(run_parameters(strategy, params)), params)
    return list(unique.values())


def search_preview(
    result: Any, ranges: Mapping[str, Any], seed: int, trades: TradeRun, real: float, step: Any
) -> tuple[SearchRun, dict[str, Any]]:
    """(the `SearchRun` of `result` with no runner yet, the arguments of its reruns),
    reading the run's bars again with the history of every combination's need. Raises
    what reading the run's bars raises, and ConfigError for an empty range."""
    record = result._record
    arguments = run_arguments(record)
    given = arguments["params"]
    grid = search_grid(record.strategy, ranges, given)
    sets = parameter_sets(record.strategy, given, grid, step)
    bars, reruns = bars_for(record.strategy, record.client, arguments, sets)
    preview = SearchRun(
        record.strategy,
        trades.capital,
        trades,
        result,
        real,
        grid,
        given,
        bars,
        None,
        seed,
        session_dates(bars),
        reruns["start"],
        reruns["end"],
    )
    return preview, reruns


def first_known(strategy: type, arguments: dict[str, Any], bars: RunBars) -> Known:
    """The indicator checks a first rerun on `bars` does, for the reruns after it."""
    _, known = rerun(strategy, arguments, bars.symbols, list(bars.frames), Known())
    return known


__all__ = ["EPOCH", "SearchRun", "first_known", "parameter_sets", "search_preview", "session_dates"]
