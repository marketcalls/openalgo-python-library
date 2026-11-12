"""The HTML tearsheet: one self-contained dashboard page for a strategy or a combined
portfolio, in one of four modes; and the metrics as a DataFrame."""

from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path

import pandas as pd

from ... import __version__
from ..errors import ConfigError
from . import _dashboard
from ._dashboard import data
from ._definitions import DEFINITIONS, GROUPS
from ._performance import Performance

DEFAULT_TITLE = "Backtest"
#: The name of a benchmark when none is given with it.
DEFAULT_BENCHMARK = "Benchmark"
#: Report modes, in the order the documentation lists them.
MODES = ("full", "basic", "metrics", "charts")


def tearsheet(
    result_like: object,
    path: str | PathLike | None = None,
    *,
    title: str | None = None,
    mode: str = "full",
    log_scale: bool = False,
) -> str:
    """Render a performance report as one self-contained HTML dashboard.

    `result_like` is a `Performance` (from `openalgo.backtest.metrics.analyse`), or any
    object with a `performance` attribute holding one. Optional attributes of that
    object are read when present:

        name            the report title (overridden by `title=`)
        assumptions     what the engine assumed, shown in the Notes: a mapping gives
                        "key: value" lines, any other iterable one line per item; a line
                        that starts with "warning:" is shown as a warning
        benchmark_name  the benchmark's name on the page ("NIFTY"); "Benchmark" when a
                        benchmark has no name
        strategies      a mapping {name: Performance} of the strategies of a combined
                        result; with two or more, a Strategies section compares them and
                        gives the correlation of their session returns (full mode)

    The page follows the layout of OpenStatz's dashboard: a sticky header with links to
    the sections, a theme toggle and Export PDF (the browser's print), the title line
    (first and last session, sessions, the benchmark), then numbered sections. `mode`
    chooses them:

        "full"     Overview (headline tiles); Performance (cumulative return with the
                   benchmark, rolling Sharpe, volatility and win rate, CAGR, maximum
                   drawdown and Sharpe by horizon, the period-return tiles); Risk
                   (drawdown chart, risk tiles, the benchmark tiles and rolling beta,
                   the ten worst drawdowns, losing streaks of sessions); Monthly
                   (monthly and weekly heatmaps, the monthly tiles, yearly returns
                   against the benchmark); Distribution (tiles, histogram of session
                   returns, box plot of session and monthly returns); Metrics (every
                   metric, the benchmark's beside it, sortable); Trades (trade tiles,
                   the trade list of `openalgo.backtest.Result.trade_list`, sortable and
                   paged, the charges by component, the R-multiples when a closed trade
                   has an initial risk); Strategies (when given); Notes (how the figures
                   were computed, the assumptions and warnings)
        "basic"    the overview, cumulative return and period returns, the drawdown
                   chart and worst drawdowns, the monthly heatmap and yearly returns,
                   the trades and charges, and the notes
        "metrics"  every metric with its value and definition, and the assumptions
        "charts"   every chart and nothing else

    Every figure is the SDK's: a metric of the Performance or a cell of its tables. A
    tile's subline is the benchmark's figure, measured as the strategy's is, where a
    benchmark has one. `log_scale` draws the cumulative return on a log scale of growth.

    The page has inline CSS, inline SVG charts and a small inline script for the theme
    toggle, the section links, sorting and paging, the weekly year picker and the chart
    crosshair; it fetches nothing and reads in full with script turned off. It follows
    the reader's light or dark setting and reads at phone width. With `path` None the
    HTML is returned; otherwise it is written there as UTF-8 and the path is returned
    as a string. Raises ConfigError when no Performance can be found or for another
    mode.
    """
    performance, name, assumptions, strategies, bench_name = _resolve(result_like)
    page = render(
        performance,
        title=title or name or DEFAULT_TITLE,
        assumptions=assumption_lines(assumptions),
        strategies=strategies,
        mode=mode,
        log_scale=log_scale,
        benchmark_name=bench_name,
    )
    if path is None:
        return page
    target = Path(path)
    target.write_text(page, encoding="utf-8")
    return str(target)


def metrics_table(result_like: object) -> pd.DataFrame:
    """Every metric of a `Performance` (or of an object with a `performance`
    attribute) as a DataFrame indexed by key, in report order: group, label, value,
    unit and definition (see `openalgo.backtest.metrics.DEFINITIONS`). Raises
    ConfigError when no Performance can be found."""
    performance = _resolve(result_like)[0]
    metrics = performance.metrics
    rows = [
        {"key": key, "group": heading, **DEFINITIONS[key]._asdict(), "value": metrics[key]}
        for heading, keys in GROUPS.items()
        for key in keys
        if key in metrics
    ]
    columns = ["key", "group", "label", "value", "unit", "definition"]
    return pd.DataFrame(rows, columns=columns).set_index("key")


def _resolve(
    result_like: object,
) -> tuple[Performance, str | None, object, Mapping | None, str | None]:
    if isinstance(result_like, Performance):
        return result_like, None, None, None, None
    performance = getattr(result_like, "performance", None)
    if not isinstance(performance, Performance):
        raise ConfigError(
            "a report needs a Performance (from openalgo.backtest.metrics.analyse) or a "
            "result with a .performance attribute holding one"
        )
    name = getattr(result_like, "name", None)
    strategies = getattr(result_like, "strategies", None)
    if not isinstance(strategies, Mapping) or len(strategies) < 2:
        strategies = None
    assumptions = getattr(result_like, "assumptions", None)
    return performance, name, assumptions, strategies, getattr(result_like, "benchmark_name", None)


def assumption_lines(assumptions: object) -> list[str]:
    """Assumptions as plain lines: a mapping gives "key: value" lines, a string one
    line, any other iterable one line per item, None none."""
    if assumptions is None:
        return []
    if isinstance(assumptions, str):
        return [assumptions]
    if isinstance(assumptions, Mapping):
        return [f"{key}: {value}" for key, value in assumptions.items()]
    if isinstance(assumptions, Iterable):
        return [str(item) for item in assumptions]
    return [str(assumptions)]


def render(
    performance: Performance,
    *,
    title: str,
    assumptions: list[str],
    strategies: Mapping[str, Performance] | None,
    mode: str = "full",
    log_scale: bool = False,
    benchmark_name: str | None = None,
) -> str:
    """The HTML page for `performance` in `mode` (see `tearsheet`). Raises ConfigError
    for an unknown mode."""
    if mode not in MODES:
        raise ConfigError(f"mode is one of {MODES}, got {mode!r}")
    view = _dashboard.View(
        performance=performance,
        bench=data.benchmark_performance(performance),
        bench_name=benchmark_name or DEFAULT_BENCHMARK,
        title=title,
        assumptions=assumptions,
        strategies=strategies,
        log_scale=log_scale,
        parts=_dashboard.MODE_PARTS[mode],
    )
    return _dashboard.render(view, __version__)


__all__ = [
    "DEFAULT_BENCHMARK",
    "DEFAULT_TITLE",
    "MODES",
    "assumption_lines",
    "metrics_table",
    "render",
    "tearsheet",
]
