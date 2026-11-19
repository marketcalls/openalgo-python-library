"""The HTML tearsheet: one self-contained dashboard page for a strategy or a combined
portfolio, in one of four modes or a custom layout (`custom`); and the metrics as a
DataFrame."""

from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path

import pandas as pd

from ... import __version__
from ..errors import ConfigError
from . import _dashboard
from ._custom import _layout
from ._dashboard import data
from ._definitions import DEFINITIONS, GROUPS
from ._performance import Performance
from .custom import Layout

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
    layout: Layout | None = None,
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
        overfitting     the quick overfitting checks already computed for this
                        performance; without it the page computes them

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
                   gross and net P&L among them; the trading cost tiles of the closed
                   trades: turnover, charges, slippage, trading cost, its share of
                   gross P&L and its basis points of that turnover; MAE and
                   MFE: the excursion tiles, a histogram of MAE percent and one of MFE
                   percent, and each against the trade's return as scatter charts,
                   winners as circles and the others as crosses, or one line when the
                   trades have no excursion columns or none closed; the trade list of
                   `openalgo.backtest.Result.trade_list` with its MAE and MFE, sortable
                   and paged; the charges by component with the slippage and the
                   trading cost under their total; the R-multiples when a closed trade
                   has an initial risk); Overfitting checks (the quick checks of the
                   trade list, `openalgo.backtest._robust._quick`: random out of
                   sample, trade reshuffle, trade resample and variance at 200 runs,
                   scored, or "too few trades to judge" below 30 closed trades);
                   Strategies (when given); Notes (how the figures were computed, the
                   assumptions and warnings)
        "basic"    the overview, cumulative return and period returns, the drawdown
                   chart and worst drawdowns, the monthly heatmap and yearly returns,
                   the trades, the trading cost, the MAE and MFE histograms and the
                   charges, the overfitting checks, and the notes
        "metrics"  every metric with its value and definition, and the assumptions
        "charts"   every chart and nothing else, the MAE and MFE histograms and
                   scatter charts included

    A money tile shows paise when the amount is under Rs 1,000 (Rs 3.60), whole rupees
    above.

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

    `layout` (a `report.custom.Layout`) makes a custom tearsheet: the built-in sections
    and parts it chooses, sections and charts of the user's, and its colours, on the
    same page shell (see `openalgo.backtest.report.custom`). Its blocks' callables
    receive `result_like`. With a layout, `mode` stays "full" (the layout's `base` is
    the mode it starts from), else ConfigError. Without one (the default) the page is
    exactly the built-in tearsheet.
    """
    return write_tearsheet(
        result_like,
        path,
        subject=result_like,
        title=title,
        mode=mode,
        log_scale=log_scale,
        layout=layout,
    )


def write_tearsheet(
    result_like: object,
    path: str | PathLike | None,
    *,
    subject: object,
    title: str | None = None,
    mode: str = "full",
    log_scale: bool = False,
    layout: Layout | None = None,
) -> str:
    """`tearsheet` (which documents the arguments), with `subject` the object a custom
    layout's callables receive: a `Result` hands itself, while its page reads the
    adapter it passes as `result_like`."""
    performance, name, assumptions, strategies, bench_name = _resolve(result_like)
    page = render(
        performance,
        title=title or name or DEFAULT_TITLE,
        assumptions=assumption_lines(assumptions),
        strategies=strategies,
        mode=mode,
        log_scale=log_scale,
        benchmark_name=bench_name,
        overfitting=getattr(result_like, "overfitting", None),
        layout=layout,
        subject=subject,
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
    overfitting: object = None,
    layout: Layout | None = None,
    subject: object = None,
) -> str:
    """The HTML page for `performance` in `mode` (see `tearsheet`); `overfitting` is its
    quick overfitting checks when already computed. With `layout`, the custom page,
    whose blocks' callables receive `subject`. Raises ConfigError for an unknown mode,
    a layout with a mode other than "full", or a layout that is not a `Layout`."""
    if mode not in MODES:
        raise ConfigError(f"mode is one of {MODES}, got {mode!r}")
    if layout is not None:
        _check_layout(layout, mode)
    parts = (
        _dashboard.MODE_PARTS[mode]
        if layout is None
        else _layout.parts_of(layout.base, layout.hide, layout.show)
    )
    view = _dashboard.View(
        performance=performance,
        bench=data.benchmark_performance(performance),
        bench_name=benchmark_name or DEFAULT_BENCHMARK,
        title=title,
        assumptions=assumptions,
        strategies=strategies,
        log_scale=log_scale,
        parts=parts,
        overfitting=overfitting,
    )
    if layout is None:
        return _dashboard.render(view, __version__)
    return _layout.render(layout, view, subject, __version__)


def _check_layout(layout: object, mode: str) -> None:
    if not isinstance(layout, Layout):
        raise ConfigError(
            f"layout is a report.custom.Layout (or None for the built-in page), got {layout!r}"
        )
    if mode != "full":
        raise ConfigError(
            f"a layout gives its own starting mode: pass Layout(base={mode!r}) and leave mode out"
        )


__all__ = [
    "DEFAULT_BENCHMARK",
    "DEFAULT_TITLE",
    "MODES",
    "assumption_lines",
    "metrics_table",
    "render",
    "tearsheet",
    "write_tearsheet",
]
