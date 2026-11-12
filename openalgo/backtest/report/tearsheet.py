"""The HTML tearsheet: one self-contained page for a strategy or a combined portfolio,
in one of four modes; and the metrics as a DataFrame."""

from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import pandas as pd

from ... import __version__
from ..errors import ConfigError
from . import _pages as pages
from . import _sections as sections
from ._definitions import DEFINITIONS, GROUPS
from ._format import text
from ._performance import Performance
from ._style import STYLESHEET

DEFAULT_TITLE = "Backtest"
#: The kicker over the title of each mode.
_KICKERS = {
    "full": "Backtest report",
    "basic": "Backtest summary",
    "metrics": "Backtest metrics",
    "charts": "Backtest charts",
}


def tearsheet(
    result_like: object,
    path: str | PathLike | None = None,
    *,
    title: str | None = None,
    mode: str = "full",
    log_scale: bool = False,
) -> str:
    """Render a performance report as one self-contained HTML page.

    `result_like` is a `Performance` (from `openalgo.backtest.metrics.analyse`), or any
    object with a `performance` attribute holding one. Optional attributes of that
    object are read when present:

        name         the report title (overridden by `title=`)
        assumptions  what the engine assumed, shown as a plain list: a mapping gives
                     "key: value" lines, any other iterable one line per item
        strategies   a mapping {name: Performance} of the strategies of a combined
                     result; with two or more, a comparison and the correlation of their
                     session returns are added (full mode)

    `mode` is one of:

        "full"     every section: headline figures and fact lists, equity with its
                   longest drawdowns shaded and the drawdown underneath, cumulative
                   return (with the benchmark), session return bars, the monthly heat
                   grid, yearly returns, the return distribution by period and its
                   histogram, the worst drawdowns, rolling volatility, Sharpe, Sortino,
                   beta and win rate, the horizon summary, the yearly benchmark
                   comparison (with a benchmark), Monte Carlo (1,000 runs from the
                   session returns, fixed seed), the trade P&L distribution, the trade
                   list, losing streaks, costs, the R-multiples (when the trades carry
                   an `r_multiple` and a closed trade has one), the strategy
                   comparison (when given), assumptions, and notes with every
                   definition used
        "basic"    headline and basic facts, equity, monthly and yearly returns, the
                   worst drawdowns, trades, costs, assumptions and notes
        "metrics"  every metric with its value and definition, and the assumptions
        "charts"   every chart and nothing else

    `log_scale` draws the cumulative return on a log scale of growth. The page has
    inline CSS and SVG, no script and no external resource; it follows the reader's
    light or dark setting and reads at phone width. With `path` None the HTML is
    returned; otherwise it is written there as UTF-8 and the path is returned as a
    string. Raises ConfigError when no Performance can be found or for another mode.
    """
    performance, name, assumptions, strategies = _resolve(result_like)
    page = render(
        performance,
        title=title or name or DEFAULT_TITLE,
        assumptions=sections.assumption_lines(assumptions),
        strategies=strategies,
        mode=mode,
        log_scale=log_scale,
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


def _resolve(result_like: object) -> tuple[Performance, str | None, object, Mapping | None]:
    if isinstance(result_like, Performance):
        return result_like, None, None, None
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
    return performance, name, getattr(result_like, "assumptions", None), strategies


def render(
    performance: Performance,
    *,
    title: str,
    assumptions: list[str],
    strategies: Mapping[str, Performance] | None,
    mode: str = "full",
    log_scale: bool = False,
) -> str:
    """The HTML page for `performance` in `mode` (see `tearsheet`)."""
    if mode not in pages.MODES:
        raise ConfigError(f"mode is one of {pages.MODES}, got {mode!r}")
    body = [
        sections.masthead(title, performance, _KICKERS[mode]),
        *pages.body(
            mode,
            performance,
            assumption_lines=assumptions,
            strategies=strategies,
            log_scale=log_scale,
        ),
        f"<footer>Generated by openalgo {text(__version__)} (openalgo.backtest). Figures come "
        "from the recorded equity and trades; the notes define each one.</footer>",
    ]
    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<meta name='color-scheme' content='light dark'>"
        f"<title>{text(title)}: backtest report</title><style>{STYLESHEET}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>\n"
    )


__all__ = ["DEFAULT_TITLE", "metrics_table", "render", "tearsheet"]
