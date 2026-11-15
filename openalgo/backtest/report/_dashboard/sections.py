"""The return sections of the dashboard: overview, performance, risk, monthly and
distribution, each a function of the `View` that returns its `Section`, or None when
the mode shows none of its parts."""

from . import bars, cards, data, heatmaps, tables
from . import timeseries as ts
from .layout import LARGE, SMALL, Section, View, card, label, stack

STRATEGY_COLOUR = "var(--text)"
BENCHMARK_COLOUR = "var(--benchmark)"
LOSS_COLOUR = "var(--pnl-neg)"
#: Chart heights, pixels: the cumulative return, a rolling statistic, the drawdown.
CUMULATIVE_PX, ROLLING_PX, DRAWDOWN_PX = 320, 200, 240
#: Stroke width of the benchmark's line, pixels (the strategy's is the default 2).
BENCHMARK_STROKE = 1.5
#: The rolling statistics charted side by side: column, title, subtitle, whether a
#: percentage, whether a dashed zero line is drawn.
ROLLING = (
    ("sharpe", "Rolling Sharpe", "{w}-session window", False, True),
    ("volatility", "Rolling Volatility", "Annualised, {w}-session", True, False),
    ("win_rate", "Rolling Win Rate", "% winning sessions, {w}-session", True, False),
)


def _grid(view: View, stats: tuple[cards.Stat, ...]) -> str:
    return cards.grid(stats, view.performance, view.bench, view.bench_name)


def overview(view: View) -> Section | None:
    """The headline tiles."""
    if not view.has("overview"):
        return None
    return Section(
        "overview", "Overview", "Headline risk &amp; return", _grid(view, cards.OVERVIEW)
    )


def _too_short(window: int) -> str:
    return f"No data: the run has fewer sessions than one {window}-session window."


def _cumulative(view: View) -> str:
    p = view.performance
    lines = [
        ts.Line("Strategy", data.cumulative(p.session_returns).to_numpy(), STRATEGY_COLOUR, "area")
    ]
    if p.benchmark_returns is not None:
        bench = data.cumulative(p.benchmark_returns).to_numpy()
        lines.append(ts.Line(view.bench_name, bench, BENCHMARK_COLOUR, "line", BENCHMARK_STROKE))
    chart = ts.chart(
        p.session_returns.index,
        lines,
        height=CUMULATIVE_PX,
        percent=True,
        label="Cumulative return",
        ident="cumulative",
        zero_line=True,
        log_scale=view.log_scale,
    )
    scale = ", log scale" if view.log_scale else ""
    return card("Cumulative Return", f"Compounded growth, indexed to start{scale}", chart)


def _rolling_chart(
    view: View, column: str, title: str, percent: bool, zero: bool, width: int = ts.THIRD
) -> str:
    p = view.performance
    window = int(p.metrics["rolling_window_sessions"])
    dates = p.session_returns.index
    values = p.rolling[column].reindex(dates).to_numpy(dtype=float)
    return ts.chart(
        dates,
        [ts.Line(title, values, STRATEGY_COLOUR)],
        height=ROLLING_PX,
        percent=percent,
        label=title,
        ident=f"rolling-{column}",
        zero_line=zero,
        nominal_width=width,
        empty=_too_short(window),
    )


def _rolling(view: View) -> str:
    window = int(view.performance.metrics["rolling_window_sessions"])
    cells = "".join(
        card(title, subtitle.format(w=window), _rolling_chart(view, column, title, pct, zero))
        for column, title, subtitle, pct, zero in ROLLING
        if column in view.performance.rolling.columns
    )
    return f"<div class='grid3'>{cells}</div>"


def performance(view: View) -> Section | None:
    """Cumulative return, the rolling statistics, the horizon table and the period
    tiles."""
    blocks = [
        ("", _cumulative(view) if view.has("cumulative") else ""),
        (SMALL, _rolling(view) if view.has("rolling") else ""),
        (
            SMALL,
            card(
                "Return &amp; Risk by Horizon",
                "CAGR, max drawdown &amp; Sharpe over trailing windows",
                tables.horizons(view.performance),
            )
            if view.has("horizons")
            else "",
        ),
        (
            LARGE,
            label("Period Returns") + _grid(view, cards.PERIODS) if view.has("periods") else "",
        ),
    ]
    body = stack(blocks)
    return (
        Section("performance", "Performance", "Equity curve &amp; rolling stats", body)
        if body
        else None
    )


def _drawdown(view: View) -> str:
    p = view.performance
    line = ts.Line("Drawdown", data.session_drawdown(p).to_numpy(), LOSS_COLOUR, "baseline")
    chart = ts.chart(
        p.session_returns.index,
        [line],
        height=DRAWDOWN_PX,
        percent=True,
        label="Drawdown",
        ident="drawdown",
    )
    return card("Drawdown", "Underwater: the deepest fall from the peak in each session", chart)


def _benchmark(view: View) -> str:
    p = view.performance
    if p.benchmark_returns is None:
        return ""
    parts = []
    if view.has("benchmark_stats"):
        parts.append(
            ("", label(f"vs Benchmark &#183; {view.bench_name}") + _grid(view, cards.BENCHMARK))
        )
    if view.has("rolling_beta") and "beta" in p.rolling.columns:
        chart = _rolling_chart(view, "beta", "Rolling Beta", False, True, ts.WIDE)
        window = int(p.metrics["rolling_window_sessions"])
        parts.append(
            (SMALL, card("Rolling Beta", f"vs {view.bench_name}, {window}-session", chart))
        )
    return stack(parts)


def _streaks(view: View) -> str:
    streaks = view.performance.losing_streaks
    counts = [(int(n), int(c)) for n, c in streaks["sessions"].items()] if len(streaks) else []
    while counts and counts[-1][1] == 0:
        counts.pop()
    if not counts or sum(c for _, c in counts) == 0:
        return ""
    total = sum(c for _, c in counts)
    mean = sum(n * c for n, c in counts) / total
    longest = max(n for n, c in counts if c)
    foot = (
        f"<div class='streak-foot nums'><span>Streaks: <b>{total}</b></span>"
        f"<span>Longest: <b>{longest}</b></span><span>Average: <b>{mean:.2f}</b></span>"
        "<span class='faint'>length = consecutive losing sessions</span></div>"
    )
    return card(
        "Consecutive Losing Streaks",
        "How often losing runs of each length occurred",
        bars.streak_bars(counts) + foot,
    )


def risk(view: View) -> Section | None:
    """The drawdown chart, the risk and benchmark tiles, rolling beta, the worst
    drawdowns and the losing streaks."""
    blocks = [
        ("", _drawdown(view) if view.has("drawdown_chart") else ""),
        (LARGE, _grid(view, cards.RISK) if view.has("risk_stats") else ""),
        (LARGE, _benchmark(view)),
        (
            LARGE,
            card(
                "Worst Drawdowns",
                "Deepest peak-to-recovery episodes",
                tables.drawdowns(view.performance),
            )
            if view.has("worst_drawdowns")
            else "",
        ),
        (LARGE, _streaks(view) if view.has("streaks") else ""),
    ]
    body = stack(blocks)
    return Section("risk", "Risk", "Drawdown, tails &amp; exposure", body) if body else None


def _annual(view: View) -> str:
    rows = data.yearly(view.performance)
    with_bench = view.performance.benchmark_returns is not None
    chart = (
        bars.yearly_bars(rows, view.bench_name if with_bench else None)
        if view.has("yearly_bars")
        else ""
    )
    table = tables.yearly(rows, with_bench) if view.has("yearly_table") else ""
    if not chart and not table:
        return ""
    body = chart + (f"<div class='under'>{table}</div>" if chart and table else table)
    return card("Annual Returns", "End-of-year, strategy vs benchmark", body)


def monthly(view: View) -> Section | None:
    """The monthly and weekly heatmaps, the monthly tiles and the yearly returns."""
    p = view.performance
    blocks = [
        (
            "",
            card(
                "Monthly Returns",
                "Calendar heatmap, % per month",
                heatmaps.monthly(p.monthly_returns),
            )
            if view.has("monthly_heatmap")
            else "",
        ),
        (
            SMALL,
            card(
                "Weekly Returns",
                "Every week of a year: pick the year",
                heatmaps.weekly(data.weekly(p.session_returns)),
            )
            if view.has("weekly_heatmap")
            else "",
        ),
        (LARGE, _grid(view, cards.MONTHLY) if view.has("monthly_stats") else ""),
        (SMALL, _annual(view)),
    ]
    body = stack(blocks)
    return Section("monthly", "Monthly", "Seasonality &amp; annual returns", body) if body else None


def distribution(view: View) -> Section | None:
    """The distribution tiles, the histogram of session returns and the box plot of
    session and monthly returns."""
    p = view.performance
    charts = []
    if view.has("histogram"):
        charts.append(
            card(
                "Daily Returns",
                "Histogram with mean marker",
                bars.histogram(p.session_returns.to_numpy(dtype=float)),
            )
        )
    if view.has("box_plot"):
        boxes = [
            bars.Box("Daily", p.session_returns.to_numpy(dtype=float)),
            bars.Box("Monthly", p.monthly_returns.to_numpy(dtype=float).ravel()),
        ]
        charts.append(card("Returns Spread", "Daily vs monthly quantiles", bars.box_plot(boxes)))
    blocks = [
        ("", _grid(view, cards.DISTRIBUTION) if view.has("distribution_stats") else ""),
        (LARGE, f"<div class='grid2'>{''.join(charts)}</div>" if charts else ""),
    ]
    body = stack(blocks)
    return (
        Section("distribution", "Distribution", "Return shape &amp; tails", body) if body else None
    )


__all__ = ["distribution", "monthly", "overview", "performance", "risk"]
