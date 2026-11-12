"""The report's text sections: masthead, headline figures, fact lists, notes,
assumptions and the comparison of several strategies."""

from collections.abc import Iterable, Mapping

import pandas as pd

from . import _format as fmt
from ._definitions import DEFINITIONS
from ._performance import Performance

#: The headline figures across the top of the report.
HEADLINE = ("net_profit", "total_return", "cagr", "max_drawdown", "sharpe", "trades")
#: The fact lists under the headline of the basic report, by heading.
BASIC_FACTS = (
    (
        "Returns and risk",
        ("total_return", "cagr", "volatility", "sharpe", "sortino", "calmar", "exposure"),
    ),
    (
        "Drawdown",
        (
            "max_drawdown",
            "max_drawdown_amount",
            "max_drawdown_peak",
            "max_drawdown_trough",
            "max_drawdown_recovery",
            "max_drawdown_days",
            "longest_drawdown_days",
        ),
    ),
    (
        "Trades",
        (
            "trades",
            "open_trades",
            "win_rate",
            "profit_factor",
            "expectancy",
            "payoff_ratio",
            "average_win",
            "average_loss",
            "largest_win",
            "largest_loss",
            "max_consecutive_wins",
            "max_consecutive_losses",
            "average_holding_seconds",
            "turnover",
        ),
    ),
    ("Benchmark", ("benchmark_total_return", "alpha", "beta", "correlation")),
    (
        "Account",
        (
            "start_equity",
            "end_equity",
            "net_cash_flows",
            "net_profit",
            "sessions",
            "missing_sessions",
        ),
    ),
)
#: The fact lists of the full report: the basic ones, the benchmark's active-return
#: figures, and the analysis groups.
FACTS = (
    *(
        (heading, (*keys, "r_squared", "tracking_error", "information_ratio", "treynor_ratio"))
        if heading == "Benchmark"
        else (heading, keys)
        for heading, keys in BASIC_FACTS
    ),
    (
        "Risk-adjusted",
        (
            "smart_sharpe",
            "smart_sortino",
            "adjusted_sortino",
            "probabilistic_sharpe",
            "omega",
            "gain_to_pain",
            "ulcer_index",
            "ulcer_performance_index",
            "serenity_index",
            "recovery_factor",
            "risk_adjusted_return",
        ),
    ),
    (
        "Session returns",
        (
            "expected_return",
            "average_session_return",
            "best_session",
            "worst_session",
            "best_month",
            "worst_month",
            "best_year",
            "worst_year",
            "session_win_rate",
            "max_consecutive_losing_sessions",
        ),
    ),
    (
        "Tail risk",
        (
            "skew",
            "kurtosis",
            "value_at_risk",
            "expected_shortfall",
            "conditional_drawdown",
            "tail_ratio",
        ),
    ),
    (
        "Trade ratios",
        ("win_loss_ratio", "kelly_criterion", "risk_of_ruin", "cpc_index", "common_sense_ratio"),
    ),
)
#: Columns of the strategy comparison.
COMPARISON = ("total_return", "cagr", "max_drawdown", "sharpe", "trades", "net_profit")


def section(anchor: str, title: str, body: str, lede: str = "") -> str:
    """One report section with its heading and an optional lede."""
    intro = f"<p class='lede'>{lede}</p>" if lede else ""
    return (
        f"<section id='{anchor}' aria-labelledby='{anchor}-title'>"
        f"<h2 id='{anchor}-title'>{title}</h2>{intro}{body}</section>"
    )


def masthead(title: str, performance: Performance, kicker: str = "Backtest report") -> str:
    """Report kicker, title and the run in one line."""
    m = performance.metrics
    parts = (
        fmt.text(", ".join(performance.exchanges)),
        f"{fmt.day(m['start'])} to {fmt.day(m['end'])}",
        f"{fmt.figure('sessions', m['sessions'])} sessions",
        f"start equity {fmt.rupees(m['start_equity'])}",
    )
    dek = "".join(f"<span>{part}</span>" for part in parts)
    return (
        f"<header class='masthead'><p class='kicker'>{fmt.text(kicker)}</p>"
        f"<h1>{fmt.text(title)}</h1><p class='dek'>{dek}</p></header>"
    )


def _headline_figure(key: str, value: object) -> str:
    # Whole rupees keep a crore-sized amount inside its tile; the fact lists show paise.
    if DEFINITIONS[key].unit == "rupees" and not fmt.is_missing(value):
        return fmt.rupees(float(value), 0)
    return fmt.figure(key, value)


def _term(key: str, value: str) -> str:
    """One label and value of a description list; the label is escaped (P&L)."""
    return f"<div><dt>{fmt.text(DEFINITIONS[key].label)}</dt><dd>{value}</dd></div>"


def headline(metrics: Mapping[str, object]) -> str:
    """The headline figures as one strip."""
    cells = "".join(_term(k, _headline_figure(k, metrics[k])) for k in HEADLINE)
    return f"<dl class='figures'>{cells}</dl>"


def facts(metrics: Mapping[str, object], lists: tuple = FACTS) -> str:
    """The fact lists `lists` ((heading, keys) pairs); a group whose keys are all absent
    (no benchmark) is left out."""
    groups = []
    for heading, keys in lists:
        present = [key for key in keys if key in metrics]
        if not present:
            continue
        rows = "".join(_term(k, fmt.figure(k, metrics[k])) for k in present)
        groups.append(f"<div class='group'><h3>{heading}</h3><dl>{rows}</dl></div>")
    return f"<div class='facts'>{''.join(groups)}</div>"


def method_notes(
    metrics: Mapping[str, object], exchanges: tuple[str, ...], lists: tuple = FACTS
) -> str:
    """How the figures were computed, in sentences, then the glossary of the keys of the
    fact lists `lists`."""
    m = metrics
    notes = [
        "Returns are time weighted: cash flows move the equity but are not returns. Each "
        "IST session date contributes one return, from the last equity point of the date.",
        f"Volatility, Sharpe, Sortino and alpha are annualised with "
        f"{fmt.figure('sessions_per_year', m['sessions_per_year'])} sessions a year: the "
        f"{fmt.text(', '.join(exchanges))} calendar in {fmt.text(m['annualisation_years'])}, "
        f"holiday data {fmt.text(m['annualisation_coverage'])}. CAGR uses elapsed calendar "
        "time.",
        f"Risk-free rate {fmt.percent(m['risk_free_rate'])} a year, compounded down to "
        f"{fmt.percent(m['risk_free_per_session'], 4)} a session.",
    ]
    if m["annualisation_template_years"]:
        notes.append(
            f"No holiday list for {fmt.text(m['annualisation_template_years'])}: every "
            "weekday was counted as a session, so the annualisation is overstated."
        )
    if m["missing_sessions"]:
        notes.append(
            f"{m['missing_sessions']} calendar session(s) have no equity point; the move "
            "of each is folded into the next session's return."
        )
    if m["ruined"]:
        notes.append(f"The account was ruined on {fmt.moment(m['ruin_time'])}.")
    notes.append(
        "Every drawdown figure is a fall, so it is zero or negative, as a fraction or in "
        "rupees. Every definition cites its source in the documentation of "
        "openalgo.backtest.metrics."
    )
    shown = [key for _, keys in lists for key in keys if key in metrics]
    glossary = "".join(_term(k, fmt.text(DEFINITIONS[k].definition)) for k in shown)
    paragraphs = "".join(f"<p>{note}</p>" for note in notes)
    return f"{paragraphs}<dl class='glossary'>{glossary}</dl>"


def assumption_lines(assumptions: object) -> list[str]:
    """Assumptions as plain lines: a mapping gives "key: value" lines, a string one
    line, any other iterable one line per item."""
    if assumptions is None:
        return []
    if isinstance(assumptions, str):
        return [assumptions]
    if isinstance(assumptions, Mapping):
        return [f"{key}: {value}" for key, value in assumptions.items()]
    if isinstance(assumptions, Iterable):
        return [str(item) for item in assumptions]
    return [str(assumptions)]


def assumption_list(lines: list[str]) -> str:
    """The assumptions as a plain list, or a sentence when none were recorded."""
    if not lines:
        return "<p class='note'>No assumptions were recorded with this result.</p>"
    items = "".join(f"<li>{fmt.text(line)}</li>" for line in lines)
    return f"<ul class='list'>{items}</ul>"


def comparison(strategies: Mapping[str, Performance], correlation: pd.DataFrame) -> str:
    """Key figures per strategy, and the correlation of their session returns."""
    head = "".join(
        f"<th scope='col' class='num'>{fmt.text(DEFINITIONS[k].label)}</th>" for k in COMPARISON
    )
    rows = "".join(
        f"<tr><th scope='row'>{fmt.text(name)}</th>"
        + "".join(f"<td class='num'>{fmt.figure(k, p.metrics[k])}</td>" for k in COMPARISON)
        + "</tr>"
        for name, p in strategies.items()
    )
    names = list(correlation.columns)
    corr_head = "".join(f"<th scope='col' class='num'>{fmt.text(n)}</th>" for n in names)
    corr_rows = "".join(
        f"<tr><th scope='row'>{fmt.text(row)}</th>"
        + "".join(f"<td class='num'>{fmt.number(correlation.loc[row, c])}</td>" for c in names)
        + "</tr>"
        for row in names
    )
    return (
        f"<div class='scroll'><table class='data'><thead><tr><th scope='col'>Strategy</th>{head}"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
        "<h3 class='sub'>Correlation of session returns</h3>"
        f"<div class='scroll'><table class='data'><thead><tr><th></th>{corr_head}</tr></thead>"
        f"<tbody>{corr_rows}</tbody></table></div>"
    )


__all__ = [
    "BASIC_FACTS",
    "FACTS",
    "HEADLINE",
    "assumption_lines",
    "assumption_list",
    "comparison",
    "facts",
    "headline",
    "masthead",
    "method_notes",
    "section",
]
