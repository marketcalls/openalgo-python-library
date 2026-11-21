"""The "Overfitting checks" section of the tearsheet (full and basic modes): the quick
checks of the run's trade list (random out of sample, trade reshuffle, trade resample
and variance, `openalgo.backtest._robust._quick`), their scorecard with how many passed,
failed and were not judged and how to read a quick check's verdict (what a failure of
the trade list shows, and what it cannot see), the trade count they need, and a line
pointing to `result.robustness()` for the full set. They are those the run already
computed (`View.overfitting`, a `Result`'s), else computed from the `Performance` the
page shows; they change none of its figures.

On a portfolio's page (`View.portfolio` set) the section shows each strategy's quick
checks instead, a row per strategy (`View.overfitting` maps each strategy's name to its
own, or to None for a strategy with no equity point), and none for the account's trades
together: the strategies' trades overlap in time, so they are not one sequence the
trade-list checks can read.
"""

from collections.abc import Mapping
from typing import Any

from ..._messages import counted
from .. import _format as fmt
from .layout import TITLES, Section, View, card
from .tables import head

#: Where a portfolio's page points for a strategy's full set of checks: a strategy of a
#: portfolio cannot rerun alone, so the full checks need a run of it alone.
PORTFOLIO_POINTER = (
    "A strategy's full set of checks: run it alone with bt.backtest and call "
    "result.robustness() on that result"
)
#: What a portfolio's page says of the account's trades together.
PORTFOLIO_SCOPE = (
    "Each strategy's quick checks read its own closed trades, as its own page shows them. "
    "None are run on the account's trades together: the strategies' trades overlap in "
    "time, so they are not one sequence to reshuffle or resample."
)


def overfitting(view: View) -> Section | None:
    """The section, or None when the mode does not show it."""
    if not view.has("overfitting_checks"):
        return None
    if view.portfolio is not None:
        return by_strategy(view.overfitting)
    from ..._robust._quick import POINTER, QUICK_READING, quick_checks
    from ..._robust._scorecard import lead, table
    from ..._robust._trade_common import MIN_TRADES

    quick = view.overfitting if view.overfitting is not None else quick_checks(view.performance)
    trades = (
        f"{fmt.indian_digits(quick.closed, 0)} closed "
        f"{'trade' if quick.closed == 1 else 'trades'}; the checks need {MIN_TRADES} to judge"
    )
    if quick.too_few_trades:
        trades = f"Too few trades to judge: {trades}."
    subtitle = (
        f"From the trade list, no rerun: {fmt.indian_digits(quick.runs, 0)} runs each, "
        f"seed {quick.seed}"
    )
    body = (
        lead(quick.counts(), QUICK_READING)
        + f"<p class='key'>{fmt.text(trades)}</p>"
        + table(quick.rows())
        + f"<p class='key'>{fmt.text(POINTER)}.</p>"
    )
    return Section(
        "overfitting",
        TITLES["overfitting"],
        "Quick checks of the trade list",
        card("Quick checks", subtitle, body),
    )


def _row(name: str, quick: Any, checks: tuple[str, ...]) -> str:
    """One strategy's row: its name, its closed trades and each check's verdict, the
    check's figure and rule in the verdict's hover text."""
    from ..._robust._scorecard import verdict_badge
    from ..._robust._trade_common import MIN_TRADES

    if quick is None:
        cells = "".join("<td class='l faint'>n/a</td>" for _ in checks)
        return f"<tr><td class='l'>{fmt.text(name)}</td><td class='r nums'>n/a</td>{cells}</tr>"
    closed = fmt.indian_digits(quick.closed, 0)
    if quick.too_few_trades:
        closed += f" of {MIN_TRADES} needed"
    results = {result.name: result for result in quick.results}
    cells = "".join(
        f"<td class='l' title='{fmt.text(results[check].figure)}; rule: "
        f"{fmt.text(results[check].rule)}'>{verdict_badge(results[check].verdict)}</td>"
        for check in checks
    )
    return f"<tr><td class='l'>{fmt.text(name)}</td><td class='r nums'>{closed}</td>{cells}</tr>"


def by_strategy(checks_by_name: Mapping[str, Any]) -> Section:
    """The section of a portfolio's page: a row per strategy, in call order, with each
    of its quick checks' verdicts (`checks_by_name`, module docstring)."""
    # Imported here, as in `overfitting`: the checks' package imports the report's.
    from ..._robust._quick import QUICK_READING
    from ..._robust._registry import BY_NAME, QUICK

    columns = [("Strategy", "l"), ("Closed trades", "r")]
    columns += [(fmt.text(BY_NAME[check].title), "l") for check in QUICK]
    rows = "".join(_row(name, quick, QUICK) for name, quick in checks_by_name.items())
    known = [quick for quick in checks_by_name.values() if quick is not None]
    subtitle = "From each strategy's trade list, no rerun"
    if known:
        subtitle += f": {fmt.indian_digits(known[0].runs, 0)} runs each, seed {known[0].seed}"
    body = (
        f"<p class='lead muted'>{fmt.text(QUICK_READING)}</p>"
        f"<p class='key'>{fmt.text(PORTFOLIO_SCOPE)}</p>"
        f"<div class='scroll'><table class='tbl hover'>{head(*columns)}<tbody>{rows}</tbody>"
        "</table></div>"
        f"<p class='key'>{fmt.text(PORTFOLIO_POINTER)}.</p>"
    )
    strategies = counted(len(checks_by_name), "strategy", "strategies")
    return Section(
        "overfitting",
        TITLES["overfitting"],
        f"Quick checks of each strategy's trade list, {strategies}",
        card("Quick checks by strategy", subtitle, body),
    )


__all__ = ["PORTFOLIO_POINTER", "PORTFOLIO_SCOPE", "by_strategy", "overfitting"]
