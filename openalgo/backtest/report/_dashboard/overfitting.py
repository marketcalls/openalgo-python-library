"""The "Overfitting checks" section of the tearsheet (full and basic modes): the quick
checks of the run's trade list (random out of sample, trade reshuffle, trade resample
and variance, `openalgo.backtest._robust._quick`), their scorecard with how many passed,
failed and were not judged and the wording rule, the trade count they need, and a line
pointing to `result.robustness()` for the full set. They are those the run already
computed (`View.overfitting`, a `Result`'s), else computed from the `Performance` the
page shows; they change none of its figures.
"""

from .. import _format as fmt
from .layout import Section, View, card


def overfitting(view: View) -> Section | None:
    """The section, or None when the mode does not show it."""
    if not view.has("overfitting_checks"):
        return None
    from ..._robust._quick import POINTER, quick_checks
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
        lead(quick.counts())
        + f"<p class='key'>{fmt.text(trades)}</p>"
        + table(quick.rows())
        + f"<p class='key'>{fmt.text(POINTER)}.</p>"
    )
    return Section(
        "overfitting",
        "Overfitting checks",
        "Quick checks of the trade list",
        card("Quick checks", subtitle, body),
    )


__all__ = ["overfitting"]
