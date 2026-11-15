"""`Result.summary()`: the few figures a beginner prints after a run, as plain text.

    Net profit   : Rs 2,00,000.00
    Return       : 4.00%
    Max drawdown : -0.97%
    Trades       : 11 (36% won)
    Benchmark    : 3.52% return (NIFTY)

Net profit is rupees with Indian digit grouping (`report._format.indian_digits`; a loss
below half a paisa reads "Rs 0.00"); the return and the maximum drawdown are those of the
time-weighted wealth index, in percent (`metrics` keys `net_profit`, `total_return`,
`max_drawdown`). The win rate is winning trades over closed trades, in whole percent
unless that would read 0% or 100% for a rate that is neither (then with the fewest
decimals that do not: "0.4% won" for 1 winner in 250). It is undefined (NaN) for a run
with no closed trade, such as a buy-and-hold or rebalanced portfolio or a run whose
entries were all rejected or skipped; the line then says "no closed trades" instead.
Positions still open at the end are counted beside the closed trades. The benchmark line
is there only for a run with `benchmark=`: the benchmark's total return
(`benchmark_total_return`), named by its index symbol, or "given closes" for closes given
as a Series or DataFrame. A figure that is undefined reads "n/a".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._result_report import GIVEN_BENCHMARK
from .report._format import MISSING, indian_digits, is_missing

# The width of the longest label, "Max drawdown", so the colons line up.
LABEL_WIDTH = 12
GIVEN_CLOSES = "given closes"
# The most decimals a win rate is shown with. k winners of n trades sit at least 1/n
# from 0 and from 1, which shows at about log10(n) decimals of a percent, so this covers
# runs of up to a million trades.
MAX_WIN_RATE_DECIMALS = 6
# A win rate that only an exact count may show.
BOUNDS_PERCENT = (0.0, 100.0)


def _percent(fraction: float, decimals: int) -> str:
    return MISSING if is_missing(fraction) else f"{fraction:.{decimals}%}"


def _win_rate(rate: float) -> str:
    """The win rate in whole percent, with the fewest decimals that keep it off 0% and
    100% when it is neither exactly: 1 winner in 250 is "0.4%", 249 in 250 "99.6%"."""
    if is_missing(rate):
        return MISSING
    for decimals in range(MAX_WIN_RATE_DECIMALS + 1):
        text = f"{rate:.{decimals}%}"
        if rate in (0.0, 1.0) or float(text.rstrip("%")) not in BOUNDS_PERCENT:
            break
    return text


def _rupees(amount: float) -> str:
    return MISSING if is_missing(amount) else f"Rs {indian_digits(amount)}"


def _trades(metrics: Mapping[str, Any]) -> str:
    """The closed trades and their win rate, and the positions still open."""
    closed, still_open = int(metrics["trades"]), int(metrics["open_trades"])
    text = f"{closed} ({_win_rate(metrics['win_rate'])} won)" if closed else "no closed trades"
    if not still_open:
        return text
    return f"{text}, {still_open} still open" if closed else f"{text} ({still_open} still open)"


def summary_text(metrics: Mapping[str, Any], benchmark: str | None) -> str:
    """The summary of the module docstring, one line per figure. `metrics` is
    `Result.metrics`; `benchmark` the run's benchmark as `_result_report.benchmark_name`
    names it (None without one)."""
    lines = [
        ("Net profit", _rupees(metrics["net_profit"])),
        ("Return", _percent(metrics["total_return"], 2)),
        ("Max drawdown", _percent(metrics["max_drawdown"], 2)),
        ("Trades", _trades(metrics)),
    ]
    if benchmark is not None:
        name = GIVEN_CLOSES if benchmark == GIVEN_BENCHMARK else benchmark
        returned = _percent(metrics["benchmark_total_return"], 2)
        lines.append(("Benchmark", f"{returned} return ({name})"))
    return "\n".join(f"{label:<{LABEL_WIDTH}} : {value}" for label, value in lines)


__all__ = ["summary_text"]
