"""Other markets (design 17, section 2.8): does the logic hold on related symbols?

The strategy runs unchanged on each symbol the user names (`others=["ICICIBANK",
"HDFCBANK"]`, on the strategy's exchange, or (symbol, exchange) pairs), with the run's
own arguments and source: the user's `data=` (a dict by symbol that holds them) or
OpenAlgo through the run's client. There is no built-in list of related symbols. Each is
an ordinary backtest of the strategy class with its `symbol` (and `exchange`) changed; an
`instrument=` given as one Instrument belongs to the run's symbol and is left out, one
given as a dict by symbol is kept.

Figures: the share of the symbols profitable (net profit of at least half a paisa) and the
median of their objectives over the run's, measured from the objective's break-even (n/a
for an objective with none, or a run not above it). Pass: at least `pass_share` (half)
of the symbols profitable. Not judged without `others=`, or when the run did not make
money or has too few closed trades.

Charts: each symbol's equity with the run's bold and each symbol's buy and hold (its
session closes over the open of its first bar of the run, the price's return, not a
return on the capital) dotted, a symbol's two lines in its colour and ending in its
shape; a table by symbol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._messages import not_supported
from .._objective import evaluate
from .._pipeline import execute, prepare
from ..errors import ConfigError
from ..reference import Instrument
from . import _words as words
from ._bar_common import BarRun, break_even, finite
from ._blocks import BOLD, DOTTED, THIN, Line, LinesBlock, TableBlock
from ._ohlc import session_days
from ._part_objective import objective_name
from ._run_bars import run_arguments
from ._setting import Setting, between
from ._trade_common import BREAK_EVEN_RUPEES, ENOUGH_TRADES, no_edge, too_few
from ._verdict import FAILED, PASSED, Outcome, not_judged

SETTINGS = (
    ENOUGH_TRADES,
    Setting(
        "pass_share",
        0.5,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of the symbols that must make money",
    ),
)


def others_value(others: Any) -> tuple[tuple[str, str | None], ...]:
    """`others=` as (symbol, exchange) pairs, exchange None for the strategy's. Raises
    ConfigError for anything but a non-empty list of symbols or (symbol, exchange) pairs."""
    if isinstance(others, str) or not isinstance(others, list | tuple) or not others:
        raise ConfigError(
            f"others takes a list of symbols, or (symbol, exchange) pairs, got {others!r}"
        )
    pairs = []
    for entry in others:
        if isinstance(entry, str) and entry:
            pairs.append((entry, None))
        elif (
            isinstance(entry, tuple | list)
            and len(entry) == 2
            and all(isinstance(part, str) and part for part in entry)
        ):
            pairs.append((entry[0], entry[1]))
        else:
            raise ConfigError(f"others takes symbols or (symbol, exchange) pairs, got {entry!r}")
    return tuple(pairs)


def check_strategy(strategy: type, run_arguments: Mapping[str, Any]) -> None:
    """Raise NotImplementedError for a strategy that does not trade one named symbol, and
    ConfigError for `data=` that cannot hold the other symbols' bars (one table, not a
    dict by symbol)."""
    if strategy.symbol is None:
        raise not_supported(
            "other markets for a strategy on several symbols, a universe or an underlying",
            "run bt.robustness for each symbol's own strategy",
        )
    data = run_arguments.get("data")
    if data is not None and not isinstance(data, dict):
        raise ConfigError(
            "others= runs the strategy on other symbols' bars, so data= must be a dict by "
            "symbol that holds the run's symbol and each of others"
        )


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    return f"at least {words.percent(settings['pass_share'], 0)} of the other symbols profitable"


def _moved(strategy: type, symbol: str, exchange: str | None) -> type:
    """The strategy class on another symbol (module docstring)."""
    attributes = {"symbol": symbol, "exchange": exchange or strategy.exchange}
    attributes["__doc__"] = f"{strategy.__name__} on {symbol}."
    return type(strategy.__name__, (strategy,), attributes)


def other_run(subject: BarRun, symbol: str, exchange: str | None) -> tuple[Any, pd.Series]:
    """(the run of the strategy on `symbol`, its buy and hold at each session close), from
    the run's own arguments and source."""
    record = subject.result._record
    arguments = run_arguments(record)
    if isinstance(arguments.get("instrument"), Instrument):
        arguments["instrument"] = None
    strategy = _moved(record.strategy, symbol, exchange)
    args = {**arguments, "client": subject.client}
    prepared = prepare(strategy, subject.client, args)
    result = execute(strategy, prepared, args)
    return result, buy_and_hold(prepared.frames[0])


def buy_and_hold(bars: pd.DataFrame) -> pd.Series:
    """The return of holding from the open of the first of `bars` to each session close, a
    fraction, by session date."""
    days = session_days(bars["ts_open"].to_numpy())
    last = np.append(days[1:] != days[:-1], True)
    closes = bars["close"].to_numpy(dtype=float)[last]
    first = float(bars["open"].iloc[0])
    return pd.Series(closes / first - 1.0, index=pd.to_datetime(days[last], unit="D"))


def equity_line(result: Any) -> pd.Series:
    """A run's cumulative return at each session close, by session date."""
    returns = result.performance.session_returns
    days = pd.DatetimeIndex(returns.index).tz_localize(None).normalize()
    return pd.Series((1.0 + returns.to_numpy(dtype=float)).cumprod() - 1.0, index=days)


def run(
    subject: BarRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    if not subject.others:
        why = "No related symbols given: name them with others=['ICICIBANK', 'HDFCBANK']."
        return not_judged(text, why)
    short = too_few(subject.trades, settings["min_trades"], text) or no_edge(subject.trades, text)
    if short is not None:
        return short
    rows, lines = [], [_own_lines(subject)]
    for group, (symbol, exchange) in enumerate(subject.others, start=1):
        result, hold = other_run(subject, symbol, exchange)
        metrics = result.metrics
        rows.append(
            {
                "symbol": symbol,
                "exchange": exchange or subject.result._record.strategy.exchange,
                "objective": evaluate(result, objective)["objective"],
                "net_profit": float(metrics["net_profit"]),
                "total_return": float(metrics["total_return"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "trades": int(metrics["trades"]),
                "buy_and_hold": float(hold.iloc[-1]) if len(hold) else np.nan,
            }
        )
        lines.append(_lines(symbol, equity_line(result), hold, group))
    table = pd.DataFrame(rows).set_index("symbol")
    return _judged(subject, table, lines, settings, objective, text)


def _own_lines(subject: BarRun) -> tuple[Line, ...]:
    symbol = subject.bars.symbols[0]
    hold = buy_and_hold(subject.bars.run_part(0))
    return (
        Line(f"{symbol}, the run", pd.DatetimeIndex(hold.index), _own_equity(subject, hold), BOLD),
        Line(f"{symbol} buy and hold", pd.DatetimeIndex(hold.index), hold.to_numpy(), DOTTED),
    )


def _own_equity(subject: BarRun, hold: pd.Series) -> np.ndarray:
    own = pd.Series(subject.trades.cumulative, index=pd.DatetimeIndex(subject.trades.sessions))
    own.index = own.index.tz_localize(None).normalize()
    return own.reindex(hold.index, method="ffill").fillna(0.0).to_numpy(dtype=float)


def _lines(symbol: str, equity: pd.Series, hold: pd.Series, group: int) -> tuple[Line, ...]:
    """The lines of the other symbol `symbol`, in its `group`'s colour and shape."""
    return (
        Line(symbol, pd.DatetimeIndex(equity.index), equity.to_numpy(), THIN, group),
        Line(
            f"{symbol} buy and hold",
            pd.DatetimeIndex(hold.index),
            hold.to_numpy(),
            DOTTED,
            group,
        ),
    )


def _judged(
    subject: BarRun,
    table: pd.DataFrame,
    lines: Sequence[tuple[Line, ...]],
    settings: Mapping[str, Any],
    objective: Any,
    text: str,
) -> Outcome:
    name, level = objective_name(objective), break_even(objective)
    share = float((table["net_profit"] >= BREAK_EVEN_RUPEES).mean())
    median = float(np.nanmedian(table["objective"])) if table["objective"].notna().any() else np.nan
    ratio = np.nan
    if level is not None and finite(subject.real) and subject.real > level:
        ratio = (median - level) / (subject.real - level)
    summary = {
        "symbols": float(len(table)),
        "profitable_share": share,
        "median_objective": median,
        "ratio": ratio,
    }
    verdict = PASSED if share >= settings["pass_share"] else FAILED
    figure = (
        f"{words.percent(share)} of {words.whole(len(table), 'symbol')} profitable; median "
        f"{name} {words.number(median)}, {words.number(ratio)} times the run's"
    )
    note = (
        "The strategy ran unchanged on each symbol with the run's own arguments and source; "
        "a symbol is profitable when its net profit is at least half a paisa."
    )
    blocks = _blocks(subject, table, lines, name)
    return Outcome(verdict, figure, text, note, table.reset_index(), summary, blocks)


#: What the lines of the other markets chart measure, for its key.
LINES_NOTE = (
    "A strategy line is its cumulative return on the capital; a buy and hold is the return "
    "of the price from the open of its first bar of the run, which a run invested in part "
    "of the time does not compare with directly."
)


def _blocks(subject: BarRun, table: pd.DataFrame, lines: Sequence, name: str) -> tuple:
    chart = LinesBlock(
        "Equity by symbol",
        "Each symbol in its colour and shape; the run bold, buy and hold dotted",
        tuple(line for group in lines for line in group),
        LINES_NOTE,
    )
    rows = tuple(
        (
            str(symbol),
            words.number(row.objective),
            words.rupees(row.net_profit),
            words.percent(row.total_return),
            words.percent(row.max_drawdown),
            words.number(row.trades, 0),
            words.percent(row.buy_and_hold),
        )
        for symbol, row in table.iterrows()
    )
    columns = (
        "Symbol",
        name.capitalize(),
        "Net profit",
        "Return",
        "Max drawdown",
        "Trades",
        "Buy and hold",
    )
    return (
        (chart,),
        (TableBlock("By symbol", "The strategy on each other symbol", columns, rows),),
    )


__all__ = [
    "SETTINGS",
    "buy_and_hold",
    "check_strategy",
    "equity_line",
    "other_run",
    "others_value",
    "rule",
    "run",
]
