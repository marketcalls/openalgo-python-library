"""The run settings of a portfolio run that are keyed by what is traded, given to each
strategy as its own (18-multi-strategy 1.1: one value for the run).

`data=`, `instrument=` and `corporate_actions=` describe symbols and `margin=` underlyings,
and a strategy's own checks refuse, or warn of, an entry for something it does not trade
(`_instruments`, `_run_corporate_actions`, `_futures_margin`). So each strategy is given the
entries of what it trades, as a run of it alone with them would be:

- `instrument=`: the instruments of the symbols it trades and, for a strategy on a futures
  underlying, the contracts of that underlying. An instrument no strategy takes is
  refused, naming the symbols the strategies trade.
- `corporate_actions=`: a dict's entries of the symbols it trades. A dict entry no
  strategy trades is refused; a list names no symbol, so it is refused when the
  strategies trade more than one symbol between them, as one table of bars is for
  `data=` (`refuse_one_frame`). A dict `data=` stays whole: each strategy reads its own
  symbols from it.
- `margin=` (once the instruments are known, `own_margins`): the part of it the strategy's
  futures use. What no strategy uses is one warning of the run, judged against the
  futures of every strategy together.

Two strategies on one symbol are both given its entries.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from ._futures import underlying_names
from ._futures_contracts import underlying_of
from ._futures_margin import margin_for, margin_warnings, parse_margin
from ._instruments import given_instruments
from ._run_check import CheckedRun
from .errors import ConfigError
from .reference import Instrument
from .strategy import Strategy


def traded_symbols(strategies: Sequence[type[Strategy]]) -> list[str]:
    """The symbols the strategies trade between them (an underlying for a futures
    strategy), sorted."""
    return sorted({symbol for each in strategies for symbol in each.traded_symbols()})


def _one_form_for_several(strategies: Sequence[type[Strategy]], what: str, form: str) -> None:
    """Raise ConfigError when the strategies trade more than one symbol between them: a
    `what` given in a form that names no symbol would be read as every symbol's; `form`
    is the form to give instead."""
    symbols = traded_symbols(strategies)
    if len(symbols) > 1:
        raise ConfigError(f"the strategies trade {', '.join(symbols)}: give {what} as {form}")


def refuse_one_frame(strategies: Sequence[type[Strategy]], data: Any) -> None:
    """Raise ConfigError for own bars given as one table (not a dict by symbol) to
    strategies that trade more than one symbol between them: the table would be read as
    the bars of every symbol."""
    if data is not None and not isinstance(data, dict):
        _one_form_for_several(
            strategies,
            "data=",
            "a dict of bars by symbol, which feeds every strategy that trades the symbol",
        )


def _takes(strategy: type[Strategy], instrument: Instrument) -> bool:
    """True when `strategy` trades `instrument`: its symbol is one the strategy names, or
    it is a contract of a futures underlying the strategy names."""
    if instrument.symbol in strategy.traded_symbols():
        return True
    return underlying_of(instrument) in underlying_names(strategy, strategy.exchange)


def own_instruments(
    strategies: Sequence[type[Strategy]], given: Any
) -> list[dict[str, Instrument] | None]:
    """Each strategy's part of `instrument=` (an Instrument or a dict of them by symbol),
    in call order: the instruments it trades by symbol, None when there are none. Raises
    ConfigError as `_instruments.given_instruments` does, and for an instrument no
    strategy trades."""
    by_symbol = given_instruments(given)
    owned = [
        {symbol: each for symbol, each in by_symbol.items() if _takes(strategy, each)}
        for strategy in strategies
    ]
    untraded = [symbol for symbol in by_symbol if not any(symbol in own for own in owned)]
    if untraded:
        raise ConfigError(
            f"instrument= gives {', '.join(untraded)}, which no strategy of the run trades "
            f"({', '.join(traded_symbols(strategies))}): pass instruments for the "
            "strategies' own symbols only"
        )
    return [own or None for own in owned]


def own_corporate_actions(strategies: Sequence[type[Strategy]], given: Any) -> list[Any]:
    """Each strategy's part of `corporate_actions=`, in call order: a dict's entries of
    the symbols it trades (None when there are none), or the list itself when the
    strategies trade one symbol between them. Raises ConfigError for a list when they
    trade more than one, and for a dict that names a symbol no strategy trades."""
    if given is None:
        return [None] * len(strategies)
    if not isinstance(given, dict):
        _one_form_for_several(
            strategies, "corporate_actions=", "a dict by symbol, {symbol: [CorporateAction, ...]}"
        )
        return [given] * len(strategies)
    symbols = traded_symbols(strategies)
    untraded = sorted(set(given) - set(symbols))
    if untraded:
        raise ConfigError(
            f"corporate_actions names {untraded}, which no strategy of the run trades "
            f"({', '.join(symbols)})"
        )
    return [
        {symbol: actions for symbol, actions in given.items() if symbol in each.traded_symbols()}
        or None
        for each in strategies
    ]


def own_margins(checked: Sequence[CheckedRun], margin: Any) -> tuple[list[CheckedRun], list[str]]:
    """(each strategy's checked run with its part of `margin=` in its plan, in call
    order; the run's warnings for what no strategy uses, judged against every strategy's
    instruments together). `margin` is the call's argument, which each plan holds
    parsed."""
    setting = parse_margin(margin)
    owned = [
        dataclasses.replace(
            run, plan=dataclasses.replace(run.plan, margin=margin_for(setting, run.instruments))
        )
        for run in checked
    ]
    every = [instrument for run in checked for instrument in run.instruments]
    return owned, margin_warnings(setting, every)


__all__ = [
    "own_corporate_actions",
    "own_instruments",
    "own_margins",
    "refuse_one_frame",
    "traded_symbols",
]
