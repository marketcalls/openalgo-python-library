"""The run settings of a portfolio run that are keyed by what is traded, given to each
strategy as its own (18-multi-strategy 1.1: one value for the run).

`data=`, `instrument=` and `corporate_actions=` describe symbols and `margin=` underlyings,
and a strategy's own checks refuse, or warn of, an entry for something it does not trade
(`_instruments`, `_futures_contracts.check_given`, `_run_corporate_actions`,
`data.splits.require_share`, `_futures_margin`). So each strategy is given the entries of
what it trades, as a run of it alone with them would be. A strategy on a futures
underlying trades that underlying's contracts, never the share or index of the same name:

- `instrument=`: for a strategy on a futures underlying, the futures contracts of that
  underlying on its exchange; for any other strategy, the instruments of the symbols it
  trades on its exchange. An instrument no strategy takes is refused, naming what the
  strategies trade.
- `corporate_actions=`: they apply to shares on NSE and BSE only, so only a strategy on
  those exchanges is given them: a dict's entries of the symbols it trades, or the list
  when the strategies trade one share between them. A dict entry no strategy trades as a
  share is refused, and so is a list when they trade several shares (it names none) or
  none.
- `data=`: a dict stays whole (each strategy reads its own symbols from it); one table of
  bars is refused when the strategies trade more than one series between them, a futures
  underlying's contracts being a series apart from the share of its name
  (`refuse_one_frame`).
- `margin=` (once the instruments are known, `own_margins`): the part of it the strategy's
  futures use. What no strategy uses is one warning of the run, judged against the
  futures of every strategy together.

Two strategies on one symbol are both given its entries. `instrument=` holds one
instrument for a symbol, so two strategies on one symbol on two exchanges cannot both be
given one: the strategy on the other exchange runs with its default instrument.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from ._futures import underlying_names
from ._futures_contracts import underlying_of
from ._futures_margin import margin_for, margin_warnings, parse_margin
from ._instruments import CASH_EXCHANGES, given_instruments
from ._run_check import CheckedRun
from .errors import ConfigError
from .reference import Instrument
from .strategy import Strategy


def _underlyings(strategy: type[Strategy]) -> tuple[str, ...]:
    """The futures underlyings `strategy` names (empty for a strategy on shares or
    contracts). Raises as `_futures.underlying_names` does."""
    return underlying_names(strategy, strategy.exchange)


def _series(strategy: type[Strategy]) -> list[str]:
    """What `strategy` trades, one name per series: its symbols, or "NAME futures" for
    each futures underlying it names."""
    names = _underlyings(strategy)
    if names:
        return [f"{name} futures" for name in names]
    return list(strategy.traded_symbols())


def _share_symbols(strategies: Sequence[type[Strategy]]) -> list[str]:
    """The shares the strategies trade between them (the symbols of the strategies on NSE
    and BSE), sorted."""
    return sorted(
        {
            symbol
            for each in strategies
            if each.exchange in CASH_EXCHANGES
            for symbol in each.traded_symbols()
        }
    )


def _one_form_for_several(symbols: Sequence[str], what: str, form: str) -> None:
    """Raise ConfigError when `symbols` (what the strategies trade between them) are more
    than one: a `what` given in a form that names no symbol would be read as every
    symbol's; `form` is the form to give instead."""
    if len(symbols) > 1:
        raise ConfigError(f"the strategies trade {', '.join(symbols)}: give {what} as {form}")


def refuse_one_frame(strategies: Sequence[type[Strategy]], data: Any) -> None:
    """Raise ConfigError for own bars given as one table (not a dict by symbol) to
    strategies that trade more than one series between them (`_series`): the table would
    be read as the bars of every series."""
    if data is not None and not isinstance(data, dict):
        _one_form_for_several(
            sorted({name for each in strategies for name in _series(each)}),
            "data=",
            "a dict of bars by symbol, which feeds every strategy that trades the symbol",
        )


def _takes(strategy: type[Strategy], instrument: Instrument) -> bool:
    """True when `strategy` trades `instrument`: a futures contract, on its exchange, of a
    futures underlying it names; or, for a strategy on no underlying, an instrument on its
    exchange of a symbol it names."""
    if instrument.exchange != strategy.exchange:
        return False
    names = _underlyings(strategy)
    if names:
        return instrument.kind == "future" and underlying_of(instrument) in names
    return instrument.symbol in strategy.traded_symbols()


def _traded_text(strategies: Sequence[type[Strategy]]) -> str:
    """What the strategies trade, each series with its exchange, sorted: "SBIN futures on
    NFO, SBIN on NSE"."""
    return ", ".join(
        sorted({f"{name} on {each.exchange}" for each in strategies for name in _series(each)})
    )


def own_instruments(
    strategies: Sequence[type[Strategy]], given: Any
) -> list[dict[str, Instrument] | None]:
    """Each strategy's part of `instrument=` (an Instrument or a dict of them by symbol),
    in call order: the instruments it trades (`_takes`) by symbol, None when there are
    none. Raises ConfigError as `_instruments.given_instruments` does, and for an
    instrument no strategy trades."""
    by_symbol = given_instruments(given)
    owned = [
        {symbol: each for symbol, each in by_symbol.items() if _takes(strategy, each)}
        for strategy in strategies
    ]
    untraded = [
        each for symbol, each in by_symbol.items() if not any(symbol in own for own in owned)
    ]
    if untraded:
        named = ", ".join(f"{each.symbol} ({each.kind}, {each.exchange})" for each in untraded)
        raise ConfigError(
            f"instrument= gives {named}, which no strategy of the run trades "
            f"({_traded_text(strategies)}): pass instruments for the strategies' own "
            "symbols on their own exchanges, and for a strategy on a futures underlying "
            "the contracts of that underlying"
        )
    return [own or None for own in owned]


def _own_entries(given: dict[str, Any], strategy: type[Strategy]) -> dict[str, Any] | None:
    """The entries of `corporate_actions=` given as a dict that `strategy` takes: those of
    the symbols it trades when it trades shares (on NSE or BSE), None when there are none."""
    if strategy.exchange not in CASH_EXCHANGES:
        return None
    traded = strategy.traded_symbols()
    return {symbol: actions for symbol, actions in given.items() if symbol in traded} or None


def own_corporate_actions(strategies: Sequence[type[Strategy]], given: Any) -> list[Any]:
    """Each strategy's part of `corporate_actions=`, in call order. They apply to shares
    on NSE and BSE (`data.splits.require_share`), so a strategy on another exchange (a
    futures strategy among them) is given None; a strategy on NSE or BSE is given a dict's
    entries of the symbols it trades (None when there are none), or the list itself when
    the strategies trade one share between them. Raises ConfigError for a list when they
    trade several shares or none, and for a dict that names a symbol no strategy trades as
    a share."""
    if given is None:
        return [None] * len(strategies)
    shares = _share_symbols(strategies)
    if given and not shares:
        raise ConfigError(
            f"corporate_actions apply to shares on {' and '.join(CASH_EXCHANGES)}, and no "
            "strategy of the run trades one: leave them out"
        )
    if not isinstance(given, dict):
        _one_form_for_several(
            shares, "corporate_actions=", "a dict by symbol, {symbol: [CorporateAction, ...]}"
        )
        return [given if each.exchange in CASH_EXCHANGES else None for each in strategies]
    untraded = sorted(set(given) - set(shares))
    if untraded:
        raise ConfigError(
            f"corporate_actions names {untraded}, which no strategy of the run trades as a "
            f"share on {' or '.join(CASH_EXCHANGES)} ({', '.join(shares)})"
        )
    return [_own_entries(given, each) for each in strategies]


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
]
