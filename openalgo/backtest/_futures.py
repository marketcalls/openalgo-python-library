"""Futures strategies: the contracts a strategy trades by underlying and rank
(14-strategy-api 2.1; 04-execution 7; decision SPEC-30; lead rulings LEAD-04, LEAD-05).

A futures strategy names an underlying, never a contract: `underlying = "NIFTY"` with a
futures exchange, or underlying names in `symbols` (a name that is not a full contract
symbol is an underlying). The class attribute `expiry` names the contract rank,
"current_month" (the default) or "next_month". Before the run this module resolves,
for each underlying:

- its monthly contracts: those `instrument=` gives for the underlying (each with its
  expiry and last trading day), else the reference tables' monthly expiry schedule
  (`ReferenceTables.expiry_schedule`, final dates, so a re-dated contract keeps the
  last trading day it really had) with each contract's terms (`ReferenceTables.
  instrument`: lot size by contract expiry, tick, freeze quantity) as on the day the
  run starts trading it;
- when the rank moves on: with `roll=N` (the default 1) at the session N sessions
  before the expiry of the nearest contract still traded, which is when a position
  rolls; with `roll=None` at the session after that contract's last trading day, a
  position being settled at expiry instead. The kernel computes both sessions
  (`futures_roll_session_day`, `futures_session_after`), so the rule has one source;
- the series the kernel receives: the contracts of the rank in trading order, each with
  the session it hands over on, and the dates each contract's bars are loaded for (from
  the run's start, or the session it takes over on, to its last trading day; with
  `roll=None` from the previous contract's last trading day, whose closes adjust the
  series).

The rank `next_month` holds the second-nearest contract, so it is never held to its own
expiry; it needs `roll=N`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass
from typing import Any

from . import reference
from ._calendar_convert import from_day, to_day
from ._engine import call
from ._native import native
from .errors import ConfigError, ReferenceDataError
from .reference import Instrument, SymbolError, format_symbol, parse_symbol

FUTURES_EXCHANGES = ("NFO", "BFO", "MCX", "CDS", "BCD")
# The contract ranks of 14-strategy-api 2.1 and how many contracts each lies past the
# nearest one still traded.
RANKS = {"current_month": 0, "next_month": 1}
DEFAULT_RANK = "current_month"
# Calendar days past the run's end for which the reference tables are asked for
# contracts: a monthly contract expires within about five weeks of any date, and the
# next-month rank needs one contract more.
LOOKAHEAD_DAYS = 75


@dataclass(frozen=True)
class SeriesPlan:
    """One underlying's series: its name as the strategy gives it, the contracts in
    trading order, the IST session each hands over to the next on (None for the last of
    the run) and the (first, last) IST dates its bars are loaded for."""

    name: str
    contracts: tuple[Instrument, ...]
    hand_over: tuple[dt.date | None, ...]
    windows: tuple[tuple[dt.date, dt.date], ...]


@dataclass(frozen=True)
class FuturesPlan:
    """The futures series of a run, the rank and the roll rule (`roll` sessions before
    expiry, or None to settle at expiry), with the Assumptions lines by contract, the
    reference-table version used (None when every contract was given) and warnings."""

    series: tuple[SeriesPlan, ...]
    rank: str
    roll: int | None
    lines: dict[str, str]
    version: str | None
    warnings: tuple[str, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every contract, series by series: the registration order."""
        return tuple(i.symbol for s in self.series for i in s.contracts)

    @property
    def instruments(self) -> list[Instrument]:
        """Every contract's instrument, in registration order."""
        return [i for s in self.series for i in s.contracts]

    @property
    def windows(self) -> dict[str, tuple[dt.date, dt.date]]:
        """The dates each contract's bars are loaded for, by symbol."""
        return {
            i.symbol: window
            for s in self.series
            for i, window in zip(s.contracts, s.windows, strict=True)
        }

    def native_series(self, index_of: dict[str, int]) -> list[dict[str, Any]]:
        """The series dicts of `Engine.add_futures_series`, with registration indexes."""
        rule = "settle_at_expiry" if self.roll is None else "roll"
        return [
            {
                "rule": rule,
                "contracts": [
                    (index_of[i.symbol], None if day is None else to_day(day))
                    for i, day in zip(s.contracts, s.hand_over, strict=True)
                ],
            }
            for s in self.series
        ]

    def assumption(self) -> str:
        """The Assumptions line: the rank, the rule and each series' contracts."""
        rule = (
            "settled at expiry (roll=None)"
            if self.roll is None
            else f"rolled {self.roll} session(s) before expiry"
        )
        parts = []
        for s in self.series:
            steps = [
                i.symbol if day is None else f"{i.symbol} to {day.isoformat()}"
                for i, day in zip(s.contracts, s.hand_over, strict=True)
            ]
            parts.append(f"{s.name}: {', then '.join(steps)}")
        return f"{self.rank} contracts, {rule}, back-adjusted series; " + "; ".join(parts)


@dataclass(frozen=True)
class _Contract:
    """A contract before its terms: its symbol, last trading day and, when the caller
    gave it, its instrument."""

    symbol: str
    last_day: dt.date
    instrument: Instrument | None


def underlying_names(strategy: type, exchange: str) -> tuple[str, ...]:
    """The underlyings a strategy names (empty for a strategy on contracts or shares).

    Raises ConfigError for `underlying` on an exchange with no futures, and
    NotImplementedError for `symbols` that mix contracts and underlyings."""
    if exchange not in FUTURES_EXCHANGES:
        if getattr(strategy, "underlying", None):
            raise ConfigError(
                f"underlying={strategy.underlying!r} trades its futures: set exchange to a "
                f"derivatives exchange ({', '.join(FUTURES_EXCHANGES)}), not {exchange}"
            )
        return ()
    if getattr(strategy, "underlying", None):
        return (strategy.underlying,)
    names = tuple(getattr(strategy, "symbols", None) or ())
    plain = tuple(name for name in names if not _is_contract(name, exchange))
    if plain and len(plain) != len(names):
        raise NotImplementedError(
            "symbols mixing futures underlyings and contract symbols are not implemented "
            "yet (Phase 1 work in progress): name every future by its underlying"
        )
    return plain


def _is_contract(name: str, exchange: str) -> bool:
    try:
        parse_symbol(name, exchange)
    except SymbolError:
        return False
    return True


def _rank(strategy: type) -> str:
    rank = getattr(strategy, "expiry", None) or DEFAULT_RANK
    if rank not in RANKS:
        raise ConfigError(f"expiry takes one of {tuple(RANKS)}, got {rank!r}")
    return rank


def check_roll(roll: Any) -> int | None:
    """`roll`: None, or the whole number of sessions before expiry (0 or more).

    Raises ConfigError otherwise."""
    if roll is None:
        return None
    if isinstance(roll, bool) or not isinstance(roll, int) or roll < 0:
        raise ConfigError(
            f"roll takes a whole number of sessions before expiry (0 or more) or None, got {roll!r}"
        )
    return roll


def _given(given: Any) -> list[Instrument]:
    if isinstance(given, Instrument):
        return [given]
    if isinstance(given, dict):
        return [value for value in given.values() if isinstance(value, Instrument)]
    return []


def _underlying_of(instrument: Instrument) -> str | None:
    if instrument.underlying:
        return instrument.underlying
    try:
        return parse_symbol(instrument.symbol, instrument.exchange).underlying
    except SymbolError:
        return None


def _last_day(instrument: Instrument) -> dt.date:
    return instrument.last_trading_day or instrument.expiry


def _contracts(name: str, plan: Any, given: Any, calendar: Any) -> list[_Contract]:
    """The underlying's monthly contracts, nearest first (see the module docstring)."""
    mine = [
        i
        for i in _given(given)
        if i.kind == "future" and i.exchange == plan.exchange and _underlying_of(i) == name
    ]
    if mine:
        mine.sort(key=_last_day)
        return [_Contract(i.symbol, _last_day(i), i) for i in mine]
    tables = reference.default_tables()
    last = plan.end + dt.timedelta(days=LOOKAHEAD_DAYS)
    try:
        found = tables.expiry_schedule(
            name, plan.exchange, plan.start, last, calendar, series="monthly"
        )
    except ReferenceDataError as exc:
        raise ReferenceDataError(
            f"{name} on {plan.exchange}: the reference tables cannot list its monthly "
            f"futures ({exc}). Pass the contracts with instrument={{symbol: Instrument(...)}} "
            "giving each one's expiry and last trading day"
        ) from None
    return [
        _Contract(format_symbol(name, item.expiry), item.last_trading_day, None) for item in found
    ]


def _move_day(day: dt.date, roll: int | None, exchange: str, calendar: Any) -> dt.date:
    """When the rank moves past a contract expiring on `day`: the roll session, or the
    session after its last trading day."""
    if roll is None:
        number = call(native.futures_session_after, calendar._cal, exchange, to_day(day))
    else:
        number = call(native.futures_roll_session_day, calendar._cal, exchange, to_day(day), roll)
    return from_day(number)


def _select(
    name: str, contracts: list[_Contract], rank: int, roll: int | None, plan: Any, calendar: Any
) -> list[tuple[_Contract, dt.date | None]]:
    """The contracts of the rank over the run and the session each hands over on.

    The rank moves past the nearest contract still traded (the anchor) on its move day.
    A contract expiring before the run starts is never the anchor. When no contract
    follows the one held, the series ends with it if the anchor expires after the run
    (the run ends before the hand-over could need the next contract)."""

    def move(j: int) -> dt.date:
        return _move_day(contracts[j].last_day, roll, plan.exchange, calendar)

    def missing() -> ReferenceDataError:
        return ReferenceDataError(
            f"{name} on {plan.exchange}: no contract is known after {contracts[-1].symbol}, "
            "which the run needs; pass the contracts with instrument="
        )

    anchor = next(
        (j for j, c in enumerate(contracts) if c.last_day >= plan.start and move(j) > plan.start),
        None,
    )
    if anchor is None:
        raise ReferenceDataError(f"{name} on {plan.exchange}: no contract trades in the run")
    chosen: list[tuple[_Contract, dt.date | None]] = []
    while True:
        held = anchor + rank
        if held >= len(contracts):
            raise missing()
        if held + 1 >= len(contracts) and contracts[anchor].last_day > plan.end:
            chosen.append((contracts[held], None))
            return chosen
        day = move(anchor)
        if day > plan.end:
            chosen.append((contracts[held], None))
            return chosen
        if held + 1 >= len(contracts):
            raise missing()
        chosen.append((contracts[held], day))
        anchor += 1


def _windows(
    chosen: list[tuple[_Contract, dt.date | None]], roll: int | None, plan: Any
) -> list[tuple[dt.date, dt.date]]:
    windows = []
    for position, (contract, _) in enumerate(chosen):
        if position == 0:
            first = plan.start
        else:
            previous, hand_over = chosen[position - 1]
            first = previous.last_day if roll is None else hand_over
        windows.append((first, min(plan.end, contract.last_day)))
    return windows


def _terms(instrument: Instrument) -> str:
    freeze = (
        "no freeze quantity" if instrument.freeze_qty is None else f"freeze {instrument.freeze_qty}"
    )
    return f"lot {instrument.lot_size}, tick {instrument.tick_size}, {freeze}"


def _instrument(contract: _Contract, plan: Any, first: dt.date, calendar: Any) -> Instrument:
    """The contract's instrument: as given, or from the reference tables as on `first`
    with its final last trading day. Raises ConfigError when its tick is unknown."""
    if contract.instrument is not None:
        return contract.instrument
    tables = reference.default_tables()
    found = tables.instrument(contract.symbol, plan.exchange, first, calendar=calendar)
    if found.tick_size is None:
        raise ConfigError(
            f"{contract.symbol}: its tick size depends on a price slab; pass "
            "instrument=Instrument(..., tick_size=...)"
        )
    return dataclasses.replace(found, last_trading_day=contract.last_day)


def _series(
    name: str, plan: Any, args: dict[str, Any], calendar: Any, rank: int, roll: int | None
) -> tuple[SeriesPlan, dict[str, str], bool]:
    contracts = _contracts(name, plan, args["instrument"], calendar)
    chosen = _select(name, contracts, rank, roll, plan, calendar)
    windows = _windows(chosen, roll, plan)
    instruments, lines, from_tables = [], {}, False
    for (contract, _), (first, _) in zip(chosen, windows, strict=True):
        instrument = _instrument(contract, plan, first, calendar)
        where = "as given" if contract.instrument is not None else "from the reference tables"
        from_tables |= contract.instrument is None
        lines[instrument.symbol] = f"{instrument.symbol}: {_terms(instrument)}, {where}"
        instruments.append(instrument)
    series = SeriesPlan(
        name=name,
        contracts=tuple(instruments),
        hand_over=tuple(day for _, day in chosen),
        windows=tuple(windows),
    )
    return series, lines, from_tables


def _warnings(series: list[SeriesPlan], roll: int | None) -> tuple[str, ...]:
    if roll is not None:
        return ()
    physical = [i.symbol for s in series for i in s.contracts if i.settlement == "physical"]
    if not physical:
        return ()
    return (
        f"{', '.join(physical)}: held to expiry (roll=None) these futures are settled in cash "
        "at the last close; the exchange settles them by physical delivery, which is not "
        "modelled",
    )


def resolve_futures(
    strategy: type, plan: Any, args: dict[str, Any], calendar: Any
) -> FuturesPlan | None:
    """The futures plan of a run, or None when the strategy names no underlying.

    Raises ConfigError for an unusable `expiry` or `roll` (and for `next_month` with
    `roll=None`), ReferenceDataError when the contracts cannot be listed, and the
    reference tables' errors for a contract they do not cover."""
    names = underlying_names(strategy, plan.exchange)
    if not names:
        return None
    rank = _rank(strategy)
    roll = check_roll(args["roll"])
    if roll is None and RANKS[rank] > 0:
        raise ConfigError(
            f"expiry={rank!r} holds a contract that is not the nearest, so it is never held "
            "to its own expiry: use roll=N (sessions before expiry) with it"
        )
    built, lines, from_tables = [], {}, False
    for name in names:
        series, series_lines, tables_used = _series(name, plan, args, calendar, RANKS[rank], roll)
        built.append(series)
        lines.update(series_lines)
        from_tables |= tables_used
    version = reference.default_tables().version if from_tables else None
    return FuturesPlan(
        series=tuple(built),
        rank=rank,
        roll=roll,
        lines=lines,
        version=version,
        warnings=_warnings(built, roll),
    )


def attach(engine: Any, futures: FuturesPlan, symbols: list[str]) -> list[dict[str, Any]]:
    """Register the plan's series with a native engine (before its first advance) and
    return the switches the kernel planned, in slice order."""
    index_of = {symbol: number for number, symbol in enumerate(symbols)}
    for series in futures.native_series(index_of):
        call(engine.add_futures_series, series)
    return list(call(engine.futures_switches))


__all__ = [
    "FUTURES_EXCHANGES",
    "RANKS",
    "FuturesPlan",
    "SeriesPlan",
    "attach",
    "check_roll",
    "resolve_futures",
    "underlying_names",
]
