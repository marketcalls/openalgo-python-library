"""Live fills against a backtest of the same bars (18-multi-strategy 12.3; unit M11): one row
per fill of the session, the price a backtest of the session's own bars fills the same order
at beside it (`_session_slippage`), and the day summary's two slippage figures from them.

Each fill the kernel took is a row (`FillRow`), named by its strategy and its version, its
order's purpose ("target" for the engine's exit at a leg's target, which an order's purpose
alone does not tell from a strategy's exit) and its price type as placed:

- a MARKET order's fill is set against the open of the bar opening at its decision, and the
  engine's target exit against the target's level, each with the run's slippage;
- a stop's fill (SL-M, or an option's stop-limit) against its trigger, or its bar's open
  when the bar opened through it, with the run's slippage; the bar is the one its fill was
  heard in, the trigger the last the broker confirmed (a trailed stop's);
- a LIMIT order's fill is not compared (its fill depends on the order book, which a bar
  does not hold), nor is a stop the price rule repriced to a LIMIT order, nor a fill of
  orders placed by hand that the reconciler adopted (its order is not the session's).

`difference` is per unit, signed against the trader (positive when live did worse: a buy
that paid more, a sale that received less) and `rupees` is it times the units and the
instrument's multiplier, to the paisa. A row not compared carries why (`reason`).

The day summary's `slippage` is the compared MARKET rows' rupees added up, and
`slippage_not_compared` the MARKET rows not compared; `stop_slippage` and
`stop_slippage_not_compared` are the stop rows' (`slippage_of`). Like the summary, the
rows cover every start of the day: each start saves its strategies' rows with their state,
and a restart's frame holds the earlier starts' saved rows before its own (`DayRows`).

Times are UTC nanoseconds (shown as IST in the frame), prices and amounts rupees, quantities
units.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ..errors import LiveError
from ._plan_catch_up import INSTRUMENT
from ._session_records import EXTERNAL, FillRecord
from ._session_slippage import (
    AGAINST_THE_TRADER,
    MARKET,
    NO_LEVEL,
    NOT_KNOWN,
    TRIGGERED,
    Priced,
    Slippage,
    Unpriced,
    is_target_exit,
    level_price,
    market_price,
)
from ._types import Purpose

#: The key of a strategy's saved state that holds its rows (`_session_saved`).
COMPARED = "fill_comparison"
#: The purpose of the engine's exit at a leg's target, and of an adopted fill by hand.
TARGET, HAND = "target", "hand"
#: Why a row is not compared, besides a missing price (`_session_slippage.Unpriced`).
LIMIT_ORDER = "a LIMIT order: its fill depends on the order book, which a bar does not hold"
REPRICED = "a stop the price rule repriced to a LIMIT order"
HAND_ORDER = "an order placed by hand, which the session adopted"
#: Decimals of a row's per-unit difference: float noise of a tick multiple dropped.
_DIFFERENCE_DECIMALS = 6
_PAISE_DECIMALS = 2


@dataclass(frozen=True)
class FillRow:
    """One fill against a backtest (module docstring). `backtest_ns` is the open of the bar
    a backtest fills the order in; it, the backtest's price, the difference and the rupees
    are None when the row is not compared."""

    strategy: str
    version_id: str
    symbol: str
    exchange: str
    order_id: str
    purpose: str
    price_type: str
    side: str
    units: int
    live_ns: int
    live_price: float
    backtest_ns: int | None
    backtest_price: float | None
    difference: float | None
    rupees: float | None
    compared: bool
    reason: str


def strategy_rows(session: Any, trader: Any) -> list[FillRow]:
    """The rows of `trader`'s fills in this start of `session` (module docstring), in the
    order they were taken. Its kernel session must not have finished (`finish` takes it).
    Raises what the kernel raises for an order or a price it cannot read."""
    placed = {venue.order_id: (kernel, venue) for kernel, venue in trader.venue.placed()}
    multiplier = trader.part.setup.instruments[INSTRUMENT].multiplier
    rows = []
    for fill in session.fills:
        if fill.strategy != trader.name:
            continue
        found = None if fill.source == EXTERNAL else placed.get(fill.order_id)
        if found is None:
            why = HAND_ORDER if fill.source == EXTERNAL else NOT_KNOWN
            rows.append(_row(fill, trader, HAND if why == HAND_ORDER else "", "", Unpriced(why)))
            continue
        rows.append(_order_row(session, trader, fill, *found, multiplier))
    return rows


def slippage_of(rows: Iterable[FillRow]) -> Slippage:
    """The MARKET rows' and the stop rows' slippage (module docstring), rupees to the
    paisa."""
    market, stops = [], []
    for row in rows:
        if row.price_type == MARKET:
            market.append(row)
        elif row.price_type in TRIGGERED:
            stops.append(row)
    return Slippage(*_total(market), *_total(stops))


class DayRows:
    """The rows the day's earlier starts saved (module docstring): `carried`, by running
    strategy, and `absent`, those of the strategies this start does not run."""

    def __init__(self, carried: Mapping[str, list[FillRow]], absent: list[FillRow]) -> None:
        self.carried = dict(carried)
        self.absent = list(absent)

    @classmethod
    def at_start(cls, saved: Mapping[str, Mapping[str, Any]], running: Sequence[str]) -> DayRows:
        """The rows `saved` holds (journal L's saved states of the day, by name) for a start
        that runs `running`. Raises LiveError for rows that cannot be read, naming the
        strategy."""
        carried: dict[str, list[FillRow]] = {}
        absent: list[FillRow] = []
        for name, state in saved.items():
            if not isinstance(state, Mapping) or COMPARED not in state:
                continue
            rows = _read(name, state[COMPARED])
            if name in running:
                carried[name] = rows
            else:
                absent += rows
        return cls(carried, absent)

    def saved(self, name: str, own: Sequence[FillRow]) -> list[dict[str, Any]]:
        """What strategy `name` saves: its carried rows, then `own`, as plain data."""
        return [asdict(row) for row in (*self.carried.get(name, ()), *own)]

    def day(self, own: Mapping[str, Sequence[FillRow]]) -> list[FillRow]:
        """Every row of the day, in the order their fills were heard: the earlier starts'
        and this start's (`own`, by strategy)."""
        rows = [*self.absent]
        for name, mine in own.items():
            rows += [*self.carried.get(name, ()), *mine]
        return sorted(rows, key=lambda row: row.live_ns)


def _order_row(
    session: Any, trader: Any, fill: FillRecord, kernel_order: int, venue: Any, multiplier: float
) -> FillRow:
    """The row of a fill of the session's own order (module docstring)."""
    intent, terms = venue.intent, venue.terms
    purpose, kind = intent.purpose.value, intent.price_type
    priced: Priced | Unpriced
    if kind == MARKET and is_target_exit(trader, kernel_order, intent):
        purpose = TARGET
        level = session.levels.level(trader.name, kernel_order)
        placed_ns = intent.created_ns
        priced = (
            Unpriced(NO_LEVEL)
            if level is None
            else _at_level(session, trader, fill, level, placed_ns)
        )
    elif kind == MARKET:
        priced = market_price(session, trader, kernel_order, intent)
    elif kind in TRIGGERED and terms.price_type in TRIGGERED:
        limit = terms.price if terms.price_type == "SL" else None
        priced = level_price(
            session,
            trader,
            side=fill.action,
            level=terms.trigger_price,
            at_ns=fill.at_ns,
            target=False,
            limit=limit,
        )
    else:
        stop = intent.purpose is Purpose.STOP
        priced = Unpriced(REPRICED if stop else LIMIT_ORDER)
    return _row(fill, trader, purpose, kind, priced, multiplier, intent.version_id)


def _at_level(
    session: Any, trader: Any, fill: FillRecord, level: float, placed_ns: int
) -> Priced | Unpriced:
    """A target exit's backtest price: at its `level` in the bar of the tick that reached
    it, the instant the exit was placed (`placed_ns`)."""
    return level_price(session, trader, side=fill.action, level=level, at_ns=placed_ns, target=True)


def _row(
    fill: FillRecord,
    trader: Any,
    purpose: str,
    kind: str,
    priced: Priced | Unpriced,
    multiplier: float = 1.0,
    version_id: str = "",
) -> FillRow:
    """The row of `fill` priced as `priced`."""
    compared = isinstance(priced, Priced)
    bar_ns = priced.bar_ns if compared else None
    price = priced.price if compared else None
    difference = rupees = None
    if compared:
        worse = AGAINST_THE_TRADER[fill.action] * (fill.price - priced.price)
        difference = round(worse, _DIFFERENCE_DECIMALS)
        rupees = round(worse * fill.units * multiplier, _PAISE_DECIMALS)
    return FillRow(
        strategy=trader.name,
        version_id=version_id,
        symbol=fill.symbol,
        exchange=fill.exchange,
        order_id=fill.order_id,
        purpose=purpose,
        price_type=kind,
        side=fill.action,
        units=fill.units,
        live_ns=fill.at_ns,
        live_price=fill.price,
        backtest_ns=bar_ns,
        backtest_price=price,
        difference=difference,
        rupees=rupees,
        compared=compared,
        reason="" if compared else priced.reason,
    )


def _total(rows: Sequence[FillRow]) -> tuple[float, int, int]:
    """(the compared rows' rupees to the paisa, rows not compared, rows compared)."""
    compared = [row.rupees for row in rows if row.compared and row.rupees is not None]
    total = round(sum(compared), _PAISE_DECIMALS)
    return total, len(rows) - len(compared), len(compared)


def _read(name: str, records: Any) -> list[FillRow]:
    """The rows of strategy `name`'s saved state. Raises LiveError for rows that cannot be
    read: the day's comparison is never guessed."""
    try:
        return [FillRow(**record) for record in records]
    except TypeError as error:
        raise LiveError(
            f"the fill comparison journal L saved for {name} cannot be read ({error}): "
            "check the account in OpenAlgo, then start under another account name (account=)"
        ) from None


__all__ = [
    "COMPARED",
    "HAND",
    "HAND_ORDER",
    "LIMIT_ORDER",
    "REPRICED",
    "TARGET",
    "DayRows",
    "FillRow",
    "slippage_of",
    "strategy_rows",
]
