"""Tick-size rules by segment, date and price slab.

A `tick_sizes` row is one price slab of the rule in force for an (exchange, segment)
pair: `price_from <= price < price_to`, prices in rupees. A rule with a single slab
from 0 with no upper bound is flat: the tick does not depend on the price.

Which price selects the slab is stated per row in `basis`, and it is never the price
being rounded: NSE fixes the slab of a share once a month from its closing price, and
the slab of a stock future or option from the close of the underlying share, not from
the premium. Where a rule has slabs, the caller must pass that reference price.

The segment of a cash instrument depends on what it is: a share uses the `equity`
rule; an ETF is excluded from it (NSE/CMTR/67133 names "all securities (excluding
Exchange Traded Funds)") and uses the `etf` rule, which the built-in tables do not
hold; an instrument whose kind is not known has no tick rule.
"""

import datetime as _dt
import math
from dataclasses import dataclass

from ..errors import ConfigError, ReferenceDataError
from ._rows import RefRow
from ._schema import SEGMENTS
from ._store import RowStore, best_origin, how_to_supply
from ._terms import Sources, underlying_kind
from .instruments import Instrument
from .rounding import round_to_tick
from .symbols import EQUITY_EXCHANGES, LISTED_DERIVATIVE_EXCHANGES

TABLE = "tick_sizes"
ETF_SEGMENT = "etf"


@dataclass(frozen=True)
class TickSlab:
    """One price slab of a tick-size rule: `price_from <= price < price_to`, in rupees.

    `price_to` None means no upper bound. `tick_size` is in rupees. `basis` names the
    price that selects the slab (empty when the rule is flat).
    """

    price_from: float
    price_to: float | None
    tick_size: float
    basis: str = ""

    def contains(self, price: float) -> bool:
        """Whether `price` (rupees) falls in this slab."""
        return price >= self.price_from and (self.price_to is None or price < self.price_to)


def check_segment(segment: str) -> str:
    """Return `segment` if it is a known tick segment. Raises `ConfigError` otherwise."""
    if segment not in SEGMENTS:
        raise ConfigError(f"segment must be one of {SEGMENTS}, got {segment!r}")
    return segment


def slab_rows(store: RowStore, exchange: str, segment: str, day: _dt.date) -> list[RefRow]:
    """The slab rows of the rule in force, lowest slab first.

    User slabs replace the built-in rule as a whole; each replaced built-in slab with a
    different tick is announced. Raises `ReferenceDataError` when no rule covers `day`;
    for the `etf` segment the message says that the share rule does not apply.
    """
    key = (exchange, segment)
    covering = store.covering(TABLE, key, day)
    rows = best_origin(covering)
    if not rows:
        detail = (
            " (ETFs are excluded from the share tick rule, NSE/CMTR/67133; the built-in "
            "tables hold no ETF tick rule)"
            if segment == ETF_SEGMENT
            else ""
        )
        raise store.missing(TABLE, key, day, detail)
    for row in rows:
        store.announce_overrides(row, covering)
    return sorted(rows, key=lambda row: row.span_low)


def is_flat(rows: list[RefRow]) -> bool:
    """True when the rule has one slab covering every price."""
    return len(rows) == 1 and rows[0].span_low == 0 and rows[0].span_high is None


def check_reference_price(reference_price: object) -> float:
    """`reference_price` as a finite positive float. Raises `ConfigError` otherwise."""
    try:
        number = float(reference_price)
    except (TypeError, ValueError):
        raise ConfigError(f"reference price must be a number, got {reference_price!r}") from None
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"reference price must be finite and positive, got {reference_price!r}")
    return number


def slab_row(
    rows: list[RefRow], exchange: str, segment: str, day: _dt.date, reference_price
) -> RefRow:
    """The slab of a rule that `reference_price` (rupees) selects.

    Raises `ConfigError` when the rule has slabs and no valid reference price is given
    (the message names the price the rule is based on), and `ReferenceDataError` when no
    slab contains the price.
    """
    if is_flat(rows):
        return rows[0]
    if reference_price is None:
        basis = rows[0].values["basis"] or "the price its circular names"
        raise ConfigError(
            f"the tick size of {segment} on {exchange} on {day.isoformat()} depends on a price "
            f"slab chosen by the {basis}; pass that price as reference_price= (the price "
            "being rounded does not choose the slab)"
        )
    price = check_reference_price(reference_price)
    for row in rows:
        if price >= row.span_low and (row.span_high is None or price < row.span_high):
            return row
    raise ReferenceDataError(
        f"No '{TABLE}' slab of {segment} on {exchange} on {day.isoformat()} contains the "
        f"reference price {price!r}. {how_to_supply(TABLE)}"
    )


def to_slab(row: RefRow) -> TickSlab:
    """A slab row as a `TickSlab`."""
    return TickSlab(row.span_low, row.span_high, row.values["tick_size"], row.values["basis"] or "")


def tick_slabs_on(
    store: RowStore, exchange: str, segment: str, day: _dt.date
) -> tuple[TickSlab, ...]:
    """The slabs of the rule in force for a segment on `day`, lowest first.

    Raises `ConfigError` for an unknown segment and `ReferenceDataError` when no rule
    covers the date.
    """
    rows = slab_rows(store, exchange, check_segment(segment), day)
    return tuple(to_slab(store.note_use(row, day)) for row in rows)


def tick_size_on(
    store: RowStore, exchange: str, segment: str, day: _dt.date, reference_price
) -> float:
    """Tick size in rupees of a segment on `day`; `reference_price` selects the slab.

    Raises the errors of `check_segment`, `slab_rows` and `slab_row`.
    """
    rows = slab_rows(store, exchange, check_segment(segment), day)
    row = slab_row(rows, exchange, segment, day, reference_price)
    return store.note_use(row, day).values["tick_size"]


def segment_of(sources: Sources, instrument: Instrument, day: _dt.date) -> str:
    """The tick-size segment of a cash instrument or of an NFO or BFO contract on `day`.

    Raises `ConfigError` for an index and for a contract whose tick comes from its own
    contract row (MCX, currency), and `ReferenceDataError` when what the instrument (or
    its underlying) is cannot be told (see `_terms.underlying_kind`).
    """
    if instrument.kind == "equity" and instrument.exchange in EQUITY_EXCHANGES:
        kind = underlying_kind(sources, instrument.symbol, instrument.exchange, day)
        return ETF_SEGMENT if kind == "etf" else "equity"
    if instrument.kind == "index" or instrument.exchange not in LISTED_DERIVATIVE_EXCHANGES:
        raise ConfigError(
            f"{instrument.symbol} on {instrument.exchange} has no slab-based tick rule"
        )
    kind = underlying_kind(sources, instrument.underlying, instrument.exchange, day)
    return f"{kind}_{instrument.kind}"


def round_price_on(
    sources: Sources,
    price: float,
    instrument: Instrument,
    day: _dt.date,
    reference_price,
    mode: str,
) -> float:
    """Round `price` (rupees) to the valid tick of `instrument` on `day`.

    Uses `instrument.tick_size` when it is set; otherwise the slab `reference_price`
    selects. Raises the errors of `segment_of` and `tick_size_on`.
    """
    tick = instrument.tick_size
    if tick is None:
        segment = segment_of(sources, instrument, day)
        tick = tick_size_on(sources.store, instrument.exchange, segment, day, reference_price)
    return round_to_tick(price, tick, mode)


__all__ = [
    "ETF_SEGMENT",
    "TickSlab",
    "check_reference_price",
    "check_segment",
    "is_flat",
    "round_price_on",
    "segment_of",
    "slab_row",
    "slab_rows",
    "tick_size_on",
    "tick_slabs_on",
    "to_slab",
]
