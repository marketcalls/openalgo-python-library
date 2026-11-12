"""Bar rows of a trace fixture as exact decimals, and aggregation of base bars.

Fixture rows follow `bars.columns`: time_ist, open, high, low, close (rupees), volume.
Prices are converted through their decimal text, so 800.5 is exactly 800.50 and tick
arithmetic has no floating-point noise.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from openalgo.backtest.reference import is_tick_valid

from .timegrid import Case, bar_open_containing, exchange_of

Row = list[Any]


@dataclass(frozen=True)
class Bar:
    """One bar: prices in rupees, volume in units."""

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


def rupees(value: float | int | str) -> Decimal:
    """A fixture price or amount as an exact decimal number of rupees."""
    return Decimal(str(value))


def is_on_tick(price: Decimal, tick_size: Decimal) -> bool:
    """True when a price is a whole number of ticks, by the package's own tick rule."""
    return is_tick_valid(float(price), float(tick_size))


def tick_size_of(case: Case, instrument: str) -> Decimal:
    """Tick size in rupees of an instrument declared by the case."""
    for declared in case["instruments"]:
        if declared["symbol"] == instrument:
            return rupees(declared["tick_size_rupees"])
    raise KeyError(f"{case['id']}: instrument {instrument!r} is not declared")


def bar_from_row(row: Row) -> Bar:
    """Build a Bar from one fixture row (the time column is dropped)."""
    _, open_, high, low, close, volume = row
    return Bar(rupees(open_), rupees(high), rupees(low), rupees(close), int(volume))


def bars_by_slice(case: Case) -> dict[str, dict[str, Bar]]:
    """Base bars of the case as {instrument: {slice_ist: Bar}}, in row order."""
    return {
        instrument: {row[0]: bar_from_row(row) for row in rows}
        for instrument, rows in case["bars"]["rows"].items()
    }


def aggregate(bars: Iterable[Bar]) -> Bar:
    """One bar from consecutive bars: first open, last close, extremes, summed volume."""
    members = list(bars)
    if not members:
        raise ValueError("cannot aggregate an empty set of bars")
    return Bar(
        open=members[0].open,
        high=max(bar.high for bar in members),
        low=min(bar.low for bar in members),
        close=members[-1].close,
        volume=sum(bar.volume for bar in members),
    )


def resample(case: Case, instrument: str, rows: list[Row], interval: str) -> dict[str, Bar]:
    """Aggregate base rows into session-aligned bars of `interval`, keyed by bar open.

    A bar is built from the base rows that exist; a bucket with no row is absent.
    """
    exchange = exchange_of(case, instrument)
    buckets: dict[str, list[Bar]] = {}
    for row in rows:
        bucket = bar_open_containing(case, exchange, row[0], interval)
        buckets.setdefault(bucket, []).append(bar_from_row(row))
    return {bucket: aggregate(members) for bucket, members in buckets.items()}


def rows_from_bars(bars: dict[str, Bar]) -> list[Row]:
    """Fixture rows (floats, as in the files) from bars keyed by their open time."""
    return [
        [time_ist, float(bar.open), float(bar.high), float(bar.low), float(bar.close), bar.volume]
        for time_ist, bar in bars.items()
    ]
