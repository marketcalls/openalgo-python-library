"""Before any data is loaded: the dated margin table must price every instrument a run
trades (04-execution 5).

The engine asks the margin table at each opening fill; an instrument with no row (a
futures underlying the table does not list) would otherwise stop the run only at its
first entry, after the data was loaded. This check asks the same table, through the
kernel, for a long position of every instrument on the run's first and last day, so it
fails first and with the engine's own message.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from ._calendar_convert import to_day
from ._engine import KIND_CODES, call
from ._native import native
from .reference import Instrument

# Kinds the margin table prices; an index is never traded.
TRADED_KINDS = ("equity", "future", "option")


def check_margin(instruments: list[Instrument], product: str, days: tuple[dt.date, ...]) -> None:
    """Raise what the engine would raise at an opening fill when the margin table has no
    rule for an instrument on any of `days`: NotImplementedError for a futures underlying
    with no row, ConfigError for a kind and product with no rule."""
    for instrument in instruments:
        if instrument.kind not in TRADED_KINDS:
            continue
        for day in dict.fromkeys(days):
            _ask(instrument, product, day)


def _ask(instrument: Instrument, product: str, day: dt.date) -> Any:
    kind = KIND_CODES[instrument.kind]
    return call(
        native.margin_check,
        instrument.exchange,
        kind,
        product,
        to_day(day),
        instrument.underlying,
    )


__all__ = ["check_margin"]
