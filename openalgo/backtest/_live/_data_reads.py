"""The data worker's reads and what their answers carry (16-session-design 2.5 B and 6.5,
unit 9): one call of the data client, the LTP of a quote answer and the candle rows of a
history answer. Part of `data_worker`.

Times are UTC nanoseconds, prices rupees.
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import Any

import pandas as pd

from ._types import LOGGER_NAME

_PRICES = ("open", "high", "low", "close")

logger = logging.getLogger(LOGGER_NAME)


class ReadFailed(Exception):
    """An answer that is not the data asked for; the text says what it was."""


def call(client: Any, endpoint: str, **arguments: Any) -> Any:
    """The answer of `client` (an `openalgo.api`) to one read of `endpoint`. Raises
    ReadFailed when the client raised: the SDK answers HTTP and transport errors with a
    dict, so an exception is unexpected, and is logged with its traceback."""
    try:
        return getattr(client, endpoint)(**arguments)
    # One failed read must not stop the others: it reaches the loop as DataFailed.
    except Exception as error:
        logger.exception("the data client raised on %s %r", endpoint, arguments)
        raise ReadFailed(f"{type(error).__name__}: {error}") from error


def ltp_of(answer: Any) -> float:
    """The last traded price of a quote answer, rupees. Raises ReadFailed for an error
    answer or a quote without a positive LTP (none traded yet)."""
    if not isinstance(answer, dict) or answer.get("status") != "success":
        raise ReadFailed(_words(answer))
    data = answer.get("data")
    ltp = data.get("ltp") if isinstance(data, dict) else None
    usable = isinstance(ltp, numbers.Real) and not isinstance(ltp, bool)
    if not (usable and math.isfinite(ltp) and ltp > 0):
        raise ReadFailed(f"the quote carries no positive last traded price (ltp {ltp!r})")
    return float(ltp)


def candle_rows(answer: Any, wanted: set[int]) -> dict[int, dict[str, Any]]:
    """The candle rows of the slices opening at `wanted` (UTC nanoseconds) in a history
    answer, by open: each an aware IST "timestamp", prices and volume, as
    `bars.candle_bar` reads a row. "No data" is an answer without rows. Raises
    ReadFailed for an error answer or a frame whose times have no zone."""
    if isinstance(answer, dict) and answer.get("error_type") == "no_data":
        return {}
    if not isinstance(answer, pd.DataFrame):
        raise ReadFailed(_words(answer))
    if getattr(answer.index, "tz", None) is None:
        raise ReadFailed("the history rows carry times without a zone: no candle is placed")
    rows: dict[int, dict[str, Any]] = {}
    for stamp, row in answer.iterrows():
        opened = int(pd.Timestamp(stamp).value)
        if opened in wanted:
            prices = {name: row[name] for name in _PRICES if name in row}
            rows[opened] = {"timestamp": stamp, **prices, "volume": row.get("volume")}
    return rows


def _words(answer: Any) -> str:
    """What an answer that is not the data said."""
    if isinstance(answer, dict):
        return str(answer.get("message") or answer)
    return f"an answer of type {type(answer).__name__}"


__all__ = ["ReadFailed", "call", "candle_rows", "ltp_of"]
