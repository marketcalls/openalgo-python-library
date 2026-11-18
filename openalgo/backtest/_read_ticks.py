"""`bt.read_ticks`: the ticks `bt.record` recorded, read back as one table. The live layer
loads only when it runs."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def read_ticks(
    day: Any,
    instruments: Sequence[Mapping[str, str]] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """The ticks recorded on an IST day, merged in the exact order they were received,
    one row per tick.

    day: the IST trading day (a date, or text such as "2026-11-16").
    instruments: [{"exchange": "NSE", "symbol": "SBIN"}, ...]; every instrument
        recorded that day when None.
    cache_dir: the folder the ticks were recorded in (the package's cache folder by
        default, as for `bt.record`).

    Columns: `exchange`, `symbol`; `exchange_time`, the last trade time the feed gave
    (IST), or the receive time where it gave none; `receive_time`, when the recording
    took the tick (IST): the order a session receives ticks in; `exchange_time_given`,
    True where `exchange_time` is the feed's; `ltp` (rupees); and the fields the
    recording mode kept: `volume` (the day's cumulative units; quote and depth modes),
    `bid` and `ask` (rupees; depth mode). A recording not finalized yet is left out for
    every instrument read, its number in `frame.attrs["left_out"]`: the next recording of
    the day finalizes it.

    Raises ConfigError for a day that is not a date, unusable instruments or a relative
    cache folder; DataError when the day has no recording (naming the folder), when a
    named instrument has none that day, and when a recorded file is damaged (naming it);
    OSError naming the path when a file cannot be read.
    """
    from ._live._ticks_frame import ticks_frame

    return ticks_frame(day, instruments, cache_dir=cache_dir)


__all__ = ["read_ticks"]
