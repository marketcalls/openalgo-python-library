"""Where a run ended, as the warnings of its `Result` say it.

The kernel ends a run at the earliest of the end of its range (`end=`), the close of
the last base bar any instrument has, and the present (`now=`, the clock by default);
see `rust/oa_bt/src/engine/run_end.rs`. When that is before the end of the range, the
equity, the trades and the metrics stop there and a position still open is reported
open, so the result says where the run ended and why.
"""

from __future__ import annotations

import pandas as pd

# What ended the run early, by the kernel's cause code.
CAUSES = {
    "last_bar": "the close of the last bar in the data",
    "present": "the present (now=)",
}
IST = "Asia/Kolkata"


def _ist(ts_ns: int) -> str:
    return pd.Timestamp(int(ts_ns), tz="UTC").tz_convert(IST).strftime("%Y-%m-%d %H:%M")


def run_end_warnings(run_end: tuple[int, int, str | None]) -> list[str]:
    """The warning for a run that ended before the end of its range, as a list of one
    line, or an empty list. `run_end` is the kernel's (close of the last slice run,
    close of the last slice of the range, cause code or None), UTC nanoseconds."""
    close, range_close, cause = run_end
    if cause is None:
        return []
    return [
        f"The run ends at {_ist(close)} IST, {CAUSES[cause]}, before the end of its range "
        f"at {_ist(range_close)} IST: equity, trades and metrics stop there, and positions "
        "still open are reported open"
    ]


__all__ = ["run_end_warnings"]
