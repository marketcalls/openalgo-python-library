"""A `Result`'s tables by name and written to files: `Result.to_dataframes`, `to_csv` and
`to_excel` (the mixin `ExportMethods`).

A spreadsheet cell holds no time zone, and pandas refuses to write a time with one to
Excel, so every time with a zone (the `time` columns and the equity table's index) is
written as a plain IST date-time: the IST wall-clock time without the zone. The `*_ts`
columns stay UTC nanoseconds. The equity table's time index becomes its first column;
the row numbers of the other tables are not written. Excel files need the optional
package openpyxl, which openalgo does not install.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pandas as pd

from ._series import IST
from .errors import MissingDependencyError

EXCEL_ENGINE = "openpyxl"
# The tables `to_dataframes` returns, in order: the `Result` properties of these names.
EXPORTED_TABLES = (
    "fills",
    "orders",
    "trades",
    "positions",
    "equity",
    "events",
    "stale_marks",
    "costs",
    "contract_notes",
)


def plain_ist(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy of `frame` with a time index as its first column and every time with a
    zone as a plain IST date-time."""
    frame = frame.reset_index() if isinstance(frame.index, pd.DatetimeIndex) else frame.copy()
    for name in frame.columns:
        if isinstance(frame[name].dtype, pd.DatetimeTZDtype):
            frame[name] = frame[name].dt.tz_convert(IST).dt.tz_localize(None)
    return frame


def write_csv(tables: dict[str, pd.DataFrame], folder: str | os.PathLike) -> list[Path]:
    """Write each table to `<folder>/<name>.csv` (the folder is created when missing) and
    return the paths, in the order of `tables`."""
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in tables.items():
        path = target / f"{name}.csv"
        plain_ist(frame).to_csv(path, index=False)
        written.append(path)
    return written


def write_excel(tables: dict[str, pd.DataFrame], path: str | os.PathLike) -> Path:
    """Write each table to a sheet of its name in one workbook at `path` (its folder is
    created when missing, as `write_csv` creates its own) and return the path. Raises
    MissingDependencyError (an ImportError) when openpyxl is missing."""
    try:
        importlib.import_module(EXCEL_ENGINE)
    except ImportError as exc:
        raise MissingDependencyError(
            "writing an Excel workbook needs the openpyxl package, which openalgo does not "
            "install: pip install openpyxl (or write CSV files with result.to_csv(folder))"
        ) from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(target, engine=EXCEL_ENGINE) as writer:
        for name, frame in tables.items():
            plain_ist(frame).to_excel(writer, sheet_name=name, index=False)
    return target


class ExportMethods:
    """`to_dataframes`, `to_csv` and `to_excel` of a `Result`, which has a property for
    each name of `EXPORTED_TABLES`."""

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Every built table, by name."""
        return {name: getattr(self, name) for name in EXPORTED_TABLES}

    def to_csv(self, folder: str | os.PathLike) -> list[Path]:
        """Write every table of `to_dataframes()` to `folder` (created when missing), one
        CSV file per table named after it (`fills.csv`, ...), and return the paths. Times
        are written as plain IST date-times, without the zone; the `*_ts` columns stay
        UTC nanoseconds."""
        return write_csv(self.to_dataframes(), folder)

    def to_excel(self, path: str | os.PathLike) -> Path:
        """Write every table of `to_dataframes()` to one Excel workbook at `path` (its
        folder is created when missing), one sheet per table named after it, and return
        the path. Times are written as plain IST date-times, without the zone (a cell
        holds none); the `*_ts` columns stay UTC nanoseconds. Needs openpyxl, which
        openalgo does not install: raises MissingDependencyError, an ImportError, saying
        `pip install openpyxl` when it is missing."""
        return write_excel(self.to_dataframes(), path)


__all__ = ["EXPORTED_TABLES", "ExportMethods", "plain_ist", "write_csv", "write_excel"]
