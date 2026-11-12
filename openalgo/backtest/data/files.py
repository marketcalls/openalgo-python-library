"""Own data on disk: `Files` reads CSV, Parquet and Feather files as a data provider.

The table a file holds goes through exactly the same column mapping, explicit
timestamp conversion and normalisation as an in-memory table (see `own.Frame`).
pyarrow is optional and imported only to read Parquet and Feather.
"""

import os
from collections.abc import Mapping

import pandas as pd

from ..errors import ConfigError, DataError, MissingDependencyError, NoDataError
from .own import Frame

FORMATS = ("csv", "parquet", "feather")
_FORMAT_BY_SUFFIX = {
    ".csv": "csv",
    ".txt": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
    ".arrow": "feather",
    ".ipc": "feather",
}
# pandas reads these compressed CSV files directly.
_COMPRESSION_SUFFIXES = (".gz", ".bz2", ".zip", ".xz", ".zst")
# What a reader raises for an unreadable or malformed file.
_READ_ERRORS = (OSError, ValueError)


class Files(Frame):
    """Bars from a CSV, Parquet or Feather file, as a data provider.

    path: a file path. It may contain the placeholders `{symbol}`, `{exchange}` and
        `{interval}`, filled in per request ("data/{symbol}_{interval}.csv").
    format: "csv", "parquet" or "feather". None reads it from the file suffix.
    read_options: keyword arguments for the reader (for CSV: `sep`, `header`,
        `names`, `usecols` and so on). The reader never parses dates: the timestamp
        column goes through the same explicit conversion as for `Frame`.
    options: the arguments of `Frame`: `tz` (required for naive timestamps),
        `columns`, `label`, `unit`, `timestamp_format`, `symbol`, `exchange`,
        `interval`, `calendar`, `now`, `regular_session_only`.

    Raises ConfigError at construction for a bad path, format or column mapping.
    Reading raises NoDataError when the file does not exist, MissingDependencyError
    when pyarrow is needed and missing, ConfigError when the format cannot be told
    and DataError when the file cannot be read.
    """

    name = "files"

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        format: str | None = None,
        read_options: Mapping[str, object] | None = None,
        **options: object,
    ) -> None:
        if not isinstance(path, str | os.PathLike):
            raise ConfigError(f"path must be a string or a path, got {type(path).__name__}")
        if format is not None and format not in FORMATS:
            raise ConfigError(f"format must be one of {FORMATS}, got {format!r}")
        if isinstance(options.get("columns"), list | tuple):
            raise ConfigError(
                "for Files, columns must be a mapping; name CSV columns with "
                "read_options={'names': [...]}"
            )
        super().__init__(None, **options)
        self._path = os.fspath(path)
        self._file_format = format
        self._read_options = dict(read_options or {})
        self._tables: dict[str, pd.DataFrame] = {}

    def _read_table(self, symbol: str, exchange: str, interval: str) -> tuple[pd.DataFrame, str]:
        try:
            path = self._path.format(symbol=symbol, exchange=exchange, interval=interval)
        except (KeyError, IndexError, ValueError) as exc:
            raise ConfigError(
                f"path {self._path!r} has a placeholder that is not symbol, exchange or "
                f"interval ({exc})"
            ) from exc
        origin = f"the file {path}"
        if path not in self._tables:
            if not os.path.isfile(path):
                raise NoDataError(f"{exchange}:{symbol} {interval}: {origin} does not exist")
            self._tables[path] = self._read(path)
        return self._tables[path], origin

    def _read(self, path: str) -> pd.DataFrame:
        kind = self._file_format
        if kind is None:
            name = os.path.basename(path).lower()
            for suffix in _COMPRESSION_SUFFIXES:
                name = name.removesuffix(suffix)
            kind = _FORMAT_BY_SUFFIX.get(os.path.splitext(name)[1])
        if kind is None:
            raise ConfigError(
                f"cannot tell the format of {path!r} from its suffix; pass format='csv', "
                "'parquet' or 'feather'"
            )
        if kind == "csv" and self._read_options.get("parse_dates"):
            raise ConfigError(
                "do not ask the CSV reader to parse dates: declare tz= (and "
                "timestamp_format= if the strings are not ISO 8601) instead"
            )
        try:
            if kind == "csv":
                return pd.read_csv(path, **self._read_options)
            from pyarrow import feather, parquet  # optional, needed only for these formats

            reader = parquet.read_table if kind == "parquet" else feather.read_table
            return reader(path, **self._read_options).to_pandas()
        except ImportError as exc:
            raise MissingDependencyError(
                f"reading the {kind} file {path} needs the optional package pyarrow, which is "
                "not installed. Install it (pip install pyarrow) or convert the file to CSV."
            ) from exc
        except _READ_ERRORS as exc:
            raise DataError(f"could not read the {kind} file {path}: {exc}") from exc


__all__ = ["FORMATS", "Files"]
