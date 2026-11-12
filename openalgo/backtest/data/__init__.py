"""The data layer of openalgo.backtest: providers, normalisation, cache, quality checks.

Typical use:

    from openalgo.backtest import data

    # bars from an OpenAlgo server (Historify if it covers the range, else the broker)
    result = data.load_bars("SBIN", "NSE", "5m", "2024-01-01", "2024-06-30", client=client)

    # own data, no server needed
    result = data.load_bars("SBIN", "NSE", "D", "2024-01-01", "2024-06-30",
                            data=frame, tz="Asia/Kolkata")

    bars, meta = result          # canonical DataFrame, LoadMetadata
    meta.source                  # "db", "api", "frame", "files" or a plugin name
    meta.warnings                # short history is said first
    meta.quality                 # QualityReport
    meta.partition_hashes()      # for the run manifest

Bars are always in the canonical layout of `openalgo.backtest._layout`: a
RangeIndex and the columns ts_open, ts_close (int64 UTC nanoseconds), open, high,
low, close, volume, oi (float64, NaN for unknown open interest).

Modules, one responsibility each:

    loader             load_bars: the choice of provider and the server-source path
    _finish            the steps every load ends with, LoadMetadata and BarsResult
    providers          the DataProvider protocol, the registry, entry-point plugins
    openalgo_provider  OpenAlgoProvider: source "api", "db" and the "auto" probe
    fetching           raw rows from the server: transport, chunking, pacing, backoff
    brokers            the per-broker chunk window table (data)
    own                Frame: own data in memory
    files              Files: own data in CSV, Parquet and Feather files
    normalise          timestamp normalisation into the canonical layout
    _placement         where each bar opens and closes, checked against the calendar
    cache              the local cache of raw rows
    quality            the quality checks and their report

Importing this package makes no network call and reads no entry points. pyarrow
and polars are optional and imported only where a file format needs them.
"""

from ._finish import BarsResult, LoadMetadata, partition_hashes
from .cache import resolve_cache_dir
from .files import Files
from .loader import load_bars
from .openalgo_provider import OpenAlgoProvider
from .own import Frame
from .providers import (
    DataProvider,
    available_providers,
    get_provider,
    register_provider,
    unregister_provider,
)
from .quality import QualityIssue, QualityReport, check_bars

__all__ = [
    "BarsResult",
    "DataProvider",
    "Files",
    "Frame",
    "LoadMetadata",
    "OpenAlgoProvider",
    "QualityIssue",
    "QualityReport",
    "available_providers",
    "check_bars",
    "get_provider",
    "load_bars",
    "partition_hashes",
    "register_provider",
    "resolve_cache_dir",
    "unregister_provider",
]
