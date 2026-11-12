"""Instrument definitions and dated reference tables for openalgo.backtest.

    from openalgo.backtest import reference as ref

    tables = ref.default_tables()
    ref.parse_symbol("NIFTY28NOV2424000CE", "NFO")
    tables.lot_size("NIFTY", "NFO", "2024-11-25", expiry="2024-11-28")
    tables.instrument("NIFTY28NOV2424000CE", "NFO", "2024-11-25")
    tables.expiries("NIFTY", "NFO", "2024-11-01", "2024-11-30", calendar)
    tables.coverage()           # what the tables cover, and what is verified
    tables.version              # for run manifests

Modules, one responsibility each:

    instruments      the `Instrument` record
    symbols          OpenAlgo symbol parsing and formatting
    rounding         tick and quantity helpers
    tables           `ReferenceTables`: dated lookups over all reference rows, and reports
    expiry           contracts and their dates from the weekday rules, the contract dates
                     and a session calendar
    tick_rules       tick-size rules by price slab
    snapshots        the symbol-master snapshot file: writing one from a client, reading it
    snapshot_store   dated lookups over a folder of snapshots

The market facts are in the CSV files under `data/` (`data/tables.csv` lists them with
the date each was last checked). Importing this package reads no data file and makes
no network call; the built-in tables are read when `default_tables()` is first called.
"""

from ._rows import RefRow
from ._store import ReferenceDataWarning
from .expiry import Expiry
from .instruments import Instrument
from .rounding import (
    check_freeze,
    is_tick_valid,
    round_to_lot,
    round_to_tick,
    split_by_freeze,
    units_to_lots,
)
from .snapshot_store import SnapshotStore
from .snapshots import read_symbol_master_snapshot, write_symbol_master_snapshot
from .symbols import ParsedSymbol, SymbolError, format_symbol, parse_symbol
from .tables import ReferenceTables, default_tables, load_tables
from .tick_rules import TickSlab

__all__ = [
    "Expiry",
    "Instrument",
    "ParsedSymbol",
    "RefRow",
    "ReferenceDataWarning",
    "ReferenceTables",
    "SnapshotStore",
    "SymbolError",
    "TickSlab",
    "check_freeze",
    "default_tables",
    "format_symbol",
    "is_tick_valid",
    "load_tables",
    "parse_symbol",
    "read_symbol_master_snapshot",
    "round_to_lot",
    "round_to_tick",
    "split_by_freeze",
    "units_to_lots",
    "write_symbol_master_snapshot",
]
