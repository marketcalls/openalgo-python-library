"""Symbol-master snapshots (FR-DATA-11), tested with an in-process fake client.

Nothing here contacts a server: the fake client returns frames built in the test.
"""

import csv
import datetime as dt
import gzip
import io
import warnings

import pandas as pd
import pytest
from test_reference_helpers import build_tables, day

from openalgo.backtest.errors import ConfigError, DataError, ReferenceDataError
from openalgo.backtest.reference import (
    ReferenceDataWarning,
    ReferenceTables,
    SnapshotStore,
    read_symbol_master_snapshot,
    write_symbol_master_snapshot,
)
from openalgo.backtest.reference.snapshot_store import TRUSTED_EXCHANGES
from openalgo.backtest.reference.snapshots import SNAPSHOT_COLUMNS

NAN = float("nan")
SDK_COLUMNS = [
    "symbol",
    "brsymbol",
    "name",
    "exchange",
    "token",
    "expiry",
    "strike",
    "lotsize",
    "instrumenttype",
    "tick_size",
]
NIFTY_FUTURE = "NIFTY24NOV26FUT"


def _frame(rows):
    return pd.DataFrame(rows, columns=SDK_COLUMNS)


def _master(nifty_lot=65):
    """A small symbol master in the shape `client.instruments(exchange=...)` returns."""
    nfo = [
        [
            "NIFTY24NOV2625000CE",
            "x",
            "NIFTY",
            "NFO",
            "101",
            "24-NOV-26",
            25000.0,
            nifty_lot,
            "CE",
            0.05,
        ],
        [NIFTY_FUTURE, "x", "NIFTY", "NFO", "102", "24-NOV-26", -1.0, nifty_lot, "FUT", 0.1],
        ["NIFTY29DEC26FUT", "x", "NIFTY", "NFO", "103", "29-DEC-26", NAN, nifty_lot, "FUT", 0.1],
        ["RELIANCE24NOV26FUT", "x", "RELIANCE", "NFO", "104", "24-NOV-26", NAN, 500, "FUT", 0.1],
        ["VEDL24NOV26452.5PE", "x", "VEDL", "NFO", "105", "24-NOV-26", 452.5, 1150, "PE", 0.05],
    ]
    nse = [
        ["RELIANCE", "RELIANCE-EQ", "RELIANCE INDUSTRIES", "NSE", "2885", None, NAN, 1, "EQ", 0.1],
        ["SMESTOCK", "SMESTOCK-SM", "AN SME COMPANY", "NSE", "9001", None, NAN, 1200, "EQ", 0.05],
    ]
    mcx = [
        ["GOLD04DEC26FUT", "x", "GOLD", "MCX", "501", "04-DEC-26", NAN, 1, "FUT", 1.0],
        ["CRUDEOIL18NOV26FUT", "x", "CRUDEOIL", "MCX", "502", "18-NOV-26", NAN, 10, "FUT", 1.0],
    ]
    return {"NFO": _frame(nfo), "NSE": _frame(nse), "MCX": _frame(mcx)}


class FakeClient:
    """Stands in for the OpenAlgo SDK client. It only knows `instruments(exchange=...)`."""

    def __init__(self, master):
        self.master = master
        self.calls = []

    def instruments(self, *, exchange=None):
        self.calls.append(exchange)
        if exchange not in self.master:
            return {"status": "error", "message": f"Invalid exchange {exchange}"}
        return self.master[exchange]


def _write(folder, as_of="2026-11-12", exchanges=("NFO",), master=None, **options):
    client = FakeClient(_master() if master is None else master)
    return write_symbol_master_snapshot(client, folder, exchanges=exchanges, as_of=as_of, **options)


def _lines(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return handle.read().splitlines()


def _stored_rows(path):
    return {row["symbol"]: row for row in csv.DictReader(io.StringIO("\n".join(_lines(path)[1:])))}


# ---- writing ------------------------------------------------------------------------


def test_snapshot_file_is_named_after_its_date_and_placed_in_the_folder(tmp_path):
    path = _write(tmp_path / "masters")
    assert path == tmp_path / "masters" / "symbol_master_2026-11-12.csv.gz"
    assert path.exists()


def test_client_is_asked_once_per_exchange_in_the_order_given(tmp_path):
    client = FakeClient(_master())
    write_symbol_master_snapshot(
        client, tmp_path, exchanges=("NSE", "NFO", "MCX"), as_of="2026-11-12"
    )
    assert client.calls == ["NSE", "NFO", "MCX"]


def test_first_line_states_the_date_broker_exchanges_and_row_count(tmp_path):
    path = _write(tmp_path, exchanges=("NSE", "NFO", "MCX"), broker="zerodha")
    assert _lines(path)[0] == (
        "# openalgo symbol-master snapshot v1; date=2026-11-12; broker=zerodha; "
        "exchanges=NSE|NFO|MCX; rows=9"
    )


def test_second_line_is_the_csv_header_without_broker_internal_columns(tmp_path):
    header = _lines(_write(tmp_path))[1].split(",")
    assert header == list(SNAPSHOT_COLUMNS)
    assert "brsymbol" not in header and "token" not in header


def test_option_row_keeps_the_server_values(tmp_path):
    assert _stored_rows(_write(tmp_path))["NIFTY24NOV2625000CE"] == {
        "symbol": "NIFTY24NOV2625000CE",
        "exchange": "NFO",
        "name": "NIFTY",
        "instrumenttype": "CE",
        "expiry": "24-NOV-26",
        "strike": "25000",
        "lotsize": "65",
        "tick_size": "0.05",
        "freeze_qty": "",
    }


@pytest.mark.parametrize("symbol", [NIFTY_FUTURE, "NIFTY29DEC26FUT"])
def test_negative_or_missing_strike_is_stored_as_empty(tmp_path, symbol):
    assert _stored_rows(_write(tmp_path))[symbol]["strike"] == ""


def test_equity_row_has_no_expiry(tmp_path):
    row = _stored_rows(_write(tmp_path, exchanges=("NSE",)))["RELIANCE"]
    assert (row["expiry"], row["lotsize"]) == ("", "1")


def test_snapshot_text_is_ascii(tmp_path):
    assert all(line.isascii() for line in _lines(_write(tmp_path, exchanges=("NSE", "NFO", "MCX"))))


def test_snapshot_bytes_depend_only_on_the_contents(tmp_path):
    assert _write(tmp_path / "a").read_bytes() == _write(tmp_path / "b").read_bytes()


def test_default_date_is_today_in_india(tmp_path):
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    before = dt.datetime.now(ist).date()
    path = _write(tmp_path, as_of=None)
    after = dt.datetime.now(ist).date()
    assert path.name in {f"symbol_master_{day_.isoformat()}.csv.gz" for day_ in (before, after)}


def test_existing_snapshot_is_not_replaced_by_default(tmp_path):
    _write(tmp_path, master=_master(75))
    with pytest.raises(ConfigError, match="already exists.*overwrite=True"):
        _write(tmp_path, master=_master(65))
    kept = read_symbol_master_snapshot(tmp_path / "symbol_master_2026-11-12.csv.gz")
    assert kept.get(NIFTY_FUTURE, "NFO").lotsize == 75


def test_existing_snapshot_is_replaced_when_overwrite_is_asked(tmp_path):
    _write(tmp_path, master=_master(75))
    path = _write(tmp_path, master=_master(65), overwrite=True)
    assert read_symbol_master_snapshot(path).get(NIFTY_FUTURE, "NFO").lotsize == 65
    assert [item.name for item in tmp_path.iterdir()] == ["symbol_master_2026-11-12.csv.gz"]


def test_uncompressed_snapshot_counts_as_the_snapshot_of_its_date(tmp_path):
    plain = _write(tmp_path, compress=False)
    assert plain.name == "symbol_master_2026-11-12.csv"
    with pytest.raises(ConfigError, match="already exists"):
        _write(tmp_path)


def test_failing_exchange_stops_the_snapshot_and_nothing_is_written(tmp_path):
    with pytest.raises(DataError, match="exchange='BFO'.*Invalid exchange BFO"):
        _write(tmp_path, exchanges=("NFO", "BFO"))
    assert list(tmp_path.iterdir()) == []


def test_failing_exchange_can_be_skipped_with_a_warning(tmp_path):
    with pytest.warns(ReferenceDataWarning, match="BFO left out"):
        path = _write(tmp_path, exchanges=("NFO", "BFO"), skip_failed=True)
    assert read_symbol_master_snapshot(path).exchanges == ("NFO",)


@pytest.mark.parametrize("result", [_frame([]), "not a table", [], {"status": "success"}])
def test_exchange_that_returns_no_instruments_is_a_data_error(tmp_path, result):
    with pytest.raises(DataError):
        _write(tmp_path, master={"NFO": result})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "client,options",
    [
        (object(), {}),  # no instruments() method
        (FakeClient(_master()), {"exchanges": ()}),
        (FakeClient(_master()), {"broker": "a;b"}),
    ],
)
def test_bad_writer_argument_is_a_config_error(tmp_path, client, options):
    with pytest.raises(ConfigError):
        write_symbol_master_snapshot(client, tmp_path, as_of="2026-11-12", **options)


@pytest.mark.parametrize("shape", ["list of rows", "server dict"])
def test_rows_given_as_a_list_or_a_server_dict_store_the_same_as_a_frame(tmp_path, shape):
    records = _master()["NFO"].to_dict("records")
    result = records if shape == "list of rows" else {"status": "success", "data": records}
    assert (
        _write(tmp_path / "a", master={"NFO": result}).read_bytes()
        == _write(tmp_path / "b").read_bytes()
    )


# ---- reading ------------------------------------------------------------------------


def test_snapshot_reads_back_its_date_broker_and_exchanges(tmp_path):
    path = _write(tmp_path, exchanges=("NSE", "NFO", "MCX"), broker="zerodha")
    snapshot = read_symbol_master_snapshot(path)
    assert (snapshot.date, snapshot.broker, snapshot.exchanges) == (
        day("2026-11-12"),
        "zerodha",
        ("NSE", "NFO", "MCX"),
    )
    assert len(snapshot.records) == 9


def test_record_values_are_typed(tmp_path):
    record = read_symbol_master_snapshot(_write(tmp_path)).get("VEDL24NOV26452.5PE", "NFO")
    assert (record.name, record.instrumenttype, record.expiry) == ("VEDL", "PE", "24-NOV-26")
    assert (record.strike, record.lotsize, record.tick_size, record.freeze_qty) == (
        452.5,
        1150,
        0.05,
        None,
    )


def test_record_is_found_by_symbol_and_exchange_together(tmp_path):
    snapshot = read_symbol_master_snapshot(_write(tmp_path, exchanges=("NSE", "NFO")))
    assert snapshot.get("RELIANCE", "NSE").lotsize == 1
    assert snapshot.get("RELIANCE", "NFO") is None


def test_uncompressed_snapshot_reads_back_the_same_values(tmp_path):
    record = read_symbol_master_snapshot(_write(tmp_path, compress=False)).get(NIFTY_FUTURE, "NFO")
    assert (record.lotsize, record.tick_size, record.strike) == (65, 0.1, None)


@pytest.mark.parametrize(
    "name,text,fragment",
    [
        ("instruments.csv", "symbol,lotsize\nX,1\n", "not a snapshot file name"),
        ("symbol_master_2026-11-12.csv", "symbol,exchange,lotsize\nX,NSE,1\n", "header line"),
        ("symbol_master_2026-13-40.csv", "", "real date"),
        (
            "symbol_master_2026-11-12.csv",
            "# openalgo symbol-master snapshot v1; date=2026-11-13\nsymbol,exchange,lotsize\n",
            "the header says 2026-11-13",
        ),
        (
            "symbol_master_2026-11-12.csv",
            "# openalgo symbol-master snapshot v1; date=2026-11-12\nsymbol,lotsize\n",
            "lacks symbol, exchange or lotsize",
        ),
    ],
)
def test_file_that_is_not_a_valid_snapshot_is_rejected(tmp_path, name, text, fragment):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ReferenceDataError, match=fragment):
        read_symbol_master_snapshot(path)


def test_missing_snapshot_folder_is_rejected(tmp_path):
    with pytest.raises(ReferenceDataError, match="does not exist"):
        SnapshotStore(tmp_path / "missing-folder")


def test_two_snapshots_for_one_date_are_rejected(tmp_path):
    _write(tmp_path)
    twin = tmp_path / "symbol_master_2026-11-12.csv"
    twin.write_text("# openalgo symbol-master snapshot v1; date=2026-11-12\n", encoding="utf-8")
    with pytest.raises(ReferenceDataError, match="two snapshots for 2026-11-12"):
        SnapshotStore(tmp_path)


# ---- the store: dated lookups -------------------------------------------------------------


@pytest.fixture
def history(tmp_path):
    """Snapshots on 10, 11 and 12 November (NIFTY lot 75, 75, 65) and on 1 December (65)."""
    folder = tmp_path / "masters"
    for as_of, lot in (
        ("2026-11-10", 75),
        ("2026-11-11", 75),
        ("2026-11-12", 65),
        ("2026-12-01", 65),
    ):
        _write(
            folder,
            as_of=as_of,
            exchanges=("NSE", "NFO", "MCX"),
            master=_master(lot),
            broker="zerodha",
        )
    return folder


def test_store_lists_its_snapshot_dates_in_order(history):
    assert SnapshotStore(history).dates == [
        day("2026-11-10"),
        day("2026-11-11"),
        day("2026-11-12"),
        day("2026-12-01"),
    ]


@pytest.mark.parametrize(
    "on_date,lot_size",
    [
        ("2026-11-09", None),  # before the first snapshot
        ("2026-11-10", 75),
        ("2026-11-11", 75),
        ("2026-11-12", 65),  # the snapshot of the day itself
        ("2026-11-19", 65),  # carried forward seven days
        ("2026-11-20", None),  # too stale
        ("2026-12-03", 65),
    ],
)
def test_lookup_uses_the_latest_snapshot_within_the_gap_limit(history, on_date, lot_size):
    row = SnapshotStore(history).lookup(NIFTY_FUTURE, "NFO", on_date)
    assert (row.values["lot_size"] if row is not None else None) == lot_size


def test_gap_limit_can_be_set_to_zero(history):
    store = SnapshotStore(history, max_gap_days=0)
    assert store.lookup(NIFTY_FUTURE, "NFO", "2026-11-12") is not None
    assert store.lookup(NIFTY_FUTURE, "NFO", "2026-11-13") is None


def test_negative_gap_limit_is_a_config_error(history):
    with pytest.raises(ConfigError, match="max_gap_days"):
        SnapshotStore(history, max_gap_days=-1)


def test_symbol_absent_from_the_snapshot_gives_no_row(history):
    assert SnapshotStore(history).lookup("NOSUCH24NOV26FUT", "NFO", "2026-11-12") is None


def test_snapshot_row_is_a_dated_symbols_row_with_snapshot_status(history):
    row = SnapshotStore(history).lookup(NIFTY_FUTURE, "NFO", "2026-11-12")
    assert (row.table, row.key, row.status, row.origin) == (
        "symbols",
        (NIFTY_FUTURE, "NFO"),
        "snapshot",
        "snapshot:2026-11-12",
    )
    assert row.source == "symbol-master snapshot 2026-11-12 (zerodha)"


@pytest.mark.parametrize(
    "on_date,valid_from,valid_to",
    [
        ("2026-11-10", "2026-11-10", "2026-11-10"),  # the next day's snapshot takes over
        ("2026-11-12", "2026-11-12", "2026-11-19"),  # no snapshot within the gap limit
        ("2026-12-03", "2026-12-01", "2026-12-08"),  # the last snapshot
    ],
)
def test_snapshot_row_is_valid_until_the_next_snapshot_or_the_gap_limit(
    history, on_date, valid_from, valid_to
):
    row = SnapshotStore(history).lookup(NIFTY_FUTURE, "NFO", on_date)
    assert (row.valid_from, row.valid_to) == (day(valid_from), day(valid_to))


def test_snapshot_tick_size_is_ignored_unless_trusted(history):
    plain = SnapshotStore(history).lookup(NIFTY_FUTURE, "NFO", "2026-11-12")
    trusting = SnapshotStore(history, trust_tick_size=True).lookup(
        NIFTY_FUTURE, "NFO", "2026-11-12"
    )
    assert (plain.values["tick_size"], trusting.values["tick_size"]) == (None, 0.1)


def test_snapshots_are_trusted_for_terms_on_equity_and_equity_derivative_exchanges_only():
    assert TRUSTED_EXCHANGES == ("NSE", "BSE", "NFO", "BFO")


def test_mcx_snapshot_row_is_kept_as_a_record_but_gives_no_terms(history):
    store = SnapshotStore(history)
    assert store.lookup("GOLD04DEC26FUT", "MCX", "2026-11-12") is None
    assert store.record("GOLD04DEC26FUT", "MCX", "2026-11-12").lotsize == 1


# ---- snapshots inside the reference tables ------------------------------------------------

LOTS = """\
underlying,exchange,valid_from,valid_to,expiry_from,expiry_to,lot_size,source,status,note
NIFTY,NFO,2026-01-01,,,,65,exchange circular,,
"""
MCX = """\
underlying,exchange,valid_from,valid_to,expiry_from,expiry_to,lot_size,multiplier,tick_size,freeze_qty,settlement,trading_unit,quote_unit,source,status,note
GOLD,MCX,2026-01-01,,,,1,100,1,10,physical,1 kg,10 g,test,user,
CRUDEOIL,MCX,2026-01-01,,,,100,1,1,10000,cash,100 barrels,1 barrel,test,user,
"""
TICKS = """\
exchange,segment,valid_from,valid_to,price_from,price_to,tick_size,basis,source,status,note
NFO,index_future,2026-01-01,,,,0.1,,test,user,
NFO,stock_future,2026-01-01,,,,0.05,,test,user,
NSE,equity,2026-01-01,,,,0.05,,test,user,
"""


def _tables(folder, snapshots):
    extra = {"lots.csv": LOTS, "mcx.csv": MCX, "ticks.csv": TICKS}
    return build_tables(
        folder,
        extra=extra,
        snapshots=snapshots,
        only=["freeze.csv", "settlement.csv", "underlyings.csv", "rules.csv"],
    )


@pytest.fixture
def tables(tmp_path, history):
    return _tables(tmp_path, history)


def test_snapshot_supplies_the_lot_of_a_contract_the_tables_do_not_cover(tables):
    with pytest.warns(ReferenceDataWarning):  # the freeze quantity of a stock is still unknown
        future = tables.instrument(
            "RELIANCE24NOV26FUT", "NFO", "2026-11-12", allow_missing=("freeze_qty",)
        )
    assert (future.lot_size, future.tick_size, future.settlement) == (500, 0.05, "physical")
    assert (
        "lot_size",
        "symbol-master snapshot 2026-11-12 (zerodha)",
        "snapshot",
    ) in future.provenance


def test_lot_size_lookup_of_a_contract_symbol_uses_the_snapshot(tables):
    assert tables.lot_size("RELIANCE24NOV26FUT", "NFO", "2026-11-12") == 500


def test_snapshot_supplies_the_lot_of_an_equity_that_trades_in_lots(tables):
    assert tables.instrument("SMESTOCK", "NSE", "2026-11-12").lot_size == 1200


def test_contract_is_uncovered_again_once_the_snapshot_is_too_old(tables):
    with pytest.raises(ReferenceDataError, match="underlying=RELIANCE, exchange=NFO on 2026-11-23"):
        tables.instrument("RELIANCE24NOV26FUT", "NFO", "2026-11-23")


def test_snapshot_that_agrees_with_the_lot_table_raises_no_warning(tables):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        future = tables.instrument(NIFTY_FUTURE, "NFO", "2026-11-12")  # snapshot 65, table 65
    assert (future.lot_size, future.tick_size, future.freeze_qty, future.settlement) == (
        65,
        0.1,
        1800,
        "cash",
    )


def test_snapshot_that_disagrees_with_a_user_lot_row_loses_with_a_warning(tables):
    with pytest.warns(ReferenceDataWarning, match="snapshot 2026-11-10 .*gives 75.*gives 65"):
        future = tables.instrument(NIFTY_FUTURE, "NFO", "2026-11-10")
    assert future.lot_size == 65


def test_mcx_snapshot_with_the_same_lot_as_the_contract_table_passes(tables):
    gold = tables.instrument("GOLD04DEC26FUT", "MCX", "2026-11-12")  # both say 1
    assert (gold.lot_size, gold.multiplier) == (1, 100.0)


def test_mcx_snapshot_with_a_different_lot_unit_stops_the_lookup(tables):
    # The contract table says one lot is 100 units; this broker's master says 10.
    with pytest.raises(ReferenceDataError) as excinfo:
        tables.instrument("CRUDEOIL18NOV26FUT", "MCX", "2026-11-12")
    message = str(excinfo.value)
    assert "lists lot size 10 for CRUDEOIL18NOV26FUT" in message
    assert "'mcx_contracts' table has 100 for CRUDEOIL" in message


def test_snapshots_are_part_of_the_version_string(tmp_path, history, tables):
    assert ";snapshots:4:" in tables.version
    assert tables.version != _tables(tmp_path / "plain", None).version


def test_coverage_reports_snapshots_per_exchange_with_their_gaps(tables):
    entries = {
        item["key"]: item for item in tables.coverage() if item["key"].startswith("snapshot")
    }
    assert set(entries) == {"snapshot MCX", "snapshot NFO", "snapshot NSE"}
    nfo = entries["snapshot NFO"]
    assert (nfo["valid_from"], nfo["valid_to"], nfo["rows"]) == ("2026-11-10", "2026-12-08", 4)
    assert nfo["gaps"] == [("2026-11-20", "2026-11-30")]


def test_coverage_marks_snapshot_exchanges_that_are_not_used_for_terms(tables):
    entries = {
        item["key"]: item for item in tables.coverage() if item["key"].startswith("snapshot")
    }
    assert entries["snapshot MCX"]["sources"] == [
        "symbol-master snapshots (stored, not used for terms)"
    ]
    assert entries["snapshot NFO"]["sources"] == ["symbol-master snapshots"]


# ---- D3: a snapshot never silently overrides a verified built-in row -----------------------


def _stale_master():
    # NIFTY 27 March 2025 traded at lot 25 until the close of 2024-12-26 (NSE/FAOP/64625).
    row = ["NIFTY27MAR2524000CE", "x", "NIFTY", "NFO", "1", "27-MAR-25", 24000.0, 25, "CE", 0.05]
    return {"NFO": _frame([row])}


def test_stale_snapshot_lot_loses_to_the_verified_built_in_row_with_a_warning(tmp_path):
    folder = tmp_path / "snaps"
    _write(folder, as_of="2024-12-24", master=_stale_master(), broker="zerodha")
    tables = ReferenceTables(snapshots=folder)
    with pytest.warns(ReferenceDataWarning) as caught:
        assert tables.lot_size("NIFTY27MAR2524000CE", "NFO", "2024-12-27") == 75
    message = " ".join(str(item.message) for item in caught)
    assert "25" in message and "75" in message


def test_snapshot_from_a_broker_whose_adapter_was_not_read_is_only_a_cross_check(tmp_path):
    folder = tmp_path / "snaps"
    _write(folder, as_of="2026-11-12", exchanges=("NFO",), broker="somebroker")
    tables = build_tables(tmp_path, snapshots=folder, only=["underlyings.csv", "rules.csv"])
    with pytest.raises(ReferenceDataError):
        tables.lot_size("RELIANCE24NOV26FUT", "NFO", "2026-11-12")


def test_version_changes_when_a_snapshot_value_changes_but_not_its_size(tmp_path):
    folder = tmp_path / "snaps"
    path = _write(folder, master=_master(65))
    before = ReferenceTables(snapshots=folder, builtin=False).version
    text = gzip.decompress(path.read_bytes()).replace(b",65,", b",75,")
    path.write_bytes(gzip.compress(text, mtime=0))
    assert ReferenceTables(snapshots=folder, builtin=False).version != before


def test_contract_absent_from_the_snapshot_never_takes_another_contracts_lot(tmp_path):
    folder = tmp_path / "snaps"
    _write(folder, broker="zerodha")
    tables = build_tables(tmp_path, snapshots=folder, only=["underlyings.csv", "rules.csv"])
    assert tables.lot_size("RELIANCE24NOV26FUT", "NFO", "2026-11-12") == 500
    with pytest.raises(ReferenceDataError):
        tables.lot_size("RELIANCE29DEC26FUT", "NFO", "2026-11-12")
