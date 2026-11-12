"""Hygiene of the CSV files shipped under openalgo/backtest/reference/data.

Every file must parse, document its columns, use ASCII only, and hold no two rows of
one key whose ranges overlap. The figures themselves are checked in
test_reference_builtin.py.
"""

import csv
import datetime as dt
import io
from importlib import resources

import pytest

from openalgo.backtest.reference._rows import find_overlaps, parse_table
from openalgo.backtest.reference._schema import TABLES

DATA = resources.files("openalgo.backtest.reference").joinpath("data")
BUILTIN_TABLES = sorted(name for name, spec in TABLES.items() if spec.builtin)
STATUSES = {"verified", "unverified"}


def _text(filename):
    return DATA.joinpath(filename).read_text(encoding="utf-8")


def _data_lines(filename):
    """Lines of a file that are neither blank nor comments."""
    lines = _text(filename).splitlines()
    return [line for line in lines if line.strip() and not line.startswith("#")]


def _records(filename):
    return list(csv.DictReader(io.StringIO("\n".join(_data_lines(filename)))))


def _registry():
    return {record["table"]: record for record in _records("tables.csv")}


def _csv_files():
    return sorted(item.name for item in DATA.iterdir() if item.name.endswith(".csv"))


def test_data_directory_holds_the_registry_and_one_file_per_built_in_table():
    assert _csv_files() == sorted(["tables.csv"] + [f"{name}.csv" for name in BUILTIN_TABLES])


def test_registry_names_every_built_in_table_and_its_file():
    registry = _registry()
    assert sorted(registry) == BUILTIN_TABLES
    assert {name: entry["file"] for name, entry in registry.items()} == {
        name: f"{name}.csv" for name in BUILTIN_TABLES
    }


@pytest.mark.parametrize("table", BUILTIN_TABLES)
def test_registry_gives_each_table_an_iso_as_of_date_and_a_description(table):
    entry = _registry()[table]
    assert dt.date.fromisoformat(entry["as_of"]) <= dt.date(2026, 11, 12)
    assert entry["description"]


@pytest.mark.parametrize("filename", _csv_files())
def test_file_is_ascii_only(filename):
    DATA.joinpath(filename).read_bytes().decode("ascii")


@pytest.mark.parametrize("filename", _csv_files())
def test_file_starts_with_a_comment_block_that_documents_its_columns(filename):
    lines = _text(filename).splitlines()
    comments = [line for line in lines if line.startswith("#")]
    header = _data_lines(filename)[0].split(",")
    assert lines[0].startswith("# ")
    assert "# Columns" in comments
    documented = " ".join(comments)
    assert [name for name in header if name not in documented] == []


@pytest.mark.parametrize("table", BUILTIN_TABLES)
def test_table_header_is_exactly_the_columns_of_its_schema(table):
    assert tuple(_data_lines(f"{table}.csv")[0].split(",")) == TABLES[table].header


@pytest.mark.parametrize("table", BUILTIN_TABLES)
def test_table_parses_in_strict_mode(table):
    rows = parse_table(TABLES[table], _text(f"{table}.csv"), f"{table}.csv", "builtin", strict=True)
    assert len(rows) == len(_data_lines(f"{table}.csv")) - 1
    assert rows


@pytest.mark.parametrize("table", BUILTIN_TABLES)
def test_table_has_no_overlapping_ranges_for_one_key(table):
    spec = TABLES[table]
    rows = parse_table(spec, _text(f"{table}.csv"), f"{table}.csv", "builtin", strict=True)
    clashes = [(first.describe(), second.describe()) for first, second in find_overlaps(spec, rows)]
    assert clashes == []


@pytest.mark.parametrize("table", BUILTIN_TABLES)
def test_every_row_has_a_source_and_a_status_of_verified_or_unverified(table):
    records = _records(f"{table}.csv")
    assert {record["status"] for record in records} <= STATUSES
    assert [record for record in records if not record["source"].strip()] == []


@pytest.mark.parametrize("table", BUILTIN_TABLES)
def test_every_validity_range_is_a_pair_of_iso_dates_in_order(table):
    for record in _records(f"{table}.csv"):
        start = dt.date.fromisoformat(record["valid_from"])
        if record["valid_to"]:
            assert start <= dt.date.fromisoformat(record["valid_to"]), record


@pytest.mark.parametrize("table", ["lot_sizes", "freeze_qty", "strike_steps", "expiry_rules"])
def test_index_table_starts_no_earlier_than_2019(table):
    starts = {dt.date.fromisoformat(record["valid_from"]) for record in _records(f"{table}.csv")}
    assert min(starts) == dt.date(2019, 1, 1)


def test_contract_date_rows_state_a_listed_expiry_no_earlier_than_the_last_trading_day():
    for record in _records("contract_dates.csv"):
        listed = dt.date.fromisoformat(record["expiry_date"])
        last = dt.date.fromisoformat(record["last_trading_day"])
        assert last <= listed, record


def test_contract_date_rows_sit_within_a_week_of_their_contract():
    # The expiry lookup reads rule dates up to a week either side of the dates it is asked
    # about, so a contract re-dated further than that would not be found.
    for record in _records("contract_dates.csv"):
        contract = dt.date.fromisoformat(record["contract"])
        listed = dt.date.fromisoformat(record["expiry_date"])
        assert abs((listed - contract).days) <= 7, record


def test_monthly_expiry_rules_start_and_end_on_month_boundaries_where_the_weekday_changes():
    records = [r for r in _records("expiry_rules.csv") if r["series"] == "monthly"]
    by_key = {}
    for record in records:
        by_key.setdefault((record["underlying"], record["exchange"]), []).append(record)
    for rows in by_key.values():
        rows.sort(key=lambda record: record["valid_from"])
        for before, after in zip(rows, rows[1:], strict=False):
            if "NONE" in (before["weekday"], after["weekday"]):
                continue
            end = dt.date.fromisoformat(before["valid_to"])
            assert (end + dt.timedelta(days=1)).day == 1, (before, after)


def test_broker_unit_rows_are_all_marked_unverified():
    assert {record["status"] for record in _records("broker_units.csv")} == {"unverified"}


def test_broker_unit_table_covers_six_brokers_and_four_segments_each():
    records = _records("broker_units.csv")
    brokers = sorted({record["broker"] for record in records})
    assert brokers == ["angel", "dhan", "fyers", "kotak", "upstox", "zerodha"]
    for broker in brokers:
        segments = sorted(record["segment"] for record in records if record["broker"] == broker)
        assert segments == ["cds_bcd", "equity", "mcx", "nfo_bfo"]
