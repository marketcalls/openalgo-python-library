"""Dated lookups, file validation, precedence and warnings of the reference tables.

These tests build their own small tables (see test_reference_helpers.py), so they do
not depend on the figures shipped in the package.
"""

import datetime as dt
import warnings

import pandas as pd
import pytest
from test_reference_helpers import FREEZE, LOTS, build_tables, write_file

from openalgo.backtest.errors import BacktestError, ConfigError, ReferenceDataError
from openalgo.backtest.reference import ReferenceDataWarning, ReferenceTables, load_tables
from openalgo.backtest.reference._loader import infer_table
from openalgo.backtest.reference._rows import find_overlaps, parse_table
from openalgo.backtest.reference._schema import TABLES

FREEZE_HEADER = "underlying,exchange,valid_from,valid_to,freeze_qty_futures,freeze_qty_options\n"


@pytest.fixture
def tables(tmp_path):
    return build_tables(tmp_path)


def _freeze_tables(tmp_path, body, header=FREEZE_HEADER):
    path = write_file(tmp_path, "freeze.csv", header + body)
    return ReferenceTables(user_files=[path], builtin=False)


# ---- validity ranges ------------------------------------------------------------------


@pytest.mark.parametrize(
    "on_date,freeze_qty",
    [
        ("2021-01-01", 2800),  # valid_from is inclusive
        ("2021-12-31", 2800),  # valid_to is inclusive
        ("2022-01-01", 1800),  # the next row starts the following day
        ("2030-06-01", 1800),  # an empty valid_to is open-ended
    ],
)
def test_row_applies_from_valid_from_to_valid_to_inclusive(tables, on_date, freeze_qty):
    assert tables.freeze_qty("NIFTY", "NFO", on_date) == freeze_qty


def test_date_before_the_first_row_raises_reference_data_error(tables):
    with pytest.raises(ReferenceDataError) as excinfo:
        tables.freeze_qty("NIFTY", "NFO", "2020-12-31")
    assert isinstance(excinfo.value, BacktestError)


def test_uncovered_date_error_names_what_is_missing_and_how_to_supply_it(tables):
    with pytest.raises(ReferenceDataError) as excinfo:
        tables.freeze_qty("NIFTY", "NFO", "2020-12-31")
    message = str(excinfo.value)
    assert "freeze_qty" in message and "underlying=NIFTY" in message and "2020-12-31" in message
    assert "Rows for this key cover 2021-01-01 to the present" in message
    assert "user file" in message and "write_symbol_master_snapshot" in message


def test_unknown_underlying_error_lists_the_keys_the_table_has(tables):
    with pytest.raises(ReferenceDataError) as excinfo:
        tables.lot_size("TATASTEEL", "NFO", "2024-06-03")
    message = str(excinfo.value)
    assert "underlying=TATASTEEL" in message
    assert "The 'lot_sizes' table has rows for: NIFTY NFO, RELIANCE NFO." in message


def test_gap_between_two_rows_is_not_bridged(tmp_path):
    body = "NIFTY,NFO,2021-01-01,2021-06-30,2800,2800\nNIFTY,NFO,2022-01-01,,1800,1800\n"
    with pytest.raises(ReferenceDataError):
        _freeze_tables(tmp_path, body).freeze_qty("NIFTY", "NFO", "2021-09-15")


@pytest.mark.parametrize(
    "value",
    [
        "2022-01-01",
        dt.date(2022, 1, 1),
        dt.datetime(2022, 1, 1, 9, 15),
        pd.Timestamp("2022-01-01 09:15"),
    ],
)
def test_date_argument_may_be_a_string_a_date_a_datetime_or_a_timestamp(tables, value):
    assert tables.freeze_qty("NIFTY", "NFO", value) == 1800


def test_date_argument_that_is_not_a_date_is_a_config_error(tables):
    with pytest.raises(ConfigError, match="ISO string"):
        tables.freeze_qty("NIFTY", "NFO", "first of January")


@pytest.mark.parametrize("underlying,exchange", [("NIFTY", "NSE_INDEX"), ("nifty", "NFO")])
def test_underlying_may_be_given_on_its_index_exchange_or_in_lower_case(
    tables, underlying, exchange
):
    assert tables.freeze_qty(underlying, exchange, "2022-01-03") == 1800


# ---- lot size: trade date and contract expiry --------------------------------------------


@pytest.mark.parametrize(
    "expiry,lot_size",
    [
        ("2024-12-05", 25),  # a contract listed before the revision keeps the old lot
        ("2024-12-26", 25),  # expiry_to is inclusive
        ("2025-01-02", 75),  # expiry_from is inclusive: the first contract on the new lot
        ("2025-01-23", 75),
        ("2025-01-30", 25),  # the monthly contract was listed before the revision
        ("2025-02-27", 75),
    ],
)
def test_lot_size_on_one_trade_date_depends_on_the_contract_expiry(tables, expiry, lot_size):
    assert tables.lot_size("NIFTY", "NFO", "2024-12-02", expiry=expiry) == lot_size


def test_lot_size_without_an_expiry_is_refused_when_listed_contracts_differ(tables):
    with pytest.raises(ConfigError) as excinfo:
        tables.lot_size("NIFTY", "NFO", "2024-12-02")
    message = str(excinfo.value)
    assert "25 for expiries any to 2024-12-26" in message
    assert "75 for expiries 2025-01-02 to 2025-01-23" in message
    assert "Pass expiry=" in message


@pytest.mark.parametrize("on_date,lot_size", [("2023-06-01", 50), ("2025-06-02", 75)])
def test_lot_size_needs_no_expiry_when_every_listed_contract_agrees(tables, on_date, lot_size):
    assert tables.lot_size("NIFTY", "NFO", on_date) == lot_size


@pytest.mark.parametrize(
    "symbol,lot_size",
    [("NIFTY26DEC2424000CE", 25), ("NIFTY02JAN2524000PE", 75), ("NIFTY30JAN25FUT", 25)],
)
def test_lot_size_of_a_full_contract_symbol_uses_the_expiry_in_the_symbol(tables, symbol, lot_size):
    assert tables.lot_size(symbol, "NFO", "2024-12-02") == lot_size


def test_lot_size_for_an_expiry_that_no_row_covers_raises(tables):
    with pytest.raises(ReferenceDataError, match="for a contract expiring on 2024-12-27"):
        tables.lot_size("NIFTY", "NFO", "2024-12-02", expiry="2024-12-27")


@pytest.mark.parametrize(
    "underlying,lot_size,multiplier",
    [("GOLD", 1, 100.0), ("CRUDEOIL", 100, 1.0), ("NATURALGAS", 1250, 1.0)],
)
def test_mcx_lot_is_in_openalgo_units_with_a_per_unit_multiplier(
    tables, underlying, lot_size, multiplier
):
    assert tables.lot_size(underlying, "MCX", "2024-11-04") == lot_size
    assert tables.multiplier(underlying, "MCX", "2024-11-04") == multiplier


def test_multiplier_is_one_on_nfo_where_price_is_quoted_per_unit(tables):
    assert tables.multiplier("NIFTY", "NFO", "2024-11-04") == 1.0


def test_multiplier_on_an_exchange_without_tables_raises(tables):
    with pytest.raises(ReferenceDataError, match="No multiplier is known for USDINR on CDS"):
        tables.multiplier("USDINR", "CDS", "2024-11-04")


@pytest.mark.parametrize(
    "underlying,exchange,kind",
    [
        ("NIFTY", "NFO", "index"),
        ("RELIANCE", "NFO", "stock"),
        ("RELIANCE", "NSE", "stock"),
        ("NIFTYBEES", "NSE", "etf"),
        ("GOLD", "MCX", "commodity"),
        ("USDINR", "CDS", "currency"),
    ],
)
def test_underlying_type_comes_from_the_underlyings_table(tables, underlying, exchange, kind):
    assert tables.underlying_type(underlying, exchange, "2024-06-03") == kind


def test_underlying_type_outside_the_dates_of_its_row_raises(tables):
    with pytest.raises(ReferenceDataError, match="underlyings"):
        tables.underlying_type("NIFTY", "NFO", "2018-12-31")


# ---- file validation ---------------------------------------------------------------------


def test_two_rows_of_one_key_with_overlapping_dates_are_rejected(tmp_path):
    body = "NIFTY,NFO,2021-01-01,2022-01-01,2800,2800\nNIFTY,NFO,2022-01-01,,1800,1800\n"
    with pytest.raises(ReferenceDataError, match="overlap"):
        _freeze_tables(tmp_path, body)


def test_user_rows_in_two_files_may_not_overlap_each_other(tmp_path):
    first = write_file(tmp_path, "one.csv", FREEZE_HEADER + "NIFTY,NFO,2021-01-01,,2800,2800\n")
    second = write_file(tmp_path, "two.csv", FREEZE_HEADER + "NIFTY,NFO,2022-01-01,,1800,1800\n")
    with pytest.raises(ReferenceDataError, match="overlap"):
        ReferenceTables(user_files=[first, second], builtin=False)


LOT_HEADER = "underlying,exchange,valid_from,valid_to,expiry_from,expiry_to,lot_size\n"


@pytest.mark.parametrize(
    "body,overlaps",
    [
        ("NIFTY,NFO,2024-01-01,,,2024-12-26,25\nNIFTY,NFO,2024-01-01,,2024-12-27,,75\n", 0),
        ("NIFTY,NFO,2024-01-01,,,2024-12-26,25\nNIFTY,NFO,2024-01-01,,2024-12-26,,75\n", 1),
        ("NIFTY,NFO,2024-01-01,,,,25\nBANKNIFTY,NFO,2024-01-01,,,,15\n", 0),
        ("NIFTY,NFO,2024-01-01,2024-06-30,,,25\nNIFTY,NFO,2024-07-01,,,,75\n", 0),
    ],
)
def test_lot_rows_overlap_only_when_dates_and_expiries_both_intersect(body, overlaps):
    spec = TABLES["lot_sizes"]
    rows = parse_table(spec, LOT_HEADER + body, "t.csv", "user:t.csv", strict=False)
    assert len(find_overlaps(spec, rows)) == overlaps


def test_tick_slabs_that_touch_at_a_price_do_not_overlap():
    spec = TABLES["tick_sizes"]
    text = (
        "exchange,segment,valid_from,valid_to,price_from,price_to,tick_size\n"
        "NSE,equity,2025-04-15,,0,250,0.01\n"
        "NSE,equity,2025-04-15,,250,1000,0.05\n"
    )
    rows = parse_table(spec, text, "t.csv", "user:t.csv", strict=False)
    assert find_overlaps(spec, rows) == []


@pytest.mark.parametrize(
    "text,fragment",
    [
        (FREEZE_HEADER + "NIFTY,NFO,01-01-2021,,1800,1800\n", "ISO dates"),
        (FREEZE_HEADER + "NIFTY,NFO,2021-01-01,2020-01-01,1800,1800\n", "before valid_from"),
        (FREEZE_HEADER + "NIFTY,NFO,2021-01-01,,many,many\n", "column 'freeze_qty_futures'"),
        (FREEZE_HEADER + "NIFTY,NFO,2021-01-01,,0,0\n", "whole number, 1 or more"),
        (FREEZE_HEADER + "NIFTY,NFO,2021-01-01,,,\n", "'freeze_qty_futures' is empty"),
        (FREEZE_HEADER + ",NFO,2021-01-01,,1800,1800\n", "key column 'underlying' is empty"),
        ("underlying,exchange,valid_to,freeze_qty_futures\nNIFTY,NFO,,1800\n", "missing column"),
        (FREEZE_HEADER + "NIFTY,NFO,2021-01-01,,1800,1800,extra\n", "expected 6 columns, found 7"),
        (
            FREEZE_HEADER[:-1] + ",status\nNIFTY,NFO,2021-01-01,,1800,1800,maybe\n",
            "cannot carry status",
        ),
        ("", "no header row"),
    ],
)
def test_malformed_user_file_is_reported_with_its_name_and_the_fault(tmp_path, text, fragment):
    path = write_file(tmp_path, "broken.csv", text)
    with pytest.raises(ReferenceDataError) as excinfo:
        ReferenceTables(user_files=[("freeze_qty", path)], builtin=False)
    assert fragment in str(excinfo.value)
    assert "broken.csv" in str(excinfo.value)


def test_malformed_row_is_reported_with_its_line_number(tmp_path):
    text = (
        "# a comment line\n"
        + FREEZE_HEADER
        + "NIFTY,NFO,2021-01-01,,1800,1800\n\nNIFTY,NFO,bad,,1,1\n"
    )
    with pytest.raises(ReferenceDataError, match="freeze.csv line 5"):
        _freeze_tables(tmp_path, "", header=text)


def test_missing_user_file_is_reported(tmp_path):
    with pytest.raises(ReferenceDataError, match="nowhere.csv"):
        ReferenceTables(user_files=[str(tmp_path / "nowhere.csv")], builtin=False)


@pytest.mark.parametrize(
    "columns,table",
    [
        (["underlying", "exchange", "valid_from", "lot_size"], "lot_sizes"),
        (["underlying", "exchange", "valid_from", "freeze_qty_futures"], "freeze_qty"),
        (["underlying", "exchange", "valid_from", "lot_size", "multiplier"], "mcx_contracts"),
        (["symbol", "exchange", "valid_from", "lot_size", "multiplier"], "symbols"),
        (["exchange", "segment", "valid_from", "price_from", "tick_size"], "tick_sizes"),
        (["exchange", "segment", "valid_from", "settlement"], "settlement"),
        (["underlying", "exchange", "series", "valid_from", "weekday"], "expiry_rules"),
        (
            ["underlying", "exchange", "contract", "valid_from", "last_trading_day"],
            "contract_dates",
        ),
        (["underlying", "exchange", "valid_from", "strike_step"], "strike_steps"),
        (["broker", "segment", "valid_from", "tick_size_unit"], "broker_units"),
    ],
)
def test_table_of_a_user_file_is_inferred_from_its_columns(columns, table):
    assert infer_table(columns) == table


def test_columns_that_match_no_table_are_rejected():
    with pytest.raises(ReferenceDataError, match="cannot tell which reference table"):
        infer_table(["a", "b"])


def test_user_files_may_be_given_as_a_mapping_of_table_to_paths(tmp_path):
    lots = write_file(tmp_path, "a.csv", LOTS)
    freeze = write_file(tmp_path, "b.csv", FREEZE)
    tables = ReferenceTables(user_files={"lot_sizes": lots, "freeze_qty": [freeze]}, builtin=False)
    assert tables.freeze_qty("NIFTY", "NFO", "2022-01-03") == 1800
    assert tables.lot_size("RELIANCE", "NFO", "2024-06-03") == 250


def test_user_file_for_an_unknown_table_is_rejected(tmp_path):
    path = write_file(tmp_path, "a.csv", LOTS)
    with pytest.raises(ReferenceDataError, match="unknown reference table 'no_such_table'"):
        ReferenceTables(user_files=[("no_such_table", path)], builtin=False)


def test_find_refuses_a_table_whose_rows_also_depend_on_a_second_range(tables):
    with pytest.raises(ConfigError, match="expiry_from"):
        tables.find("lot_sizes", ("NIFTY", "NFO"), "2024-12-02")


# ---- status and warnings -----------------------------------------------------------------

# The built-in freeze table holds verified and unverified rows; FINNIFTY at launch is
# unverified (its circular writes the limit as 5001). TESTINDEX comes from a user file.
USER_FREEZE = FREEZE_HEADER + "TESTINDEX,NFO,2022-01-01,,900,900\n"


@pytest.fixture
def mixed(tmp_path):
    return ReferenceTables(user_files=[write_file(tmp_path, "freeze.csv", USER_FREEZE)])


def test_unverified_row_warns_and_names_the_row_and_its_source(mixed):
    with pytest.warns(ReferenceDataWarning) as caught:
        assert mixed.freeze_qty("FINNIFTY", "NFO", "2021-01-20") == 5000
    message = str(caught[0].message)
    assert "freeze_qty[FINNIFTY NFO] 2021-01-11 to 2021-01-31" in message
    assert "is unverified (source: NSE/FAOP/46924" in message


def test_unverified_row_warns_on_every_use_not_only_the_first(mixed):
    for _ in range(3):
        with pytest.warns(ReferenceDataWarning):
            mixed.freeze_qty("FINNIFTY", "NFO", "2021-01-20")


@pytest.mark.parametrize("underlying", ["FINNIFTY", "TESTINDEX"])
def test_verified_row_and_plain_user_row_do_not_warn(mixed, underlying):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mixed.freeze_qty(underlying, "NFO", "2022-06-01")


def test_used_rows_can_be_listed_by_status(mixed):
    with pytest.warns(ReferenceDataWarning):
        mixed.freeze_qty("FINNIFTY", "NFO", "2021-01-20")
    mixed.freeze_qty("TESTINDEX", "NFO", "2022-06-01")
    assert [row.values["freeze_qty_options"] for row in mixed.used_rows("unverified")] == [5000]
    assert [row.values["freeze_qty_options"] for row in mixed.used_rows("user")] == [900]
    assert len(mixed.used_rows()) == 2


def test_row_without_a_status_or_source_is_a_user_row_named_after_its_file(mixed):
    (row,) = mixed.rows("freeze_qty", ("TESTINDEX", "NFO"))
    assert (row.status, row.source, row.origin) == (
        "user",
        "user file freeze.csv",
        "user:freeze.csv",
    )


def test_verification_summary_counts_rows_by_table_and_status(mixed):
    rows = mixed.rows("freeze_qty")
    expected = {
        status: sum(row.status == status for row in rows) for status in {r.status for r in rows}
    }
    assert mixed.verification_summary()["freeze_qty"] == expected
    assert expected["user"] == 1 and expected["verified"] > 0 and expected["unverified"] > 0


# ---- coverage ------------------------------------------------------------------------------


def test_coverage_reports_the_span_and_the_status_counts_of_a_key(mixed):
    entry = next(item for item in mixed.coverage() if item["key"] == "TESTINDEX NFO")
    assert entry == {
        "table": "freeze_qty",
        "key": "TESTINDEX NFO",
        "valid_from": "2022-01-01",
        "valid_to": None,
        "rows": 1,
        "verified": 0,
        "unverified": 0,
        "user": 1,
        "gaps": [],
        "as_of": "2026-11-12",
        "sources": ["user file freeze.csv"],
    }


def test_coverage_lists_the_uncovered_dates_between_rows(tmp_path):
    body = "NIFTY,NFO,2021-01-01,2021-06-30,2800,2800\nNIFTY,NFO,2022-01-01,,1800,1800\n"
    (entry,) = _freeze_tables(tmp_path, body).coverage()
    assert entry["gaps"] == [("2021-07-01", "2021-12-31")]


def test_coverage_has_one_entry_per_table_and_key(tables):
    keys = {(item["table"], item["key"]) for item in tables.coverage()}
    assert {
        ("lot_sizes", "NIFTY NFO"),
        ("lot_sizes", "RELIANCE NFO"),
        ("freeze_qty", "NIFTY NFO"),
        ("expiry_rules", "NIFTY NFO weekly"),
        ("expiry_rules", "BANKNIFTY NFO monthly"),
        ("tick_sizes", "NSE equity"),
        ("settlement", "NFO index_option"),
        ("mcx_contracts", "GOLD MCX"),
    } <= keys


# ---- version ---------------------------------------------------------------------------------


def test_version_without_any_data_says_so():
    assert ReferenceTables(builtin=False).version == "builtin:none"


def test_version_is_the_same_for_the_same_user_data_whatever_the_line_endings(tmp_path):
    unix = write_file(tmp_path, "one.csv", FREEZE)
    windows = write_file(tmp_path, "two.csv", FREEZE.replace("\n", "\r\n"))
    first = ReferenceTables(user_files=[unix], builtin=False).version
    second = ReferenceTables(user_files=[windows], builtin=False).version
    assert first == second
    assert first.startswith("builtin:none;user:") and len(first) == len("builtin:none;user:") + 12


def test_version_changes_when_a_user_figure_changes(tmp_path):
    original = write_file(tmp_path, "one.csv", FREEZE)
    changed = write_file(tmp_path, "two.csv", FREEZE.replace("1800", "1200"))
    first = ReferenceTables(user_files=[original], builtin=False).version
    assert first != ReferenceTables(user_files=[changed], builtin=False).version


def test_load_tables_with_extras_builds_a_new_object_each_time(tmp_path):
    path = write_file(tmp_path, "f.csv", FREEZE)
    first = load_tables(user_files=[path], builtin=False)
    assert first is not load_tables(user_files=[path], builtin=False)


# ---- tick sizes ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exchange,segment,on_date,reference_price",
    [
        ("NFO", "index_option", "2024-06-03", None),
        ("NSE", "equity", "2025-04-14", None),
        ("NSE", "equity", "2025-04-14", 120.0),  # a price is accepted and ignored
    ],
)
def test_flat_tick_rule_needs_no_price(tables, exchange, segment, on_date, reference_price):
    assert tables.tick_size(exchange, segment, on_date, reference_price) == 0.05


@pytest.mark.parametrize(
    "reference_price,tick",
    [
        (249.99, 0.01),
        (250.0, 0.05),  # price_from is inclusive, price_to exclusive
        (999.95, 0.05),
        (1000.0, 0.10),
        (4999.95, 0.10),
    ],
)
def test_slab_tick_rule_uses_the_slab_of_the_reference_price(tables, reference_price, tick):
    assert tables.tick_size("NSE", "equity", "2025-04-15", reference_price) == tick


def test_tick_slabs_are_listed_lowest_first_with_their_basis(tables):
    slabs = tables.tick_slabs("NSE", "equity", "2025-04-15")
    assert [(slab.price_from, slab.price_to, slab.tick_size) for slab in slabs] == [
        (0.0, 250.0, 0.01),
        (250.0, 1000.0, 0.05),
        (1000.0, 5000.0, 0.10),
    ]
    assert {slab.basis for slab in slabs} == {"reference close"}


def test_slab_tick_rule_without_a_price_is_a_config_error_naming_the_price_it_needs(tables):
    with pytest.raises(ConfigError, match="chosen by the reference close; pass that price"):
        tables.tick_size("NSE", "equity", "2025-04-15")


@pytest.mark.parametrize(
    "exchange,on_date,reference_price",
    [
        ("NSE", "2025-04-15", 7500.0),  # above the highest slab of this small table
        ("BSE", "2025-04-15", 100.0),  # no rule for the exchange
        ("NSE", "2018-12-31", 100.0),  # before the first rule
    ],
)
def test_tick_lookup_that_no_rule_or_slab_covers_raises(tables, exchange, on_date, reference_price):
    with pytest.raises(ReferenceDataError, match="tick_sizes"):
        tables.tick_size(exchange, "equity", on_date, reference_price)


def test_unknown_tick_segment_is_a_config_error(tables):
    with pytest.raises(ConfigError, match="segment must be one of"):
        tables.tick_size("NSE", "bonds", "2025-04-15")


# ---- freeze quantity by kind ----------------------------------------------------------------

SPLIT_LIMITS = (
    "underlying,exchange,valid_from,valid_to,freeze_qty_futures,freeze_qty_options\n"
    "SENSEX,BFO,2023-06-22,,500,1000\n"
)


@pytest.fixture
def split_limits(tmp_path):
    path = write_file(tmp_path, "limits.csv", SPLIT_LIMITS)
    return ReferenceTables(user_files=[path], builtin=False)


@pytest.mark.parametrize("kind,limit", [("future", 500), ("option", 1000)])
def test_freeze_quantity_is_looked_up_per_kind(split_limits, kind, limit):
    assert split_limits.freeze_qty("SENSEX", "BFO", "2024-06-03", kind=kind) == limit


@pytest.mark.parametrize(
    "symbol,limit", [("SENSEX28JUN24FUT", 500), ("SENSEX28JUN2477000CE", 1000)]
)
def test_freeze_quantity_of_a_contract_symbol_uses_its_own_kind(split_limits, symbol, limit):
    assert split_limits.freeze_qty(symbol, "BFO", "2024-06-03") == limit


def test_freeze_quantity_without_a_kind_is_refused_when_the_limits_differ(split_limits):
    with pytest.raises(ConfigError, match="Pass kind='future' or kind='option'"):
        split_limits.freeze_qty("SENSEX", "BFO", "2024-06-03")


def test_tick_slab_contains_its_lower_bound_but_not_its_upper_bound(tables):
    slab = tables.tick_slabs("NSE", "equity", "2025-04-15")[1]  # 250 up to 1000
    assert (slab.contains(250.0), slab.contains(999.95), slab.contains(1000.0)) == (
        True,
        True,
        False,
    )


# ---- D3: user rows override built-in rows openly -------------------------------------------


def test_user_row_overriding_a_built_in_row_warns_naming_both_values(tmp_path):
    path = write_file(
        tmp_path,
        "lots.csv",
        "underlying,exchange,valid_from,valid_to,lot_size\nNIFTY,NFO,2024-01-01,2024-12-31,50\n",
    )
    custom = ReferenceTables(user_files=[path])
    with pytest.warns(ReferenceDataWarning) as caught:
        assert custom.lot_size("NIFTY", "NFO", "2024-12-02", expiry="2025-01-02") == 50
    message = " ".join(str(item.message) for item in caught)
    assert "lots.csv" in message and "50" in message and "75" in message


def test_user_row_cannot_claim_to_be_verified(tmp_path):
    path = write_file(
        tmp_path,
        "lots.csv",
        "underlying,exchange,valid_from,lot_size,status,source\n"
        "NIFTY,NFO,2024-01-01,50,verified,trust me\n",
    )
    with pytest.raises(ReferenceDataError, match="status"):
        ReferenceTables(user_files=[path], builtin=False)


@pytest.mark.parametrize("value", ["inf", "nan", "1e400"])
def test_non_finite_lot_in_a_user_file_is_reported_with_the_file_and_line(tmp_path, value):
    path = write_file(
        tmp_path,
        "bad.csv",
        f"underlying,exchange,valid_from,lot_size\nNIFTY,NFO,2024-01-01,{value}\n",
    )
    with pytest.raises(ReferenceDataError, match="bad.csv line 2"):
        ReferenceTables(user_files=[path], builtin=False)


@pytest.mark.parametrize("price", [float("nan"), float("inf"), -10.0, 0.0])
def test_reference_price_that_is_not_a_positive_number_is_a_config_error(tables, price):
    with pytest.raises(ConfigError, match="reference price"):
        tables.tick_size("NSE", "equity", "2025-04-15", price)


# ---- finding 9: every built-in row used past its as-of date warns --------------------------


def test_closed_built_in_row_used_after_its_as_of_date_warns():
    fresh = ReferenceTables()
    # The row runs to 2026-12-31 but was last checked on 2026-11-12.
    with pytest.warns(ReferenceDataWarning, match="last checked on 2026-11-12"):
        assert fresh.lot_size("SENSEX", "BFO", "2026-12-15", expiry="2026-12-31") == 20


def test_each_distinct_built_in_row_warns_once_past_its_as_of_date():
    fresh = ReferenceTables()
    with pytest.warns(ReferenceDataWarning) as first:
        fresh.lot_size("SENSEX", "BFO", "2026-12-15", expiry="2026-12-31")
    with pytest.warns(ReferenceDataWarning) as second:
        fresh.freeze_qty("NIFTY", "NFO", "2026-12-15")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fresh.lot_size("SENSEX", "BFO", "2026-12-16", expiry="2026-12-31")
    assert len(first) == 1 and len(second) == 1
