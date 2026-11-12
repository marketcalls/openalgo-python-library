"""`instrument(symbol, exchange, on_date)`: the parser combined with the tables.

Uses the small made-up tables in test_reference_helpers.py: NIFTY (an index, lot 25
then 75), RELIANCE (a stock, lot 250, no freeze quantity row) and three MCX contracts.
"""

import warnings

import pytest
from test_reference_helpers import FakeCalendar, build_tables, day, write_file

from openalgo.backtest.errors import ConfigError, ReferenceDataError
from openalgo.backtest.reference import (
    Instrument,
    ReferenceDataWarning,
    ReferenceTables,
    SymbolError,
)

SYMBOLS_HEADER = (
    "symbol,exchange,valid_from,valid_to,lot_size,tick_size,freeze_qty,multiplier,"
    "underlying_type,settlement\n"
)


@pytest.fixture
def tables(tmp_path):
    return build_tables(tmp_path)


def _with_symbols(tmp_path, body):
    return build_tables(tmp_path, extra={"symbols.csv": SYMBOLS_HEADER + body})


def _terms(instrument):
    return (
        instrument.lot_size,
        instrument.multiplier,
        instrument.tick_size,
        instrument.freeze_qty,
        instrument.settlement,
    )


# ---- index derivatives ------------------------------------------------------------------


def test_index_option_takes_its_terms_from_the_tables_of_its_underlying(tables):
    option = tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-02")
    assert _terms(option) == (25, 1.0, 0.05, 1800, "cash")


def test_parsed_parts_of_the_symbol_are_carried_on_the_instrument(tables):
    option = tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-02")
    assert (option.symbol, option.exchange, option.kind) == ("NIFTY26DEC2424000CE", "NFO", "option")
    assert (option.underlying, option.expiry, option.strike, option.option_type) == (
        "NIFTY",
        day("2024-12-26"),
        24000.0,
        "CE",
    )


@pytest.mark.parametrize(
    "symbol,lot_size",
    [("NIFTY26DEC2424000CE", 25), ("NIFTY02JAN2524000CE", 75), ("NIFTY30JAN25FUT", 25)],
)
def test_lot_size_of_a_contract_follows_its_own_expiry_on_the_same_trade_date(
    tables, symbol, lot_size
):
    assert tables.instrument(symbol, "NFO", "2024-12-02").lot_size == lot_size


def test_provenance_names_the_source_and_status_of_every_table_value(tables):
    option = tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-02")
    assert option.provenance == (
        ("last_trading_day", "test", "user"),
        ("lot_size", "test", "user"),
        ("multiplier", "price is quoted per unit of the underlying", "assumed"),
        ("tick_size", "test", "user"),
        ("freeze_qty", "test", "user"),
        ("settlement", "test", "user"),
    )
    assert option.generic is False


def test_contract_can_be_resolved_on_its_expiry_day(tables):
    assert tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-26").lot_size == 25


def test_expired_contract_has_no_terms(tables):
    with pytest.raises(ConfigError, match="expired on 2024-12-26"):
        tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-27")


def test_malformed_symbol_is_a_symbol_error(tables):
    with pytest.raises(SymbolError):
        tables.instrument("NIFTY26DEC24", "NFO", "2024-12-02")


def test_strike_in_non_canonical_form_is_accepted_only_when_not_strict(tables):
    with pytest.raises(SymbolError):
        tables.instrument("NIFTY26DEC2424000.0CE", "NFO", "2024-12-02")
    option = tables.instrument("NIFTY26DEC2424000.0CE", "NFO", "2024-12-02", strict=False)
    assert (option.symbol, option.strike) == ("NIFTY26DEC2424000.0CE", 24000.0)


# ---- expiry check against the rules -------------------------------------------------------


def test_expiry_in_the_symbol_that_the_rules_list_passes_the_calendar_check(tables):
    option = tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-02", calendar=FakeCalendar())
    assert option.expiry == day("2024-12-26")


def test_expiry_in_the_symbol_that_the_rules_do_not_list_is_a_symbol_error(tables):
    with pytest.raises(SymbolError, match="2024-12-24 is not an expiry date of NIFTY"):
        tables.instrument("NIFTY24DEC2424000CE", "NFO", "2024-12-02", calendar=FakeCalendar())


def test_contract_carries_its_last_trading_day_and_identity(tables):
    option = tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-02")
    assert (option.last_trading_day, option.contract_date) == (day("2024-12-26"), day("2024-12-26"))


def test_contract_whose_expiry_day_is_a_holiday_stops_trading_the_day_before(tmp_path):
    calendar = FakeCalendar(holidays=[day("2024-12-26")])
    tables = build_tables(tmp_path, calendar=calendar)
    with pytest.raises(SymbolError, match="not an expiry"):
        tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-02")
    option = tables.instrument("NIFTY25DEC2424000CE", "NFO", "2024-12-02")
    assert (option.last_trading_day, option.contract_date) == (day("2024-12-25"), day("2024-12-26"))


def test_stock_contract_without_rules_or_a_row_for_it_cannot_be_checked(tmp_path):
    tables = build_tables(tmp_path, only=["lots.csv", "underlyings.csv", "freeze.csv"])
    with pytest.raises(ReferenceDataError, match="cannot be checked"):
        tables.instrument("RELIANCE29MAY25FUT", "NFO", "2025-05-02")


# ---- stock derivatives and missing terms ----------------------------------------------------


def test_term_the_tables_do_not_cover_raises_and_names_the_table(tables):
    with pytest.raises(ReferenceDataError, match="No 'freeze_qty' reference row covers"):
        tables.instrument("RELIANCE29MAY25FUT", "NFO", "2025-05-02")


def test_allowed_missing_term_is_none_and_warns(tables):
    with pytest.warns(ReferenceDataWarning, match="freeze_qty of RELIANCE29MAY25FUT"):
        future = tables.instrument(
            "RELIANCE29MAY25FUT", "NFO", "2025-05-02", allow_missing=("freeze_qty",)
        )
    assert future.freeze_qty is None
    assert ("freeze_qty", "not covered", "missing") in future.provenance


def test_lot_size_can_never_be_allowed_to_be_missing(tables):
    with pytest.raises(ConfigError, match="allow_missing accepts"):
        tables.instrument("RELIANCE29MAY25FUT", "NFO", "2025-05-02", allow_missing=("lot_size",))


@pytest.fixture
def stock_future(tables):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)
        return tables.instrument(
            "RELIANCE29MAY25FUT", "NFO", "2025-05-02", allow_missing=("freeze_qty",)
        )


def test_stock_future_is_physically_settled(stock_future):
    assert (stock_future.lot_size, stock_future.settlement) == (250, "physical")


def test_tick_that_depends_on_a_price_slab_is_none_without_a_reference_price(stock_future):
    assert stock_future.tick_size is None
    assert (
        "tick_size",
        "by price slab; no reference price was given",
        "unresolved",
    ) in stock_future.provenance


def test_reference_price_selects_the_tick_slab(tables):
    with pytest.warns(ReferenceDataWarning):  # the freeze quantity is still missing
        future = tables.instrument(
            "RELIANCE29MAY25FUT",
            "NFO",
            "2025-05-02",
            reference_price=1300.0,
            allow_missing=("freeze_qty",),
        )
    assert future.tick_size == 0.05


def test_round_price_resolves_a_slab_tick_from_the_reference_price(tables, stock_future):
    assert tables.round_price(120.337, stock_future, "2025-05-02", 120.0) == 120.34


def test_round_price_never_takes_the_slab_from_the_price_being_rounded(tables, stock_future):
    with pytest.raises(ConfigError, match="underlying close"):
        tables.round_price(1300.07, stock_future, "2025-05-02")


def test_round_price_uses_the_tick_already_on_the_instrument(tables):
    option = tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-02")
    assert tables.round_price(101.37, option, "2024-12-02") == 101.35


# ---- equity and index -----------------------------------------------------------------------


def test_equity_trades_in_single_shares_with_delivery_settlement(tables):
    share = tables.instrument("RELIANCE", "NSE", "2024-06-03")
    assert (share.kind, share.lot_size, share.freeze_qty, share.settlement) == (
        "equity",
        1,
        None,
        "delivery",
    )
    assert ("lot_size", "cash equity trades in single shares", "assumed") in share.provenance


@pytest.mark.parametrize(
    "on_date,reference_price,tick",
    [
        ("2024-06-03", None, 0.05),  # flat rule
        ("2025-05-02", None, None),  # slab rule and no price
        ("2025-05-02", 1300.0, 0.10),
        ("2025-05-02", 120.0, 0.01),
    ],
)
def test_equity_tick_follows_the_rule_in_force_on_the_date(tables, on_date, reference_price, tick):
    share = tables.instrument("RELIANCE", "NSE", on_date, reference_price=reference_price)
    assert share.tick_size == tick


def test_equity_on_an_exchange_without_a_tick_rule_raises(tables):
    with pytest.raises(ReferenceDataError, match="exchange=BSE, segment=equity"):
        tables.instrument("RELIANCE", "BSE", "2024-06-03")


def test_index_is_not_tradable_and_has_no_tick(tables):
    index = tables.instrument("NIFTY", "NSE_INDEX", "2024-06-03")
    assert (index.kind, index.settlement, index.tick_size, index.is_tradable) == (
        "index",
        "none",
        None,
        False,
    )


# ---- MCX ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol,terms",
    [
        ("GOLD05DEC24FUT", (1, 100.0, 1.0, 10, "physical")),
        ("CRUDEOIL19NOV24FUT", (100, 1.0, 1.0, 10000, "cash")),
    ],
)
def test_mcx_future_takes_its_terms_from_the_contract_table(tables, symbol, terms):
    assert _terms(tables.instrument(symbol, "MCX", "2024-11-04")) == terms


def test_mcx_contract_with_no_freeze_quantity_in_its_row_raises(tables):
    with pytest.raises(ReferenceDataError, match="with a freeze_qty value"):
        tables.instrument("NATURALGAS26NOV24FUT", "MCX", "2024-11-04")


def test_mcx_underlying_not_in_the_contract_table_raises(tables):
    with pytest.raises(ReferenceDataError, match="underlying=SILVER, exchange=MCX"):
        tables.instrument("SILVER05DEC24FUT", "MCX", "2024-11-04")


@pytest.mark.parametrize(
    "symbol,exchange",
    [("CRUDEOIL15NOV246000CE", "MCX"), ("USDINR27NOV24FUT", "CDS"), ("USDINR27NOV2484CE", "BCD")],
)
def test_contract_outside_the_built_in_tables_needs_an_exact_symbol_row(tables, symbol, exchange):
    with pytest.raises(ReferenceDataError) as excinfo:
        tables.instrument(symbol, exchange, "2024-11-04")
    message = str(excinfo.value)
    assert f"No lot_size is known for {symbol} on {exchange}" in message
    assert "a 'symbols' row must state lot_size, multiplier" in message


# ---- exact-symbol rows ----------------------------------------------------------------------


def test_exact_symbol_row_wins_over_the_rows_of_its_underlying(tmp_path):
    tables = _with_symbols(tmp_path, "NIFTY26DEC2424000CE,NFO,2024-12-01,2024-12-26,9999,,,,,\n")
    assert tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-02").lot_size == 9999
    assert tables.lot_size("NIFTY26DEC2424000CE", "NFO", "2024-12-02") == 9999


def test_terms_an_exact_symbol_row_leaves_empty_still_come_from_the_tables(tmp_path):
    tables = _with_symbols(tmp_path, "NIFTY26DEC2424000CE,NFO,2024-12-01,2024-12-26,9999,,,,,\n")
    option = tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-02")
    assert (option.tick_size, option.freeze_qty, option.settlement) == (0.05, 1800, "cash")


def test_exact_symbol_row_applies_only_inside_its_validity_range(tmp_path):
    tables = _with_symbols(tmp_path, "NIFTY26DEC2424000CE,NFO,2024-12-10,2024-12-26,9999,,,,,\n")
    assert tables.instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-09").lot_size == 25


def test_exact_symbol_row_defines_a_stock_contract_the_tables_lack(tmp_path):
    tables = _with_symbols(
        tmp_path, "TATASTEEL28NOV24FUT,NFO,2024-11-01,2024-11-28,5500,0.01,27500,,stock,\n"
    )
    future = tables.instrument("TATASTEEL28NOV24FUT", "NFO", "2024-11-04")
    assert _terms(future) == (5500, 1.0, 0.01, 27500, "physical")


def test_exact_symbol_row_defines_a_currency_contract_in_full(tmp_path):
    tables = _with_symbols(
        tmp_path, "USDINR27NOV24FUT,CDS,2024-11-01,2024-11-27,1000,0.0025,10000,1,currency,cash\n"
    )
    future = tables.instrument("USDINR27NOV24FUT", "CDS", "2024-11-04")
    assert _terms(future) == (1000, 1.0, 0.0025, 10000, "cash")


@pytest.mark.parametrize(
    "row,missing",
    [
        ("SILVER05DEC24FUT,MCX,2024-11-01,2024-12-05,30,1,600,,,physical\n", "multiplier"),
        ("SILVER05DEC24FUT,MCX,2024-11-01,2024-12-05,30,1,600,1,,\n", "settlement"),
    ],
)
def test_exact_symbol_row_on_mcx_must_state_its_multiplier_and_settlement(tmp_path, row, missing):
    tables = _with_symbols(tmp_path, row)
    with pytest.raises(ReferenceDataError, match=f"No {missing} is known for SILVER05DEC24FUT"):
        tables.instrument("SILVER05DEC24FUT", "MCX", "2024-11-04")


def test_exact_symbol_row_on_mcx_replaces_the_contract_table(tmp_path):
    tables = _with_symbols(
        tmp_path, "GOLD05DEC24FUT,MCX,2024-11-01,2024-12-05,100,1,5,1,,physical\n"
    )
    gold = tables.instrument("GOLD05DEC24FUT", "MCX", "2024-11-04")
    # Lot 100 with multiplier 1: the same contract value per rupee as lot 1 with 100.
    assert _terms(gold) == (100, 1.0, 1.0, 5, "physical")


# ---- MCX terms by contract expiry ---------------------------------------------------------


@pytest.mark.parametrize(
    "symbol,lot_size",
    [("NICKEL29AUG25FUT", 1500), ("NICKEL30SEP25FUT", 250)],
)
def test_mcx_terms_follow_the_specification_of_the_contract_expiry(tables, symbol, lot_size):
    # Two specifications trade side by side on this date: the old one to the August
    # 2025 expiry and the revised one from the September 2025 expiry.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)  # the old row has no freeze value
        future = tables.instrument(symbol, "MCX", "2025-08-20", allow_missing=("freeze_qty",))
    assert future.lot_size == lot_size


def test_mcx_lot_of_an_underlying_is_refused_while_two_specifications_trade(tables):
    with pytest.raises(ConfigError, match="lot_size of NICKEL on MCX on 2025-08-20 depends"):
        tables.lot_size("NICKEL", "MCX", "2025-08-20")


def test_mcx_lot_of_an_underlying_is_given_by_the_expiry_when_stated(tables):
    assert tables.lot_size("NICKEL", "MCX", "2025-08-20", expiry="2025-09-30") == 250
    assert tables.multiplier("NICKEL", "MCX", "2025-08-20", expiry="2025-09-30") == 1.0


# ---- D4: the kind of an underlying is data, never a default -------------------------------


def test_underlying_in_no_table_is_not_assumed_to_be_a_stock(tables):
    with pytest.raises(ReferenceDataError, match="underlyings"):
        tables.underlying_type("TATASTEEL", "NFO", "2024-06-03")


def test_index_without_built_in_lots_is_still_an_index(tmp_path):
    # NIFTYFPI is an index (NSE/FAOP/75239); a user supplies only an exact-contract lot.
    path = write_file(
        tmp_path,
        "symbols.csv",
        "symbol,exchange,valid_from,lot_size\nNIFTYFPI29DEC26FUT,NFO,2026-01-01,1100\n",
    )
    custom = ReferenceTables(user_files=[path])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)
        future = custom.instrument(
            "NIFTYFPI29DEC26FUT", "NFO", "2026-11-11", reference_price=1500.0
        )
    assert (future.lot_size, future.settlement, future.tick_size) == (1100, "cash", 0.05)


# ---- D5: a slab tick needs the reference price its circular names --------------------------


def test_stock_option_tick_needs_the_underlying_close_not_the_premium(tables):
    option = Instrument(
        symbol="RELIANCE30DEC251300CE",
        exchange="NFO",
        kind="option",
        lot_size=500,
        expiry=day("2025-12-30"),
        strike=1300.0,
        option_type="CE",
        underlying="RELIANCE",
        settlement="physical",
    )
    with pytest.raises(ConfigError, match="underlying close"):
        tables.round_price(40.03, option, "2025-12-01")
    assert tables.round_price(40.03, option, "2025-12-01", reference_price=1300.0) == 40.05


def test_tick_of_an_etf_is_not_taken_from_the_share_rule(tables):
    etf = tables.instrument("NIFTYBEES", "NSE", "2025-06-02", reference_price=275.0)
    assert etf.tick_size is None
    with pytest.raises(ReferenceDataError, match="ETF"):
        tables.round_price(275.12, etf, "2025-06-02", reference_price=275.0)


def test_tick_of_a_share_of_unknown_kind_is_left_unresolved(tables):
    share = tables.instrument("INFY", "NSE", "2025-06-02", reference_price=1500.0)
    assert share.tick_size is None
    with pytest.raises(ReferenceDataError, match="underlyings"):
        tables.round_price(1500.07, share, "2025-06-02", reference_price=1500.0)


# ---- D6: the expiry in a symbol is checked against the rules -------------------------------


def test_symbol_with_a_date_that_is_no_expiry_is_rejected_without_a_calendar(tables):
    # 27 November 2024 is a Wednesday; NIFTY expired on Thursdays then.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)
        with pytest.raises(SymbolError, match="not an expiry"):
            tables.instrument("NIFTY27NOV2424000CE", "NFO", "2024-11-25")


def test_four_digit_year_typo_is_rejected(tables):
    # Parses as 28 November 2020 (a Saturday) with strike 242400.
    with pytest.raises(SymbolError, match="not an expiry"):
        tables.instrument("NIFTY28NOV20242400CE", "NFO", "2020-11-02", calendar=FakeCalendar())
