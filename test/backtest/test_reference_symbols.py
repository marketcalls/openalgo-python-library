"""OpenAlgo symbol parsing and formatting (FR-INST-01)."""

import datetime as dt
import re
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openalgo.backtest.errors import BacktestError, ConfigError
from openalgo.backtest.reference.symbols import (
    ParsedSymbol,
    SymbolError,
    derivative_exchange,
    format_expiry,
    format_strike,
    format_symbol,
    parse_expiry,
    parse_symbol,
)

MONTH_CODES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# (symbol, exchange, kind, underlying, expiry, strike, option_type), taken from the
# OpenAlgo symbol format documentation.
DOCUMENTED = [
    ("INFY", "NSE", "equity", None, None, None, None),
    ("TATAMOTORS", "BSE", "equity", None, None, None, None),
    ("M&M", "NSE", "equity", None, None, None, None),
    ("BAJAJ-AUTO", "NSE", "equity", None, None, None, None),
    ("NIFTY", "NSE_INDEX", "index", None, None, None, None),
    ("SENSEX", "BSE_INDEX", "index", None, None, None, None),
    ("BSEOIL&GAS", "BSE_INDEX", "index", None, None, None, None),
    ("BANKNIFTY24APR24FUT", "NFO", "future", "BANKNIFTY", dt.date(2024, 4, 24), None, None),
    ("SENSEX24APR24FUT", "BFO", "future", "SENSEX", dt.date(2024, 4, 24), None, None),
    ("USDINR10MAY24FUT", "CDS", "future", "USDINR", dt.date(2024, 5, 10), None, None),
    ("CRUDEOILM20MAY24FUT", "MCX", "future", "CRUDEOILM", dt.date(2024, 5, 20), None, None),
    ("726GS203325APR24FUT", "CDS", "future", "726GS2033", dt.date(2024, 4, 25), None, None),
    ("NIFTY28NOV2424000CE", "NFO", "option", "NIFTY", dt.date(2024, 11, 28), 24000.0, "CE"),
    ("NIFTY28MAR2420800CE", "NFO", "option", "NIFTY", dt.date(2024, 3, 28), 20800.0, "CE"),
    ("VEDL25APR24292.5CE", "NFO", "option", "VEDL", dt.date(2024, 4, 25), 292.5, "CE"),
    ("USDINR19APR2482CE", "CDS", "option", "USDINR", dt.date(2024, 4, 19), 82.0, "CE"),
    ("USDINR19APR2482.25PE", "CDS", "option", "USDINR", dt.date(2024, 4, 19), 82.25, "PE"),
    ("CRUDEOIL17APR246750CE", "MCX", "option", "CRUDEOIL", dt.date(2024, 4, 17), 6750.0, "CE"),
    ("726GS203225APR2497PE", "CDS", "option", "726GS2032", dt.date(2024, 4, 25), 97.0, "PE"),
    ("M&M28NOV243000PE", "NFO", "option", "M&M", dt.date(2024, 11, 28), 3000.0, "PE"),
    ("BAJAJ-AUTO28NOV24FUT", "NFO", "future", "BAJAJ-AUTO", dt.date(2024, 11, 28), None, None),
    ("NIFTYNXT5029NOV24FUT", "NFO", "future", "NIFTYNXT50", dt.date(2024, 11, 29), None, None),
]


@pytest.mark.parametrize("symbol,exchange,kind,underlying,expiry,strike,option_type", DOCUMENTED)
def test_documented_symbol_parses_into_its_parts(
    symbol, exchange, kind, underlying, expiry, strike, option_type
):
    expected = ParsedSymbol(symbol, exchange, kind, underlying, expiry, strike, option_type)
    assert parse_symbol(symbol, exchange) == expected


@pytest.mark.parametrize(
    "symbol,underlying,expiry,strike,option_type",
    [(row[0], row[3], row[4], row[5], row[6]) for row in DOCUMENTED if row[3] is not None],
)
def test_documented_derivative_symbol_is_rebuilt_exactly_from_its_parts(
    symbol, underlying, expiry, strike, option_type
):
    assert format_symbol(underlying, expiry, strike, option_type) == symbol


@pytest.mark.parametrize("symbol,exchange", [(row[0], row[1]) for row in DOCUMENTED])
def test_parsed_symbol_reports_its_canonical_spelling(symbol, exchange):
    assert parse_symbol(symbol, exchange).canonical == symbol


# ---- round trips over generated symbols --------------------------------------------------

_DATE_LIKE = re.compile(r"\d{2}(" + "|".join(MONTH_CODES) + ")")
_bases = st.from_regex(r"[A-Z][A-Z0-9&\-]{0,11}", fullmatch=True).filter(
    lambda text: not _DATE_LIKE.search(text)
)
_expiries = st.dates(min_value=dt.date(2000, 1, 1), max_value=dt.date(2099, 12, 31))
# A strike as (the text a symbol carries, its value): whole numbers and decimals with
# up to four places, never with a trailing zero.
_whole_strikes = st.integers(min_value=1, max_value=250000).map(lambda n: (str(n), float(n)))
_decimal_strikes = st.tuples(
    st.integers(min_value=0, max_value=99999),
    st.from_regex(r"[0-9]{0,3}[1-9]", fullmatch=True),
).map(lambda parts: (f"{parts[0]}.{parts[1]}", float(Decimal(f"{parts[0]}.{parts[1]}"))))
_strikes = st.one_of(_whole_strikes, _decimal_strikes)


def _expiry_code(expiry):
    return f"{expiry.day:02d}{MONTH_CODES[expiry.month - 1]}{expiry.year % 100:02d}"


@settings(max_examples=300, deadline=None)
@given(base=_bases, expiry=_expiries)
def test_any_future_symbol_round_trips(base, expiry):
    symbol = f"{base}{_expiry_code(expiry)}FUT"
    parsed = parse_symbol(symbol, "NFO")
    assert (parsed.kind, parsed.underlying, parsed.expiry) == ("future", base, expiry)
    assert format_symbol(parsed.underlying, parsed.expiry) == symbol


@settings(max_examples=300, deadline=None)
@given(base=_bases, expiry=_expiries, strike=_strikes, option_type=st.sampled_from(["CE", "PE"]))
def test_any_option_symbol_round_trips(base, expiry, strike, option_type):
    strike_text, strike_value = strike
    symbol = f"{base}{_expiry_code(expiry)}{strike_text}{option_type}"
    parsed = parse_symbol(symbol, "NFO")
    assert (parsed.kind, parsed.underlying, parsed.expiry) == ("option", base, expiry)
    assert (parsed.strike, parsed.option_type) == (strike_value, option_type)
    assert format_symbol(parsed.underlying, parsed.expiry, parsed.strike, option_type) == symbol


# ---- malformed input --------------------------------------------------------------------

MALFORMED = [
    ("NIFTY", "NFO", "expected [Base][DDMMMYY]FUT"),
    ("NIFTY28NOV24", "NFO", "expected [Base][DDMMMYY]FUT"),
    ("NIFTY28NOV2424000", "NFO", "expected [Base][DDMMMYY]FUT"),
    ("NIFTY28NOV2424000XX", "NFO", "expected [Base][DDMMMYY]FUT"),
    ("NIFTY28NOV24CE", "NFO", "expected [Base][DDMMMYY]FUT"),
    ("NIFTY28NOVEMBER24FUT", "NFO", "expected [Base][DDMMMYY]FUT"),
    ("NIFTY2024-11-28FUT", "NFO", "expected [Base][DDMMMYY]FUT"),
    ("28NOV24FUT", "NFO", "expected [Base][DDMMMYY]FUT"),
    ("NIFTY31NOV2424000CE", "NFO", "not a real calendar date"),
    ("NIFTY30FEB24FUT", "NFO", "not a real calendar date"),
    ("NIFTY00JAN24FUT", "NFO", "not a real calendar date"),
    ("NIFTY28NOV24024000CE", "NFO", "no padding and no trailing zeros"),
    ("VEDL25APR24292.50CE", "NFO", "no padding and no trailing zeros"),
    ("VEDL25APR24292.0CE", "NFO", "no padding and no trailing zeros"),
    ("NIFTY28NOV240CE", "NFO", "positive"),
    ("nifty28NOV2424000CE", "NFO", "upper case"),
    ("NIFTY28nov2424000CE", "NFO", "upper case"),
    ("NIFTY 28NOV24FUT", "NFO", "spaces"),
    ("INFY ", "NSE", "spaces"),
    ("infy", "NSE", "upper case"),
]


@pytest.mark.parametrize("symbol,exchange,fragment", MALFORMED)
def test_malformed_symbol_raises_an_error_naming_the_symbol_and_the_fault(
    symbol, exchange, fragment
):
    with pytest.raises(SymbolError) as excinfo:
        parse_symbol(symbol, exchange)
    assert fragment in str(excinfo.value)
    assert symbol in str(excinfo.value)


@pytest.mark.parametrize(
    "symbol,exchange", [("", "NFO"), ("", "NSE"), (None, "NSE"), (12345, "NFO")]
)
def test_empty_or_non_string_symbol_is_rejected(symbol, exchange):
    with pytest.raises(SymbolError, match="non-empty string"):
        parse_symbol(symbol, exchange)


def test_symbol_error_can_be_caught_as_a_config_error_or_a_value_error():
    assert issubclass(SymbolError, ConfigError)
    assert issubclass(SymbolError, BacktestError)
    assert issubclass(SymbolError, ValueError)


@pytest.mark.parametrize(
    "exchange,fragment",
    [
        ("NYSE", "unknown exchange"),
        ("nfo", "unknown exchange"),
        ("", "exchange must be a string"),
        ("NCO", "not supported"),
        ("CRYPTO", "not supported"),
    ],
)
def test_unknown_or_unsupported_exchange_is_rejected(exchange, fragment):
    with pytest.raises(SymbolError) as excinfo:
        parse_symbol("BTCUSDFUT", exchange)
    assert fragment in str(excinfo.value)


def test_lenient_parsing_keeps_the_given_string_and_reports_the_canonical_one():
    parsed = parse_symbol("VEDL25APR24292.50CE", "NFO", strict=False)
    assert parsed.symbol == "VEDL25APR24292.50CE"
    assert parsed.strike == 292.5
    assert parsed.canonical == "VEDL25APR24292.5CE"


@pytest.mark.parametrize(
    "symbol,exchange,kind", [("RELIANCE", "NSE", "equity"), ("NIFTY", "NSE_INDEX", "index")]
)
def test_the_exchange_decides_the_family(symbol, exchange, kind):
    parsed = parse_symbol(symbol, exchange)
    assert (parsed.kind, parsed.underlying, parsed.expiry) == (kind, None, None)


# ---- expiry codes, strikes and exchanges ----------------------------------------------------


@pytest.mark.parametrize(
    "expiry,code",
    [
        (dt.date(2024, 11, 28), "28NOV24"),
        (dt.date(2025, 1, 2), "02JAN25"),
        (dt.date(2030, 12, 31), "31DEC30"),
    ],
)
def test_expiry_is_written_as_a_seven_character_code(expiry, code):
    assert format_expiry(expiry) == code


@pytest.mark.parametrize("code", ["28NOV24", "28-NOV-24", "28-nov-24", " 28NOV24 "])
def test_expiry_code_is_read_in_symbol_and_symbol_master_forms(code):
    assert parse_expiry(code) == dt.date(2024, 11, 28)


@pytest.mark.parametrize("code", ["28NOV2024", "2024-11-28", "31NOV24", "", "NOV24", None])
def test_bad_expiry_code_is_rejected(code):
    with pytest.raises(SymbolError):
        parse_expiry(code)


def test_expiry_year_outside_two_digit_range_is_rejected():
    with pytest.raises(SymbolError, match="two digits"):
        format_expiry(dt.date(1999, 12, 30))


@pytest.mark.parametrize(
    "strike,text",
    [
        (24000, "24000"),
        (24000.0, "24000"),
        (292.5, "292.5"),
        ("292.50", "292.5"),
        (Decimal("82.2500"), "82.25"),
        (0.0025, "0.0025"),
        (100000, "100000"),
    ],
)
def test_strike_is_written_without_padding_or_trailing_zeros(strike, text):
    assert format_strike(strike) == text


@pytest.mark.parametrize("strike", [0, -5, float("nan"), float("inf"), "abc", True])
def test_strike_that_is_not_a_positive_number_is_rejected(strike):
    with pytest.raises(SymbolError):
        format_strike(strike)


@pytest.mark.parametrize(
    "parts",
    [
        ("NIFTY", None, 24000, "CE"),
        ("NIFTY", dt.date(2024, 11, 28), 24000, None),
        ("NIFTY", dt.date(2024, 11, 28), None, "CE"),
        ("NIFTY", dt.date(2024, 11, 28), 24000, "XX"),
        ("", dt.date(2024, 11, 28), None, None),
    ],
)
def test_format_symbol_rejects_inconsistent_parts(parts):
    with pytest.raises(SymbolError):
        format_symbol(*parts)


def test_format_symbol_with_only_a_base_is_the_equity_or_index_symbol():
    assert format_symbol("NIFTY") == "NIFTY"


@pytest.mark.parametrize(
    "exchange,derivatives_exchange",
    [
        ("NSE_INDEX", "NFO"),
        ("NSE", "NFO"),
        ("BSE_INDEX", "BFO"),
        ("BSE", "BFO"),
        ("NFO", "NFO"),
        ("MCX", "MCX"),
    ],
)
def test_underlying_exchange_maps_to_its_derivatives_exchange(exchange, derivatives_exchange):
    assert derivative_exchange(exchange) == derivatives_exchange


# ---- D6: a derivative symbol is never a share -------------------------------------------


@pytest.mark.parametrize(
    "symbol,exchange",
    [
        ("NIFTY28NOV2424000CE", "NSE"),
        ("BANKNIFTY24APR24FUT", "BSE"),
        ("NIFTY28NOV24FUT", "NSE_INDEX"),
    ],
)
def test_derivative_shaped_symbol_on_a_cash_or_index_exchange_is_rejected(symbol, exchange):
    with pytest.raises(SymbolError, match="derivative"):
        parse_symbol(symbol, exchange)
