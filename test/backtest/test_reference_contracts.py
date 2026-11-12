"""Contracts as they stood on each trade date: listed expiry, last trading day, identity.

Every listed expiry below is copied from an exchange F&O bhavcopy (the file named in
the row): the date in its expiry column for that index on that trade date. A row of
the form (trade date, file, symbol, listed expiry) says the exchange listed a contract
of that index with that expiry on that trade date.
"""

import warnings

import pytest
from test_reference_helpers import RULES, FakeCalendar, day, write_file

from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import ReferenceDataError
from openalgo.backtest.reference import ReferenceDataWarning, ReferenceTables, SymbolError

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Trading holidays used by the fake calendar for years the real calendar has no
# holiday list for yet (NSE holiday circulars FAOP/46625 for 2021, FAOP/59723 for 2024).
HOLIDAYS_2021_2024 = ["2021-10-15", "2021-11-04", "2021-11-05", "2024-01-22", "2024-05-20"]


@pytest.fixture(scope="module")
def tables():
    return ReferenceTables()


@pytest.fixture(scope="module")
def real_calendar():
    """The built-in calendar, `openalgo.backtest.calendar.default_calendar()`."""
    return default_calendar()


@pytest.fixture(scope="module")
def calendar():
    """Weekdays open, the holidays above closed, every year reported as sourced."""
    return FakeCalendar(holidays=[day(text) for text in HOLIDAYS_2021_2024])


def _future(underlying, expiry):
    expiry = day(expiry)
    return f"{underlying}{expiry.day:02d}{MONTHS[expiry.month - 1]}{expiry.year % 100:02d}FUT"


def _resolve(tables, underlying, exchange, expiry, on_date, calendar):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)
        return tables.instrument(
            _future(underlying, expiry),
            exchange,
            on_date,
            calendar=calendar,
            allow_missing=("freeze_qty",),
        )


# (trade date, bhavcopy file, symbol, exchange, listed expiry, last trading day, contract identity)
LISTED = [
    # NSE/FAOP/65336: BANKNIFTY monthly contracts moved from Wednesday to Thursday at
    # the close of 2025-01-01.
    (
        "2024-12-02",
        "BhavCopy_NSE_FO_0_0_0_20241202_F_0000.csv",
        "BANKNIFTY",
        "NFO",
        "2025-01-29",
        "2025-01-29",
        "2025-01-30",
    ),
    (
        "2025-01-01",
        "BhavCopy_NSE_FO_0_0_0_20250101_F_0000.csv",
        "BANKNIFTY",
        "NFO",
        "2025-01-29",
        "2025-01-29",
        "2025-01-30",
    ),
    (
        "2025-01-02",
        "BhavCopy_NSE_FO_0_0_0_20250102_F_0000.csv",
        "BANKNIFTY",
        "NFO",
        "2025-01-30",
        "2025-01-30",
        "2025-01-30",
    ),
    (
        "2025-01-01",
        "BhavCopy_NSE_FO_0_0_0_20250101_F_0000.csv",
        "MIDCPNIFTY",
        "NFO",
        "2025-03-31",
        "2025-03-31",
        "2025-03-27",
    ),
    (
        "2025-01-02",
        "BhavCopy_NSE_FO_0_0_0_20250102_F_0000.csv",
        "MIDCPNIFTY",
        "NFO",
        "2025-03-27",
        "2025-03-27",
        "2025-03-27",
    ),
    (
        "2025-01-01",
        "BhavCopy_NSE_FO_0_0_0_20250101_F_0000.csv",
        "NIFTYNXT50",
        "NFO",
        "2025-01-31",
        "2025-01-31",
        "2025-01-30",
    ),
    # The December 2025 BANKNIFTY quarterly: Wednesday 31, then Thursday 25 (a holiday,
    # so 24), then the last Tuesday (NSE/FAOP/68747, at the close of 2025-07-31).
    (
        "2025-01-01",
        "BhavCopy_NSE_FO_0_0_0_20250101_F_0000.csv",
        "BANKNIFTY",
        "NFO",
        "2025-12-31",
        "2025-12-31",
        "2025-12-30",
    ),
    (
        "2025-01-02",
        "BhavCopy_NSE_FO_0_0_0_20250102_F_0000.csv",
        "BANKNIFTY",
        "NFO",
        "2025-12-24",
        "2025-12-24",
        "2025-12-30",
    ),
    (
        "2025-07-31",
        "BhavCopy_NSE_FO_0_0_0_20250731_F_0000.csv",
        "BANKNIFTY",
        "NFO",
        "2025-12-24",
        "2025-12-24",
        "2025-12-30",
    ),
    (
        "2025-08-01",
        "BhavCopy_NSE_FO_0_0_0_20250801_F_0000.csv",
        "BANKNIFTY",
        "NFO",
        "2025-12-30",
        "2025-12-30",
        "2025-12-30",
    ),
    # NSE/FAOP/68747: long-dated NIFTY contracts realigned to Tuesday at the close of 2025-07-31.
    (
        "2025-07-01",
        "BhavCopy_NSE_FO_0_0_0_20250701_F_0000.csv",
        "NIFTY",
        "NFO",
        "2025-09-25",
        "2025-09-25",
        "2025-09-30",
    ),
    (
        "2025-08-01",
        "BhavCopy_NSE_FO_0_0_0_20250801_F_0000.csv",
        "NIFTY",
        "NFO",
        "2025-09-30",
        "2025-09-30",
        "2025-09-30",
    ),
    (
        "2025-07-31",
        "BhavCopy_NSE_FO_0_0_0_20250731_F_0000.csv",
        "NIFTY",
        "NFO",
        "2027-06-24",
        "2027-06-24",
        "2027-06-29",
    ),
    (
        "2026-11-11",
        "BhavCopy_NSE_FO_0_0_0_20261111_F_0000.csv",
        "NIFTY",
        "NFO",
        "2027-06-29",
        "2027-06-29",
        "2027-06-29",
    ),
    # 31 March 2026 became a holiday after the March contracts were listed.
    (
        "2025-12-26",
        "BhavCopy_NSE_FO_0_0_0_20251226_F_0000.csv",
        "NIFTY",
        "NFO",
        "2026-03-31",
        "2026-03-31",
        "2026-03-31",
    ),
    (
        "2025-12-29",
        "BhavCopy_NSE_FO_0_0_0_20251229_F_0000.csv",
        "NIFTY",
        "NFO",
        "2026-03-30",
        "2026-03-30",
        "2026-03-31",
    ),
    # Contracts listed into 2027, a year the calendar has no holiday list for.
    (
        "2026-11-11",
        "BhavCopy_NSE_FO_0_0_0_20261111_F_0000.csv",
        "NIFTY",
        "NFO",
        "2027-01-25",
        "2027-01-25",
        "2027-01-26",
    ),
    (
        "2026-11-11",
        "BhavCopy_NSE_FO_0_0_0_20261111_F_0000.csv",
        "FINNIFTY",
        "NFO",
        "2027-01-25",
        "2027-01-25",
        "2027-01-26",
    ),
    (
        "2026-11-11",
        "BhavCopy_BSE_FO_0_0_0_20261111_F_0000.CSV",
        "SENSEX",
        "BFO",
        "2027-03-25",
        "2027-03-25",
        "2027-03-25",
    ),
    # NSE/MSD/60340: only the maturity of the 22 January 2024 MIDCPNIFTY contract moved.
    (
        "2024-01-19",
        "BhavCopy_NSE_FO_0_0_0_20240119_F_0000.csv",
        "MIDCPNIFTY",
        "NFO",
        "2024-01-22",
        "2024-01-22",
        "2024-01-22",
    ),
    (
        "2024-01-20",
        "BhavCopy_NSE_FO_0_0_0_20240120_F_0000.csv",
        "MIDCPNIFTY",
        "NFO",
        "2024-01-22",
        "2024-01-20",
        "2024-01-22",
    ),
    # BSE notice 20240119-45: BSE changed the BANKEX expiry date itself.
    (
        "2024-01-19",
        "BhavCopy_BSE_FO_0_0_0_20240119_F_0000.CSV",
        "BANKEX",
        "BFO",
        "2024-01-22",
        "2024-01-22",
        "2024-01-22",
    ),
    (
        "2024-01-20",
        "BhavCopy_BSE_FO_0_0_0_20240120_F_0000.CSV",
        "BANKEX",
        "BFO",
        "2024-01-20",
        "2024-01-20",
        "2024-01-22",
    ),
    # The June 2023 holiday moved from the 28th to the 29th after NIFTY June was listed.
    ("2023-06-27", "fo27JUN2023bhav.csv", "NIFTY", "NFO", "2023-06-29", "2023-06-29", "2023-06-29"),
    ("2023-06-28", "fo28JUN2023bhav.csv", "NIFTY", "NFO", "2023-06-28", "2023-06-28", "2023-06-29"),
    (
        "2023-06-26",
        "fo26JUN2023bhav.csv",
        "MIDCPNIFTY",
        "NFO",
        "2023-06-27",
        "2023-06-27",
        "2023-06-28",
    ),
    # NSE/FAOP/49595: FINNIFTY contracts preponed to the Tuesday of their week at the
    # close of 2021-10-14; the November monthly future became the 23 November contract.
    (
        "2021-10-14",
        "fo14OCT2021bhav.csv",
        "FINNIFTY",
        "NFO",
        "2021-11-25",
        "2021-11-25",
        "2021-11-23",
    ),
    (
        "2021-10-18",
        "fo18OCT2021bhav.csv",
        "FINNIFTY",
        "NFO",
        "2021-11-23",
        "2021-11-23",
        "2021-11-23",
    ),
    (
        "2021-10-14",
        "fo14OCT2021bhav.csv",
        "FINNIFTY",
        "NFO",
        "2021-12-02",
        "2021-12-02",
        "2021-11-30",
    ),
    # NSE/FAOP/57538: MIDCPNIFTY monthly contracts moved to the last Monday of their month.
    (
        "2023-08-10",
        "fo10AUG2023bhav.csv",
        "MIDCPNIFTY",
        "NFO",
        "2023-10-25",
        "2023-10-25",
        "2023-10-30",
    ),
    (
        "2023-09-01",
        "fo01SEP2023bhav.csv",
        "MIDCPNIFTY",
        "NFO",
        "2023-10-30",
        "2023-10-30",
        "2023-10-30",
    ),
    # BSE notice 20241128-70: SENSEX moved from Friday to Tuesday at the close of 2024-12-31.
    (
        "2024-12-31",
        "BhavCopy_BSE_FO_0_0_0_20241231_F_0000.CSV",
        "SENSEX",
        "BFO",
        "2025-01-31",
        "2025-01-31",
        "2025-01-28",
    ),
    (
        "2025-01-01",
        "BhavCopy_BSE_FO_0_0_0_20250101_F_0000.CSV",
        "SENSEX",
        "BFO",
        "2025-01-28",
        "2025-01-28",
        "2025-01-28",
    ),
    # BSE notice 20260112-8: the 15 January 2026 SENSEX contract preponed to the 14th.
    (
        "2026-01-12",
        "BhavCopy_BSE_FO_0_0_0_20260112_F_0000.CSV",
        "SENSEX",
        "BFO",
        "2026-01-15",
        "2026-01-15",
        "2026-01-15",
    ),
    (
        "2026-01-14",
        "BhavCopy_BSE_FO_0_0_0_20260114_F_0000.CSV",
        "SENSEX",
        "BFO",
        "2026-01-14",
        "2026-01-14",
        "2026-01-15",
    ),
]


@pytest.mark.parametrize(
    "on_date,source,underlying,exchange,expiry,last_trading_day,contract", LISTED
)
def test_contract_listed_in_a_bhavcopy_resolves_with_the_dates_in_force_that_day(
    tables, calendar, on_date, source, underlying, exchange, expiry, last_trading_day, contract
):
    future = _resolve(tables, underlying, exchange, expiry, on_date, calendar)
    assert (future.expiry, future.last_trading_day, future.contract_date) == (
        day(expiry),
        day(last_trading_day),
        day(contract),
    )


NOT_LISTED = [
    # NSE moved only the maturity of this contract; no 20 January contract ever existed.
    ("2024-01-20", "MIDCPNIFTY", "NFO", "2024-01-20"),
    # The June 2023 MIDCPNIFTY contract was always listed for the 27th.
    ("2023-06-26", "MIDCPNIFTY", "NFO", "2023-06-28"),
    # NSE/FAOP/39894: NIFTY weekly options were listed from 2019-02-11.
    ("2019-02-08", "NIFTY", "NFO", "2019-02-14"),
    # BSE notice 20230830-8: Monday BANKEX contracts were listed from 2023-10-16.
    ("2023-10-13", "BANKEX", "BFO", "2023-10-16"),
    # First listed (as 10 January) in the BSE bhavcopy of 2024-12-02.
    ("2024-11-25", "SENSEX", "BFO", "2025-01-10"),
]


@pytest.mark.parametrize("on_date,underlying,exchange,expiry", NOT_LISTED)
def test_contract_not_listed_on_the_trade_date_is_a_symbol_error(
    tables, calendar, on_date, underlying, exchange, expiry
):
    with pytest.raises(SymbolError, match="not an expiry"):
        _resolve(tables, underlying, exchange, expiry, on_date, calendar)


# (trade date, symbol given, exchange, symbol in force that day, listed expiry, contract)
ALIASES = [
    # Stored data can hold a re-dated contract under its later symbol for earlier dates.
    ("2024-12-02", "BANKNIFTY30JAN25FUT", "NFO", "BANKNIFTY29JAN25FUT", "2025-01-29", "2025-01-30"),
    ("2025-01-02", "BANKNIFTY29JAN25FUT", "NFO", "BANKNIFTY30JAN25FUT", "2025-01-30", "2025-01-30"),
    ("2025-07-01", "NIFTY30SEP25FUT", "NFO", "NIFTY25SEP25FUT", "2025-09-25", "2025-09-30"),
    ("2025-08-01", "NIFTY25SEP25FUT", "NFO", "NIFTY30SEP25FUT", "2025-09-30", "2025-09-30"),
    ("2024-01-20", "BANKEX22JAN24FUT", "BFO", "BANKEX20JAN24FUT", "2024-01-20", "2024-01-22"),
    # BSE notice 20241128-70: listed for 14 February, re-dated to the 11th on 2024-12-31.
    ("2025-01-01", "SENSEX14FEB25FUT", "BFO", "SENSEX11FEB25FUT", "2025-02-11", "2025-02-11"),
]


@pytest.mark.parametrize("on_date,given,exchange,in_force,expiry,contract", ALIASES)
def test_other_symbol_of_a_re_dated_contract_resolves_to_the_contract_in_force(
    tables, calendar, on_date, given, exchange, in_force, expiry, contract
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)
        future = tables.instrument(
            given, exchange, on_date, calendar=calendar, allow_missing=("freeze_qty",)
        )
    assert (future.symbol, future.expiry, future.contract_date) == (
        in_force,
        day(expiry),
        day(contract),
    )
    assert ("symbol", f"{given} is an alias of {in_force}", "alias") in future.provenance


def test_alias_is_resolved_without_a_warning(calendar):
    fresh = ReferenceTables()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fresh.instrument("BANKNIFTY30JAN25FUT", "NFO", "2024-12-02", calendar=calendar)


def test_alias_that_could_name_two_contracts_raises(tmp_path):
    rules = write_file(tmp_path, "rules.csv", RULES)
    dates = write_file(
        tmp_path,
        "dates.csv",
        "underlying,exchange,contract,valid_from,valid_to,expiry_date,last_trading_day\n"
        "NIFTY,NFO,2024-11-14,2024-10-01,2024-10-15,2024-11-15,2024-11-15\n"
        "NIFTY,NFO,2024-11-21,2024-10-16,2024-10-31,2024-11-15,2024-11-15\n",
    )
    custom = ReferenceTables(user_files=[rules, dates], builtin=False)
    with pytest.raises(ReferenceDataError, match="two NIFTY contracts"):
        custom.instrument("NIFTY15NOV24FUT", "NFO", "2024-11-05", calendar=FakeCalendar())


def test_sensex_contract_listed_for_14_february_2025_resolves_on_its_listing_day(tables, calendar):
    future = _resolve(tables, "SENSEX", "BFO", "2025-02-14", "2024-12-30", calendar)
    assert (future.expiry, future.contract_date) == (day("2025-02-14"), day("2025-02-11"))


# ---- when contracts were listed ----------------------------------------------------------


@pytest.mark.parametrize(
    "underlying,exchange,as_of,start,end,expected",
    [
        # NSE/FAOP/39894: the first weekly contracts (14 and 21 February) from 2019-02-11.
        ("NIFTY", "NFO", "2019-02-08", "2019-02-08", "2019-02-21", []),
        ("NIFTY", "NFO", "2019-02-11", "2019-02-11", "2019-02-21", ["2019-02-14", "2019-02-21"]),
        # BSE notice 20230830-8: Friday contracts end on 2023-10-13; Monday ones from 10-16.
        ("BANKEX", "BFO", "2023-10-13", "2023-10-13", "2023-10-26", ["2023-10-13"]),
        ("BANKEX", "BFO", "2023-10-16", "2023-10-16", "2023-10-29", ["2023-10-16", "2023-10-23"]),
        # BSE notice 20230327-65: weekly and monthly Friday contracts listed from 2023-05-15.
        ("SENSEX", "BFO", "2023-05-15", "2023-05-15", "2023-05-28", ["2023-05-19", "2023-05-26"]),
    ],
)
def test_schedule_as_of_a_trade_date_holds_only_contracts_listed_by_then(
    tables, calendar, underlying, exchange, as_of, start, end, expected
):
    schedule = tables.expiry_schedule(underlying, exchange, start, end, calendar, as_of=as_of)
    assert [item.expiry for item in schedule] == [day(text) for text in expected]


def test_old_and_new_symbol_of_a_re_dated_contract_are_the_same_contract(tables, calendar):
    before = _resolve(tables, "BANKNIFTY", "NFO", "2025-01-29", "2024-12-02", calendar)
    after = _resolve(tables, "BANKNIFTY", "NFO", "2025-01-30", "2025-01-02", calendar)
    assert before.symbol != after.symbol
    assert before.contract_key == after.contract_key


def test_schedule_as_of_a_trade_date_shows_the_dates_then_in_force(tables, calendar):
    then = tables.expiry_schedule(
        "BANKNIFTY", "NFO", "2025-01-25", "2025-01-31", calendar, as_of="2024-12-02"
    )
    final = tables.expiry_schedule("BANKNIFTY", "NFO", "2025-01-25", "2025-01-31", calendar)
    assert [(item.expiry, item.last_trading_day) for item in then] == [
        (day("2025-01-29"), day("2025-01-29"))
    ]
    assert [(item.expiry, item.last_trading_day) for item in final] == [
        (day("2025-01-30"), day("2025-01-30"))
    ]


def test_expiries_are_the_last_trading_days(tables, calendar):
    # The MIDCPNIFTY contract listed as 22 January 2024 stopped trading on Saturday the 20th.
    found = tables.expiries(
        "MIDCPNIFTY", "NFO", "2024-01-16", "2024-01-26", calendar, series="weekly"
    )
    assert found == [day("2024-01-20")]


# ---- expiry rules: one monthly expiry per month (findings 2 and 3) ---------------------------


def test_midcpnifty_april_2023_has_one_monthly_expiry(tables, calendar):
    found = tables.expiries(
        "MIDCPNIFTY", "NFO", "2023-04-01", "2023-04-30", calendar, series="monthly"
    )
    assert found == [day("2023-04-25")]  # NSE/FAOP/56128: Wednesday expiry from May


def test_midcpnifty_june_2023_expired_on_tuesday_the_27th(tables):
    # Bhavcopies of 26, 27 and 28 June 2023; the holiday was finally on the 29th.
    calendar = FakeCalendar(holidays=[day("2023-06-29")])
    found = tables.expiries(
        "MIDCPNIFTY", "NFO", "2023-06-01", "2023-06-30", calendar, series="monthly"
    )
    assert found == [day("2023-06-27")]


def test_niftynxt50_launch_cycle_starts_in_may_2024(tables, calendar):
    # NSE/FAOP/61629 lists May 31, June 28 and July 26 2024 for the launch.
    found = tables.expiries("NIFTYNXT50", "NFO", "2024-04-24", "2024-07-31", calendar)
    assert found == [day("2024-05-31"), day("2024-06-28"), day("2024-07-26")]


@pytest.mark.parametrize(
    "underlying,exchange,listed_from,first_month",
    [
        ("NIFTY", "NFO", "2019-01-01", (2019, 1)),
        ("BANKNIFTY", "NFO", "2019-01-01", (2019, 1)),
        ("FINNIFTY", "NFO", "2021-01-11", (2021, 1)),
        ("MIDCPNIFTY", "NFO", "2022-01-24", (2022, 1)),
        ("NIFTYNXT50", "NFO", "2024-04-24", (2024, 5)),
        ("NIFTYFPI", "NFO", "2026-08-12", (2026, 8)),
        ("SENSEX", "BFO", "2023-05-15", (2023, 5)),
        ("BANKEX", "BFO", "2023-05-15", (2023, 5)),
    ],
)
def test_every_month_has_exactly_one_monthly_expiry(
    tables, underlying, exchange, listed_from, first_month
):
    found = tables.expiries(
        underlying, exchange, listed_from, "2026-12-31", FakeCalendar(), series="monthly"
    )
    months = [(item.year, item.month) for item in found]
    year, month = first_month
    assert len(months) == len(set(months))
    assert months[0] == first_month
    assert len(months) == (2026 - year) * 12 + 12 - month + 1


# ---- D1: never derive an expiry from a template calendar date -----------------------------


def test_expiry_that_needs_a_template_calendar_date_raises(tables):
    calendar = FakeCalendar(template_years=[2023])
    with pytest.raises(ReferenceDataError) as excinfo:
        tables.expiries("NIFTY", "NFO", "2023-06-01", "2023-06-23", calendar)
    message = str(excinfo.value)
    assert "NFO" in message and "2023" in message and "holiday list" in message


def test_explicit_contract_row_is_used_where_the_calendar_is_template(tables):
    calendar = FakeCalendar(template_years=[2023])
    assert tables.expiries("NIFTY", "NFO", "2023-06-28", "2023-06-28", calendar) == [
        day("2023-06-28")
    ]


def test_expiry_from_an_unverified_calendar_date_warns(tables):
    calendar = FakeCalendar(unverified_years=[2024])
    with pytest.warns(ReferenceDataWarning, match="unverified calendar"):
        tables.expiries("NIFTY", "NFO", "2024-07-01", "2024-07-31", calendar)


def test_real_calendar_gives_the_listed_expiries_of_november_2026(tables, real_calendar):
    # Bhavcopy of 2026-11-11: 17 November and 23 November (24 November is a holiday). The
    # contract of the week of 10 November expired on the 9th: the 10th was a holiday
    # (Diwali-Balipratipada, NSE/CMTR/71775).
    found = tables.expiries("NIFTY", "NFO", "2026-11-01", "2026-11-30", real_calendar)
    assert found == [day("2026-11-03"), day("2026-11-09"), day("2026-11-17"), day("2026-11-23")]


@pytest.mark.parametrize(
    "underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYFPI"]
)
def test_real_calendar_january_2027_monthly_expiry_comes_from_the_listed_contract(
    tables, real_calendar, underlying
):
    # Bhavcopy of 2026-11-11 lists 25 January 2027 (26 January is Republic Day).
    found = tables.expiries(
        underlying, "NFO", "2027-01-25", "2027-01-26", real_calendar, series="monthly"
    )
    assert found == [day("2027-01-25")]


def test_real_calendar_weekly_contracts_of_2027_need_the_2027_holiday_list(tables, real_calendar):
    # A weekly contract of early February 2027 falls back into the range only if every
    # day from 27 January is a holiday, which only a 2027 holiday list can say.
    with pytest.raises(ReferenceDataError, match="NFO in 2027"):
        tables.expiries("NIFTY", "NFO", "2027-01-25", "2027-01-26", real_calendar)


def test_real_calendar_2027_expiry_without_a_listed_contract_raises(tables, real_calendar):
    with pytest.raises(ReferenceDataError, match="2027"):
        tables.expiries("NIFTY", "NFO", "2027-02-01", "2027-02-28", real_calendar)


@pytest.mark.parametrize(
    "underlying,start,end,listed",
    [
        # Bhavcopy of 2023-06-28: NIFTY June 2023 expired on Wednesday the 28th.
        ("NIFTY", "2023-06-20", "2023-06-30", ["2023-06-22", "2023-06-28"]),
        # Bhavcopy of 2023-08-10: FINNIFTY expired on Monday 14 August (15th a holiday).
        ("FINNIFTY", "2023-08-10", "2023-08-20", ["2023-08-14"]),
        # NSE/FAOP/39894: NIFTY expired on 20 March 2019 (21st a holiday).
        ("NIFTY", "2019-03-15", "2019-03-25", ["2019-03-20"]),
    ],
)
def test_real_calendar_never_returns_an_expiry_the_bhavcopy_contradicts(
    tables, real_calendar, underlying, start, end, listed
):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ReferenceDataWarning)
            found = tables.expiries(underlying, "NFO", start, end, real_calendar)
    except ReferenceDataError:
        return  # no holiday list for that year yet: refusing is the correct answer
    assert found == [day(text) for text in listed]


def test_instrument_without_a_calendar_uses_the_real_one(tables, real_calendar):
    # 2026 has a sourced holiday list, so no warning and no error.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        option = tables.instrument("NIFTY17NOV2625000CE", "NFO", "2026-11-11")
    assert (option.expiry, option.last_trading_day) == (day("2026-11-17"), day("2026-11-17"))
