"""The market checks of a live session's preflight (08-live-trading section 4): each
symbol found with its lot size, tick size and freeze quantity, the MIS product, an
interval the broker offers, today a trading day whose session has not closed, and the
clock check that could not be measured.

The trading days come from the package's calendar: 2026-11-16 is a Monday on which NSE
and NFO trade from 09:15 to 15:30 IST, 2026-11-14 a Saturday and 2026-11-24 a holiday.
"""

from __future__ import annotations

import pytest
from preflight_fakes import (
    NOW_NS,
    FakeOpenAlgo,
    ist_ns,
    make_plan,
    sdk_http_error,
    symbol_found,
    symbol_missing,
)

from openalgo.backtest._live.preflight import (
    WARNING,
    InstrumentFacts,
    PreflightReport,
    run_preflight,
)
from openalgo.backtest.calendar import Calendar
from openalgo.backtest.errors import PreflightError

NIFTY_FUTURE = ("NIFTY30DEC25FUT", "NFO")


def preflight(client=None, *, at_ns=NOW_NS, calendar=None, **changes) -> PreflightReport:
    client = FakeOpenAlgo() if client is None else client
    return run_preflight(client, make_plan(**changes), now=lambda: at_ns, calendar=calendar)


def failures_of(client=None, **options) -> tuple[str, ...]:
    with pytest.raises(PreflightError) as raised:
        preflight(client, **options)
    return raised.value.failures


def check_named(report: PreflightReport, name: str):
    (check,) = [check for check in report.checks if check.name == name]
    return check


def with_symbols(*answers) -> FakeOpenAlgo:
    """A client that knows the instruments of `answers`, symbol_found answers."""
    return FakeOpenAlgo(
        symbols={
            (answer["data"]["symbol"], answer["data"]["exchange"]): answer for answer in answers
        }
    )


# ----- symbols -----------------------------------------------------------------------


def test_each_instrument_is_reported_with_its_lot_tick_and_freeze_quantity_in_order():
    client = with_symbols(
        symbol_found("SBIN", "NSE"),
        symbol_found(*NIFTY_FUTURE, lotsize=75, tick_size=0.1, freeze_qty=1800),
    )

    report = preflight(client, instruments=[NIFTY_FUTURE, ("SBIN", "NSE")])

    assert report.instruments == (
        InstrumentFacts("NIFTY30DEC25FUT", "NFO", 75, 0.1, 1800),
        InstrumentFacts("SBIN", "NSE", 1, 0.05, None),
    )


def test_facts_given_as_text_are_read_as_numbers():
    client = with_symbols(
        symbol_found("SBIN", "NSE", lotsize="1", tick_size="0.05", freeze_qty="0")
    )

    report = preflight(client)

    assert report.instruments == (InstrumentFacts("SBIN", "NSE", 1, 0.05, None),)


def test_an_answer_without_a_freeze_quantity_means_no_freeze_limit_is_known():
    answer = symbol_found("SBIN", "NSE")
    del answer["data"]["freeze_qty"]

    report = preflight(with_symbols(answer))

    assert report.instruments[0].freeze_quantity is None


def test_the_symbol_check_says_what_it_found():
    client = with_symbols(symbol_found(*NIFTY_FUTURE, lotsize=75, tick_size=0.1, freeze_qty=1800))

    report = preflight(client, instruments=[NIFTY_FUTURE])

    assert check_named(report, "symbol:NFO:NIFTY30DEC25FUT").message == (
        "NFO:NIFTY30DEC25FUT: lot size 75 units, tick size 0.1 rupees, freeze quantity 1800 units"
    )


def test_a_symbol_openalgo_does_not_know_fails_with_the_servers_message():
    client = FakeOpenAlgo(symbols={("SBIN", "NSE"): symbol_missing("SBIN", "NSE")})

    (failure,) = failures_of(client)

    assert failure.startswith(
        "NSE:SBIN was not found (HTTP 404: Symbol SBIN not found in exchange NSE): write the "
        "symbol as OpenAlgo does"
    )


def test_every_unknown_symbol_is_reported():
    failures = failures_of(instruments=[("SBIN", "NSE"), ("TYPO", "NSE"), ("OTHER", "BSE")])

    assert [failure.split(" ")[0] for failure in failures] == ["NSE:TYPO", "BSE:OTHER"]


@pytest.mark.parametrize(
    ("facts", "problem"),
    [
        ({"lotsize": 0}, "no usable lot size (0)"),
        ({"lotsize": 1.5}, "no usable lot size (1.5)"),
        ({"lotsize": None}, "no usable lot size (None)"),
        ({"lotsize": "one"}, "no usable lot size ('one')"),
        ({"tick_size": 0}, "no usable tick size (0)"),
        ({"tick_size": -0.05}, "no usable tick size (-0.05)"),
        ({"tick_size": "nan"}, "no usable tick size ('nan')"),
        ({"freeze_qty": -1}, "no usable freeze quantity (-1)"),
        ({"freeze_qty": 2.5}, "no usable freeze quantity (2.5)"),
    ],
)
def test_facts_that_cannot_be_used_fail_the_symbol_check(facts, problem):
    client = with_symbols(symbol_found("SBIN", "NSE", **facts))

    (failure,) = failures_of(client)

    assert failure == (
        f"OpenAlgo gives NSE:SBIN {problem}: download the broker's master contract again in "
        "OpenAlgo"
    )


# ----- the product -------------------------------------------------------------------


@pytest.mark.parametrize("product", ["CNC", "NRML", "mis", ""])
def test_a_product_other_than_mis_fails(product):
    (failure,) = failures_of(product=product)

    assert failure.startswith(f"product {product!r} cannot trade live in this release")
    assert failure.endswith("set the strategy's product to 'MIS'")


# ----- the interval ------------------------------------------------------------------


@pytest.mark.parametrize("interval", ["1m", "15m", "D"])
def test_an_interval_the_broker_offers_passes(interval):
    report = preflight(interval=interval)

    assert check_named(report, "interval").message == f"the broker offers {interval} candles"


def test_an_interval_the_broker_does_not_offer_fails_and_lists_the_ones_it_offers():
    (failure,) = failures_of(interval="2m")

    assert failure == (
        "the broker offers no 2m candles (it offers 1m, 3m, 5m, 15m, 30m, 60m, D): choose "
        "one of those as the strategy's interval"
    )


@pytest.mark.parametrize(
    ("interval", "advice"),
    [
        ("5min", "unknown interval '5min': use forms such as '1m', '5m'"),
        ("", "interval must be a string such as '5m' or 'D', got ''"),
        (5, "interval must be a string such as '5m' or 'D', got 5"),
    ],
)
def test_an_interval_written_wrong_fails_with_how_to_write_it(interval, advice):
    (failure,) = failures_of(interval=interval)

    assert failure.startswith(advice)


def test_intervals_that_fail_are_reported_with_the_servers_message():
    missing = sdk_http_error(
        404, {"status": "error", "message": "Broker-specific module not found"}
    )

    (failure,) = failures_of(FakeOpenAlgo(intervals=missing))

    assert failure.startswith(
        "OpenAlgo did not list the broker's intervals (HTTP 404: Broker-specific module not found)"
    )


# ----- the trading day ---------------------------------------------------------------


def test_a_trading_day_passes_with_its_session_times():
    report = preflight()

    assert check_named(report, "trading_day:NSE").message == (
        "NSE trades today, 2026-11-16, from 09:15 to 15:30 IST"
    )


def test_before_the_open_the_trading_day_passes():
    report = preflight(at_ns=ist_ns(2026, 11, 16, 8, 0))

    assert check_named(report, "trading_day:NSE").ok


@pytest.mark.parametrize(
    ("at_ns", "reason"),
    [
        (ist_ns(2026, 11, 24, 10, 0), "NSE does not trade today, 2026-11-24 (IST)"),
        (ist_ns(2026, 11, 14, 10, 0), "NSE does not trade today, 2026-11-14 (IST)"),
        (ist_ns(2026, 11, 16, 15, 30), "the NSE session of 2026-11-16 closed at 15:30 IST"),
        (ist_ns(2026, 11, 16, 23, 0), "the NSE session of 2026-11-16 closed at 15:30 IST"),
    ],
)
def test_a_day_without_a_session_still_to_come_fails(at_ns, reason):
    (failure,) = failures_of(at_ns=at_ns)

    assert failure.startswith(reason)


def test_one_second_before_the_close_the_trading_day_passes():
    report = preflight(at_ns=ist_ns(2026, 11, 16, 15, 29, 59))

    assert check_named(report, "trading_day:NSE").ok


def test_each_exchange_of_the_instruments_is_checked_once():
    client = with_symbols(
        symbol_found("SBIN", "NSE"), symbol_found("INFY", "NSE"), symbol_found(*NIFTY_FUTURE)
    )

    report = preflight(client, instruments=[("SBIN", "NSE"), NIFTY_FUTURE, ("INFY", "NSE")])

    names = [check.name for check in report.checks if check.name.startswith("trading_day")]
    assert names == ["trading_day:NSE", "trading_day:NFO"]


def test_an_exchange_the_calendar_does_not_know_fails_its_trading_day():
    client = with_symbols(symbol_found("SBIN", "XNSE"))

    (failure,) = failures_of(client, instruments=[("SBIN", "XNSE")])

    assert failure.startswith("the trading day of XNSE cannot be told: ")


def test_a_year_the_calendar_holds_no_holiday_list_for_passes_with_a_warning():
    report = preflight(calendar=Calendar(builtin=False))

    check = check_named(report, "trading_day:NSE")
    assert check.ok
    assert check.message.startswith(f"{WARNING}NSE trades today, 2026-11-16")
    assert check.message.endswith(
        "but the calendar holds no holiday list for 2026: check that today is not a holiday of NSE"
    )


# ----- the clock ---------------------------------------------------------------------


def test_the_clock_check_passes_with_a_warning_that_the_skew_could_not_be_measured():
    check = check_named(preflight(), "clock")

    assert check.ok
    assert check.message.startswith(f"{WARNING}the local clock could not be compared")
    assert "within 2 seconds" in check.message
