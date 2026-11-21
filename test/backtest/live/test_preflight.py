"""The preflight of a live session (08-live-trading section 4): a passing plan and its
report, the broker's session, the supported brokers, the mode funds is answered in, the
funds against the capital, the risk limits, the alert channel, every failure reported
together, the plan itself and the countdown before the first order.

The market checks (symbols, product, interval, trading day, clock) are in
test_preflight_market.py, the account lock in test_preflight_lock.py.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from preflight_fakes import (
    EXPIRED_FUNDS,
    KEY_REFUSED,
    NOW_NS,
    ROOT_URL,
    UNREACHABLE,
    FakeAlerts,
    FakeOpenAlgo,
    live_funds,
    make_plan,
    pong,
    quote,
    sandbox_funds,
    sdk_http_error,
    symbol_found,
    symbol_missing,
)

from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.orders import UNSUPPORTED_BROKERS
from openalgo.backtest._live.preflight import (
    COUNTDOWN_SECONDS,
    WARNING,
    InstrumentFacts,
    PreflightPlan,
    PreflightReport,
    countdown,
    run_preflight,
)
from openalgo.backtest.errors import ConfigError, LiveError, PreflightError


def preflight(client: FakeOpenAlgo | None = None, **changes) -> PreflightReport:
    """The preflight of `make_plan(**changes)` against `client` at NOW_NS."""
    client = FakeOpenAlgo() if client is None else client
    return run_preflight(client, make_plan(**changes), now=lambda: NOW_NS)


def failures_of(client: FakeOpenAlgo | None = None, **changes) -> tuple[str, ...]:
    """The failures of a preflight expected to fail."""
    with pytest.raises(PreflightError) as raised:
        preflight(client, **changes)
    return raised.value.failures


def check_named(report: PreflightReport, name: str):
    (check,) = [check for check in report.checks if check.name == name]
    return check


# ----- a passing plan --------------------------------------------------------------


def test_a_plan_that_passes_every_check_reports_the_broker_the_funds_and_the_facts():
    report = preflight()

    assert report.broker == "zerodha"
    assert report.funds_available == 250_000.0
    assert report.instruments == (InstrumentFacts("SBIN", "NSE", 1, 0.05, None),)


def test_a_passed_preflight_reports_every_check_in_the_order_it_ran():
    report = preflight()

    assert [check.name for check in report.checks] == [
        "broker_session",
        "broker_supported",
        "mode",
        "funds",
        "symbol:NSE:SBIN",
        "broker_quotes",
        "interval",
        "capital",
        "max_daily_loss",
        "account_max_daily_loss",
        "product",
        "trading_day:NSE",
        "holidays",
        "clock",
        "alerts",
    ]
    assert all(check.ok for check in report.checks)


def test_the_preflight_asks_for_ping_funds_each_symbol_one_quote_the_intervals_holidays_time():
    client = FakeOpenAlgo(
        symbols={
            ("SBIN", "NSE"): symbol_found("SBIN", "NSE"),
            ("INFY", "NSE"): symbol_found("INFY", "NSE"),
        }
    )

    preflight(client, instruments=[("SBIN", "NSE"), ("INFY", "NSE")])

    assert client.calls == [
        ("ping",),
        ("funds",),
        ("symbol", "SBIN", "NSE"),
        ("symbol", "INFY", "NSE"),
        ("quotes", "SBIN", "NSE"),
        ("intervals",),
        ("holidays", 2026),
        ("HEAD", ROOT_URL),
    ]


def test_the_report_lists_the_warnings_of_the_checks_that_passed_with_one():
    client = FakeOpenAlgo(funds=sandbox_funds(), head_headers={})

    report = preflight(client, mode=Mode.SANDBOX, alerts=None)

    clock, alerts = check_named(report, "clock"), check_named(report, "alerts")
    assert report.warnings == (clock.message, alerts.message)
    assert all(message.startswith(WARNING) for message in report.warnings)


def test_a_preflight_without_warnings_lists_none():
    assert preflight().warnings == ()


def test_a_report_is_frozen():
    report = preflight()

    with pytest.raises(FrozenInstanceError):
        report.broker = "another"  # type: ignore[misc]


# ----- the broker's session ----------------------------------------------------------


def test_a_ping_that_cannot_reach_openalgo_fails_the_broker_session_and_asks_nothing_more():
    client = FakeOpenAlgo(ping=UNREACHABLE)
    alerts = FakeAlerts()

    (failure,) = failures_of(client, alerts=alerts)

    assert failure.startswith("OpenAlgo's ping failed (Failed to connect to the server.")
    assert "the test alert wait for ping" in failure
    assert client.calls == [("ping",)]
    assert alerts.tests == 0


def test_a_refused_api_key_is_reported_with_the_servers_own_message():
    (failure,) = failures_of(FakeOpenAlgo(ping=KEY_REFUSED))

    assert "(HTTP 403: Invalid openalgo apikey)" in failure
    assert '"status"' not in failure


def test_a_ping_that_names_no_broker_fails_the_broker_session():
    ping = {"status": "success", "data": {"message": "pong"}}

    (failure,) = failures_of(FakeOpenAlgo(ping=ping))

    assert "answered without naming the broker" in failure


def test_funds_answered_without_data_fail_the_broker_session_as_an_expired_login():
    (failure,) = failures_of(FakeOpenAlgo(funds=EXPIRED_FUNDS))

    assert failure.startswith("OpenAlgo is logged in to zerodha, but the funds call failed")
    assert "login may have expired" in failure


def test_funds_that_fail_are_reported_with_the_servers_message():
    busy = sdk_http_error(500, {"status": "error", "message": "Broker busy, try again"})

    (failure,) = failures_of(FakeOpenAlgo(funds=busy))

    assert "(HTTP 500: Broker busy, try again)" in failure


def test_an_answer_that_is_not_openalgos_own_is_quoted_short():
    page = {
        "status": "error",
        "message": "HTTP 502: <html>" + "x" * 1000 + "</html>",
        "code": 502,
        "error_type": "http_error",
    }

    (failure,) = failures_of(FakeOpenAlgo(funds=page))

    assert "(HTTP 502: <html>xxx" in failure
    assert "</html>" not in failure
    assert len(failure) < 600


def test_a_broker_the_order_path_cannot_serve_is_refused_with_its_reason():
    (failure,) = failures_of(FakeOpenAlgo(ping=pong("fivepaisa")))

    assert failure.startswith("bt.live cannot trade through fivepaisa: ")
    assert UNSUPPORTED_BROKERS["fivepaisa"] in failure
    assert "ExchOrderID" in failure


def test_an_unsupported_broker_is_refused_whatever_the_case_of_its_name():
    (failure,) = failures_of(FakeOpenAlgo(ping=pong("FivePaisa")))

    assert failure.startswith("bt.live cannot trade through FivePaisa: ")


@pytest.mark.parametrize("broker", ["pocketful", "mstock", "MStock"])
def test_a_broker_whose_trade_book_order_ids_are_unchecked_passes_with_a_warning(broker):
    # Design 16 3.1 step 7: the session matches the trade book's trades to its orders by
    # order ID, which has not been checked for these brokers (merge notes, order path
    # round 5). The warning names the broker, says so, and says what to watch.
    report = preflight(FakeOpenAlgo(ping=pong(broker)))

    check = check_named(report, "broker_supported")
    assert check.ok
    assert report.warnings == (check.message,)
    assert check.message.startswith(f"{WARNING}bt.live trades through {broker}, ")
    for words in ("trade book", "by order ID", "not been checked for", "RECONCILE"):
        assert words in check.message


def test_a_broker_whose_trade_book_order_ids_are_checked_passes_with_no_warning():
    report = preflight(FakeOpenAlgo(ping=pong("upstox")))

    check = check_named(report, "broker_supported")
    assert (check.ok, check.message) == (True, "bt.live can trade through upstox")
    assert report.warnings == ()


# ----- the mode ----------------------------------------------------------------------


def test_funds_answered_in_sandbox_mode_fail_a_live_session_as_a_switch_change():
    (failure,) = failures_of(FakeOpenAlgo(funds=sandbox_funds()), mode=Mode.LIVE)

    assert failure.startswith(
        "OpenAlgo answered funds in SANDBOX mode, but the session runs in LIVE mode"
    )


def test_funds_answered_in_live_mode_fail_a_sandbox_session_as_a_switch_change():
    (failure,) = failures_of(FakeOpenAlgo(funds=live_funds()), mode=Mode.SANDBOX)

    assert failure.startswith(
        "OpenAlgo answered funds in LIVE mode, but the session runs in SANDBOX mode"
    )


def test_a_sandbox_session_passes_on_the_sandboxs_funds():
    report = preflight(FakeOpenAlgo(funds=sandbox_funds(5_000_000.0)), mode=Mode.SANDBOX)

    assert report.funds_available == 5_000_000.0
    assert check_named(report, "mode").ok


# ----- the funds against the capital -------------------------------------------------


def test_funds_below_the_capital_fail_with_both_amounts():
    (failure,) = failures_of(FakeOpenAlgo(funds=live_funds("99999.50")), capital=100_000)

    assert failure == (
        "OpenAlgo reports 99,999.50 rupees of available cash (collateral not counted), less "
        "than the capital of 100,000.00 rupees: lower capital, or add funds to the broker "
        "account"
    )


def test_sandbox_funds_below_the_capital_point_at_the_sandboxs_funds():
    client = FakeOpenAlgo(funds=sandbox_funds(50_000.0))

    (failure,) = failures_of(client, mode=Mode.SANDBOX, capital=100_000)

    assert failure.endswith("add funds to the sandbox's funds in OpenAlgo")


def test_funds_equal_to_the_capital_cover_it():
    report = preflight(FakeOpenAlgo(funds=live_funds("100000.00")), capital=100_000)

    assert check_named(report, "funds").ok


def test_without_a_usable_capital_the_funds_are_not_compared():
    with pytest.raises(PreflightError) as raised:
        preflight(capital=None)

    assert len(raised.value.failures) == 1
    assert raised.value.failures[0].startswith("capital is required")


def test_no_available_cash_at_all_also_points_at_an_expired_broker_login():
    (failure,) = failures_of(FakeOpenAlgo(funds=live_funds("0.00")))

    assert failure.startswith("OpenAlgo reports 0.00 rupees of available cash")
    assert failure.endswith(
        "No cash at all is also what some brokers report for an expired login: if the "
        "account holds funds, log in to OpenAlgo again"
    )


def test_funds_short_of_the_capital_but_above_zero_do_not_point_at_the_login():
    (failure,) = failures_of(FakeOpenAlgo(funds=live_funds("50.00")))

    assert "expired login" not in failure


# ----- the broker's login, by a quote ------------------------------------------------


def test_a_quote_at_a_positive_price_shows_the_brokers_login_works():
    report = preflight()

    assert check_named(report, "broker_quotes").message == (
        "the broker quoted NSE:SBIN at 812.5 rupees"
    )


def test_a_sandbox_session_whose_broker_gives_no_quote_fails_as_an_expired_login():
    failed = sdk_http_error(500, {"status": "error", "message": "Invalid session token"})
    client = FakeOpenAlgo(funds=sandbox_funds(), quotes=failed)

    (failure,) = failures_of(client, mode=Mode.SANDBOX)

    assert failure == (
        "the broker gave no quote for NSE:SBIN (HTTP 500: Invalid session token): its login "
        "may have expired (it ends around 03:00 IST each day), which funds do not show in "
        "SANDBOX mode. Log in to OpenAlgo again, then start the session again"
    )


@pytest.mark.parametrize("ltp", [0, 0.0, "0", None, -1, "nan"])
def test_a_quote_without_a_positive_price_fails_the_broker_login(ltp):
    (failure,) = failures_of(FakeOpenAlgo(quotes=quote(ltp=ltp)))

    assert failure.startswith(
        "the broker gave no quote for NSE:SBIN (it answered without a positive price: "
    )


def test_the_quote_is_asked_of_the_first_instrument_found():
    client = FakeOpenAlgo(symbols={("INFY", "NSE"): symbol_found("INFY", "NSE")})

    with pytest.raises(PreflightError):
        preflight(client, instruments=[("SBIN", "NSE"), ("INFY", "NSE")])

    assert [call for call in client.calls if call[0] == "quotes"] == [("quotes", "INFY", "NSE")]


def test_no_quote_is_asked_when_no_instrument_is_found():
    client = FakeOpenAlgo(symbols={})

    with pytest.raises(PreflightError):
        preflight(client)

    assert [call for call in client.calls if call[0] == "quotes"] == []


# ----- the risk limits ---------------------------------------------------------------


@pytest.mark.parametrize("capital", [None, 0, -5_000, math.nan, math.inf, True, "100000"])
def test_a_capital_that_is_not_a_positive_amount_fails(capital):
    (failure,) = failures_of(capital=capital)

    assert failure.startswith(
        f"capital is required and must be a positive number of rupees, not {capital!r}"
    )


@pytest.mark.parametrize("limit", [None, 0, -1, math.nan])
def test_a_max_daily_loss_that_is_not_a_positive_amount_fails(limit):
    (failure,) = failures_of(max_daily_loss=limit)

    assert failure.startswith(
        f"max_daily_loss is required and must be a positive number of rupees, not {limit!r}"
    )


@pytest.mark.parametrize("limit", [0, -2_000, math.inf, "5000"])
def test_an_account_max_daily_loss_given_must_be_a_positive_amount(limit):
    (failure,) = failures_of(account_max_daily_loss=limit)

    assert failure.startswith(
        f"account_max_daily_loss must be a positive number of rupees when given, not {limit!r}"
    )


@pytest.mark.parametrize("limit", [None, 5_000, 7_500.5])
def test_an_account_max_daily_loss_passes_when_left_out_or_positive(limit):
    report = preflight(account_max_daily_loss=limit)

    assert check_named(report, "account_max_daily_loss").ok


# ----- the alert channel -------------------------------------------------------------


def test_a_live_session_whose_test_alert_is_delivered_passes_the_alerts_check():
    alerts = FakeAlerts(delivered=True)

    report = preflight(alerts=alerts)

    assert check_named(report, "alerts").message == "a test alert was delivered"
    assert alerts.tests == 1


def test_a_live_session_without_an_alert_channel_fails():
    (failure,) = failures_of(alerts=None)

    assert failure.startswith("the session has no alert channel, and a LIVE session needs")


def test_a_live_session_whose_test_alert_is_not_delivered_fails():
    (failure,) = failures_of(alerts=FakeAlerts(delivered=False))

    assert failure.startswith("the test alert was not delivered, and a LIVE session needs")


def test_a_sandbox_session_without_an_alert_channel_passes_with_a_warning():
    report = preflight(FakeOpenAlgo(funds=sandbox_funds()), mode=Mode.SANDBOX, alerts=None)

    check = check_named(report, "alerts")
    assert check.ok
    assert check.message.startswith(f"{WARNING}the session has no alert channel")


def test_a_sandbox_session_whose_test_alert_fails_passes_with_a_warning():
    alerts = FakeAlerts(delivered=False)

    report = preflight(FakeOpenAlgo(funds=sandbox_funds()), mode=Mode.SANDBOX, alerts=alerts)

    check = check_named(report, "alerts")
    assert check.ok
    assert check.message.startswith(f"{WARNING}the test alert was not delivered")
    assert alerts.tests == 1


def test_a_missing_alert_channel_is_reported_even_when_ping_fails():
    failures = failures_of(FakeOpenAlgo(ping=UNREACHABLE), alerts=None)

    assert [failure.split(" ")[0:2] for failure in failures] == [
        ["OpenAlgo's", "ping"],
        ["the", "session"],
    ]


# ----- every failure together --------------------------------------------------------


def test_every_failed_check_is_reported_together_in_the_order_the_checks_ran():
    client = FakeOpenAlgo(
        funds=live_funds("1000.00"), symbols={("SBIN", "NSE"): symbol_missing("SBIN", "NSE")}
    )

    failures = failures_of(client, interval="2m", max_daily_loss=None, product="CNC", alerts=None)

    assert [failure.split(" ")[0:2] for failure in failures] == [
        ["OpenAlgo", "reports"],
        ["NSE:SBIN", "was"],
        ["the", "broker"],
        ["max_daily_loss", "is"],
        ["product", "'CNC'"],
        ["the", "session"],
    ]


def test_the_preflight_error_lists_every_failure_in_its_message():
    with pytest.raises(PreflightError) as raised:
        preflight(product="NRML", capital=0)

    error = raised.value
    assert isinstance(error, LiveError)
    assert str(error).startswith("the live session cannot start: 2 preflight checks failed.")
    for failure in error.failures:
        assert f"\n- {failure}" in str(error)


# ----- the plan ----------------------------------------------------------------------


def test_a_plan_keeps_its_instruments_as_a_tuple_of_pairs():
    plan = make_plan(instruments=[["SBIN", "NSE"], ("NIFTY30DEC25FUT", "NFO")])

    assert plan.instruments == (("SBIN", "NSE"), ("NIFTY30DEC25FUT", "NFO"))


@pytest.mark.parametrize(
    "instruments",
    [[], (), "SBIN", [("SBIN",)], [("SBIN", "")], [("SBIN", 5)], [("SBIN", "NSE", "EQ")], None],
)
def test_a_plan_refuses_instruments_that_are_not_symbol_and_exchange_pairs(instruments):
    with pytest.raises(ConfigError):
        make_plan(instruments=instruments)


def test_a_plan_refuses_a_mode_that_is_not_a_mode():
    with pytest.raises(ConfigError, match="Mode.SANDBOX or Mode.LIVE"):
        make_plan(mode="live")


def test_a_plan_is_frozen():
    plan = make_plan()

    with pytest.raises(FrozenInstanceError):
        plan.capital = 1.0  # type: ignore[misc]


def test_a_plan_takes_keyword_arguments_only():
    with pytest.raises(TypeError):
        PreflightPlan([("SBIN", "NSE")], "5m", "MIS", 1.0, 1.0, None, Mode.LIVE)  # type: ignore[misc]


def test_the_preflight_refuses_anything_but_a_plan():
    with pytest.raises(ConfigError, match="plan must be a PreflightPlan"):
        run_preflight(FakeOpenAlgo(), {"product": "MIS"}, now=lambda: NOW_NS)  # type: ignore[arg-type]


# ----- the countdown -----------------------------------------------------------------


def test_the_countdown_emits_one_line_a_second_down_to_one():
    events: list[tuple[str, object]] = []

    countdown(
        3,
        emit=lambda line: events.append(("emit", line)),
        sleep=lambda s: events.append(("sleep", s)),
    )

    assert events == [
        ("emit", "Starting in 3 seconds. Press Ctrl+C to stop."),
        ("sleep", 1.0),
        ("emit", "Starting in 2 seconds. Press Ctrl+C to stop."),
        ("sleep", 1.0),
        ("emit", "Starting in 1 second. Press Ctrl+C to stop."),
        ("sleep", 1.0),
    ]


def test_a_countdown_of_zero_seconds_emits_nothing_and_waits_for_nothing():
    events: list[object] = []

    countdown(0, emit=events.append, sleep=events.append)

    assert events == []


def test_the_live_countdown_lasts_ten_seconds():
    lines: list[str] = []
    slept: list[float] = []

    countdown(COUNTDOWN_SECONDS, emit=lines.append, sleep=slept.append)

    assert lines[0] == "Starting in 10 seconds. Press Ctrl+C to stop."
    assert sum(slept) == 10.0


@pytest.mark.parametrize("seconds", [-1, 2.5, True, "10", None])
def test_a_countdown_refuses_anything_but_whole_seconds_from_zero(seconds):
    with pytest.raises(ConfigError):
        countdown(seconds, emit=lambda line: None, sleep=lambda s: None)  # type: ignore[arg-type]
