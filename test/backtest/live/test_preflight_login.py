"""A preflight that waits for the broker's login (16-session-design 3.2): every check of a
run, passed or failed (`check_preflight`); the failures a login in OpenAlgo may fix,
no funds at all with a working ping among them; and a run again while the login is
awaited, which reports the first run's alerts check instead of sending a second test
alert."""

from __future__ import annotations

import pytest
from preflight_fakes import (
    EXPIRED_FUNDS,
    KEY_REFUSED,
    NOW_NS,
    FakeAlerts,
    FakeOpenAlgo,
    live_funds,
    make_plan,
    quote,
    sandbox_funds,
)

from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.preflight import (
    WARNING,
    Check,
    PreflightOutcome,
    check_preflight,
    run_preflight,
)
from openalgo.backtest.errors import ConfigError, PreflightError

ZERO_FUNDS = live_funds("0.00")


def outcome_of(client: FakeOpenAlgo | None = None, **changes) -> PreflightOutcome:
    """Every check of `make_plan(**changes)` against `client` at NOW_NS."""
    client = FakeOpenAlgo() if client is None else client
    return check_preflight(client, make_plan(**changes), now=lambda: NOW_NS)


# ----- every check of a run ----------------------------------------------------------


def test_check_preflight_returns_the_failures_run_preflight_raises():
    short = live_funds("50.00")
    with pytest.raises(PreflightError) as raised:
        run_preflight(FakeOpenAlgo(funds=short), make_plan(product="CNC"), now=lambda: NOW_NS)

    outcome = outcome_of(FakeOpenAlgo(funds=short), product="CNC")

    assert outcome.failures == raised.value.failures
    assert len(outcome.failures) == 2


def test_the_report_of_a_run_that_passed_is_the_preflights_report():
    report = outcome_of().report()

    assert report == run_preflight(FakeOpenAlgo(), make_plan(), now=lambda: NOW_NS)


def test_the_report_of_a_run_that_failed_raises_its_failures():
    outcome = outcome_of(FakeOpenAlgo(funds=ZERO_FUNDS))

    with pytest.raises(PreflightError) as raised:
        outcome.report()

    assert raised.value.failures == outcome.failures


def test_a_check_is_found_by_its_name_and_one_that_did_not_run_is_none():
    outcome = outcome_of(FakeOpenAlgo(ping=KEY_REFUSED))

    assert outcome.check("broker_session").ok is False
    assert outcome.check("alerts") is None


# ----- login failures ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("client", "check"),
    [
        (FakeOpenAlgo(ping=KEY_REFUSED), "broker_session"),
        (FakeOpenAlgo(funds=EXPIRED_FUNDS), "broker_session"),
        (FakeOpenAlgo(quotes=quote(ltp=0)), "broker_quotes"),
        (FakeOpenAlgo(funds=ZERO_FUNDS), "funds"),
    ],
    ids=["api key refused", "funds without data", "no quote", "zero funds"],
)
def test_a_failure_a_broker_login_may_fix_waits_for_the_login(client, check):
    outcome = outcome_of(client)

    assert outcome.check(check).login
    assert outcome.waits_for_login


def test_zero_funds_with_a_working_ping_are_a_login_failure_of_the_funds_check():
    outcome = outcome_of(FakeOpenAlgo(funds=ZERO_FUNDS))

    ((name, login),) = [(check.name, check.login) for check in outcome.checks if not check.ok]
    assert (name, login) == ("funds", True)
    assert outcome.check("broker_session").ok


@pytest.mark.parametrize(
    ("client", "changes"),
    [
        (FakeOpenAlgo(funds=live_funds("50.00")), {}),
        (FakeOpenAlgo(funds=sandbox_funds(0.0)), {"mode": Mode.SANDBOX}),
    ],
    ids=["funds short of the capital", "zero sandbox funds"],
)
def test_funds_short_of_the_capital_are_not_a_login_failure(client, changes):
    outcome = outcome_of(client, **changes)

    assert not outcome.check("funds").login
    assert not outcome.waits_for_login


def test_a_login_failure_with_any_other_failure_does_not_wait():
    outcome = outcome_of(FakeOpenAlgo(funds=ZERO_FUNDS), product="CNC")

    assert not outcome.waits_for_login


def test_a_run_that_passed_does_not_wait_for_the_login():
    outcome = outcome_of()

    assert (outcome.failures, outcome.waits_for_login) == ((), False)


# ----- the test alert, once ----------------------------------------------------------


def test_a_run_again_after_the_login_passes_with_one_test_alert():
    alerts = FakeAlerts()
    first = outcome_of(FakeOpenAlgo(funds=EXPIRED_FUNDS), alerts=alerts)

    again = make_plan(alerts=alerts, tested_alerts=first.check("alerts"))
    report = run_preflight(FakeOpenAlgo(), again, now=lambda: NOW_NS)

    assert first.waits_for_login
    assert alerts.tests == 1
    assert [check.name for check in report.checks][-1] == "alerts"


def test_a_run_again_reports_the_first_runs_alerts_check():
    alerts = FakeAlerts()
    first = outcome_of(FakeOpenAlgo(funds=ZERO_FUNDS), alerts=alerts)

    again = outcome_of(FakeOpenAlgo(), alerts=alerts, tested_alerts=first.check("alerts"))

    assert again.check("alerts") == first.check("alerts")


def test_a_run_again_in_sandbox_mode_keeps_the_first_runs_alerts_warning():
    alerts = FakeAlerts(delivered=False)
    sandbox = {"mode": Mode.SANDBOX, "alerts": alerts}
    first = outcome_of(FakeOpenAlgo(funds=sandbox_funds(), quotes=quote(ltp=0)), **sandbox)

    again = outcome_of(
        FakeOpenAlgo(funds=sandbox_funds()), **sandbox, tested_alerts=first.check("alerts")
    )

    assert again.report().warnings[-1].startswith(f"{WARNING}the test alert was not delivered")
    assert alerts.tests == 1


def test_a_run_whose_ping_failed_sent_no_test_alert_to_hand_on():
    alerts = FakeAlerts()

    outcome = outcome_of(FakeOpenAlgo(ping=KEY_REFUSED), alerts=alerts)

    assert (outcome.check("alerts"), alerts.tests) == (None, 0)


def test_the_first_runs_alerts_check_is_reported_even_when_ping_fails_again():
    tested = Check("alerts", True, "a test alert was delivered")

    outcome = outcome_of(FakeOpenAlgo(ping=KEY_REFUSED), tested_alerts=tested)

    assert outcome.check("alerts") is tested


@pytest.mark.parametrize(
    "tested", [Check("funds", True, "covered"), "a test alert was delivered", True]
)
def test_a_plan_refuses_anything_but_an_alerts_check_as_tested_alerts(tested):
    with pytest.raises(ConfigError, match="tested_alerts is the alerts check"):
        make_plan(tested_alerts=tested)
