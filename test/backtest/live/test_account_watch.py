"""The account's daily rules of a live session (`_live/_account_watch.py`; 16-session-design
6.3, 18-multi-strategy 2.5 and 2.5a rule 6) against the shared table of cases the backtest's
fund is pinned by (`test/backtest/data/account_rule_cases.json`, design rule 4): each
case's measure and whether it breaches the daily loss limit or reaches the daily profit
target, worked by hand in the table, so the live rule and the fund's cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openalgo.backtest._live._account_watch import AccountRules, account_measure, reached
from openalgo.backtest._live._types import AlertKind

TABLE = Path(__file__).resolve().parent.parent / "data" / "account_rule_cases.json"
CASES = json.loads(TABLE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_every_case_of_the_shared_table_gives_its_measure_breach_and_profit(case):
    day = case["round"]["date"]
    measure = account_measure(
        day, ((engine["measure_rupees"], engine["date"]) for engine in case["engines"])
    )
    # A rule that fired earlier on the round's day does not fire again that day.
    limit = None if case["breached_on"] == day else case["limit_rupees"]
    target = case.get("profit_target_rupees")
    if case.get("profit_reached_on") == day:
        target = None
    rule = reached(measure, limit, target)
    expected = case["expected"]
    assert measure == expected["measure_rupees"]
    assert (rule == "account_loss") is expected["breach"]
    assert (rule == "account_profit") is expected.get("profit", False)


def test_the_table_holds_cases_of_both_rules():
    assert sum(1 for case in CASES if case["expected"]["breach"]) >= 4
    assert sum(1 for case in CASES if case["expected"].get("profit")) >= 1


@pytest.mark.parametrize(
    ("measure", "limit", "target", "rule"),
    [
        (-10_000.0, 10_000.0, None, "account_loss"),
        (-9_999.99, 10_000.0, None, None),
        (8_000.0, None, 8_000.0, "account_profit"),
        (7_999.99, 10_000.0, 8_000.0, None),
        (-50_000.0, None, None, None),
    ],
)
def test_a_rule_is_reached_in_whole_paise(measure, limit, target, rule):
    assert reached(measure, limit, target) == rule


def account_session(limit: float | None, target: float | None, measures: list[float]):
    """A session of strategies whose daily loss measures are `measures` (rupees), with the
    account's daily loss `limit` and profit `target`, recording its alerts and the rules
    the kill switch is fired for."""
    arguments = SimpleNamespace(account_max_daily_loss=limit, account_daily_profit_target=target)
    traders = [
        SimpleNamespace(engine=SimpleNamespace(daily_loss_measure=lambda m=m: m)) for m in measures
    ]
    session = SimpleNamespace(
        parts=SimpleNamespace(spec=SimpleNamespace(arguments=arguments)),
        traders=traders,
        locked=False,
        alerts=[],
        fired=[],
        event=lambda *_, **__: None,
    )
    session.alert = lambda kind, text: session.alerts.append((kind, text))
    kill = SimpleNamespace(fire=lambda trader, rule: session.fired.append(rule))
    return session, kill


def test_the_accounts_loss_alert_states_rupees_in_indian_digit_grouping():
    # Two strategies down Rs 75,000.00 and Rs 50,000.50: the account's Rs 1,25,000.50 is
    # past its limit of Rs 1,20,000.00, and the kill switch fires for both.
    session, kill = account_session(120_000.0, None, [-75_000.0, -50_000.5])
    AccountRules(session, kill).duties()
    assert session.alerts == [
        (
            AlertKind.ERROR,
            "the account's daily loss Rs 1,25,000.50 is at or past its limit Rs 1,20,000.00: "
            "every strategy's entries are locked for the day and the kill switch exits every "
            "strategy's position",
        )
    ]
    assert session.fired == ["account_loss", "account_loss"]


def test_the_accounts_profit_alert_states_rupees_in_indian_digit_grouping():
    # Rs 12,00,000.00 and Rs 34,567.89 up: Rs 12,34,567.89 is past a target of Rs 10,00,000.
    session, kill = account_session(None, 1_000_000.0, [1_200_000.0, 34_567.89])
    AccountRules(session, kill).duties()
    assert session.alerts == [
        (
            AlertKind.ERROR,
            "the account's daily profit Rs 12,34,567.89 is at or past its target Rs "
            "10,00,000.00: every strategy's entries are locked for the day and the kill switch "
            "exits every strategy's position",
        )
    ]
