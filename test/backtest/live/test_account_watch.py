"""The account's daily rules of a live session (`_live/_account_watch.py`; 16-session-design
6.3, 18-multi-strategy 2.5 and 2.5a rule 6) against the shared table of cases the backtest's
fund is pinned by (`test/backtest/data/account_rule_cases.json`, design rule 4): each
case's measure and whether it breaches the daily loss limit or reaches the daily profit
target, worked by hand in the table, so the live rule and the fund's cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openalgo.backtest._live._account_watch import account_measure, reached

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
