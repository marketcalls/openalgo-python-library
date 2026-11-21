"""The preflight's check of each futures and options contract's expiry against OpenAlgo's
master contract (`_live/_preflight_expiry.py`; 18-multi-strategy 12.2, unit M11).

The contract is the NIFTY future of December 2026 on NFO, whose reference tables' listed
expiry and last trading day are both Tuesday 2026-12-29; OpenAlgo writes it "29-DEC-26".
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest
from preflight_fakes import NOW_NS, FakeOpenAlgo, make_plan, symbol_found

from openalgo.backtest._expiry_check import master_expiry
from openalgo.backtest._live._preflight_expiry import expiry_checks
from openalgo.backtest._live._preflight_plan import WARNING, InstrumentFacts
from openalgo.backtest._live.preflight import check_preflight
from openalgo.backtest.reference import default_tables

FUTURE = "NIFTY29DEC26FUT"
#: The tables' contract, resolved for the session's day as a session resolves it.
CONTRACT = default_tables().instrument(FUTURE, "NFO", "2026-11-16")
DEC_29 = dt.date(2026, 12, 29)


def facts(expiry) -> list[InstrumentFacts]:
    return [InstrumentFacts(FUTURE, "NFO", 65, 0.1, 1800, expiry)]


def test_the_tables_give_the_contract_its_listed_expiry_and_last_trading_day():
    assert (CONTRACT.expiry, CONTRACT.last_trading_day) == (DEC_29, DEC_29)


@pytest.mark.parametrize(
    ("given", "day"),
    [
        ("29-DEC-26", DEC_29),
        ("29-Dec-2026", DEC_29),
        ("2026-12-29", DEC_29),
        ("29DEC26", DEC_29),
        ("", None),
        (None, None),
        ("DEC-26", None),
        (20261229, None),
    ],
)
def test_a_master_contracts_expiry_is_read_in_each_form_it_is_written(given, day):
    assert master_expiry(given) == day


def test_an_expiry_that_agrees_passes():
    [check] = expiry_checks([CONTRACT], facts("29-DEC-26"))
    assert check.ok and not check.message.startswith(WARNING)
    assert check.name == f"expiry:NFO:{FUTURE}"


def test_an_expiry_that_disagrees_fails_naming_both_dates_and_what_to_do():
    [check] = expiry_checks([CONTRACT], facts("24-DEC-26"))
    assert not check.ok
    assert f"NFO:{FUTURE}: OpenAlgo's master contract gives the expiry 2026-12-24" in check.message
    assert "the reference tables listed expiry 2026-12-29" in check.message
    assert "pip install -U openalgo" in check.message
    assert "report the reference table row" in check.message


def test_the_last_trading_day_agrees_where_it_differs_from_the_listed_expiry():
    moved = dataclasses.replace(CONTRACT, last_trading_day=dt.date(2026, 12, 28))
    [check] = expiry_checks([moved], facts("28-DEC-26"))
    assert check.ok and "last trading day 2026-12-28" in check.message


@pytest.mark.parametrize("given", ["", None, "DEC-26"])
def test_a_missing_or_unreadable_expiry_is_a_warning_not_a_refusal(given):
    [check] = expiry_checks([CONTRACT], facts(given))
    assert check.ok and check.message.startswith(WARNING)


def test_a_contract_two_strategies_trade_is_checked_once():
    # Two strategies on one contract (a shared instrument) name it twice.
    [check] = expiry_checks([CONTRACT, CONTRACT], facts("24-DEC-26"))
    assert not check.ok


def test_a_contract_openalgo_did_not_find_has_no_expiry_check():
    # Its own symbol check has failed already.
    assert expiry_checks([CONTRACT], []) == []


def test_cash_equity_has_no_expiry_check():
    sbin = [InstrumentFacts("SBIN", "NSE", 1, 0.05, None, "")]
    assert expiry_checks([], sbin) == []


# ----- through the preflight -----------------------------------------------------------------


def future_found(expiry: str) -> dict:
    found = symbol_found(FUTURE, "NFO", lotsize=65, tick_size=0.1, freeze_qty=1800)
    found["data"] |= {"expiry": expiry, "instrumenttype": "FUT"}
    return found


def run(expiry: str, *, contracts=(CONTRACT,)):
    client = FakeOpenAlgo(symbols={(FUTURE, "NFO"): future_found(expiry)})
    plan = make_plan(instruments=[(FUTURE, "NFO")], contracts=contracts)
    return client, check_preflight(client, plan, now=lambda: NOW_NS)


def test_a_disagreeing_expiry_refuses_the_start():
    _, outcome = run("24-DEC-26")
    assert any("gives the expiry 2026-12-24" in failure for failure in outcome.failures)


def test_a_missing_expiry_is_a_warning_of_the_report():
    _, outcome = run("")
    assert any("gives NFO:NIFTY29DEC26FUT no expiry" in note for note in outcome.report().warnings)


def test_the_expiry_check_makes_no_call_of_its_own():
    # The symbol answer the preflight reads anyway holds the expiry.
    with_check, _ = run("29-DEC-26")
    without, _ = run("29-DEC-26", contracts=())
    assert with_check.calls == without.calls
