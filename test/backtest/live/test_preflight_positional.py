"""The preflight's checks of a session with positions held overnight (`_live/_preflight_gtt.py`,
`_live/_preflight_carry.py`; 20-positional-live 3.5, 4.2, 5 and 6, the lead's ruling N2-3):
the GTT capability check against the fake OpenAlgo's GTT book in each answer, the trading
days missed since a day-close record, the other ledger's carried positions named for the
banner, the funds check of a carry, and a kept refusal refusing its strategy.

The messages the design quotes are checked word for word. The rupees a carried position
holds are the margin its engine blocked when it entered, as its day-close record keeps it
(`margin_rupees`), each worked by hand from the position: a delivery buy's whole cost (units
times average price), a future's default margin of 20 percent of its contract value, cut
with the units a part exit sold, and an option's premium (units times price). Nothing
contacts a server.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fake_openalgo.fixtures import *  # noqa: F403

from openalgo.backtest._live._journal_carry import DayClose, KeptRefusal
from openalgo.backtest._live._preflight_carry import (
    CarriedPosition,
    carried_positions,
    carry_funds_check,
    held_rupees,
    kept_refusal_checks,
    missed_days,
    other_ledger_lines,
)
from openalgo.backtest._live._preflight_gtt import REARM_DAILY_SENTENCE, gtt_check
from openalgo.backtest._live._preflight_plan import WARNING
from openalgo.backtest._live._types import Mode
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import ConfigError, LiveError
from openalgo.backtest.reference import Instrument

BROKER = "zerodha"
#: The design's banner line of a sandbox session (20-positional-live 5).
SANDBOX_LINE = "GTT support of your live broker is checked when you start in live mode."
#: The design's sentence for rearm_daily (20-positional-live 6).
SENTENCE = (
    "Positions held overnight have no stop at the broker from the close until your next "
    "start; start bt.live before the open on every trading day you hold a position."
)
TODAY = dt.date(2026, 11, 24)


def gtt_calls(fake) -> int:
    return len(fake.log.calls("gttorderbook"))


# ----- (a) the GTT capability check ----------------------------------------------------------


def test_a_live_broker_with_gtt_orders_passes(fake_openalgo):
    check = gtt_check(fake_openalgo.client(), mode=Mode.LIVE, broker=BROKER, positional_stops="gtt")

    assert (check.name, check.ok, gtt_calls(fake_openalgo)) == ("gtt", True, 1)
    assert check.message == (
        "zerodha holds GTT orders: each CNC or NRML position's stop rests at the broker as a "
        "GTT overnight"
    )


def test_a_live_broker_without_gtt_orders_refuses_naming_it_and_both_choices(make_fake_openalgo):
    fake = make_fake_openalgo(gtt=False)

    check = gtt_check(fake.client(), mode=Mode.LIVE, broker=BROKER, positional_stops="gtt")

    assert not check.ok
    assert check.message == (
        "OpenAlgo answered HTTP 501: GTT orders are not supported for broker 'zerodha' yet: a "
        "CNC or NRML position keeps its stop at the broker overnight as a GTT, which zerodha "
        'does not hold. Trade MIS, or pass positional_stops="rearm_daily" to place each '
        f"position's stop as a day order at every start. {SENTENCE}"
    )


def test_the_refusal_quotes_the_designs_sentence_for_rearm_daily():
    assert REARM_DAILY_SENTENCE == SENTENCE


def test_a_live_broker_without_gtt_orders_passes_with_rearm_daily(make_fake_openalgo):
    fake = make_fake_openalgo(gtt=False)

    check = gtt_check(fake.client(), mode=Mode.LIVE, broker=BROKER, positional_stops="rearm_daily")

    assert check.ok
    assert check.message == (
        "zerodha holds no GTT orders in OpenAlgo (HTTP 501: GTT orders are not supported for "
        "broker 'zerodha' yet): positional_stops=\"rearm_daily\" places each carried "
        "position's stop as a day order at each start"
    )


def test_rearm_daily_on_a_broker_with_gtt_orders_warns_that_gtts_would_protect(fake_openalgo):
    client = fake_openalgo.client()

    check = gtt_check(client, mode=Mode.LIVE, broker=BROKER, positional_stops="rearm_daily")

    assert check.ok
    assert check.message == (
        f"{WARNING}zerodha holds GTT orders, which would protect your positions overnight; "
        'positional_stops="rearm_daily" places a day stop at each start instead'
    )


def test_any_other_answer_refuses_the_start_with_the_servers_message(fake_openalgo):
    fake_openalgo.faults.gateway_error("gttorderbook", status=503)

    check = gtt_check(fake_openalgo.client(), mode=Mode.LIVE, broker=BROKER, positional_stops="gtt")

    assert not check.ok
    assert check.message.startswith(
        "the start cannot tell whether zerodha holds GTT orders: OpenAlgo's GTT book answered "
        "HTTP 503: Endpoint request timed out."
    )


def test_with_rearm_daily_another_answer_still_refuses_in_words_true_of_that_run(fake_openalgo):
    # No stop is a GTT with rearm_daily, but the session reads the GTT book for every CNC
    # and NRML leg (a strategy's own GTT orders among them), so the start cannot go on.
    fake_openalgo.faults.gateway_error("gttorderbook", status=503)

    check = gtt_check(
        fake_openalgo.client(), mode=Mode.LIVE, broker=BROKER, positional_stops="rearm_daily"
    )

    assert not check.ok
    assert check.message == (
        "the start cannot tell whether zerodha holds GTT orders: OpenAlgo's GTT book answered "
        'HTTP 503: Endpoint request timed out. With positional_stops="rearm_daily" no stop is '
        "a GTT, but the session reads the GTT book for its CNC and NRML positions and the "
        "strategies' own GTT orders: check OpenAlgo and the broker's login, then start again"
    )


def test_a_gateways_page_is_quoted_with_its_http_status_once(fake_openalgo):
    fake_openalgo.faults.bad_gateway_page("gttorderbook")

    check = gtt_check(fake_openalgo.client(), mode=Mode.LIVE, broker=BROKER, positional_stops="gtt")

    assert not check.ok
    assert check.message.startswith(
        "the start cannot tell whether zerodha holds GTT orders: OpenAlgo's GTT book answered "
        "HTTP 502: <html>"
    )


def test_an_expired_brokers_login_refuses_the_start_with_the_brokers_words(fake_openalgo):
    fake_openalgo.faults.expire_session()

    check = gtt_check(fake_openalgo.client(), mode=Mode.LIVE, broker=BROKER, positional_stops="gtt")

    assert not check.ok
    assert check.message.startswith(
        "the start cannot tell whether zerodha holds GTT orders: OpenAlgo's GTT book answered "
        "HTTP 403: Incorrect `api_key` or `access_token`."
    )


@pytest.mark.parametrize("stops", ["gtt", "rearm_daily"])
def test_a_live_session_whose_gtt_book_the_sandbox_answered_is_refused(make_fake_openalgo, stops):
    # The analyzer switch was turned on after the session read the mode: the sandbox
    # answers every GTT call, which says nothing of the live broker.
    fake = make_fake_openalgo(analyzer=True, gtt=False)

    check = gtt_check(fake.client(), mode=Mode.LIVE, broker=BROKER, positional_stops=stops)

    assert not check.ok
    assert check.message == (
        "OpenAlgo answered the GTT book in SANDBOX mode, but the session runs in LIVE mode: "
        "its analyzer switch changed after the session read it. Set the switch in OpenAlgo to "
        "the mode you want, then start the session again"
    )


@pytest.mark.parametrize("stops", ["gtt", "rearm_daily"])
def test_a_sandbox_session_asks_nothing_and_says_when_the_live_broker_is_checked(
    make_fake_openalgo, stops
):
    fake = make_fake_openalgo(analyzer=True, gtt=False)

    check = gtt_check(fake.client(), mode=Mode.SANDBOX, broker=BROKER, positional_stops=stops)

    assert (check.ok, check.message, gtt_calls(fake)) == (True, SANDBOX_LINE, 0)


def test_a_positional_stops_choice_the_session_does_not_know_is_refused(fake_openalgo):
    with pytest.raises(ConfigError, match='"gtt".*"rearm_daily".*not \'daily\''):
        gtt_check(fake_openalgo.client(), mode=Mode.LIVE, broker=BROKER, positional_stops="daily")


# ----- (b) missed trading days -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("closed", "today", "missed"),
    [
        (dt.date(2026, 11, 13), dt.date(2026, 11, 16), ()),
        (dt.date(2026, 11, 16), dt.date(2026, 11, 16), ()),
        (dt.date(2026, 11, 12), dt.date(2026, 11, 16), (dt.date(2026, 11, 13),)),
        (
            dt.date(2026, 11, 13),
            dt.date(2026, 11, 18),
            (dt.date(2026, 11, 16), dt.date(2026, 11, 17)),
        ),
        # 2026-11-10 is an NSE holiday: no session ran that day to miss.
        (dt.date(2026, 11, 9), dt.date(2026, 11, 12), (dt.date(2026, 11, 11),)),
    ],
    ids=["previous trading day", "same day", "one day", "two days", "a holiday between"],
)
def test_the_trading_days_missed_since_a_day_close_record(closed, today, missed):
    assert missed_days("NSE", closed, today, default_calendar()) == missed


def test_a_day_close_record_of_a_later_day_is_refused():
    with pytest.raises(LiveError, match="after today .* check this computer's date"):
        missed_days("NSE", dt.date(2026, 11, 25), TODAY, default_calendar())


# ----- the carried state ------------------------------------------------------------------------


def a_record(strategy, positions, *, mode=Mode.SANDBOX, instrument=("SBIN", "NSE")):
    """A day-close record of 2026-11-23 whose carried state holds `positions`, each (product,
    signed units, the rupees of margin its engine blocked) of `instrument`, as the kernel's
    saved state writes a position (rust/oa_bt/src/engine/venue_state/book.rs)."""
    carried = {
        "instruments": [{"symbol": instrument[0], "exchange": instrument[1]}],
        "positions": [
            {
                "instrument": 0,
                "product": product,
                "quantity_units": units,
                "average_price_rupees": 800.0,
                "multiplier": 1.0,
                "margin_rupees": margin,
            }
            for product, units, margin in positions
        ],
    }
    return DayClose(strategy, dt.date(2026, 11, 23), mode, carried, {}, "0a1b2c3d4e5f", 0)


def test_the_open_positions_of_a_carried_state_are_read_flat_ones_left_out():
    record = a_record("hold", [("CNC", 40.0, 32_000.0), ("MIS", 0.0, 0.0)])

    assert carried_positions(record) == (
        CarriedPosition("hold", "SBIN", "NSE", "CNC", 40.0, 32_000.0),
    )


def test_a_carried_state_that_cannot_be_read_is_refused_naming_its_strategy():
    record = DayClose("hold", dt.date(2026, 11, 23), Mode.LIVE, {"positions": [{}]}, {}, "v", 0)

    with pytest.raises(
        LiveError, match="day-close record of hold of 2026-11-23 in the live ledger"
    ):
        carried_positions(record)


# ----- (c) the other ledger's carried positions -----------------------------------------------


def test_the_other_ledgers_carried_positions_are_named_for_the_banner():
    records = {
        "hold": a_record("hold", [("CNC", 40.0, 32_000.0)]),
        "flat": a_record("flat", [("CNC", 0.0, 0.0)]),
        "swing": a_record(
            "swing", [("NRML", -130.0, 624_000.0)], instrument=("NIFTY29DEC26FUT", "NFO")
        ),
    }

    assert other_ledger_lines(records, Mode.LIVE) == [
        "hold: 40 SBIN NSE (CNC) held in sandbox, not continued in live",
        "swing: -130 NIFTY29DEC26FUT NFO (NRML) held in sandbox, not continued in live",
    ]


@pytest.mark.parametrize(
    ("units", "shown"),
    [(1_000_000.0, "10,00,000"), (1_234_567.0, "12,34,567"), (-2_500_000.0, "-25,00,000")],
    ids=["a million", "above a million", "a short of millions"],
)
def test_carried_units_above_a_million_are_named_whole(units, shown):
    records = {"hold": a_record("hold", [("CNC", units, 0.0)], instrument=("YESBANK", "NSE"))}

    assert other_ledger_lines(records, Mode.LIVE) == [
        f"hold: {shown} YESBANK NSE (CNC) held in sandbox, not continued in live"
    ]


# ----- (d) the funds check on a carry ---------------------------------------------------------

#: 40 SBIN bought at 800.00: delivery blocks the whole cost, 32,000.00.
SBIN_CNC = CarriedPosition("hold", "SBIN", "NSE", "CNC", 40.0, 32_000.0)
NIFTY_FUTURE = Instrument(
    symbol="NIFTY29DEC26FUT",
    exchange="NFO",
    kind="future",
    lot_size=65,
    tick_size=0.1,
    expiry=dt.date(2026, 12, 29),
    underlying="NIFTY",
)
NIFTY_CALL = Instrument(
    symbol="NIFTY29DEC2624000CE",
    exchange="NFO",
    kind="option",
    lot_size=75,
    tick_size=0.05,
    expiry=dt.date(2026, 12, 29),
    strike=24000.0,
    option_type="CE",
    underlying="NIFTY",
)
#: Two lots of the future sold at 24,000.00: the engine blocked 20 percent of the contract
#: value of 130 x 24,000 = 31,20,000 when it entered, 6,24,000.00.
SHORT_FUTURE = CarriedPosition("swing", NIFTY_FUTURE.symbol, "NFO", "NRML", -130.0, 624_000.0)


@pytest.mark.parametrize(
    ("product", "units", "margin", "instrument"),
    [
        ("CNC", 40.0, 32_000.0, Instrument.cash_equity("SBIN")),
        ("NRML", -130.0, 624_000.0, NIFTY_FUTURE),
        # A future carried after a part exit: the engine cut its margin with its units.
        ("NRML", -65.0, 312_000.0, NIFTY_FUTURE),
        # A lot of the call bought at 120.00: the premium paid, 75 x 120.00.
        ("NRML", 75.0, 9_000.0, NIFTY_CALL),
    ],
    ids=["cnc cost", "future margin", "future after a part exit", "option premium"],
)
def test_the_cash_a_carried_position_holds_is_the_margin_its_engine_blocked(
    product, units, margin, instrument
):
    record = a_record("held", [(product, units, margin)], instrument=(instrument.symbol, "NFO"))
    (position,) = carried_positions(record)

    assert held_rupees(position, instrument) == margin


def test_a_sold_option_carried_is_refused():
    sold = CarriedPosition("calls", NIFTY_CALL.symbol, "NFO", "NRML", -75.0, 0.0)

    with pytest.raises(LiveError, match="carries -75 NIFTY29DEC2624000CE, a sold option: sold"):
        held_rupees(sold, NIFTY_CALL)


def test_available_cash_covering_the_capital_less_what_is_carried_passes():
    check = carry_funds_check(70_000.0, 100_000.0, [(SBIN_CNC, 32_000.0)], Mode.LIVE)

    assert check.ok
    assert check.message == (
        "70,000.00 rupees of available cash cover the capital of 1,00,000.00 rupees less the "
        "32,000.00 rupees the carried positions hold (hold SBIN CNC 32,000.00), 68,000.00 rupees"
    )


def test_available_cash_short_of_the_capital_less_what_is_carried_fails_with_the_figures():
    held = [(SBIN_CNC, 32_000.0), (SHORT_FUTURE, 624_000.0)]

    check = carry_funds_check(300_000.0, 1_000_000.0, held, Mode.LIVE)

    # 10,00,000 less 6,56,000 carried leaves 3,44,000 to cover.
    assert not check.ok
    assert check.message == (
        "OpenAlgo reports 3,00,000.00 rupees of available cash (collateral not counted), less "
        "than the capital of 10,00,000.00 rupees less the 6,56,000.00 rupees the carried "
        "positions hold (hold SBIN CNC 32,000.00, swing NIFTY29DEC26FUT NRML 6,24,000.00), "
        "3,44,000.00 rupees: lower capital, or add funds to the broker account"
    )


def test_no_available_cash_on_a_carry_says_a_broker_may_report_an_expired_login_so():
    check = carry_funds_check(0.0, 100_000.0, [(SBIN_CNC, 32_000.0)], Mode.LIVE)

    assert (check.ok, check.login) == (False, True)
    assert check.message.endswith(
        "68,000.00 rupees: lower capital, or add funds to the broker account. No cash at all is "
        "also what some brokers report for an expired login: if the account holds funds, log in "
        "to OpenAlgo again"
    )


def test_carried_positions_holding_more_than_the_capital_leave_nothing_to_cover():
    check = carry_funds_check(0.0, 20_000.0, [(SBIN_CNC, 32_000.0)], Mode.SANDBOX)

    assert check.ok


# ----- (e) kept refusals ----------------------------------------------------------------------

REFUSAL = KeptRefusal(
    "hold", "SBIN", "NSE", "CNC", "GTT is not allowed for this segment", dt.date(2026, 11, 20), 0
)
STRATEGIES = [("hold", "SBIN", "NSE", "CNC"), ("day", "SBIN", "NSE", "MIS")]


def test_a_kept_refusal_refuses_its_strategy_with_the_brokers_message():
    (check,) = kept_refusal_checks([REFUSAL], STRATEGIES, "gtt")

    assert (check.name, check.ok) == ("kept_refusal:hold", False)
    assert check.message == (
        "hold: on 2026-11-20 the broker refused the GTT that holds its stop on NSE:SBIN (CNC): "
        '"GTT is not allowed for this segment". It is refused at every start until you pass '
        'positional_stops="rearm_daily" or trade it in another product'
    )


@pytest.mark.parametrize(
    ("strategies", "stops"),
    [
        (STRATEGIES, "rearm_daily"),
        ([("hold", "SBIN", "NSE", "NRML")], "gtt"),
        ([("day", "SBIN", "NSE", "MIS")], "gtt"),
    ],
    ids=["rearm daily", "another product", "another strategy"],
)
def test_a_kept_refusal_refuses_nothing_once_its_leg_or_choice_changed(strategies, stops):
    assert kept_refusal_checks([REFUSAL], strategies, stops) == []
