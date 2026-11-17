"""The alert of an exit that failed every attempt, and the operator who follows it
(`_live/_session_exits.py`, `_live/_stops_report.py`; 16-session-design 3.5, 4.8 and
4.10): the alert says what `protect` found at the broker for the leg, and to end the
session before any close by hand, since while the session runs the engine sends its exit
again at a later open and places again a stop the operator cancels. An operator who ends
the session first, then cancels the stop and sells by hand, is never short.

The scenarios run on the fake OpenAlgo: the strategies buy 10 SBIN at 14:42:01 IST, with
a stop at 497.00 unless a test says otherwise (`stops_samples`). The words of the alert
for each outcome of `protect` are checked on their own, with no server.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from stops_samples import STOP_TRIGGER, StopHold, stop_world

from openalgo.backtest._live._session_exits import EXIT_ATTEMPTS, ExitRetries
from openalgo.backtest._live._types import LOGGER_NAME, OrderIntent, Purpose

#: What every alert sent while the session runs says before a close by hand, and why.
END_THE_SESSION = (
    "to close it by hand, end the session first (Ctrl+C, and a second Ctrl+C if it does not end)"
)
WHY = "while the session runs, the engine may send its exit or its stop again"
IF_OPEN = "if its closing lines still show the leg open"
#: The price the operator's sale by hand is limited to: under the market, so it fills.
HAND_SALE_LIMIT = 495.0


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


# ----- the words, by what protect found (no server) -------------------------------------


class _Kernel:
    """A kernel session whose `protect` answers `outcome`."""

    def __init__(self, outcome: str) -> None:
        self.outcome = outcome

    def protect(self, instrument: int, product: str) -> str:
        return self.outcome


class _Leg:
    """A strategy holding 10 SBIN whose kernel's `protect` answers `outcome`."""

    name, symbol, exchange, finished = "StopHold", "SBIN", "NSE", False

    def __init__(self, outcome: str) -> None:
        self.engine = _Kernel(outcome)
        self.reasons: dict[int, str] = {}
        self.units_at_place: dict[int, float] = {}

    def open_units(self) -> float:
        return 10.0

    def lock(self, reason: str) -> bool:
        return True


class _Session:
    """The session's part the retry schedule uses: its clock, events and alerts."""

    def __init__(self) -> None:
        self.alerts: list[str] = []

    def now(self) -> int:
        return 0

    def event(self, kind: str, **fields: Any) -> None:
        del kind, fields

    def alert(self, kind: Any, text: str) -> None:
        self.alerts.append(text)


SQUARE_OFF = OrderIntent(
    intent_id="20261116-live-1-StopHold-o3",
    strategy="StopHold",
    symbol="SBIN",
    exchange="NSE",
    action="SELL",
    quantity=10,
    price_type="MARKET",
    product="MIS",
    purpose=Purpose.SQUARE_OFF,
    created_ns=0,
)


def give_up_alert(outcome: str) -> str:
    """The alert after the square-off of 10 SBIN is rejected every time, `protect`
    answering `outcome`."""
    session, leg = _Session(), _Leg(outcome)
    retries = ExitRetries(session)
    for _ in range(EXIT_ATTEMPTS):
        retries.rejected(leg, 3, SQUARE_OFF, "RMS blocked")
    (alert,) = session.alerts
    return alert


STOP_PLACED = "its stop is placed again at the broker and protects the position"
NO_STOP = "no stop at the broker"


@pytest.mark.parametrize(
    ("outcome", "words"),
    [
        ("placed", f"{STOP_PLACED}; {END_THE_SESSION}, then, {IF_OPEN}, cancel that stop"),
        (
            "waits_for_cancels",
            f"{STOP_PLACED}; {END_THE_SESSION}, then, {IF_OPEN}, cancel that stop",
        ),
        (
            "stop_works",
            "its stop works at the broker and protects the position; "
            f"{END_THE_SESSION}, then, {IF_OPEN}, cancel that stop",
        ),
        (
            "exiting",
            f"an exit of it waits or works; {END_THE_SESSION}, then, {IF_OPEN}, "
            "check the order book, cancelling what still works",
        ),
        ("no_stop_level", f"{NO_STOP}; {END_THE_SESSION}"),
        ("market_closed", f"{NO_STOP}; {END_THE_SESSION}"),
    ],
)
def test_the_give_up_alert_says_what_protect_found_and_to_end_the_session_first(outcome, words):
    assert give_up_alert(outcome) == (
        "exit of SBIN NSE failed 5 times (RMS blocked): StopHold's entries are locked; "
        f"{words}: {WHY}"
    )


# ----- the operator who follows the alert (on the fake) ----------------------------------


class ExitsAtItsThirdDecision(StopHold):
    """StopHold that exits at its third decision (14:44:01)."""

    def on_bar(self, bar):
        super().on_bar(bar)
        if self.calls == 3:
            self.exit()


class HoldsWithNoStop(StopHold):
    """Buys 10 SBIN at its first decision with no stop."""

    rules: dict[str, Any] = {}


def refuse_sales_at_market(world, count: int, reason: str) -> None:
    """The broker refuses the next `count` SBIN orders at market."""
    for _ in range(count):
        world.fake.reject_next(reason, price_type="MARKET")


def the_give_up_alert(caplog) -> str:
    (alert,) = [text for text in alerts(caplog, "error") if "failed 5 times" in text]
    return alert


def close_by_hand_after_the_session(world) -> None:
    """The operator, once the session has ended, cancels the SBIN stop still working and
    sells what the broker still holds with a LIMIT under the market; then SBIN trades
    through the old stop's trigger."""
    client = world.fake.client()
    for order in world.orders():
        if order.price_type == "SL-M" and order.status == "trigger pending":
            client.cancelorder(order_id=order.order_id, strategy="manual")
    held = world.fake.position("SBIN", "NSE")
    if held:
        client.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="SELL",
            exchange="NSE",
            price_type="LIMIT",
            product="MIS",
            quantity=held,
            price=HAND_SALE_LIMIT,
        )
    world.fake.trade("SBIN", "NSE", STOP_TRIGGER - 1, volume=2 * StopHold.qty)


def never_short(world) -> bool:
    """Whether the broker's SBIN position stayed at or above zero at every change of an
    order, the session's and the operator's alike."""
    return all(positions.get("SBIN", 0) >= 0 for _, _, positions in world.timeline.states())


def test_after_a_signal_exit_fails_ending_the_session_before_the_sale_by_hand_never_sells_short(
    world, caplog
):
    # The exit at 14:44:01 is refused five times; the stop is placed again at about
    # 14:44:20. The operator presses Ctrl+C at 14:44:40 (on_stop="keep" ends the session),
    # then cancels the stop and sells by hand.
    world.script.at(
        ist(14, 43, 50), lambda: refuse_sales_at_market(world, EXIT_ATTEMPTS, "RMS blocked")
    )
    world.stop_at(ist(14, 44, 40))
    outcome = world.run(ExitsAtItsThirdDecision, on_stop="keep")

    assert END_THE_SESSION in the_give_up_alert(caplog)
    assert [(p.symbol, p.units, p.stop_confirmed) for p in outcome.open_positions] == [
        ("SBIN", 10.0, True)
    ]
    close_by_hand_after_the_session(world)
    assert world.fake.position("SBIN", "NSE") == 0
    assert never_short(world)


def test_after_a_square_off_fails_ending_the_session_before_the_sale_by_hand_never_sells_short(
    world, caplog
):
    # The square-off at 15:00:01 is refused five times; the stop is placed again at about
    # 15:00:20. The operator presses Ctrl+C at 15:00:40: the flatten's exit is refused too
    # and the stop is placed again, so the session does not end, and a second Ctrl+C at
    # 15:00:50 ends it. Then the operator cancels the stop and sells by hand.
    world.script.at(
        ist(14, 59, 50),
        lambda: refuse_sales_at_market(world, EXIT_ATTEMPTS + 1, "RMS: square-off blocked"),
    )
    world.stop_at(ist(15, 0, 40))
    world.stop_at(ist(15, 0, 50))
    outcome = world.run(StopHold)

    assert END_THE_SESSION in the_give_up_alert(caplog)
    assert [(p.symbol, p.units, p.stop_confirmed) for p in outcome.open_positions] == [
        ("SBIN", 10.0, True)
    ]
    close_by_hand_after_the_session(world)
    assert world.fake.position("SBIN", "NSE") == 0
    assert never_short(world)


def test_after_a_square_off_of_a_leg_with_no_stop_fails_the_sale_by_hand_never_sells_short(
    world, caplog
):
    # As above for a leg with no stop: no stop can be placed again, and the operator
    # sells by hand once the second Ctrl+C has ended the session.
    world.script.at(
        ist(14, 59, 50),
        lambda: refuse_sales_at_market(world, EXIT_ATTEMPTS + 1, "RMS: square-off blocked"),
    )
    world.stop_at(ist(15, 0, 40))
    world.stop_at(ist(15, 0, 50))
    outcome = world.run(HoldsWithNoStop)

    gave_up = the_give_up_alert(caplog)
    assert gave_up.endswith(f"{NO_STOP}; {END_THE_SESSION}: {WHY}")
    assert [(p.symbol, p.units) for p in outcome.open_positions] == [("SBIN", 10.0)]
    close_by_hand_after_the_session(world)
    assert world.fake.position("SBIN", "NSE") == 0
    assert never_short(world)
