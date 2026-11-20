"""The kill switch on an instrument two strategies of the session share, on a stand-in
session (`_live/_kill_switch.py`; 18-multi-strategy 4.9, unit M6): the position book shows
the net of both, so a member's exit is agreed only when the broker holds at least the
members' sum on its side, and a flat member is flat at the broker when the book shows
what the other holds.

"A" and "B" trade SBIN in MIS; the account's daily loss limit fires the kill switch of
both. Expected values are worked by hand: signed units, long above 0.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openalgo.backtest._live._kill_switch import ACCOUNT_LOSS, KillSwitch
from openalgo.backtest._live.inbox import PositionRows


class Member:
    """A strategy's kernel session as the kill switch sees it: `units` signed SBIN, no
    working order."""

    symbol, exchange, finished, square_off_ns = "SBIN", "NSE", False, 0

    def __init__(self, name: str, units: int) -> None:
        self.name, self.units = name, units
        self.engine = SimpleNamespace(position=self.position)

    def position(self, instrument: Any, product: str) -> dict[str, Any] | None:
        del instrument, product
        return {"quantity_units": float(self.units), "tranches": [(1, 0.0, 0.0, 0)]}

    def open_units(self) -> float:
        return float(self.units)

    def working(self) -> list[int]:
        return []

    def entry_orders(self) -> list[int]:
        return []

    def lock(self, reason: str) -> bool:
        del reason
        return True

    def cancel(self, kernel_order: int) -> None:
        del kernel_order


class Session:
    """What the kill switch calls on its session: two members of one group, the clock
    (`clock_ns`), and what it asked of the margin gate (`exits`) and alerted (`alerts`)."""

    locked = False

    def __init__(self, a_units: int, b_units: int) -> None:
        self.a, self.b = Member("A", a_units), Member("B", b_units)
        self.traders = (self.a, self.b)
        self.groups = SimpleNamespace(
            members=lambda trader: self.traders, shared=lambda trader: True
        )
        self.clock_ns = 0
        self.exits: list[str] = []
        self.alerts: list[str] = []
        self.margin = SimpleNamespace(request_exit=self.request_exit)

    def request_exit(self, trader: Member, reason: str) -> str:
        self.exits.append(f"{trader.name} {reason}")
        return "exit_requested"

    def now(self) -> int:
        return self.clock_ns

    def trader_named(self, name: str) -> Member:
        return self.a if name == "A" else self.b

    def event(self, kind: str, **fields: Any) -> None:
        del kind, fields

    def alert(self, kind: Any, text: str) -> None:
        self.alerts.append(text)

    def hand_over(self, requests: list[Any]) -> None:
        del requests


def fired_and_read(session: Session, broker_units: int) -> KillSwitch:
    """Fire the kill switch of both members, then let one position-book read showing
    `broker_units` SBIN (net, MIS) count for both."""
    switch = KillSwitch(session)
    switch.fire(session.a, ACCOUNT_LOSS)
    switch.fire(session.b, ACCOUNT_LOSS)
    session.clock_ns = 1
    switch.duties(session.clock_ns)
    row = {"symbol": "SBIN", "exchange": "NSE", "product": "MIS", "quantity": broker_units}
    switch.positions(PositionRows({"status": "success", "data": [row]}, 2))
    session.clock_ns = 3
    switch.duties(session.clock_ns)
    return switch


@pytest.mark.parametrize(
    ("a", "b", "broker", "exits"),
    [
        (100, 50, 150, ["A account_loss", "B account_loss"]),
        (100, 50, 170, ["A account_loss", "B account_loss"]),
        (100, 50, 120, []),
        (100, 50, 0, []),
        (100, 0, 150, ["A account_loss"]),
    ],
    ids=[
        "the broker holds the sum",
        "the broker holds more: an excess",
        "a sale by hand of 30: each alone would agree",
        "the broker flat",
        "B flat, the broker holds more than A",
    ],
)
def test_a_members_kill_exit_is_agreed_only_on_the_groups_sum(a, b, broker, exits):
    session = Session(a, b)
    fired_and_read(session, broker)
    assert session.exits == exits


def test_a_flat_member_is_reported_flat_when_the_book_shows_the_other_members_units():
    session = Session(0, 50)
    fired_and_read(session, 50)
    [flat] = [text for text in session.alerts if "A is flat" in text]
    assert "OpenAlgo's position book shows only the 50 units of B" in flat
    assert "close" not in flat


def test_a_flat_member_names_what_the_book_shows_beyond_the_other_members_units():
    session = Session(0, 50)
    switch = fired_and_read(session, 80)
    # Two reads in a row show the same figure before the flat state is reported.
    session.clock_ns = 2 * 10**12
    row = {"symbol": "SBIN", "exchange": "NSE", "product": "MIS", "quantity": 80}
    switch.positions(PositionRows({"status": "success", "data": [row]}, session.clock_ns))
    switch.duties(session.clock_ns + 1)
    [flat] = [text for text in session.alerts if "A is flat" in text]
    assert "still shows 80 units, 50 of them B's: close the rest by hand" in flat
