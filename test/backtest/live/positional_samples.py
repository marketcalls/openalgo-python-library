"""CNC and NRML strategies on the fake OpenAlgo, shared by the positional session tests
(`test_session_positional*.py`; 20-positional-live 2 and 3, the lead's rulings N3b-1 to
N3b-6).

The market is `stops_samples`': SBIN on NSE on Monday 2026-11-16, the session started at
14:40:05 IST, SBIN trading 10 units every 10 seconds at the script's price (500.00 unless
a test moves it). `CncHold` buys 10 SBIN CNC at its first decision (14:42:01, filled at
once at 500.00) with a stop 3 points below, held by the broker as a SINGLE GTT: sell 10
when SBIN trades at 497.00 or below, the child's limit 492.00 (the kernel's buffer of
1 percent beyond the trigger, on the tick). The fake's GTT book keeps history rows or
shows the active GTTs only (`BOOKS`), in the broker's book or the sandbox's (`MODES`).

Expected values come from the fake's books and the scenario, never from the session's
code.
"""

from __future__ import annotations

import functools
from typing import Any

import pytest
from session_samples import ist
from stops_samples import STOP_POINTS, StopHold, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._types import Mode

#: The stop GTT's trigger and its child's limit, rupees.
GTT_TRIGGER, GTT_LIMIT = 497.0, 492.0
#: The fake broker's first trigger ID on Monday 2026-11-16.
FIRST_TRIGGER = "261116000001"
#: When the session is asked to stop (Ctrl+C), unless a test says otherwise.
STOP_AT = ist(14, 46)

BOOKS = pytest.mark.parametrize("history", [True, False], ids=["history rows", "active only"])
MODES = pytest.mark.parametrize("analyzer", [False, True], ids=["broker", "sandbox"])


class CncHold(StopHold):
    """StopHold in CNC: its stop is a GTT."""

    product = "CNC"


class CncTarget(CncHold):
    """CncHold with a target 2 points above (502.00), held by the session on ticks."""

    rules = {"stop": bt.points(STOP_POINTS), "target": bt.points(2)}


class CncTrail(CncHold):
    """CncHold whose stop trails 3 points below the highest price since the entry."""

    rules = {"stop": bt.points(STOP_POINTS), "trail": bt.trail_points(STOP_POINTS)}


class CncTwoTargets(CncHold):
    """CncHold with two targets, half at 2 points above (502.00) and the rest at 4
    (504.00): the first cuts the stop GTT to the 5 units left before its exit."""

    rules = {"stop": bt.points(STOP_POINTS), "targets": [(bt.points(2), 0.5), (bt.points(4), 0.5)]}


class MisHold(StopHold):
    """StopHold in MIS, beside a CNC strategy on the same symbol."""


def gtt_world(
    make_fake_openalgo: Any,
    monkeypatch: Any,
    tmp_path: Any,
    *,
    history: bool = True,
    analyzer: bool = False,
    **options: Any,
) -> Any:
    """`stops_samples.stop_world` whose fake keeps GTT history rows or not (`history`), in
    the broker's book or the sandbox's (`analyzer`)."""
    build = functools.partial(make_fake_openalgo, gtt_history=history)
    return stop_world(build, monkeypatch, tmp_path, analyzer=analyzer, **options)


def mode_of(world: Any) -> Mode:
    """The mode the fake runs in."""
    return world.fake.mode


def gtts(world: Any) -> list[tuple[str, int, float, str]]:
    """(status, units, trigger, action) of each GTT of the fake's book of its mode."""
    return [(g.status, g.quantity, g.trigger, g.action) for g in world.fake.gtts(mode_of(world))]


def calls(world: Any, *endpoints: str) -> list[Any]:
    """The fake's log entries of `endpoints`, in order."""
    return [entry for entry in world.fake.log.calls() if entry.what in endpoints]


def sells(world: Any) -> list[tuple[str, int, str]]:
    """(price type, units, status) of each SELL order of SBIN at the fake, oldest first."""
    return [
        (o.price_type, o.quantity, o.status)
        for o in world.orders()
        if o.symbol == "SBIN" and o.action == "SELL"
    ]


__all__ = [
    "BOOKS",
    "FIRST_TRIGGER",
    "GTT_LIMIT",
    "GTT_TRIGGER",
    "MODES",
    "STOP_AT",
    "CncHold",
    "CncTarget",
    "CncTrail",
    "CncTwoTargets",
    "MisHold",
    "calls",
    "gtt_world",
    "gtts",
    "mode_of",
    "sells",
]
