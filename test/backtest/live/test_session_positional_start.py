"""The start of a live session with a CNC or NRML strategy, on the fake OpenAlgo
(20-positional-live 3.5, 5 and 6; the lead's ruling N3b-3): in LIVE mode the preflight
reads OpenAlgo's GTT book to learn whether the broker holds GTT orders, and refuses a
broker that does not with the design's words; in SANDBOX mode it cannot tell, and the
banner says so; a stop GTT refusal the journal kept from an earlier day refuses the
strategy at start with the broker's message. An MIS-only start makes no GTT call.

The market and the strategies are `positional_samples`'; Ctrl+C comes at 14:46.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import STOP_AT, CncHold, MisHold, calls, gtt_world
from session_samples import DAY

from openalgo.backtest._live._preflight_gtt import REARM_DAILY_SENTENCE, SANDBOX_GTT_LINE
from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.journal import JOURNAL_FILE_NAME, Journal
from openalgo.backtest.errors import PreflightError

#: The broker's refusal kept on an earlier trading day.
KEPT = "GTT orders are not allowed on this segment"


def test_a_live_start_with_a_cnc_strategy_on_a_broker_without_gtt_is_refused(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(lambda **o: make_fake_openalgo(gtt=False, **o), monkeypatch, tmp_path)

    with pytest.raises(PreflightError) as refused:
        world.run(CncHold)

    (failure,) = [f for f in refused.value.failures if "GTT" in f]
    assert failure == (
        f"OpenAlgo answered HTTP 501: GTT orders are not supported for broker '{BROKER}' "
        f"yet: a CNC or NRML position keeps its stop at the broker overnight as a GTT, which "
        f'{BROKER} does not hold. Trade MIS, or pass positional_stops="rearm_daily" to place '
        f"each position's stop as a day order at every start. {REARM_DAILY_SENTENCE}"
    )
    assert calls(world, "placeorder", "placegttorder") == []


def test_a_live_start_with_a_cnc_strategy_reads_the_gtt_book_once_before_the_banner(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(STOP_AT)

    world.run(CncHold)

    first = calls(world, "gttorderbook", "placeorder")[0]
    assert first.what == "gttorderbook"
    assert SANDBOX_GTT_LINE not in world.lines


def test_a_sandbox_start_with_a_cnc_strategy_says_live_gtt_support_is_checked_in_live_mode(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=True)
    world.stop_at(STOP_AT)

    world.run(CncHold)

    assert SANDBOX_GTT_LINE in world.lines


def test_a_stop_gtt_refusal_kept_from_an_earlier_day_refuses_the_strategy_at_start(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    folder = tmp_path / "live" / BROKER
    folder.mkdir(parents=True)
    earlier = DAY - dt.timedelta(days=3)
    with Journal.open(
        folder / JOURNAL_FILE_NAME,
        account=BROKER,
        broker=BROKER,
        mode=Mode.LIVE,
        now=world.fake.clock.now,
    ) as journal:
        journal.start_session(earlier, Mode.LIVE, BROKER)
        journal.keep_refusal("CncHold", "SBIN", "NSE", "CNC", KEPT, day=earlier)

    with pytest.raises(PreflightError) as refused:
        world.run(CncHold)

    assert refused.value.failures == (
        f"CncHold: on {earlier} the broker refused the GTT that holds its stop on NSE:SBIN "
        f'(CNC): "{KEPT}". It is refused at every start until you pass '
        'positional_stops="rearm_daily" or trade it in another product',
    )
    assert calls(world, "placeorder", "placegttorder") == []


def test_an_mis_only_start_makes_no_gtt_call(make_fake_openalgo, monkeypatch, tmp_path):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(STOP_AT)

    world.run(MisHold)

    assert calls(world, "gttorderbook", "placegttorder") == []
