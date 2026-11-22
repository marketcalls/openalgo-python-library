"""What a live session writes about each strategy bar, on the fake OpenAlgo
(`_live/_session_decisions.py`, `_live/_session_trader.py`; 2.4.0 item P2-2): one line per
bar in the day's decisions file, a decision with its bar, the bar's source, the
indicators, the position and what it submitted, a bar not decided on with why, and the
log lines that say why a decision sent nothing. Diagnostic only: nothing here changes a
decision, an order or a figure.

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16 and SBIN trades 10 units every 10 s, at :05, :15 and so on, at
500.00. A 1-minute strategy's first decision is at the close of 14:41's bar, 14:42, and
NSE's MIS entries end at 14:50.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import PRICE, ist, rig

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.series import LiveRunContext


class AverageBuy(bt.Strategy):
    """A 1-minute strategy with a two-bar average of the close (`self.I`) that buys 10
    units at its first decision."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.average = self.I(ta.sma, self.close, 2)
        self.bought = False

    def on_bar(self, bar):
        if not self.bought:
            self.bought = True
            self.buy(10)


class LateBuy(bt.Strategy):
    """Buys 10 units at its first decision after NSE's MIS entry cut-off (14:50 IST), an
    entry the kernel rejects at submission."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.tried = False

    def on_bar(self, bar):
        if bar.ts_close >= ist(14, 51) and not self.tried:
            self.tried = True
            self.buy(10)


class LongAverage(bt.Strategy):
    """A 1-minute strategy with a 150-bar EMA, whose warm-up needs more history than the
    rig's broker serves (two sessions and today to the start): no bar of the session is
    decided on."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.line = self.I(ta.ema, self.close, 150)

    def on_bar(self, bar):
        raise AssertionError("no decision is made before the indicators are ready")


class BuysThenRaises(bt.Strategy):
    """Buys 10 units at its first decision, then raises in the same `on_bar`."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        self.buy(10)
        raise RuntimeError("the strategy's own error")


class LateHugeBuy(LateBuy):
    """`LateBuy` of 1,234,567 units, rejected at submission as the cut-off has passed."""

    def on_bar(self, bar):
        if bar.ts_close >= ist(14, 51) and not self.tried:
            self.tried = True
            self.buy(1_234_567)


class AverageTwice(AverageBuy):
    """`AverageBuy` that registers its average twice, as two uses of one call would."""

    def init(self):
        super().init()
        self.again = self.I(ta.sma, self.close, 2)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path)


def decision_lines(outcome: Any) -> list[dict[str, Any]]:
    """The decisions file's lines, oldest first, without their write time."""
    with open(outcome.paths["decisions"], encoding="utf-8") as stream:
        return [
            {key: value for key, value in json.loads(text).items() if key != "ts_ns"}
            for text in stream
        ]


def records(caplog: Any, words: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if words in record.getMessage()]


def test_the_result_names_the_decisions_file_beside_the_event_stream(world):
    world.stop_at(ist(14, 42, 30))
    outcome = world.run(AverageBuy)
    decisions, events = (outcome.paths[name] for name in ("decisions", "events"))
    assert decisions == events.replace(".events.jsonl", ".decisions.jsonl")


def test_a_decided_bar_writes_its_bar_source_indicators_position_and_orders(world):
    world.stop_at(ist(14, 42, 30))
    outcome = world.run(AverageBuy)
    # The 14:41 bar: six trades of 10 units at 500.00; the average of 14:40's close
    # (the seed minute's candle, 500.00) and 14:41's, to the rounding of a running sum.
    # The buy is the kernel's first order, working once sent; nothing is held yet.
    line = decision_lines(outcome)[0]
    assert line.pop("indicators") == {"sma(SBIN.close, 2)": pytest.approx(PRICE)}
    assert line == {
        "kind": "decision",
        "strategy": "AverageBuy",
        "decided_at_ns": ist(14, 42),
        "decided_at": "2026-11-16 14:42 IST",
        "bar": {
            "ts_open_ns": ist(14, 41),
            "ts_close_ns": ist(14, 42),
            "open": PRICE,
            "high": PRICE,
            "low": PRICE,
            "close": PRICE,
            "volume": 60.0,
        },
        "source": "ticks",
        "position_units": 0.0,
        "submitted": [
            {
                "order": "o1",
                "side": "BUY",
                "units": 10.0,
                "order_type": "MARKET",
                "status": "Submitted",
                "reject_code": None,
            }
        ],
    }


def test_a_decision_line_gives_the_position_held_at_the_decision(world):
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(AverageBuy)
    later = decision_lines(outcome)[1]
    assert (later["decided_at_ns"], later["position_units"], later["submitted"]) == (
        ist(14, 43),
        10.0,
        [],
    )


def test_decision_lines_stay_off_the_event_stream(world):
    world.stop_at(ist(14, 42, 30))
    outcome = world.run(AverageBuy)
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        kinds = {json.loads(text)["kind"] for text in stream}
    assert not kinds & {"decision", "skipped"}


def test_a_rejection_at_submission_is_in_its_decision_line(world):
    world.stop_at(ist(14, 51, 30))
    outcome = world.run(LateBuy)
    line = next(line for line in decision_lines(outcome) if line["decided_at_ns"] == ist(14, 51))
    assert [(o["side"], o["units"], o["status"], o["reject_code"]) for o in line["submitted"]] == [
        ("BUY", 10.0, "Rejected", "entry_window")
    ]


def test_a_rejection_at_submission_is_logged_at_info_with_the_side_units_and_code(world, caplog):
    world.stop_at(ist(14, 51, 30))
    world.run(LateBuy)
    (record,) = records(caplog, "rejected at submission")
    assert record.levelno == logging.INFO
    assert (
        record.getMessage()
        == "LateBuy: order o1 rejected at submission: BUY 10 units, entry_window"
    )


def test_a_huge_rejected_order_is_logged_with_its_exact_units(world, caplog):
    world.stop_at(ist(14, 51, 30))
    world.run(LateHugeBuy)
    (record,) = records(caplog, "rejected at submission")
    assert record.getMessage().endswith("BUY 1234567 units, entry_window")


def test_a_decision_whose_on_bar_raises_writes_its_line_with_the_error(world):
    # The bar the strategy failed on is the one most worth a line: what it submitted
    # before the exception, and the exception.
    with pytest.raises(RuntimeError, match="the strategy's own error"):
        world.run(BuysThenRaises)
    (path,) = (world.folder / "live").rglob("*.decisions.jsonl")
    with open(path, encoding="utf-8") as stream:
        first = json.loads(stream.readline())
    assert (first["kind"], first["decided_at_ns"]) == ("decision", ist(14, 42))
    assert [(o["side"], o["units"]) for o in first["submitted"]] == [("BUY", 10.0)]
    assert first["error"] == "RuntimeError: the strategy's own error"


def test_a_decision_line_names_each_indicator_output_once_its_label_repeats(world):
    world.stop_at(ist(14, 42, 30))
    outcome = world.run(AverageTwice)
    assert list(decision_lines(outcome)[0]["indicators"]) == [
        "sma(SBIN.close, 2)",
        "sma(SBIN.close, 2)#2",
    ]


def test_a_decision_skipped_for_missing_data_writes_its_reason(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # Started at 14:40:58, watching begins in the 14:41 minute, whose candle reads at
    # 14:42:02, :04 and :07 fail: the 14:43 decision reads a bar before it with no data.
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, start=(14, 40, 58))

    def candle_reads_fail():
        for _ in range(3):
            world.fake.faults.gateway_error("history", status=503)

    world.script.at(ist(14, 42), candle_reads_fail)
    world.stop_at(ist(14, 44, 30))
    outcome = world.run(AverageBuy)
    assert [(line["kind"], line["decided_at_ns"]) for line in decision_lines(outcome)] == [
        ("skipped", ist(14, 43)),
        ("decision", ist(14, 44)),
    ]
    assert decision_lines(outcome)[0]["reason"] == "missing data"


def test_each_bar_whose_indicators_are_not_ready_writes_its_reason(world):
    world.stop_at(ist(14, 44, 30))
    outcome = world.run(LongAverage)
    assert [(line["kind"], line["reason"]) for line in decision_lines(outcome)] == [
        ("skipped", "indicators not ready")
    ] * 3


def test_only_the_first_bar_whose_indicators_are_not_ready_is_logged_at_info(world, caplog):
    world.stop_at(ist(14, 44, 30))
    world.run(LongAverage)
    levels = [record.levelno for record in records(caplog, "indicators not ready")]
    assert levels == [logging.INFO, logging.DEBUG, logging.DEBUG]


def test_a_bar_that_closes_while_the_session_stops_writes_its_reason(world):
    # Ctrl+C at 14:42:59 with 10 SBIN held: the safe stop's flatten still runs when the
    # 14:42 slice closes at 14:43:01, so that bar is revealed and decided on no more.
    world.stop_at(ist(14, 42, 59))
    outcome = world.run(AverageBuy)
    assert decision_lines(outcome)[-1] == {
        "kind": "skipped",
        "strategy": "AverageBuy",
        "decided_at_ns": ist(14, 43),
        "decided_at": "2026-11-16 14:43 IST",
        "reason": "session stopping",
    }


def test_indicator_values_that_cannot_be_read_leave_the_decision_and_the_session_as_they_are(
    world, monkeypatch, caplog
):
    # The decisions file is diagnostic: a failure reading the indicators for its line is
    # logged (at WARNING once) and named in the line; the session goes on as before.
    def unreadable(context):
        raise ValueError("an indicator failed to compute")

    monkeypatch.setattr(LiveRunContext, "indicator_values", unreadable)
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(AverageBuy)
    assert [line["indicators"] for line in decision_lines(outcome)] == [
        "not read: ValueError: an indicator failed to compute"
    ] * 2
    assert outcome.stopped_by == "ctrl_c"
    told = records(caplog, "indicator values could not be read")
    assert [record.levelno for record in told] == [logging.WARNING, logging.DEBUG]
