"""The data worker (`_live/data_worker.py`; 16-session-design 2.5 B and 6.5, unit 9),
driven step by step on the fake OpenAlgo's manual clock: quotes before candles, one
history read per instrument per close covering every slice asked for, paced at
`HISTORY_CALLS_PER_SECOND`, read again at the close plus 4 and 7 s while a candle is
missing, none while the feed is dead, gap candles, and the quote fallback; every result
reaches the loop inbox as a message.

The market: SBIN on NSE on Monday 2026-11-16. Its 10:15 IST minute trades at 500.00
(10 units), 501.50 (5) and 499.00 (5), so the broker's candle of the 10:15 slice is
500.00, 501.50, 499.00, 499.00 with 20 units. The checks are asked at 10:16:01, the
slice's close plus the bar builders' one-second grace.
"""

from __future__ import annotations

import threading

import pytest
from fake_openalgo import ist_ns
from fake_openalgo.fixtures import *  # noqa: F403

from openalgo.backtest._live.bars import candle_bar
from openalgo.backtest._live.data_worker import (
    THREAD_NAME,
    CandleRead,
    DataWorker,
    QuoteRead,
)
from openalgo.backtest._live.inbox import Candles, DataFailed, Inbox, Quote
from openalgo.backtest.errors import ConfigError

NS = 1_000_000_000
MINUTE_NS = 60 * NS
DAY = "2026-11-16"
SLICE_1015 = ist_ns(2026, 11, 16, 10, 15)
SLICE_1016 = SLICE_1015 + MINUTE_NS
CLOSE_1015 = SLICE_1016
ASKED_AT = CLOSE_1015 + NS


def move_to(fake, instant_ns: int) -> None:
    """Move the fake's clock on to `instant_ns` (UTC nanoseconds)."""
    fake.clock.advance((instant_ns - fake.clock.now()) / NS)


def trade_at(fake, instant_ns: int, price: float, volume: float, symbol: str = "SBIN") -> None:
    move_to(fake, instant_ns)
    fake.trade(symbol, "NSE", price, volume=volume)


@pytest.fixture
def fake(fake_openalgo):
    """The market of the module docstring, the clock at 10:16:01."""
    for symbol in ("SBIN", "INFY", "TCS"):
        fake_openalgo.add_instrument(symbol, "NSE")
    trade_at(fake_openalgo, SLICE_1015, 500.0, 10)
    trade_at(fake_openalgo, SLICE_1015 + 20 * NS, 501.5, 5)
    trade_at(fake_openalgo, SLICE_1015 + 40 * NS, 499.0, 5)
    move_to(fake_openalgo, ASKED_AT)
    return fake_openalgo


@pytest.fixture
def inbox(fake):
    return Inbox(fake.clock.now)


@pytest.fixture
def worker(fake, inbox):
    return DataWorker(fake.client(), inbox, day=DAY, now=fake.clock.now)


def taken(inbox: Inbox) -> list:
    """Every message on the inbox, oldest first."""
    messages = []
    while (envelope := inbox.take()) is not None:
        messages.append(envelope.message)
    return messages


def history_calls(fake) -> list[int]:
    """When the fake answered each history read (UTC nanoseconds)."""
    return [entry.ts_ns for entry in fake.log.calls("history")]


def checked_at_two_seconds(fake, worker, *slices: int, symbol: str = "SBIN") -> None:
    """A check of `slices` asked at 10:16:01 and stepped at its first read, 10:16:02."""
    worker.check_candles(symbol, "NSE", slices or (SLICE_1015,), CLOSE_1015)
    move_to(fake, CLOSE_1015 + 2 * NS)
    worker.step()


# ----- candle checks -------------------------------------------------------------------


def test_a_candle_check_is_not_read_before_the_close_plus_two_seconds(fake, worker):
    worker.check_candles("SBIN", "NSE", [SLICE_1015], CLOSE_1015)
    assert worker.step() == CLOSE_1015 + 2 * NS
    assert history_calls(fake) == []


def test_a_candle_check_is_read_at_the_close_plus_two_seconds(fake, worker):
    checked_at_two_seconds(fake, worker)
    assert history_calls(fake) == [CLOSE_1015 + 2 * NS]


def test_the_candles_reach_the_loop_as_rows_a_live_bar_is_read_from(fake, worker, inbox):
    checked_at_two_seconds(fake, worker)
    (message,) = taken(inbox)
    assert isinstance(message, Candles)
    assert (message.symbol, message.exchange, message.read_ns) == ("SBIN", "NSE", fake.clock.now())
    bar = candle_bar("SBIN", "NSE", message.rows[0], MINUTE_NS)
    assert (bar.ts_open_ns, bar.ts_close_ns) == (SLICE_1015, CLOSE_1015)
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (500.0, 501.5, 499.0, 499.0, 20)


def test_only_the_slices_asked_for_are_given_never_the_forming_one(fake, worker, inbox):
    trade_at(fake, CLOSE_1015 + 1.5 * NS, 498.0, 3)
    checked_at_two_seconds(fake, worker)
    (message,) = taken(inbox)
    assert [candle_bar("SBIN", "NSE", row, MINUTE_NS).ts_open_ns for row in message.rows] == [
        SLICE_1015
    ]


def test_one_history_read_covers_every_slice_asked_for(fake, worker, inbox):
    earlier = SLICE_1015 - 2 * MINUTE_NS
    fake.seed_candles("SBIN", "NSE", [(earlier, 497, 498, 496, 497.5, 4)])
    fake.seed_candles("SBIN", "NSE", [(earlier + MINUTE_NS, 497.5, 499, 497, 498, 6)])
    checked_at_two_seconds(fake, worker, earlier, earlier + MINUTE_NS, SLICE_1015)
    (message,) = taken(inbox)
    assert len(history_calls(fake)) == 1
    assert len(message.rows) == 3


def test_the_candle_of_a_gap_slice_is_given(fake, worker, inbox):
    # No tick of the 10:14 slice reached the session (a feed outage), but the broker has
    # its candle: the check that covers it gives it, so the series takes the candle.
    gap = SLICE_1015 - MINUTE_NS
    fake.seed_candles("SBIN", "NSE", [(gap, 498, 499, 497.5, 498.5, 9)])
    checked_at_two_seconds(fake, worker, gap)
    (message,) = taken(inbox)
    assert candle_bar("SBIN", "NSE", message.rows[0], MINUTE_NS).close == 498.5


def test_a_missing_candle_is_read_again_at_four_and_seven_seconds_then_left(fake, worker):
    quiet = SLICE_1015 - MINUTE_NS
    worker.check_candles("SBIN", "NSE", [quiet], CLOSE_1015)
    for seconds in (2, 4, 7, 12):
        move_to(fake, CLOSE_1015 + seconds * NS)
        worker.step()
    assert history_calls(fake) == [CLOSE_1015 + seconds * NS for seconds in (2, 4, 7)]


def test_a_check_that_found_every_candle_is_not_read_again(fake, worker):
    checked_at_two_seconds(fake, worker)
    move_to(fake, CLOSE_1015 + 7 * NS)
    assert worker.step() is None
    assert len(history_calls(fake)) == 1


def test_a_second_read_gives_only_the_candles_still_missing(fake, worker, inbox):
    late = SLICE_1015 - MINUTE_NS
    checked_at_two_seconds(fake, worker, late, SLICE_1015)
    fake.seed_candles("SBIN", "NSE", [(late, 498, 499, 497.5, 498.5, 9)])
    move_to(fake, CLOSE_1015 + 4 * NS)
    worker.step()
    first, second = taken(inbox)
    opens = [
        [candle_bar("SBIN", "NSE", row, MINUTE_NS).ts_open_ns for row in m.rows]
        for m in (first, second)
    ]
    assert opens == [[SLICE_1015], [late]]


def test_a_newer_check_of_an_instrument_replaces_the_waiting_one(fake, worker, inbox):
    quiet = SLICE_1015 - MINUTE_NS
    worker.check_candles("SBIN", "NSE", [quiet], CLOSE_1015)
    worker.check_candles("SBIN", "NSE", [SLICE_1015], CLOSE_1015)
    move_to(fake, CLOSE_1015 + 2 * NS)
    worker.step()
    (message,) = taken(inbox)
    assert len(message.rows) == 1
    assert worker.step() is None


def test_history_reads_are_paced_at_two_per_second_across_instruments(fake, worker):
    for symbol in ("SBIN", "INFY", "TCS"):
        worker.check_candles(symbol, "NSE", [SLICE_1015], CLOSE_1015)
    start = CLOSE_1015 + 2 * NS
    move_to(fake, start)
    assert worker.step() == start + NS // 2
    move_to(fake, start + NS // 2)
    assert worker.step() == start + NS
    move_to(fake, start + NS)
    worker.step()
    assert history_calls(fake) == [start, start + NS // 2, start + NS]


def test_a_failed_history_read_reaches_the_loop_and_is_read_again(fake, worker, inbox):
    fake.faults.gateway_error("history")
    checked_at_two_seconds(fake, worker)
    move_to(fake, CLOSE_1015 + 4 * NS)
    worker.step()
    failed, candles = taken(inbox)
    assert isinstance(failed, DataFailed)
    assert failed.request == CandleRead("SBIN", "NSE", (SLICE_1015,), CLOSE_1015)
    assert isinstance(candles, Candles)


def test_a_day_without_candles_yet_is_no_failure_and_is_read_again(fake, worker, inbox):
    # INFY has not traded today: the SDK answers "no data", which is no failed read.
    checked_at_two_seconds(fake, worker, SLICE_1015, symbol="INFY")
    assert taken(inbox) == []
    assert worker.step() == CLOSE_1015 + 4 * NS


def test_a_check_without_slices_is_refused(worker):
    with pytest.raises(ConfigError, match="at least one slice"):
        worker.check_candles("SBIN", "NSE", [], CLOSE_1015)


# ----- the feed dead ---------------------------------------------------------------------


def test_no_history_read_is_made_while_the_feed_is_dead(fake, worker):
    worker.check_candles("SBIN", "NSE", [SLICE_1015], CLOSE_1015)
    worker.feed_dead()
    worker.check_candles("INFY", "NSE", [SLICE_1015], CLOSE_1015)
    move_to(fake, CLOSE_1015 + 7 * NS)
    assert worker.step() is None
    assert history_calls(fake) == []


def test_the_first_check_after_the_feed_returns_is_read(fake, worker, inbox):
    worker.feed_dead()
    worker.step()
    worker.feed_alive()
    checked_at_two_seconds(fake, worker)
    assert [type(message) for message in taken(inbox)] == [Candles]


def test_quotes_are_read_while_the_feed_is_dead(fake, worker, inbox):
    worker.feed_dead()
    worker.quote([("SBIN", "NSE")])
    worker.step()
    assert [type(message) for message in taken(inbox)] == [Quote]


# ----- the quote fallback ----------------------------------------------------------------


def test_a_quote_reaches_the_loop_as_the_last_traded_price(fake, worker, inbox):
    worker.quote([("SBIN", "NSE")])
    worker.step()
    assert taken(inbox) == [Quote("SBIN", "NSE", 499.0, fake.clock.now())]


def test_quotes_are_read_every_two_seconds(fake, worker):
    worker.quote([("SBIN", "NSE")])
    for seconds in (0, 1, 2, 3, 4):
        move_to(fake, ASKED_AT + seconds * NS)
        worker.step()
    assert [entry.ts_ns for entry in fake.log.calls("quotes")] == [
        ASKED_AT,
        ASKED_AT + 2 * NS,
        ASKED_AT + 4 * NS,
    ]


def test_each_instrument_named_is_quoted(fake, worker, inbox):
    trade_at(fake, ASKED_AT, 1_500.0, 1, symbol="INFY")
    worker.quote([("SBIN", "NSE"), ("INFY", "NSE")])
    worker.step()
    assert [(m.symbol, m.ltp) for m in taken(inbox)] == [("SBIN", 499.0), ("INFY", 1_500.0)]


def test_an_empty_list_stops_the_quotes(fake, worker):
    worker.quote([("SBIN", "NSE")])
    worker.step()
    worker.quote([])
    move_to(fake, ASKED_AT + 2 * NS)
    assert worker.step() is None
    assert len(fake.log.calls("quotes")) == 1


def test_due_quotes_are_read_before_a_due_candle_check(fake, worker):
    worker.check_candles("SBIN", "NSE", [SLICE_1015], CLOSE_1015)
    worker.quote([("SBIN", "NSE")])
    move_to(fake, CLOSE_1015 + 2 * NS)
    worker.step()
    reads = [entry.what for entry in fake.log.calls() if entry.what in ("quotes", "history")]
    assert reads == ["quotes", "history"]


def test_quotes_due_after_a_slow_history_read_go_before_the_next(fake, worker):
    for symbol in ("SBIN", "INFY"):
        worker.check_candles(symbol, "NSE", [SLICE_1015], CLOSE_1015)
    move_to(fake, CLOSE_1015 + 2 * NS)
    worker.quote([("SBIN", "NSE")])
    fake.faults.slow("history", 3.0)
    worker.step()
    reads = [entry.what for entry in fake.log.calls() if entry.what in ("quotes", "history")]
    assert reads == ["quotes", "history", "quotes", "history"]


def test_a_failed_quote_reaches_the_loop_as_data_failed(fake, worker, inbox):
    fake.faults.expire_session()
    worker.quote([("SBIN", "NSE")])
    worker.step()
    (failed,) = taken(inbox)
    assert failed.request == QuoteRead("SBIN", "NSE")
    assert "Error fetching quotes" in failed.error


def test_a_quote_before_any_trade_is_a_failed_read(fake, worker, inbox):
    worker.quote([("TCS", "NSE")])
    worker.step()
    (failed,) = taken(inbox)
    assert isinstance(failed, DataFailed)
    assert "no positive last traded price" in failed.error


# ----- the thread --------------------------------------------------------------------------


def running_names() -> set[str]:
    return {thread.name for thread in threading.enumerate()}


def test_the_thread_reads_what_is_asked_and_ends_on_close(worker, inbox):
    worker.start()
    try:
        assert THREAD_NAME in running_names()
        worker.quote([("SBIN", "NSE")])
        envelope = inbox.take(wait_s=10.0)
    finally:
        worker.close()
    assert isinstance(envelope.message, Quote)
    assert THREAD_NAME not in running_names()


def test_the_thread_is_started_once(worker):
    worker.start()
    try:
        with pytest.raises(ConfigError, match="started once"):
            worker.start()
    finally:
        worker.close()
