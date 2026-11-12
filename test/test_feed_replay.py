"""Which subscriptions the server accepted after a replay (openalgo._feed_replay)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openalgo._feed_replay import ReplayTracker  # noqa: E402

SBIN = {"symbol": "SBIN", "exchange": "NSE"}
INFY = {"symbol": "INFY", "exchange": "NSE"}

NOTHING = {"ltp": [], "quote": [], "depth": [], "orders": False}


def requested(ltp=(), quote=(), depth=(), orders=False):
    return {"ltp": list(ltp), "quote": list(quote), "depth": list(depth), "orders": orders}


def accepted(instrument, mode):
    return dict(instrument, status="success", mode=mode)


def refused(instrument):
    return dict(instrument, status="error", message="Subscription failed")


def reply(*entries):
    return {"type": "subscribe", "subscriptions": list(entries)}


def test_replay_of_nothing_is_done_at_once():
    assert ReplayTracker(NOTHING).done is True


def test_replay_is_not_done_while_a_reply_is_missing():
    tracker = ReplayTracker(requested(ltp=[SBIN], quote=[SBIN]))
    tracker.subscribe_reply(reply(accepted(SBIN, "LTP")))
    assert tracker.done is False


def test_replay_is_done_once_every_instrument_was_answered():
    tracker = ReplayTracker(requested(ltp=[SBIN], quote=[SBIN]))
    tracker.subscribe_reply(reply(accepted(SBIN, "LTP")))
    tracker.subscribe_reply(reply(accepted(SBIN, "Quote")))
    assert tracker.done is True


def test_accepted_instruments_are_listed_under_their_mode():
    tracker = ReplayTracker(requested(ltp=[SBIN, INFY], depth=[SBIN]))
    tracker.subscribe_reply(reply(accepted(SBIN, "LTP"), accepted(INFY, "LTP")))
    tracker.subscribe_reply(reply(accepted(SBIN, "Depth")))
    assert tracker.accepted == requested(ltp=[SBIN, INFY], depth=[SBIN])


def test_refused_instrument_is_answered_but_not_accepted():
    tracker = ReplayTracker(requested(ltp=[SBIN, INFY]))
    tracker.subscribe_reply(reply(accepted(SBIN, "LTP"), refused(INFY)))
    assert (tracker.done, tracker.accepted["ltp"]) == (True, [SBIN])


def test_refusal_without_a_mode_counts_for_the_first_mode_still_waiting():
    tracker = ReplayTracker(requested(ltp=[SBIN], quote=[SBIN]))
    tracker.subscribe_reply(reply(refused(SBIN)))
    assert tracker.unanswered() == requested(quote=[SBIN])


def test_numeric_mode_in_a_reply_is_understood():
    tracker = ReplayTracker(requested(quote=[SBIN]))
    tracker.subscribe_reply(reply(accepted(SBIN, 2)))
    assert tracker.accepted["quote"] == [SBIN]


def test_reply_about_an_instrument_that_was_not_replayed_is_ignored():
    tracker = ReplayTracker(requested(ltp=[SBIN]))
    tracker.subscribe_reply(reply(accepted(INFY, "LTP")))
    assert (tracker.done, tracker.accepted["ltp"]) == (False, [])


def test_reply_for_another_mode_does_not_answer_this_one():
    tracker = ReplayTracker(requested(ltp=[SBIN]))
    tracker.subscribe_reply(reply(accepted(SBIN, "Quote")))
    assert tracker.done is False


def test_order_stream_is_accepted_by_a_successful_reply():
    tracker = ReplayTracker(requested(orders=True))
    tracker.orders_reply({"type": "subscribe_orders", "status": "success"})
    assert (tracker.done, tracker.accepted["orders"]) == (True, True)


def test_order_stream_reply_that_is_not_a_success_is_not_accepted():
    tracker = ReplayTracker(requested(orders=True))
    tracker.orders_reply({"type": "subscribe_orders", "status": "error"})
    assert (tracker.done, tracker.accepted["orders"]) == (True, False)


def test_order_stream_reply_is_ignored_when_the_stream_was_not_replayed():
    tracker = ReplayTracker(requested(ltp=[SBIN]))
    tracker.orders_reply({"type": "subscribe_orders", "status": "success"})
    assert tracker.accepted["orders"] is False


def test_unanswered_lists_what_the_server_has_not_answered():
    tracker = ReplayTracker(requested(ltp=[SBIN, INFY], orders=True))
    tracker.subscribe_reply(reply(accepted(SBIN, "LTP")))
    assert tracker.unanswered() == requested(ltp=[INFY], orders=True)


def test_reply_without_entries_changes_nothing():
    tracker = ReplayTracker(requested(ltp=[SBIN]))
    tracker.subscribe_reply({"type": "subscribe"})
    assert tracker.unanswered() == requested(ltp=[SBIN])
