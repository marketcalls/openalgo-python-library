# -*- coding: utf-8 -*-
"""Offline check that a tick keeps the fields the broker sent.

Issue #5 (and marketcalls/openalgo#1527): a depth tick reached the callback
with only ltp, timestamp and depth, so OI, volume and OHLC were dropped even
though the server frame carries them and the API Playground shows them.

Feeds synthetic frames straight into FeedAPI._process_message, so it needs no
running server, no broker session and no market hours. Run it with:

    python test/feed_passthrough_test.py

Prints one line per check and exits non-zero on the first failure.
"""

import os
import sys

# Prefer the source tree over any installed openalgo, so this tests the repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from openalgo.feed import FeedAPI

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def make_feed():
    """A FeedAPI that never opens a socket. _process_message does not need one."""
    return FeedAPI(api_key="offline-test", host="http://127.0.0.1:5000")


def frame(mode, data):
    return json.dumps({
        "type": "market_data",
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "mode": mode,
        "data": data,
    })


DEPTH_BOOK = {
    "buy": [{"price": 1402.5, "quantity": 50, "orders": 2}],
    "sell": [{"price": 1403.0, "quantity": 75, "orders": 3}],
}


def test_depth_keeps_broker_fields():
    feed = make_feed()
    received = []
    feed.depth_callback = received.append

    feed._process_message(frame(3, {
        "ltp": 1402.75,
        "timestamp": 1757800000000,
        "depth": DEPTH_BOOK,
        "oi": 812500,
        "volume": 4821000,
        "open": 1395.0,
        "high": 1410.2,
        "low": 1391.05,
        "close": 1398.6,
    }))

    check("depth callback fired", len(received) == 1)
    if not received:
        return
    data = received[0]["data"]

    for field, expected in (("oi", 812500), ("volume", 4821000),
                            ("open", 1395.0), ("high", 1410.2),
                            ("low", 1391.05), ("close", 1398.6)):
        check(f"depth tick carries {field}", data.get(field) == expected,
              f"got {data.get(field)!r}")

    check("depth tick keeps ltp", data.get("ltp") == 1402.75)
    check("depth tick keeps timestamp", data.get("timestamp") == 1757800000000)
    check("depth tick keeps depth book", data.get("depth") == DEPTH_BOOK)

    snapshot = feed.get_depth("NSE", "RELIANCE")["depth"]["NSE"]["RELIANCE"]
    check("get_depth shape unchanged",
          set(snapshot) == {"timestamp", "ltp", "buyBook", "sellBook"},
          f"got {sorted(snapshot)}")
    check("get_depth buyBook level 1 intact",
          snapshot["buyBook"]["1"] == {"price": 1402.5, "qty": 50, "orders": 2})
    check("get_depth pads to five levels", len(snapshot["sellBook"]) == 5)


def test_depth_defaults_when_server_omits():
    """A frame missing ltp must still produce the documented keys."""
    feed = make_feed()
    received = []
    feed.depth_callback = received.append

    feed._process_message(frame(3, {"depth": DEPTH_BOOK, "oi": 12}))

    check("sparse depth callback fired", len(received) == 1)
    if not received:
        return
    data = received[0]["data"]
    check("sparse depth defaults ltp to 0", data.get("ltp") == 0,
          f"got {data.get('ltp')!r}")
    check("sparse depth fills timestamp", isinstance(data.get("timestamp"), int))
    check("sparse depth still carries oi", data.get("oi") == 12)


def test_quote_keeps_broker_fields():
    feed = make_feed()
    received = []
    feed.quote_callback = received.append

    feed._process_message(frame(2, {
        "open": 1395.0, "high": 1410.2, "low": 1391.05, "close": 1398.6,
        "ltp": 1402.75, "volume": 4821000, "last_trade_quantity": 25,
        "avg_trade_price": 1400.1, "change": 4.15, "change_percent": 0.3,
        "timestamp": 1757800000000,
        "oi": 812500,
        "upper_circuit": 1538.4,
    }))

    check("quote callback fired", len(received) == 1)
    if not received:
        return
    data = received[0]["data"]
    check("quote tick carries oi", data.get("oi") == 812500)
    check("quote tick carries upper_circuit", data.get("upper_circuit") == 1538.4)
    check("quote tick keeps ltp", data.get("ltp") == 1402.75)
    check("quote tick keeps volume", data.get("volume") == 4821000)

    snapshot = feed.get_quotes("NSE", "RELIANCE")["quote"]["NSE"]["RELIANCE"]
    check("get_quotes shape unchanged",
          set(snapshot) == {"timestamp", "open", "high", "low", "close", "ltp",
                            "volume", "last_trade_quantity", "avg_trade_price",
                            "change", "change_percent"},
          f"got {sorted(snapshot)}")


def test_ltp_keeps_broker_fields():
    feed = make_feed()
    received = []
    feed.ltp_callback = received.append

    feed._process_message(frame(1, {
        "ltp": 1402.75,
        "timestamp": 1757800000000,
        "ltt": 1757799998000,
        "oi": 812500,
    }))

    check("ltp callback fired", len(received) == 1)
    if not received:
        return
    data = received[0]["data"]
    check("ltp tick carries oi", data.get("oi") == 812500)
    check("ltp tick keeps ltt", data.get("ltt") == 1757799998000)
    check("ltp tick keeps ltp", data.get("ltp") == 1402.75)

    snapshot = feed.get_ltp("NSE", "RELIANCE")["ltp"]["NSE"]["RELIANCE"]
    check("get_ltp shape unchanged",
          set(snapshot) == {"timestamp", "ltp"}, f"got {sorted(snapshot)}")
    check("get_ltp reads the mapped price", snapshot["ltp"] == 1402.75)


def main():
    test_depth_keeps_broker_fields()
    test_depth_defaults_when_server_omits()
    test_quote_keeps_broker_fields()
    test_ltp_keeps_broker_fields()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
