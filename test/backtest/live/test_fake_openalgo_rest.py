"""The fake OpenAlgo's account and data endpoints, called through a real `openalgo.api`:
each answer is the server's (the expected bodies are written out from the server's code:
services/ping_service.py, analyzer_service.py, funds_service.py, symbol_service.py,
intervals_service.py, quotes_service.py, history_service.py; restx_api/telegram_bot.py),
as the SDK returns it. Nothing leaves the process.
"""

from __future__ import annotations

import email.utils
import json

import pandas as pd
from fake_openalgo import BROKER, HOST, USERNAME, FakeOpenAlgo, ist_ns
from fake_openalgo.fixtures import *  # noqa: F403
from fake_openalgo.fixtures import SESSION_START_NS

NS = 1_000_000_000
MINUTE_NS = 60 * NS


def http_body(answer: dict) -> dict:
    """The server's JSON body inside the SDK's "HTTP <code>: <body>" message."""
    code, _, body = answer["message"].partition(": ")
    assert code == f"HTTP {answer['code']}"
    return json.loads(body)


def test_ping_answers_pong_naming_the_broker(fake_openalgo):
    answer = fake_openalgo.client().ping()

    assert answer == {"status": "success", "data": {"message": "pong", "broker": BROKER}}


def test_analyzerstatus_reports_live_mode_while_the_switch_is_off(fake_openalgo):
    answer = fake_openalgo.client().analyzerstatus()

    assert answer == {
        "status": "success",
        "data": {"mode": "live", "analyze_mode": False, "total_logs": 0},
    }


def test_analyzerstatus_follows_the_switch_flipped_mid_session(fake_openalgo):
    client = fake_openalgo.client()
    client.analyzerstatus()

    fake_openalgo.set_analyzer(True)

    assert client.analyzerstatus()["data"] == {
        "mode": "analyze",
        "analyze_mode": True,
        "total_logs": 0,
    }


def test_live_funds_are_the_brokers_text_amounts(fake_openalgo):
    answer = fake_openalgo.client().funds()

    assert answer == {
        "status": "success",
        "data": {
            "availablecash": "250000.00",
            "collateral": "0.00",
            "m2mrealized": "0.00",
            "m2munrealized": "0.00",
            "utiliseddebits": "0.00",
        },
    }


def test_sandbox_funds_are_numbers_marked_analyze(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True)

    answer = fake.client().funds()

    assert answer["mode"] == "analyze"
    assert answer["data"]["availablecash"] == 10_000_000.0
    assert answer["data"]["m2munrealized"] == 0.0


def test_symbol_gives_the_lot_tick_and_freeze_sizes(fake_openalgo):
    option = "NIFTY26NOV2624000CE"
    fake_openalgo.add_instrument(
        option, "NFO", lot_size=75, freeze_qty=1800, instrument_type="CE", strike=24000.0
    )

    data = fake_openalgo.client().symbol(symbol=option, exchange="NFO")["data"]

    assert data["symbol"] == option
    assert (data["lotsize"], data["tick_size"], data["freeze_qty"]) == (75, 0.05, 1800)
    assert (data["instrumenttype"], data["strike"]) == ("CE", 24000.0)


def test_an_unknown_symbol_is_the_servers_404(fake_openalgo):
    answer = fake_openalgo.client().symbol(symbol="NOPE", exchange="NSE")

    assert (answer["status"], answer["code"], answer["error_type"]) == ("error", 404, "http_error")
    assert http_body(answer) == {
        "status": "error",
        "message": "Symbol NOPE not found in exchange NSE",
    }


def test_intervals_list_the_brokers_minutes_hours_and_days(fake_openalgo):
    answer = fake_openalgo.client().intervals()

    assert answer == {
        "status": "success",
        "data": {
            "seconds": [],
            "minutes": ["1m", "3m", "5m", "10m", "15m", "30m"],
            "hours": ["1h"],
            "days": ["D"],
            "weeks": [],
            "months": [],
        },
    }


def test_quotes_give_the_last_trade_and_the_days_range(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", 810.0, volume=100)
    fake_openalgo.trade("SBIN", "NSE", 815.5, volume=50, bid=815.4, ask=815.6)
    fake_openalgo.trade("SBIN", "NSE", 812.0, volume=25)

    answer = fake_openalgo.client().quotes(symbol="SBIN", exchange="NSE")

    assert answer == {
        "status": "success",
        "data": {
            "ask": 812.0,
            "bid": 812.0,
            "high": 815.5,
            "low": 810.0,
            "ltp": 812.0,
            "open": 810.0,
            "prev_close": 0.0,
            "volume": 175.0,
            "oi": 0,
        },
    }


def test_quotes_carry_the_best_bid_and_ask_given_with_the_trade(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", 815.5, bid=815.4, ask=815.6)

    data = fake_openalgo.client().quotes(symbol="SBIN", exchange="NSE")["data"]

    assert (data["bid"], data["ask"]) == (815.4, 815.6)


def test_quotes_of_an_unknown_instrument_fail_in_the_brokers_words(fake_openalgo):
    answer = fake_openalgo.client().quotes(symbol="NOPE", exchange="NSE")

    assert answer["code"] == 500
    assert http_body(answer) == {
        "status": "error",
        "message": "Error fetching quotes: Could not find exchange token for NSE:NOPE",
    }


def test_history_gives_one_minute_candles_stamped_at_their_start_in_ist(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    yesterday = ist_ns(2026, 11, 13, 15, 29)
    fake_openalgo.seed_candles("SBIN", "NSE", [(yesterday, 800.0, 801.0, 799.5, 800.5, 1200)])
    fake_openalgo.trade("SBIN", "NSE", 810.0, volume=10)
    fake_openalgo.clock.advance(20)
    fake_openalgo.trade("SBIN", "NSE", 812.0, volume=5)
    fake_openalgo.clock.advance(50)
    fake_openalgo.trade("SBIN", "NSE", 809.0, volume=7)

    frame = fake_openalgo.client().history(
        symbol="SBIN", exchange="NSE", interval="1m", start_date="2026-11-13", end_date="2026-11-16"
    )

    starts = [stamp.strftime("%Y-%m-%d %H:%M %z") for stamp in frame.index]
    assert starts == ["2026-11-13 15:29 +0530", "2026-11-16 10:15 +0530", "2026-11-16 10:16 +0530"]
    assert frame[["open", "high", "low", "close", "volume"]].values.tolist() == [
        [800.0, 801.0, 799.5, 800.5, 1200],
        [810.0, 812.0, 810.0, 812.0, 15],
        [809.0, 809.0, 809.0, 809.0, 7],
    ]


def test_history_of_dates_without_candles_is_the_sdks_no_data(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")

    answer = fake_openalgo.client().history(
        symbol="SBIN", exchange="NSE", interval="1m", start_date="2026-11-16", end_date="2026-11-16"
    )

    assert not isinstance(answer, pd.DataFrame)
    assert (answer["status"], answer["error_type"]) == ("error", "no_data")


def test_history_refuses_an_interval_the_server_does_not_accept(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")

    answer = fake_openalgo.client().history(
        symbol="SBIN", exchange="NSE", interval="4m", start_date="2026-11-16", end_date="2026-11-16"
    )

    assert answer["code"] == 400
    assert list(http_body(answer)["message"]) == ["interval"]


def test_telegram_notify_queues_the_message_for_the_linked_user(fake_openalgo):
    answer = fake_openalgo.client().telegram(username=USERNAME, message="entry SBIN", priority=8)

    assert answer == {"status": "success", "message": "Notification queued for delivery"}
    sent = fake_openalgo.telegram_sent
    assert [(m.username, m.message, m.priority) for m in sent] == [(USERNAME, "entry SBIN", 8)]


def test_telegram_notify_waiting_for_delivery_answers_sent(fake_openalgo):
    client = fake_openalgo.client()

    answer = client.telegram(username=USERNAME, message="exit SBIN", wait_for_delivery=True)

    assert answer == {"status": "success", "message": "Notification sent successfully"}


def test_telegram_notify_to_a_user_not_linked_is_the_servers_404(fake_openalgo):
    answer = fake_openalgo.client().telegram(username="someone", message="hello")

    assert answer["code"] == 404
    assert http_body(answer) == {
        "status": "error",
        "message": "User not found or not linked to Telegram",
    }


def test_telegram_notify_while_the_bot_is_stopped_is_the_servers_409(fake_openalgo):
    fake_openalgo.set_telegram_bot(False)

    answer = fake_openalgo.client().telegram(username=USERNAME, message="hello")

    assert answer["code"] == 409
    assert fake_openalgo.telegram_sent == []


def test_an_unknown_api_key_is_refused_with_403(fake_openalgo):
    answer = fake_openalgo.client(api_key="not-the-fakes-key").funds()

    assert answer["code"] == 403
    assert http_body(answer) == {"status": "error", "message": "Invalid openalgo apikey"}


def test_every_answer_is_dated_by_the_fakes_clock(fake_openalgo):
    fake_openalgo.clock.advance(7)
    client = fake_openalgo.client()

    head = client.client.head(f"{HOST}/")
    post = client.client.post(f"{HOST}/api/v1/ping", json={"apikey": "x"})

    expected = email.utils.formatdate(SESSION_START_NS // NS + 7, usegmt=True)
    assert head.headers["date"] == post.headers["date"] == expected


def test_a_service_answer_is_compact_json_with_sorted_keys_and_a_newline(fake_openalgo):
    raw = fake_openalgo.client().client.post(f"{HOST}/api/v1/ping", json={"apikey": "x"})

    assert raw.text == '{"message":"Invalid openalgo apikey","status":"error"}\n'


def test_every_call_is_logged_without_its_api_key(fake_openalgo):
    fake_openalgo.client().ping()

    [call] = fake_openalgo.log.calls("ping")
    assert (call.detail["request"], call.detail["status"], call.detail["mode"]) == ({}, 200, "live")


def test_a_call_the_fake_does_not_model_is_logged_as_a_problem():
    fake = FakeOpenAlgo(start_ns=SESSION_START_NS)
    try:
        answer = fake.client().holidays(2026)

        assert answer["error_type"] == "unknown_error"
        [problem] = fake.log.problems()
        assert problem.what == "/api/v1/market/holidays"
    finally:
        fake.close()
