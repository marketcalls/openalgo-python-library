"""The session logs of a live session: the daily log file on the `openalgo.backtest.live`
logger, the JSON-lines event stream with no key or token in it, and a clean close
(08-live-trading section 8)."""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np
import pytest

from openalgo.backtest._live._types import LOGGER_NAME, AlertKind, Mode, OrderStatus
from openalgo.backtest._live.alerts import Alerts, configure_session_logs
from openalgo.backtest.errors import ConfigError, LiveError

DAY = dt.date(2026, 11, 16)
# 2026-11-16 09:15:00.250 IST is 03:45:00.250 UTC.
OPEN_NS = int(dt.datetime(2026, 11, 16, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9 + 250_000_000
SECRET = "s3cr3t-value"
live_logger = logging.getLogger(LOGGER_NAME)


class ManualClock:
    """UTC nanoseconds that move only when a test moves them."""

    def __init__(self, ns: int = OPEN_NS) -> None:
        self.ns = ns

    def __call__(self) -> int:
        return self.ns


@dataclass(frozen=True)
class Fill:
    symbol: str
    quantity: int
    status: OrderStatus


@pytest.fixture
def clock():
    return ManualClock()


@pytest.fixture
def logs(tmp_path, clock):
    with configure_session_logs(tmp_path / "logs", DAY, now=clock) as opened:
        yield opened


def file_handlers() -> list[logging.Handler]:
    return [handler for handler in live_logger.handlers if isinstance(handler, logging.FileHandler)]


def log_lines(logs) -> list[str]:
    return logs.log_path.read_text(encoding="utf-8").splitlines()


def events(logs) -> list[dict]:
    return [json.loads(text) for text in logs.events_path.read_text(encoding="utf-8").splitlines()]


# The daily log file


def test_the_files_are_named_after_the_ist_day_under_the_folder(logs, tmp_path):
    assert logs.log_path == tmp_path / "logs" / "2026-11-16.log"
    assert logs.events_path == tmp_path / "logs" / "2026-11-16.events.jsonl"


def test_a_day_given_as_a_utc_time_is_read_as_its_ist_day(tmp_path, clock):
    # 20:00 UTC on 15 November is 01:30 IST on 16 November.
    evening_utc = dt.datetime(2026, 11, 15, 20, 0, tzinfo=dt.UTC)
    with configure_session_logs(tmp_path, evening_utc, now=clock) as logs:
        assert logs.log_path.name == "2026-11-16.log"


def test_a_record_of_the_live_logger_is_written_with_the_session_time_in_ist(logs):
    live_logger.warning("feed dead: no tick for 30 s")
    assert log_lines(logs)[-1] == "2026-11-16 09:15:00.250 IST WARNING feed dead: no tick for 30 s"


def test_an_information_record_reaches_the_log_file(logs):
    live_logger.info("entry sent")
    assert log_lines(logs)[-1].endswith("INFO entry sent")


def test_a_debug_record_stays_out_of_the_log_file(logs):
    live_logger.debug("tick SBIN 812.40")
    assert not any("tick SBIN" in text for text in log_lines(logs))


def test_the_log_file_records_when_the_logs_opened(logs):
    assert log_lines(logs) == [
        "2026-11-16 09:15:00.250 IST INFO session logs of 2026-11-16 opened "
        "(events in 2026-11-16.events.jsonl)"
    ]


def test_an_alert_without_a_username_is_written_to_the_log_file(logs, clock):
    alerts = Alerts(client=None, username=None, mode=Mode.SANDBOX, now=clock, strategies=["orb"])
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert (
        log_lines(logs)[-1]
        == "2026-11-16 09:15:00.250 IST INFO alert entry: SANDBOX (orb): Bought 10 SBIN"
    )


def test_a_session_restarted_on_the_same_day_appends_to_both_files(tmp_path, clock):
    for kind in ("first_session", "second_session"):
        with configure_session_logs(tmp_path, DAY, now=clock) as logs:
            logs.event(kind)
    assert [event["kind"] for event in events(logs)] == ["first_session", "second_session"]
    assert sum("opened" in text for text in log_lines(logs)) == 2


def test_logs_opened_while_another_session_has_them_open_are_refused(logs, tmp_path, clock):
    with pytest.raises(LiveError, match="close them before opening new ones"):
        configure_session_logs(tmp_path / "other", DAY, now=clock)


def test_a_folder_that_cannot_be_made_is_refused(tmp_path, clock):
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder", encoding="utf-8")
    with pytest.raises(LiveError, match="Give a folder you can write to"):
        configure_session_logs(blocker / "logs", DAY, now=clock)
    assert file_handlers() == []


def test_a_day_that_is_not_a_date_is_refused(tmp_path, clock):
    with pytest.raises(ConfigError, match="cannot read '16/11/2026' as a date"):
        configure_session_logs(tmp_path, "16/11/2026", now=clock)


# The event stream


def test_an_event_line_holds_the_utc_nanosecond_time_the_kind_and_the_fields(logs):
    logs.event("order_sent", symbol="SBIN", quantity=10, price=812.4)
    assert events(logs) == [
        {"ts_ns": OPEN_NS, "kind": "order_sent", "symbol": "SBIN", "quantity": 10, "price": 812.4}
    ]


def test_each_event_is_one_line_on_disk_before_the_call_returns(logs, clock):
    logs.event("session_start")
    clock.ns += 10**9
    logs.event("bar", symbol="SBIN")
    assert [(event["ts_ns"], event["kind"]) for event in events(logs)] == [
        (OPEN_NS, "session_start"),
        (OPEN_NS + 10**9, "bar"),
    ]


@pytest.mark.parametrize(
    "name",
    ["api_key", "apikey", "API_KEY", "access_token", "auth-token", "client_secret", "password"],
)
def test_a_field_named_like_a_key_or_token_is_redacted(logs, name):
    logs.event("login", **{name: SECRET})
    assert events(logs)[0][name] == "[redacted]"
    assert SECRET not in logs.events_path.read_text(encoding="utf-8")


def test_a_key_inside_a_nested_field_is_redacted(logs):
    logs.event("request", payload={"apikey": SECRET, "legs": [{"symbol": "SBIN", "token": SECRET}]})
    assert events(logs)[0]["payload"] == {
        "apikey": "[redacted]",
        "legs": [{"symbol": "SBIN", "token": "[redacted]"}],
    }


def test_a_redacted_field_is_warned_about_by_name_without_its_value(logs):
    logs.event("login", api_key=SECRET)
    log_text = logs.log_path.read_text(encoding="utf-8")
    assert "field api_key looks like a key or token" in log_text
    assert SECRET not in log_text


@pytest.mark.parametrize("name", ["symbol", "order_id", "tokens_left", "ticks"])
def test_a_field_not_named_like_a_key_is_written_as_given(logs, name):
    logs.event("note", **{name: "SBIN"})
    assert events(logs)[0][name] == "SBIN"


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (Mode.LIVE, "live"),
        (dt.date(2026, 11, 16), "2026-11-16"),
        (dt.datetime(2026, 11, 16, 9, 15), "2026-11-16T09:15:00"),
        ((1, 2), [1, 2]),
        (np.int64(5), 5),
        (np.float64(812.4), 812.4),
        (float("nan"), None),
        (float("inf"), None),
        (PurePosixPath("logs/2026-11-16.log"), "logs/2026-11-16.log"),
        (None, None),
        (True, True),
    ],
    ids=[
        "an enum",
        "a date",
        "a time",
        "a tuple",
        "an integer of numpy",
        "a float of numpy",
        "not a number",
        "infinity",
        "a path",
        "none",
        "a boolean",
    ],
)
def test_a_value_is_written_in_the_json_form_of_its_type(logs, value, written):
    logs.event("value", value=value)
    assert events(logs)[0]["value"] == written


def test_a_dataclass_is_written_as_an_object_of_its_fields(logs):
    logs.event("fill", fill=Fill("SBIN", 10, OrderStatus.COMPLETE))
    assert events(logs)[0]["fill"] == {"symbol": "SBIN", "quantity": 10, "status": "complete"}


def test_a_value_json_cannot_hold_is_written_as_its_repr_with_a_warning(logs):
    logs.event("odd", value=complex(1, 2))
    assert events(logs)[0]["value"] == "(1+2j)"
    assert "field value is a complex, written as its repr" in logs.log_path.read_text("utf-8")


def test_a_field_named_ts_ns_is_refused(logs):
    with pytest.raises(ConfigError, match="'ts_ns' is the event's time"):
        logs.event("bar", ts_ns=5)


def test_an_empty_event_kind_is_refused(logs):
    with pytest.raises(ConfigError, match="non-empty string"):
        logs.event("")


# Closing


def test_close_detaches_the_log_file_from_the_logger(tmp_path, clock):
    logs = configure_session_logs(tmp_path, DAY, now=clock)
    logs.close()
    live_logger.warning("after the session")
    assert log_lines(logs)[-1].endswith("INFO session logs closed")
    assert file_handlers() == []


def test_close_twice_is_safe(tmp_path, clock):
    logs = configure_session_logs(tmp_path, DAY, now=clock)
    logs.close()
    logs.close()
    assert sum("closed" in text for text in log_lines(logs)) == 1


def test_an_event_after_close_is_refused(tmp_path, clock):
    logs = configure_session_logs(tmp_path, DAY, now=clock)
    logs.close()
    with pytest.raises(LiveError, match="after the session logs were closed"):
        logs.event("late")


@pytest.mark.parametrize(
    "level", [logging.NOTSET, logging.WARNING, logging.DEBUG], ids=["unset", "warning", "debug"]
)
def test_close_leaves_the_logger_at_the_level_it_had(tmp_path, clock, level):
    live_logger.setLevel(level)
    try:
        configure_session_logs(tmp_path, DAY, now=clock).close()
        assert live_logger.level == level
    finally:
        live_logger.setLevel(logging.NOTSET)
