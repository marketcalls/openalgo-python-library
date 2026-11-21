"""The decisions file of a live session's logs (`_live/_alerts_logs.py`; 2.4.0 item P2-2):
one JSON line per strategy bar, beside the event stream and off it, and a line that
cannot be written never stops the session."""

from __future__ import annotations

import datetime as dt
import json
import logging

import pytest

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.alerts import configure_session_logs
from openalgo.backtest.errors import ConfigError

DAY = dt.date(2026, 11, 16)
# 2026-11-16 09:15:00.250 IST is 03:45:00.250 UTC.
OPEN_NS = int(dt.datetime(2026, 11, 16, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9 + 250_000_000


def clock() -> int:
    return OPEN_NS


@pytest.fixture
def logs(tmp_path):
    with configure_session_logs(tmp_path / "logs", DAY, now=clock) as opened:
        yield opened


def lines(path) -> list[dict]:
    return [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines()]


class BrokenStream:
    """A file whose every write fails, as on a full disk."""

    def write(self, text: str) -> int:
        raise OSError(28, "No space left on device")

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_the_decisions_file_is_named_after_the_ist_day_beside_the_event_stream(logs, tmp_path):
    assert logs.decisions_path == tmp_path / "logs" / "2026-11-16.decisions.jsonl"


def test_a_decision_line_holds_the_utc_nanosecond_time_the_kind_and_the_fields(logs):
    logs.decision("skipped", strategy="Flip", reason="missing data")
    assert lines(logs.decisions_path) == [
        {"ts_ns": OPEN_NS, "kind": "skipped", "strategy": "Flip", "reason": "missing data"}
    ]


def test_a_decision_line_stays_off_the_event_stream(logs):
    logs.decision("decision", strategy="Flip")
    assert logs.events_path.read_text(encoding="utf-8") == ""


def test_a_decision_line_that_cannot_be_written_does_not_stop_the_session(logs, monkeypatch):
    monkeypatch.setattr(logs, "_decisions", BrokenStream())
    logs.decision("decision", strategy="Flip")
    logs.decision("decision", strategy="Flip")


def test_only_the_first_line_that_cannot_be_written_is_logged_as_an_error(
    logs, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    monkeypatch.setattr(logs, "_decisions", BrokenStream())
    logs.decision("decision", strategy="Flip")
    logs.decision("decision", strategy="Flip")
    failed = [record for record in caplog.records if "decisions file" in record.getMessage()]
    assert [record.levelno for record in failed] == [logging.ERROR, logging.DEBUG]


def test_a_decision_line_after_the_logs_closed_is_dropped(tmp_path):
    logs = configure_session_logs(tmp_path, DAY, now=clock)
    logs.close()
    logs.decision("decision", strategy="Flip")
    assert logs.decisions_path.read_text(encoding="utf-8") == ""


def test_a_decision_line_with_a_field_named_ts_ns_is_refused(logs):
    with pytest.raises(ConfigError, match="'ts_ns' is the event's time"):
        logs.decision("decision", ts_ns=5)
