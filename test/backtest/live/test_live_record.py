"""The public `bt.record` (`run.record`, `_live/_record_verb.py`; 16-session-design 1.3,
unit 13): the tick recorder on a feed client of its own, until a time of day, raising when
its ticks could not be written.

A recording runs on the recorder tests' fakes (`recorder_fakes`): manual clocks and a feed
the test plays, given through `record_day`'s test seam; a write failure is injected at
the raw file's one write call. The public verb's own work (its client, its clocks, its
stop) is checked with the recorder itself stood in. Nothing opens a socket.
"""

from __future__ import annotations

import datetime as dt
import errno
import time

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from recorder_fakes import DAY, IST, SBIN, Clocks, FakeFeed, Script, ist_ns, quote, raw_files

import openalgo.backtest as bt
from openalgo.backtest._live import _record_verb
from openalgo.backtest._live._recorder_loop import Recording
from openalgo.backtest._live._tick_raw import RawFile
from openalgo.backtest._live.recorder import Stop, read_ticks
from openalgo.backtest.errors import ConfigError, LiveError

START = "10:00:00"


def recording(tmp_path, *, until: object = "10:01", failing: bool = False, monkeypatch=None):
    """Record SBIN from 10:00:00 IST on the fakes, two ticks played at 10:00:00.200 and
    10:00:00.500; with `failing`, every write of the raw file fails as on a full disk."""
    clocks = Clocks(ist_ns(START))
    feed, script = FakeFeed(), Script(clocks)
    script.at(ist_ns("10:00:00.200"), lambda: feed.push(quote("SBIN", 812.4)))
    script.at(ist_ns("10:00:00.500"), lambda: feed.push(quote("SBIN", 812.5)))
    if failing:

        def full(raw, data):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(RawFile, "_write_bytes", full)
    return _record_verb.record_day(
        None,
        [SBIN],
        cache_dir=tmp_path,
        until=until,
        now=clocks.wall,
        monotonic=clocks.mono,
        wait=script,
        feed=feed,
    )


# ----- until -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("until", "stop_at"),
    [
        (None, None),
        ("15:30", ist_ns("15:30:00")),
        (dt.datetime(2026, 11, 16, 15, 30), ist_ns("15:30:00")),
        (dt.datetime(2026, 11, 16, 10, 0, tzinfo=dt.UTC), ist_ns("15:30:00")),
    ],
    ids=["the day's close", "a time of day", "a naive datetime, IST", "an aware datetime"],
)
def test_until_names_the_instant_the_recording_stops(until, stop_at):
    assert _record_verb.stop_at_of(until, ist_ns("09:00:00")) == stop_at


@pytest.mark.parametrize("until", ["25:00", "3:30 pm", 1530, dt.date(2026, 11, 16)])
def test_an_until_that_is_no_time_is_refused_saying_what_to_pass(until):
    with pytest.raises(ConfigError, match="until takes an IST time of day"):
        _record_verb.stop_at_of(until, ist_ns("09:00:00"))


# ----- a recording on the fakes ------------------------------------------------------------


def test_a_recording_writes_the_ticks_and_reads_them_back_verified(tmp_path):
    summary = recording(tmp_path)
    assert (summary.stopped_by, summary.stopped_ns) == (Stop.STOP_AT, ist_ns("10:01:00"))
    assert [tick.ltp for tick in read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)] == [
        812.4,
        812.5,
    ]
    assert dict(summary.ticks) == {("NSE", "SBIN"): 2}


def test_a_write_that_fails_is_raised_chained_from_its_error(tmp_path, monkeypatch):
    with pytest.raises(LiveError, match="could not be written") as failed:
        recording(tmp_path, failing=True, monkeypatch=monkeypatch)
    assert isinstance(failed.value.__cause__, OSError)


def test_a_write_that_fails_names_the_raw_files_kept_for_the_next_recording(tmp_path, monkeypatch):
    # Finalizing fails too (the disk is full), so the recording's raw file stays.
    def full(recording):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Recording, "finalize", full)
    with pytest.raises(LiveError) as failed:
        recording(tmp_path, failing=True, monkeypatch=monkeypatch)
    [kept] = raw_files(tmp_path)
    assert f"Raw files kept: {kept}." in str(failed.value)
    assert "the next recording of the day finalizes them" in str(failed.value)


# ----- the public verb's own work --------------------------------------------------------


class Recorder:
    """Stands in for the tick recorder: keeps what it was called with and returns a
    summary of a recording that stopped as planned."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict]] = []

    def __call__(self, client: object, instruments: object, **keywords: object) -> object:
        self.calls.append((client, {"instruments": instruments, **keywords}))
        return type("Summary", (), {"stopped_by": Stop.STOP_AT})()


@pytest.fixture
def recorder(monkeypatch):
    stand_in = Recorder()
    monkeypatch.setattr(_record_verb, "record", stand_in)
    return stand_in


def test_bt_record_records_on_a_feed_client_of_its_own(fake_openalgo, recorder):
    user = fake_openalgo.client()
    bt.record(user, [SBIN], until="15:30")
    [(client, _)] = recorder.calls
    assert client is not user
    assert (client.api_key, client.base_url) == (user.api_key, user.base_url)


def test_bt_record_closes_its_own_client_and_leaves_the_users_open(fake_openalgo, recorder):
    user = fake_openalgo.client()
    bt.record(user, [SBIN], until="15:30")
    [(client, _)] = recorder.calls
    assert client.client.is_closed and not user.client.is_closed


def test_bt_record_runs_on_the_wall_clock_until_the_time_given(fake_openalgo, recorder):
    bt.record(fake_openalgo.client(), [SBIN], mode="ltp", until="15:30")
    [(_, keywords)] = recorder.calls
    today = dt.datetime.now(IST).date()
    stop_at = dt.datetime.combine(today, dt.time(15, 30), tzinfo=IST).timestamp() * 10**9
    assert (keywords["now"], keywords["monotonic"]) == (time.time_ns, time.monotonic_ns)
    assert (keywords["mode"], keywords["stop_at_ns"]) == ("ltp", round(stop_at))


def test_bt_record_refuses_a_client_that_is_not_an_openalgo_client(recorder):
    with pytest.raises(ConfigError, match="no API key"):
        bt.record(object(), [SBIN])
    assert recorder.calls == []
