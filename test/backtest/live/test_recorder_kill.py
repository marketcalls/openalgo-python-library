"""A recording killed for real: a child process records on a fake feed and ends with
`os._exit` in the middle of the recording, with no chance to clean up. The next
finalize or recording recovers the ticks it had flushed and marks its span as ended by
a crash. One child process runs for the whole module; each test works on a copy of the
cache it left.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from recorder_fakes import (
    DAY,
    KILLED_EXIT_CODE,
    NS,
    SBIN,
    Clocks,
    FakeFeed,
    Script,
    ist_ns,
    quote,
    raw_files,
)

from openalgo.backtest._live.recorder import Ending, finalize, read_day, read_ticks, record

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
#: Seconds the child may take to start Python, import the package and record.
CHILD_TIMEOUT_S = 120


@pytest.fixture(scope="module")
def killed_cache(tmp_path_factory) -> Path:
    """The cache a recording killed at 10:00:02 left, with two ticks flushed."""
    cache = tmp_path_factory.mktemp("killed")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(HERE)]))
    code = "import sys, recorder_fakes; recorder_fakes.killed_recording(sys.argv[1])"
    child = subprocess.run(
        [sys.executable, "-c", code, str(cache)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
    )
    assert child.returncode == KILLED_EXIT_CODE, child.stderr
    return cache


@pytest.fixture
def cache(killed_cache, tmp_path) -> Path:
    copy = tmp_path / "cache"
    shutil.copytree(killed_cache, copy)
    return copy


def test_a_killed_recording_leaves_its_raw_file(cache):
    assert len(raw_files(cache)) == 1


def test_finalize_recovers_the_ticks_a_killed_recording_flushed(cache):
    finalize(DAY, cache_dir=cache)

    assert [tick.ltp for tick in read_ticks("NSE", "SBIN", DAY, cache_dir=cache)] == [812.4, 812.5]


def test_a_killed_recordings_span_ends_at_its_last_tick_by_a_crash(cache):
    finalize(DAY, cache_dir=cache)

    (span,) = read_day("NSE", "SBIN", DAY, cache_dir=cache).spans

    assert (span.start_ns, span.stop_ns, span.ended_by) == (
        ist_ns("10:00:00"),
        ist_ns("10:00:00.500"),
        Ending.CRASH,
    )


def test_the_next_recording_recovers_a_killed_recordings_ticks_before_its_own(cache):
    clocks = Clocks(ist_ns("12:00:00"))
    feed, script = FakeFeed(), Script(clocks)
    script.at(ist_ns("12:00:01"), lambda: feed.push(quote("SBIN", 815.0)))

    record(
        feed,
        [SBIN],
        now=clocks.wall,
        monotonic=clocks.mono,
        wait=script,
        cache_dir=cache,
        stop_at_ns=ist_ns("12:00:00") + 10 * NS,
    )

    day = read_day("NSE", "SBIN", DAY, cache_dir=cache)
    assert (day.records["ltp"].tolist(), [span.ended_by for span in day.spans]) == (
        [812.4, 812.5, 815.0],
        [Ending.CRASH, Ending.STOP],
    )
