"""Fault-suite sessions recorded as they ran on the fake OpenAlgo and replayed through fresh
kernel sessions with no fake and no threads (`session_replay`; 16-session-design section 8
unit 16; 15-phase2-plan step 1: "replaying a recorded session gives the same orders and
fills twice").

The sessions are `replay_samples`': a trailing stop, a part fill then a rejected stop, a
crash and a restart on the same journal, and a sandbox session whose warm-up reaches into
the days before and whose limits are distances. Each record is written to its file and
read back, then replayed twice: each replay gives, line by line, the commands the kernel
issued at each take of the loop (an empty take too, so each command is pinned to the input
after which the kernel issued it), its answer to every fill, rejection, cancel and modify,
every decision drive, and the kernel's whole run output at the close (or its state where a
run crashed), and these must equal the live run's and each other's. A replay reads its
warm-up from the record, whatever the data layer's cache holds, and writes nothing into
that cache.
"""

from __future__ import annotations

import logging
from typing import Any

from fake_openalgo.fixtures import *  # noqa: F403
from replay_fixtures import *  # noqa: F403
from replay_samples import recorded
from session_record import candle_rows
from session_replay import replay

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.data.cache import CACHE_ENV


def first_difference(left: list, right: list) -> Any:
    """The first output (line, strategy, call, value) where two runs differ, or None."""
    for one, other in zip(left, right, strict=False):
        if one != other:
            return one, other
    return None if len(left) == len(right) else (len(left), len(right))


def test_a_replay_gives_the_commands_and_fills_of_the_live_run_command_by_command(session):
    for record in session.records:
        outputs = replay(record).outputs
        assert any(output[2] == "commands" and output[3] for output in outputs)
        assert first_difference(outputs, record.outputs()) is None


def test_two_replays_give_the_same_commands_and_fills(session):
    one = [replay(record).outputs for record in session.records]
    two = [replay(record).outputs for record in session.records]
    assert [first_difference(a, b) for a, b in zip(one, two, strict=True)] == [None] * len(one)


def test_a_replay_writes_nothing_into_the_data_cache(session, monkeypatch, tmp_path):
    cache = tmp_path / "user-cache"
    cache.mkdir()
    monkeypatch.setenv(CACHE_ENV, str(cache))
    for record in session.records:
        replay(record)
    assert list(cache.iterdir()) == []


def test_a_replay_reads_its_warm_up_from_the_record_whatever_the_cache_holds(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # The data layer's cache holds the days before as the session read them; the record
    # says they were a rupee higher. The replay's warm-up is the record's.
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    (record,) = recorded("sandbox", make_fake_openalgo, monkeypatch, tmp_path).records
    (history,) = [line for line in record.lines if line["kind"] == "history"]
    assert history["base"] and any((tmp_path / "cache").iterdir())
    history["base"] = [[row[0], *(p + 1.0 for p in row[1:5]), row[5]] for row in history["base"]]
    part = replay(record).parts[history["strategy"]]
    kept = part.setup.history.symbols[history["symbol"]].base
    assert candle_rows(kept) == history["base"]


def test_a_replay_finds_a_command_the_record_says_the_kernel_issued_otherwise(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # The check itself: a record whose first command names one unit more than the kernel
    # issued differs from its replay at that command.
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    (record,) = recorded("trailing", make_fake_openalgo, monkeypatch, tmp_path).records
    at, line = next(
        (i, line)
        for i, line in enumerate(record.lines)
        if line["kind"] == "commands" and line["commands"]
    )
    line["commands"][0]["quantity_units"] += 1
    live, replayed = first_difference(record.outputs(), replay(record).outputs)
    assert (live[0], live[2], replayed[0]) == (at, "commands", at)


def test_a_replay_finds_a_command_the_record_says_the_kernel_issued_a_take_earlier(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # Every take of the loop is recorded, an empty one too: a record whose first command
    # is moved to the take before the one that gave it differs from its replay at that
    # earlier take, where the replay's kernel had issued nothing yet.
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    (record,) = recorded("trailing", make_fake_openalgo, monkeypatch, tmp_path).records
    takes = [at for at, line in enumerate(record.lines) if line["kind"] == "commands"]
    gave = next(at for at in takes if record.lines[at]["commands"])
    earlier = max((at for at in takes if at < gave), default=None)
    assert earlier is not None, "the takes before the first command are recorded"
    lines = record.lines
    lines[earlier]["commands"], lines[gave]["commands"] = lines[gave]["commands"], []
    live, replayed = first_difference(record.outputs(), replay(record).outputs)
    assert (live[0], live[2], replayed[0], replayed[3]) == (earlier, "commands", earlier, [])


def test_a_replay_finds_a_working_stop_the_record_says_the_run_ended_with_otherwise(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # The run output at the close is compared whole: a record whose working stop rests a
    # tick higher than the kernel left it differs from its replay at the `finished` line.
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    (record,) = recorded("trailing", make_fake_openalgo, monkeypatch, tmp_path).records
    at, line = next((i, line) for i, line in enumerate(record.lines) if line["kind"] == "finished")
    orders = line["output"]["orders"]
    orders["trigger_rupees"][orders["status"].index("Submitted")] += 0.05
    live, replayed = first_difference(record.outputs(), replay(record).outputs)
    assert (live[0], live[2], replayed[0]) == (at, "finished", at)
