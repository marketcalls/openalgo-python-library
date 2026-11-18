"""Fault-suite sessions recorded as they ran on the fake OpenAlgo and replayed through fresh
kernel sessions with no fake and no threads (`session_replay`; 16-session-design section 8
unit 16; 15-phase2-plan step 1: "replaying a recorded session gives the same orders and
fills twice").

Three sessions of `stops_samples`' market (SBIN on NSE from 14:40:05 IST, trading 10 units
every 10 s):

- **Trailing.** `Trails` buys 10 SBIN at 14:42:01 with a stop 3 points below that trails
  3 points below the best price; SBIN rises 0.50 a trade from 14:43 to 14:46, so the
  broker's stop is modified up (F25). Ctrl+C at 14:48 keeps the leg.
- **A part fill and a rejected stop.** The entry fills 4 units at once and 6 at the next
  trade, its stop placed for 4 and modified to 10 (F17); at 14:43 another client cancels
  the stop, the session places it again (F79), and the broker rejects that one right after
  accepting it, so the leg exits at market (F11).
- **A crash and a restart.** `TrailsWhenFlat` is holding 10 SBIN with its stop when the
  session crashes at 14:42:30; SBIN rises from 14:43 and a new session on the same journal
  resumes the leg and trails its stop (F50). Two runs, two records.

Each record is written to its file and read back, then replayed twice: each replay gives,
line by line, the commands the kernel issued, its answer to every fill, rejection, cancel
and modify, every decision drive, and the run's fills at its end, and these must equal
the live run's and each other's.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from restart_samples import TrailsWhenFlat, crash_at, crashing, restart, run_to_crash
from session_record import FORMAT, VERSION, ReplayError, SessionRecord
from session_recorder import record_sessions
from session_replay import replay
from session_samples import ist
from stops_samples import StopHold, stop_world
from test_stops_exits import Trails, rising

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.data.cache import CACHE_ENV


def trailing(world: Any) -> None:
    rising(world, start=ist(14, 43), step=0.5, until=ist(14, 46))
    world.stop_at(ist(14, 48))
    world.run(Trails, on_stop="keep")


def part_fill_then_rejected_stop(world: Any) -> None:
    other = world.fake.client()

    def cancel_the_stop() -> None:
        world.fake.reject_next("trigger invalid", price_type="SL-M")
        stop = next(o for o in world.orders() if o.price_type == "SL-M")
        other.cancelorder(order_id=stop.order_id, strategy="StopHold")

    world.fake.split_next_fill(4)
    world.script.at(ist(14, 43), cancel_the_stop)
    world.stop_at(ist(14, 45))
    world.run(StopHold, on_stop="keep")


def crash_and_restart(world: Any) -> None:
    crash_at(world, ist(14, 42, 30))
    run_to_crash(world, TrailsWhenFlat)
    rising(world, start=ist(14, 43), step=0.5, until=ist(14, 46))
    restart(world, TrailsWhenFlat, stop_ns=ist(14, 48), on_stop="keep")


#: Each session, and the kinds of line its records must hold between them.
SESSIONS = {
    "trailing": (trailing, {"tick", "fill", "modified", "commands", "drive", "finished"}),
    "part fill, then a rejected stop": (
        part_fill_then_rejected_stop,
        {"fill", "modified", "rejected", "cancelled", "commands", "finished"},
    ),
    "crash and restart": (
        crash_and_restart,
        {"tick", "fill", "modified", "control", "commands", "end", "finished"},
    ),
}


@pytest.fixture(params=list(SESSIONS), ids=list(SESSIONS))
def session(request, make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """The session the parameter names, run on the fake under the recorder: (its records,
    each written to a file and read back; the kinds of line they must hold)."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    play, kinds = SESSIONS[request.param]
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    crashing(monkeypatch)
    recorder = record_sessions(monkeypatch)
    play(world)
    records = []
    for at, record in enumerate(recorder.finish()):
        path = tmp_path / f"session-{at}.jsonl"
        record.dump(path)
        records.append(SessionRecord.load(path))
    return records, kinds


def replayed(records: list[SessionRecord], monkeypatch: Any, folder: Any) -> list[list]:
    """Each record's replay outputs, the data layer's cache an empty `folder`, so the
    warm-up is read from the record."""
    monkeypatch.setenv(CACHE_ENV, str(folder))
    return [replay(record).outputs for record in records]


def first_difference(left: list, right: list) -> Any:
    """The first output (line, strategy, call, value) where two runs differ, or None."""
    for one, other in zip(left, right, strict=False):
        if one != other:
            return one, other
    return None if len(left) == len(right) else (len(left), len(right))


def test_a_record_holds_the_sessions_ticks_news_results_and_commands(session):
    records, kinds = session
    held = set().union(*(record.kinds() for record in records))
    assert kinds | {"news", "history", "series", "bar", "release"} <= held
    assert all(record.header["format"] == FORMAT for record in records)


def test_a_replay_gives_the_commands_and_fills_of_the_live_run_command_by_command(
    session, monkeypatch, tmp_path
):
    records, _ = session
    (first,) = [replayed(records, monkeypatch, tmp_path / "replay-1")]
    for record, outputs in zip(records, first, strict=True):
        assert sum(1 for output in outputs if output[2] == "commands") > 0
        assert first_difference(outputs, record.outputs()) is None


def test_two_replays_give_the_same_commands_and_fills(session, monkeypatch, tmp_path):
    records, _ = session
    one = replayed(records, monkeypatch, tmp_path / "replay-1")
    two = replayed(records, monkeypatch, tmp_path / "replay-2")
    assert [first_difference(a, b) for a, b in zip(one, two, strict=True)] == [None] * len(one)


def test_a_replay_finds_a_command_the_record_says_the_kernel_issued_otherwise(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The check itself: a record whose first command names one unit more than the kernel
    # issued differs from its replay at that command.
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    recorder = record_sessions(monkeypatch)
    trailing(world)
    (record,) = recorder.finish()
    at, line = next((i, line) for i, line in enumerate(record.lines) if line["kind"] == "commands")
    line["commands"][0]["quantity_units"] += 1
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "replay"))
    live, replayed_ = first_difference(record.outputs(), replay(record).outputs)
    assert (live[0], live[2], replayed_[0]) == (at, "commands", at)


def test_a_record_of_another_format_or_version_is_refused(tmp_path):
    path = tmp_path / "other.jsonl"
    path.write_text(f'{{"format": "{FORMAT}", "version": {VERSION + 1}}}\n', encoding="utf-8")
    with pytest.raises(ReplayError, match="not a openalgo-live-session record of version 1"):
        SessionRecord.load(path)
