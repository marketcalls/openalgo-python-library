"""Fault-suite sessions recorded on the fake OpenAlgo for the replay's tests
(`test_session_replay.py`, `test_session_record.py`; 16-session-design section 8 unit 16;
15-phase2-plan step 1).

Four sessions of `stops_samples`' market (SBIN on NSE from 14:40:05 IST, trading 10 units
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
- **Sandbox.** The trailing session with OpenAlgo's analyzer switch on, its strategy
  (`AveragedTrails`) holding a 400-bar average whose warm-up reaches into the sessions
  before today, its limits given as distances (`max_daily_loss=bt.rupees(5000)`,
  `drawdown_halt=bt.pct(10)`) and its charges as a brokerage preset
  (`costs=bt.costs.flat(20)`, for a broker with no preset of its name), as a recorded
  sandbox day may have them.

`recorded` runs one under the recorder and writes each of its records to a file;
`replay_fixtures.session` is the pytest fixture of all four. Each session's `defining` check asserts
the lines that make it the session it is, so a change to the fake's faults that makes a
session degenerate fails its record's tests. Expected values come from the scenario.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from restart_samples import TrailsWhenFlat, crash_at, crashing, restart, run_to_crash
from session_record import SessionRecord
from session_recorder import record_sessions
from session_samples import ist
from stops_samples import StopHold, stop_world
from test_stops_exits import Trails, rising

import openalgo.backtest as bt
from openalgo import ta

#: The bars of `AveragedTrails`' average: more than today holds before 14:40 (325).
WARM_UP_BARS = 400
#: The sandbox session's `bt.live` arguments given as objects: its limits as distances,
#: its charges as a brokerage preset (Rs 20 an executed order).
OBJECT_ARGUMENTS = {
    "max_daily_loss": bt.rupees(5_000),
    "drawdown_halt": bt.pct(10),
    "costs": bt.costs.flat(20),
}


class AveragedTrails(Trails):
    """`Trails` holding a `WARM_UP_BARS` average it never reads: its warm-up reaches into
    the sessions before today."""

    def init(self):
        super().init()
        self.average = self.I(ta.sma, self.close, WARM_UP_BARS)


def trailing(world: Any, strategy: Any = Trails, **arguments: Any) -> None:
    """The trailing session (module docstring)."""
    rising(world, start=ist(14, 43), step=0.5, until=ist(14, 46))
    world.stop_at(ist(14, 48))
    world.run(strategy, on_stop="keep", **arguments)


def part_fill_then_rejected_stop(world: Any) -> None:
    """The part fill and the rejected stop (module docstring)."""
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
    """The crash and the restart (module docstring)."""
    crash_at(world, ist(14, 42, 30))
    run_to_crash(world, TrailsWhenFlat)
    rising(world, start=ist(14, 43), step=0.5, until=ist(14, 46))
    restart(world, TrailsWhenFlat, stop_ns=ist(14, 48), on_stop="keep")


def sandbox(world: Any) -> None:
    """The sandbox session (module docstring)."""
    trailing(world, AveragedTrails, **OBJECT_ARGUMENTS)


# ----- the lines that define each session ----------------------------------------------------


def calls(record: SessionRecord, call: str) -> list[list[Any]]:
    """The arguments of each kernel input `call` the record holds, in order."""
    return [line["args"] for line in record.lines if line.get("call") == call]


def commands(record: SessionRecord) -> list[dict[str, Any]]:
    """Every command the kernel issued in the record, in order."""
    return [c for line in record.lines if line["kind"] == "commands" for c in line["commands"]]


def modifies_its_stop_up(records: list[SessionRecord]) -> None:
    """The trailing stop: the broker's stop is modified up at least twice, each modify
    confirmed to the kernel."""
    (record,) = records
    triggers = [c["trigger_price"] for c in commands(record) if c["command"] == "modify"]
    assert len(triggers) >= 2 and triggers == sorted(set(triggers)), triggers
    assert len(calls(record, "apply_modify")) == len(triggers)


def fills_in_parts_then_exits_on_the_rejected_stop(records: list[SessionRecord]) -> None:
    """The entry fills 4 then 6; the stop placed for 4 is cancelled by the other client;
    the one placed again for 10 is rejected; the leg exits at market for that reason."""
    (record,) = records
    placed = [c for c in commands(record) if c["command"] == "place"]
    assert len(placed) == 4, placed
    entry, first_stop, second_stop, exit_ = placed
    fills = [args for args in calls(record, "apply_fill") if args[0] == entry["order_id"]]
    assert [units for _, units, *_ in fills] == [4.0, 6.0]
    assert [(c["role"], c["quantity_units"]) for c in (first_stop, second_stop)] == [
        ("leg_stop", 4.0),
        ("leg_stop", 10.0),
    ]
    assert calls(record, "apply_cancel") == [[first_stop["order_id"]]]
    assert [args[0] for args in calls(record, "apply_reject")] == [second_stop["order_id"]]
    assert (exit_["role"], exit_["reason"], exit_["quantity_units"]) == (
        "risk_exit",
        "stop_rejected",
        10.0,
    )


def resumes_the_leg_after_the_crash(records: list[SessionRecord]) -> None:
    """The first run ends where it crashed, with no close; the second restores the
    kernel's state from the journal, protects the leg and trails its stop, entering
    nothing anew."""
    assert len(records) == 2, "a crash and a restart are two runs, two records"
    crashed, restarted = records
    assert crashed.lines[-1]["kind"] == "end" and "finished" not in crashed.kinds()
    assert len(calls(restarted, "restore_live_state")) == 1
    assert calls(restarted, "protect")
    assert not [c for c in commands(restarted) if c["command"] == "place" and c["role"] == "entry"]
    assert [c for c in commands(restarted) if c["command"] == "modify"]


@dataclass(frozen=True)
class Scenario:
    """A session to record: how it is played, the kinds of line its records hold between
    them, the check of its defining lines, the units of SBIN it holds at its end, the
    mode OpenAlgo is in, and the `bt.live` arguments it gives as objects."""

    play: Callable[[Any], None]
    kinds: frozenset[str]
    defining: Callable[[list[SessionRecord]], None]
    held_at_end: float
    analyzer: bool = False
    objects: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        """The mode the records' header must name."""
        return "sandbox" if self.analyzer else "live"


SCENARIOS = {
    "trailing": Scenario(
        trailing,
        frozenset({"tick", "fill", "modified", "commands", "drive", "finished"}),
        modifies_its_stop_up,
        held_at_end=10.0,
    ),
    "part fill, then a rejected stop": Scenario(
        part_fill_then_rejected_stop,
        frozenset({"fill", "modified", "rejected", "cancelled", "commands", "finished"}),
        fills_in_parts_then_exits_on_the_rejected_stop,
        held_at_end=0.0,
    ),
    "crash and restart": Scenario(
        crash_and_restart,
        frozenset({"tick", "fill", "modified", "control", "commands", "end", "finished"}),
        resumes_the_leg_after_the_crash,
        held_at_end=10.0,
    ),
    "sandbox": Scenario(
        sandbox,
        frozenset({"tick", "fill", "modified", "commands", "drive", "finished"}),
        modifies_its_stop_up,
        held_at_end=10.0,
        analyzer=True,
        objects=OBJECT_ARGUMENTS,
    ),
}


@dataclass
class Recorded:
    """A scenario's records, each written to its file (`paths`) and read back."""

    scenario: Scenario
    records: list[SessionRecord]
    paths: list[Path]


def recorded(name: str, make_fake_openalgo: Any, monkeypatch: Any, tmp_path: Path) -> Recorded:
    """Run the scenario `name` on the fake under the recorder (module docstring)."""
    scenario = SCENARIOS[name]
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=scenario.analyzer)
    crashing(monkeypatch)
    recorder = record_sessions(monkeypatch)
    scenario.play(world)
    records, paths = [], []
    for at, record in enumerate(recorder.finish()):
        path = tmp_path / f"session-{at}.jsonl"
        record.dump(path)
        records.append(SessionRecord.load(path))
        paths.append(path)
    return Recorded(scenario, records, paths)


__all__ = [
    "OBJECT_ARGUMENTS",
    "SCENARIOS",
    "WARM_UP_BARS",
    "AveragedTrails",
    "Recorded",
    "Scenario",
    "calls",
    "commands",
    "recorded",
    "trailing",
]
