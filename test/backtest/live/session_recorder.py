"""Records each live session run in a test as it runs on the fake OpenAlgo, one
`SessionRecord` a run, in the format `session_record` describes (16-session-design
section 8 unit 16).

`record_sessions(monkeypatch)` installs the recorder for the rest of the test: it wraps
the session's start (the arguments, the mode the guard adopts, the broker's name, the
plan's build and each strategy's part of it), each strategy's kernel session and live
series, the loop's decision drives, the catch-up's, the close's stop and the order news.
A crash and a restart are two runs, two records. `SessionRecorder.finish` ends each run
that never closed with its kernels' states.

A sandbox session is saved in the same format by running it with the recorder installed;
its replay (`session_replay`) needs the strategy classes importable.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any

from record_objects import KEPT
from session_record import (
    FORMAT,
    INPUTS,
    SERIES_CALLS,
    VERSION,
    ReplayError,
    SessionRecord,
    candle_rows,
    failure_of,
    plain,
    run_output,
)

from openalgo.backtest._live import _session_close, _session_start
from openalgo.backtest._live import plan as plan_module
from openalgo.backtest._live._plan_catch_up import StrategyPlan
from openalgo.backtest._live._session_news import News
from openalgo.backtest._live._session_trader import StrategyFailed, Trader
from openalgo.backtest._live._types import LiveBar
from openalgo.backtest._live.guard import ModeGuard


class SessionRecorder:
    """Records each session run in a test (`record_sessions`), one `SessionRecord` a run."""

    def __init__(self) -> None:
        self.records: list[SessionRecord] = []
        self._engines: dict[int, dict[str, Any]] = {}
        self._quiet = 0

    @property
    def current(self) -> SessionRecord:
        """The record of the run under way (the last one started)."""
        return self.records[-1]

    def line(self, kind: str, strategy: str, **fields: Any) -> dict[str, Any]:
        """Add a line to the current run's record, unless a unit is under way."""
        entry = {"kind": kind, "strategy": strategy, **plain(fields)}
        if not self._quiet:
            self.current.lines.append(entry)
        return entry

    @contextlib.contextmanager
    def unit(self) -> Iterator[None]:
        """What runs inside is one line already: its own calls are not recorded."""
        self._quiet += 1
        try:
            yield
        finally:
            self._quiet -= 1

    def finish(self) -> list[SessionRecord]:
        """The records, each run that never closed ended with its kernels' states."""
        for at, record in enumerate(self.records):
            if any(line["kind"] == "finished" for line in record.lines):
                continue
            if record.lines and record.lines[-1]["kind"] == "end":
                continue
            for name, engine in self._engines.get(at, {}).items():
                state = engine.live_state()
                record.lines.append({"kind": "end", "strategy": name, "live_state": plain(state)})
        return self.records

    def watch(self, name: str, part: StrategyPlan) -> None:
        """Record what `part`'s kernel session and live series are given from now on."""
        self._engines.setdefault(len(self.records) - 1, {})[name] = part.engine
        part.engine = _RecordingEngine(part.engine, self, name)
        series = part.series
        for call in SERIES_CALLS:
            original = getattr(series, call)

            def recorded(*args: Any, original: Any = original, call: str = call) -> Any:
                shown = [dataclasses.asdict(a) if isinstance(a, LiveBar) else a for a in args]
                self.line("series", name, call=call, args=shown)
                return original(*args)

            setattr(series, call, recorded)


class _RecordingEngine:
    """A kernel session whose input calls and commands are recorded (`session_record`);
    every other attribute is the kernel's own."""

    def __init__(self, engine: Any, recorder: SessionRecorder, name: str) -> None:
        self._engine, self._recorder, self._name = engine, recorder, name

    def __getattr__(self, call: str) -> Any:
        attribute = getattr(self._engine, call)
        if call == "take_commands":
            return self._take(attribute)
        if call not in INPUTS:
            return attribute
        recorder, name = self._recorder, self._name

        def recorded(*args: Any) -> Any:
            try:
                result = attribute(*args)
            except Exception as error:
                outcome = failure_of(error)
                recorder.line(INPUTS[call], name, call=call, args=list(args), **outcome)
                raise
            recorder.line(INPUTS[call], name, call=call, args=list(args), result=result)
            return result

        return recorded

    def _take(self, attribute: Any) -> Any:
        def taken() -> Any:
            commands = attribute()
            self._recorder.line("commands", self._name, commands=list(commands))
            return commands

        return taken


def record_sessions(monkeypatch: Any) -> SessionRecorder:
    """Record every session run in the test from now on (module docstring)."""
    recorder = SessionRecorder()
    specify, build, broker = _session_start.specify, _session_start.build, _session_start._broker
    start = ModeGuard.start

    def specified(strategy: Any, *, clock: Any, **arguments: Any) -> Any:
        present = clock()
        spec = specify(strategy, clock=lambda: present, **arguments)
        classes = [{"name": s.name, "class": _path(s.strategy)} for s in spec.strategies]
        header = {
            "format": FORMAT,
            "version": VERSION,
            "day": spec.day.isoformat(),
            "strategies": classes,
            "arguments": _kept(arguments),
            "specified_ns": present,
        }
        recorder.records.append(SessionRecord(header))
        return spec

    def started(guard: ModeGuard) -> Any:
        mode = start(guard)
        recorder.current.header["mode"] = mode.value
        return mode

    def named(client: Any, account: Any) -> str:
        found = broker(client, account)
        recorder.current.header["broker"] = found
        return found

    def built(spec: Any, client: Any, *, facts: Any, clock: Any, user_states: Any = None) -> Any:
        reads: list[int] = []

        def read() -> int:
            reads.append(clock())
            return reads[-1]

        header = recorder.current.header
        header["facts"] = [dataclasses.asdict(fact) for fact in facts]
        header["user_states"] = plain(user_states or {})
        plan = build(spec, client, facts=facts, clock=read, user_states=user_states)
        header["built_ns"] = reads[0]
        return plan

    strategy_plan = plan_module._strategy_plan

    def planned(spec: Any, setup: Any, calendar: Any, saved: Any) -> Any:
        part = strategy_plan(spec, setup, calendar, saved)
        kept = setup.history.symbols.get(spec.symbol)
        base = candle_rows(kept.base) if kept is not None else []
        recorder.line(
            "history", spec.name, symbol=spec.symbol, exchange=spec.exchange, base=base,
            today=candle_rows(setup.today),
        )  # fmt: skip
        recorder.watch(spec.name, part)
        return part

    monkeypatch.setattr(_session_start, "specify", specified)
    monkeypatch.setattr(ModeGuard, "start", started)
    monkeypatch.setattr(_session_start, "_broker", named)
    monkeypatch.setattr(_session_start, "build", built)
    monkeypatch.setattr(plan_module, "_strategy_plan", planned)
    _record_units(recorder, monkeypatch)
    return recorder


def _record_units(recorder: SessionRecorder, monkeypatch: Any) -> None:
    """Record the drives, the catch-up's, the close's stop and the order news."""
    drive, catch_up, stop = Trader.drive, StrategyPlan._drive, _session_close._stop_engine
    apply_update = News.apply_update

    def driven(trader: Trader, *, deciding: bool) -> None:
        entry = recorder.line(
            "drive", trader.name, deciding=deciding, decide_from_ns=trader.decide_from_ns,
            raised=False,
        )  # fmt: skip
        with recorder.unit():
            try:
                drive(trader, deciding=deciding)
            except StrategyFailed:
                entry["raised"] = True
                raise

    def caught_up(part: StrategyPlan) -> None:
        recorder.line("catch_up", part.spec.name)
        with recorder.unit():
            catch_up(part)

    def stopped(trader: Trader, carry: Any = None) -> Any:
        recorder.line("stop", trader.name)
        with recorder.unit():
            output, figures = stop(trader, carry)
        recorder.line("finished", trader.name, output=run_output(output))
        return output, figures

    def news(handler: News, trader: Trader, update: Any) -> Any:
        recorder.line(
            "news", trader.name, order_id=update.order_id, status=update.status.value,
            filled=update.filled_quantity, average_price=update.average_price,
            source=update.source, at_ns=update.ts_recv_ns,
        )  # fmt: skip
        return apply_update(handler, trader, update)

    monkeypatch.setattr(Trader, "drive", driven)
    monkeypatch.setattr(StrategyPlan, "_drive", caught_up)
    monkeypatch.setattr(_session_close, "_stop_engine", stopped)
    monkeypatch.setattr(News, "apply_update", news)


def _kept(arguments: dict[str, Any]) -> dict[str, Any]:
    """`bt.live`'s keyword arguments as the header keeps them. Raises ReplayError naming
    an argument a record cannot keep, and what to pass instead."""
    kept = {}
    for name, value in arguments.items():
        try:
            kept[name] = plain(value)
        except ReplayError as error:
            raise ReplayError(
                f"bt.live's {name}= cannot be kept in a session record ({error}): pass {KEPT}"
            ) from error
    return kept


def _path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


__all__ = ["SessionRecorder", "record_sessions"]
