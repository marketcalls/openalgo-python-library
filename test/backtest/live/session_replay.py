"""A live session's record, and its replay through fresh kernel sessions (16-session-design
section 8 unit 16; 15-phase2-plan step 1: "replaying a recorded session gives the same
orders and fills twice").

**What is recorded.** Everything a session gives each strategy's kernel session and live
series, in the order the loop gave it, and what the kernel gave back: every tick as a
price with its receive time, every bar and slice release, every piece of order news and
the result the kernel was given for it (a fill's increment, a rejection, a cancel, a
modify confirmed or refused), every control (entry locks, exits asked for, protection,
repricing, the sizing cap, an adopted fill, a restored state), each drive of the
strategy's decisions, and the commands the kernel issued. `record_sessions` records every
session run in a test, one `SessionRecord` per run (a crash and a restart are two runs);
`replay` builds the strategy's kernel session, series and instance afresh from the
record's header and history, gives them the same calls in the same order, with no fake
and no threads, and returns what the kernel gave back (`ReplayOutcome.outputs`), to be
compared with the record's own (`SessionRecord.outputs`) command by command.

**The format** ("openalgo-live-session", version 1): JSON lines, UTF-8, one object a line,
times UTC nanoseconds, prices rupees, quantities units.

- Line 1, the header: `format`, `version`, `day` (the IST trading day, ISO), `broker`,
  `strategies` (each `name` and `class`, "module:qualname", importable where the record
  is replayed), `arguments` (`bt.live`'s keyword arguments, JSON values), `specified_ns`
  and `built_ns` (the clock's reads when the arguments were resolved and the plan built),
  `facts` (OpenAlgo's instrument facts: `symbol`, `exchange`, `lot_size`, `tick_size`,
  `freeze_quantity`) and `user_states` (each strategy's saved `self.state`).
- `history` (one per strategy): `base` and `today`, rows [ts_open_ns, open, high, low,
  close, volume] of the broker's 1-minute candles the plan's warm-up kept and of today's
  read when it was built: what a replay's data reads are answered from.
- A kernel input: `call` (the kernel's method), `args`, and `result`, or `refused` (a
  venue refusal's kind) or `error` (the error's type), with `kind` naming it: `bar`
  (add_bar), `release`, `tick` (push_tick: instrument, price, receive time), `fill`
  (apply_fill: order, units, price, receive time), `rejected`, `cancelled`, `modified`,
  `modify_refused`, `cancel_refused`, and `control` for the others.
- `series`: `call` ("take", "mark_missing", "no_trade") and `args` (a bar's fields, or a
  slice's open) of the strategy's live series.
- `news`: an order update as the loop took it (`order_id`, `status`, `filled`,
  `average_price`, `source`, `at_ns`); for the reader: the kernel inputs that follow it
  are what the loop made of it.
- `drive`: the loop ran the kernel's pauses with the strategy (`deciding`,
  `decide_from_ns`, `raised` when the strategy's code raised); `catch_up`: the plan's
  catch-up ran them deciding nothing.
- `commands`: what one `take_commands` gave, when it gave any.
- `stop` then `finished`: the close stopped the kernel session, and the `fills` of its
  run output; `end`, for a run that never closed (a crash): each strategy's `live_state`.

Every line but the header carries `strategy`. A sandbox session is saved in the same
format by running it with the recorder installed (`record_sessions`); its replay needs
the strategy classes importable and the data layer's cache (OPENALGO_BT_CACHE) an empty
folder, so the warm-up is read from the record.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import importlib
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from openalgo.backtest._live import _session_close, _session_start
from openalgo.backtest._live import plan as plan_module
from openalgo.backtest._live._plan_catch_up import StrategyPlan
from openalgo.backtest._live._plan_setup import strategy_setup
from openalgo.backtest._live._preflight_plan import InstrumentFacts
from openalgo.backtest._live._session_news import News
from openalgo.backtest._live._session_trader import StrategyFailed, Trader
from openalgo.backtest._live._types import LiveBar
from openalgo.backtest.calendar import default_calendar

FORMAT = "openalgo-live-session"
VERSION = 1
#: The kernel's input calls, each with the kind of line that records it.
INPUTS = MappingProxyType(
    {
        "add_bar": "bar",
        "release": "release",
        "push_tick": "tick",
        "apply_fill": "fill",
        "apply_reject": "rejected",
        "apply_cancel": "cancelled",
        "apply_modify": "modified",
        "apply_modify_refused": "modify_refused",
        "apply_cancel_refused": "cancel_refused",
        **dict.fromkeys(
            (
                "lock_entries",
                "unlock_entries",
                "request_exit",
                "cancel",
                "protect",
                "reprice_exit",
                "cap_sizing_equity",
                "apply_external_fill",
                "restore_live_state",
            ),
            "control",
        ),
    }
)
#: The live series' calls that change what the strategy sees.
SERIES_CALLS = ("take", "mark_missing", "no_trade")
#: The lines whose values a replay must give back.
OUTPUT_KINDS = frozenset({"commands", "finished", "end", "drive"})

# What the replay builds and drives a strategy's part with: the session's own functions,
# as they were before a test's recorder wrapped them.
_STRATEGY_PLAN = plan_module._strategy_plan
_STOP_ENGINE = _session_close._stop_engine
_DRIVE = Trader.drive
_CATCH_UP = StrategyPlan._drive


class ReplayError(AssertionError):
    """A record that cannot be replayed (an unknown format, a read it does not hold, a
    warm-up its candles do not rebuild); the message says which."""


def plain(value: Any) -> Any:
    """`value` as JSON gives it back: tuples as lists, keys as text, NumPy as Python."""
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise ReplayError(f"a record keeps JSON values only, not {type(value).__name__}: {value!r}")


@dataclass
class SessionRecord:
    """One session run's record (module docstring): its header and its lines in order."""

    header: dict[str, Any]
    lines: list[dict[str, Any]] = field(default_factory=list)

    def dump(self, path: os.PathLike[str] | str) -> None:
        """Write the record to `path` as JSON lines."""
        with open(path, "w", encoding="utf-8") as stream:
            for line in (self.header, *self.lines):
                stream.write(json.dumps(line, default=_json_default) + "\n")

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> SessionRecord:
        """Read a record written by `dump`. Raises ReplayError for another format or
        version."""
        with open(path, encoding="utf-8") as stream:
            header, *lines = (json.loads(text) for text in stream if text.strip())
        if (header.get("format"), header.get("version")) != (FORMAT, VERSION):
            raise ReplayError(f"not a {FORMAT} record of version {VERSION}: {header!r:.200}")
        return cls(header, lines)

    def outputs(self) -> list[tuple[Any, ...]]:
        """What the kernel gave back, line by line: (line, strategy, call or kind, value)."""
        found = []
        for index, line in enumerate(self.lines):
            value = _output_of(line)
            if value is not None:
                found.append((index, line["strategy"], line.get("call", line["kind"]), value))
        return found

    def kinds(self) -> set[str]:
        """The kinds of line the record holds."""
        return {line["kind"] for line in self.lines}


def _output_of(line: Mapping[str, Any]) -> Any:
    kind = line["kind"]
    if kind == "commands":
        return line["commands"]
    if kind == "finished":
        return line["fills"]
    if kind == "end":
        return line["live_state"]
    if kind == "drive":
        return {"raised": line["raised"]}
    if "call" in line and kind != "series":
        return {name: line[name] for name in ("result", "refused", "error") if name in line}
    return None


# ----- recording ----------------------------------------------------------------------------


class SessionRecorder:
    """Records each session run in a test (`record_sessions`), one `SessionRecord` a run."""

    def __init__(self) -> None:
        self.records: list[SessionRecord] = []
        self._engines: dict[int, dict[str, Any]] = {}
        self._quiet = 0

    @property
    def current(self) -> SessionRecord:
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
    """A kernel session whose input calls and commands are recorded (module docstring);
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
                outcome = _failure(error)
                recorder.line(INPUTS[call], name, call=call, args=list(args), **outcome)
                raise
            recorder.line(INPUTS[call], name, call=call, args=list(args), result=result)
            return result

        return recorded

    def _take(self, attribute: Any) -> Any:
        def taken() -> Any:
            commands = attribute()
            if commands:
                self._recorder.line("commands", self._name, commands=list(commands))
            return commands

        return taken


def _failure(error: BaseException) -> dict[str, str]:
    kind = getattr(error, "kind", None)
    return {"refused": kind} if isinstance(kind, str) else {"error": type(error).__name__}


def record_sessions(monkeypatch: Any) -> SessionRecorder:
    """Record every session run in the test from now on (module docstring)."""
    recorder = SessionRecorder()
    specify, build, broker = _session_start.specify, _session_start.build, _session_start._broker

    def specified(strategy: Any, *, clock: Any, **arguments: Any) -> Any:
        present = clock()
        spec = specify(strategy, clock=lambda: present, **arguments)
        classes = [{"name": s.name, "class": _path(s.strategy)} for s in spec.strategies]
        header = {
            "format": FORMAT,
            "version": VERSION,
            "day": spec.day.isoformat(),
            "strategies": classes,
            "arguments": plain(arguments),
            "specified_ns": present,
        }
        recorder.records.append(SessionRecord(header))
        return spec

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

    def planned(spec: Any, setup: Any, calendar: Any, saved: Any) -> Any:
        part = _STRATEGY_PLAN(spec, setup, calendar, saved)
        kept = setup.history.symbols.get(spec.symbol)
        base = _rows(kept.base) if kept is not None else []
        recorder.line(
            "history", spec.name, symbol=spec.symbol, exchange=spec.exchange, base=base,
            today=_rows(setup.today),
        )  # fmt: skip
        recorder.watch(spec.name, part)
        return part

    monkeypatch.setattr(_session_start, "specify", specified)
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

    def stopped(trader: Trader) -> Any:
        recorder.line("stop", trader.name)
        with recorder.unit():
            output, figures = stop(trader)
        recorder.line("finished", trader.name, fills=_fill_rows(output))
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


def _path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _rows(frame: Any) -> list[list[float]]:
    columns = ("ts_open", "open", "high", "low", "close", "volume")
    values = frame.loc[:, columns].to_numpy().tolist()
    return [[int(row[0]), *map(float, row[1:])] for row in values]


def _fill_rows(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The run output's fills table as rows."""
    table = output["fills"]
    names = list(table)
    columns = [plain(table[name]) for name in names]
    return [dict(zip(names, values, strict=True)) for values in zip(*columns, strict=True)]


# ----- the replay ---------------------------------------------------------------------------


@dataclass
class ReplayOutcome:
    """What a replay's kernels gave back (`SessionRecord.outputs`' shape), and the replay's
    strategy parts by name."""

    outputs: list[tuple[Any, ...]]
    parts: dict[str, StrategyPlan]


class _History:
    """The data reads a replay's plan makes, answered from the record's candles: the
    broker's name for `ping`, the candles of the dates asked for `history`."""

    api_key = "replay"

    def __init__(self, record: SessionRecord) -> None:
        self._broker = record.header.get("broker", "")
        self._rows: dict[tuple[str, str], list[list[float]]] = {}
        for line in record.lines:
            if line["kind"] == "history":
                key = (line["symbol"], line["exchange"])
                rows = {row[0]: row for row in (*line["base"], *line["today"])}
                self._rows[key] = [rows[ts] for ts in sorted(rows)]

    def _post(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if endpoint == "ping":
            return {"status": "success", "data": {"broker": self._broker, "message": "pong"}}
        if endpoint != "history":
            raise ReplayError(f"a replay answers ping and history only, not {endpoint}")
        first, last = (dt.date.fromisoformat(payload[name]) for name in ("start_date", "end_date"))
        rows = self._rows.get((payload["symbol"], payload["exchange"]), [])
        data = [
            dict(zip(("open", "high", "low", "close", "volume"), row[1:], strict=True))
            | {"timestamp": row[0] // 10**9, "oi": 0}
            for row in rows
            if first <= _ist_date(row[0]) <= last
        ]
        return {"status": "success", "data": data}


def _ist_date(ts_ns: int) -> dt.date:
    return (dt.datetime.fromtimestamp(ts_ns / 10**9, dt.UTC) + dt.timedelta(hours=5.5)).date()


def replay(record: SessionRecord) -> ReplayOutcome:
    """Replay `record` through fresh kernel sessions, series and strategies (module
    docstring), with no fake and no threads. Raises ReplayError when the record cannot be
    replayed."""
    header = record.header
    classes = [_resolve(each["class"]) for each in header["strategies"]]
    spec = plan_module.specify(classes, clock=lambda: header["specified_ns"], **header["arguments"])
    calendar = default_calendar()
    facts = [InstrumentFacts(**fact) for fact in header["facts"]]
    reads = _History(record)
    parts, traders = {}, {}
    for each in spec.strategies:
        setup = strategy_setup(each, reads, facts, header["built_ns"], calendar)
        part = _STRATEGY_PLAN(each, setup, calendar, header["user_states"].get(each.name))
        _check_history(record, each.name, part)
        parts[each.name] = part
        traders[each.name] = Trader(
            part, None, square_off_ns=0, close_ns=0, now=lambda: 0, event=lambda *a, **k: None
        )
    outputs = []
    stopped: dict[str, Any] = {}
    for index, line in enumerate(record.lines):
        value = _step(line, parts, traders, stopped)
        if value is not None:
            outputs.append((index, line["strategy"], line.get("call", line["kind"]), value))
    return ReplayOutcome(outputs, parts)


def _step(line: dict[str, Any], parts: dict, traders: dict, stopped: dict) -> Any:
    """Give one line to the replay's part; what the kernel gave back, when it is an
    output (`_output_of`)."""
    kind, name = line["kind"], line["strategy"]
    part, trader = parts[name], traders[name]
    if kind in ("history", "news"):
        return None
    if kind == "series":
        args = [LiveBar(**arg) if isinstance(arg, dict) else arg for arg in line["args"]]
        getattr(part.series, line["call"])(*args)
        return None
    if kind == "catch_up":
        _CATCH_UP(part)
        return None
    if kind == "drive":
        trader.decide_from_ns = line["decide_from_ns"]
        try:
            _DRIVE(trader, deciding=line["deciding"])
        except StrategyFailed:
            return {"raised": True}
        return {"raised": False}
    if kind == "commands":
        return plain(part.engine.take_commands())
    if kind == "stop":
        stopped[name] = _STOP_ENGINE(trader)[0]
        return None
    if kind == "finished":
        return _fill_rows(stopped.pop(name))
    if kind == "end":
        return plain(part.engine.live_state())
    return _input(part.engine, line)


def _input(engine: Any, line: Mapping[str, Any]) -> dict[str, Any]:
    args = list(line["args"])
    if line["call"] == "add_bar":
        args[1] = tuple(args[1])
    try:
        return {"result": plain(getattr(engine, line["call"])(*args))}
    except Exception as error:
        return _failure(error)


def _resolve(path: str) -> type:
    module, _, qualname = path.partition(":")
    found: Any = importlib.import_module(module)
    for name in qualname.split("."):
        found = getattr(found, name)
    return found


def _check_history(record: SessionRecord, name: str, part: StrategyPlan) -> None:
    """The replay's warm-up must be the session's: the record's candles rebuild it."""
    histories = (each for each in record.lines if each["kind"] == "history")
    line = next(each for each in histories if each["strategy"] == name)
    kept = part.setup.history.symbols.get(part.spec.symbol)
    if (_rows(kept.base) if kept is not None else []) != line["base"]:
        raise ReplayError(f"{name}: the record's candles do not rebuild the session's warm-up")


__all__ = [
    "FORMAT",
    "INPUTS",
    "OUTPUT_KINDS",
    "SERIES_CALLS",
    "VERSION",
    "ReplayError",
    "ReplayOutcome",
    "SessionRecord",
    "SessionRecorder",
    "plain",
    "record_sessions",
    "replay",
]
