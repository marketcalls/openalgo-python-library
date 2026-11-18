"""A recorded live session replayed through fresh kernel sessions (16-session-design
section 8 unit 16; 15-phase2-plan step 1: "replaying a recorded session gives the same
orders and fills twice").

`replay` builds each strategy's kernel session, live series and instance afresh from a
record's header and history (`session_record`), gives them the record's calls in the
record's order, with no fake and no threads, and returns what the kernel gave back
(`ReplayOutcome.outputs`), to be compared with the record's own (`SessionRecord.outputs`)
command by command. Its data reads (the plan's warm-up and today's candles) are answered
from the record's candles, through a data cache (OPENALGO_BT_CACHE) of its own, an empty
folder for the replay's duration: what the caller's cache holds is neither read nor
written.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from session_record import (
    ReplayError,
    SessionRecord,
    candle_rows,
    failure_of,
    plain,
    run_output,
    with_distances,
)

from openalgo.backtest._live import _session_close
from openalgo.backtest._live import plan as plan_module
from openalgo.backtest._live._plan_catch_up import StrategyPlan
from openalgo.backtest._live._plan_setup import strategy_setup
from openalgo.backtest._live._preflight_plan import InstrumentFacts
from openalgo.backtest._live._session_trader import StrategyFailed, Trader
from openalgo.backtest._live._types import LiveBar
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.data.cache import CACHE_ENV

# What the replay builds and drives a strategy's part with: the session's own functions,
# as they were before a test's recorder wrapped them (this module is imported when a
# test module is collected, before any test runs).
_STRATEGY_PLAN = plan_module._strategy_plan
_STOP_ENGINE = _session_close._stop_engine
_DRIVE = Trader.drive
_CATCH_UP = StrategyPlan._drive


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
    with _own_cache():
        parts = _parts(record)
        traders = {
            name: Trader(
                part, None, square_off_ns=0, close_ns=0, now=lambda: 0, event=lambda *a, **k: None
            )
            for name, part in parts.items()
        }
        outputs = []
        stopped: dict[str, Any] = {}
        for index, line in enumerate(record.lines):
            value = _step(line, parts, traders, stopped)
            if value is not None:
                outputs.append((index, line["strategy"], line.get("call", line["kind"]), value))
    return ReplayOutcome(outputs, parts)


@contextlib.contextmanager
def _own_cache() -> Iterator[None]:
    """The data layer's cache an empty folder of the replay's own while it runs: its
    warm-up is read from the record whatever the caller's cache holds, and nothing is
    written into that cache."""
    before = os.environ.get(CACHE_ENV)
    with tempfile.TemporaryDirectory(prefix="replay-cache-", ignore_cleanup_errors=True) as folder:
        os.environ[CACHE_ENV] = folder
        try:
            yield
        finally:
            if before is None:
                del os.environ[CACHE_ENV]
            else:
                os.environ[CACHE_ENV] = before


def _parts(record: SessionRecord) -> dict[str, StrategyPlan]:
    """Each strategy's part of the plan, built afresh from the record's header and
    candles, by name. Raises ReplayError when the candles do not rebuild a warm-up."""
    header = record.header
    classes = [_resolve(each["class"]) for each in header["strategies"]]
    arguments = with_distances(header["arguments"])
    spec = plan_module.specify(classes, clock=lambda: header["specified_ns"], **arguments)
    calendar = default_calendar()
    facts = [InstrumentFacts(**fact) for fact in header["facts"]]
    reads = _History(record)
    parts = {}
    for each in spec.strategies:
        setup = strategy_setup(each, reads, facts, header["built_ns"], calendar)
        part = _STRATEGY_PLAN(each, setup, calendar, header["user_states"].get(each.name))
        _check_history(record, each.name, part)
        parts[each.name] = part
    return parts


def _step(line: dict[str, Any], parts: dict, traders: dict, stopped: dict) -> Any:
    """Give one line to the replay's part; what the kernel gave back, when it is an
    output (`session_record.output_of`)."""
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
        return run_output(stopped.pop(name))
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
        return failure_of(error)


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
    if (candle_rows(kept.base) if kept is not None else []) != line["base"]:
        raise ReplayError(f"{name}: the record's candles do not rebuild the session's warm-up")


__all__ = ["ReplayOutcome", "replay"]
