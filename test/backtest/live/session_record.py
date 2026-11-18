"""A live session run's record and its format (16-session-design section 8 unit 16;
15-phase2-plan step 1: "replaying a recorded session gives the same orders and fills
twice"). `session_recorder` writes records as sessions run on the fake; `session_replay`
replays them through fresh kernel sessions.

**What is recorded.** Everything a session gives each strategy's kernel session and live
series, in the order the loop gave it, and what the kernel gave back: every tick as a
price with its receive time, every bar and slice release, every piece of order news and
the result the kernel was given for it (a fill's increment, a rejection, a cancel, a
modify confirmed or refused), every control (entry locks, exits asked for, protection,
repricing, the sizing cap, an adopted fill, a restored state), each drive of the
strategy's decisions, and the commands the kernel issued.

**The format** ("openalgo-live-session", version 1): JSON lines, UTF-8, one object a line,
strict JSON (a number with no value, NaN, is written null; an infinite one is refused),
times UTC nanoseconds, prices rupees, quantities units. A distance (`bt.points`,
`bt.pct`, `bt.rupees`, `bt.ticks`, `bt.atr`) is written as a tagged object,
{"tag": "distance", "unit", "value", "period"}, and read back as one (`with_distances`).

- Line 1, the header: `format`, `version`, `day` (the IST trading day, ISO), `mode`
  ("live" or "sandbox": the mode OpenAlgo's analyzer switch was in at the start),
  `broker`, `strategies` (each `name` and `class`, "module:qualname", importable where
  the record is replayed), `arguments` (`bt.live`'s keyword arguments, JSON values and
  distances), `specified_ns` and `built_ns` (the clock's reads when the arguments were
  resolved and the plan built), `facts` (OpenAlgo's instrument facts: `symbol`,
  `exchange`, `lot_size`, `tick_size`, `freeze_quantity`) and `user_states` (each
  strategy's saved `self.state`).
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
- `stop` then `finished`: the close stopped the kernel session, and its whole run output
  (`output`, `finish()`'s tables: among them `orders`, `fills`, `trades`, `levels`,
  `equity`, `events` and `final_positions`); `end`, for a run that never closed (a
  crash): each strategy's `live_state`.

Every line but the header carries `strategy`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from openalgo.backtest.units import Distance

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
#: The columns of a `history` row, in order.
CANDLE_COLUMNS = ("ts_open", "open", "high", "low", "close", "volume")
_PRICES = CANDLE_COLUMNS[1:]
#: The tag of a `Distance` written as an object (module docstring).
DISTANCE_TAG = "distance"
_DISTANCE_KEYS = frozenset({"tag", "unit", "value", "period"})


class ReplayError(AssertionError):
    """A record that cannot be written or replayed (an unknown format, a read it does not
    hold, a warm-up its candles do not rebuild); the message says which."""


def plain(value: Any) -> Any:
    """`value` as a record keeps it (module docstring): tuples as lists, keys as text,
    NumPy as Python, NaN as None, a distance as its tagged object."""
    return _nan_as_none(json.loads(json.dumps(value, default=_json_default)))


def _nan_as_none(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _nan_as_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nan_as_none(item) for item in value]
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, Distance):
        return {"tag": DISTANCE_TAG, **dataclasses.asdict(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise ReplayError(f"a record keeps JSON values only, not {type(value).__name__}: {value!r}")


def with_distances(value: Any) -> Any:
    """`value` read from a record with each tagged distance (module docstring) a
    `Distance` again."""
    if isinstance(value, dict):
        if value.get("tag") == DISTANCE_TAG and set(value) == _DISTANCE_KEYS:
            return Distance(value["unit"], value["value"], value["period"])
        return {key: with_distances(item) for key, item in value.items()}
    if isinstance(value, list):
        return [with_distances(item) for item in value]
    return value


def failure_of(error: BaseException) -> dict[str, str]:
    """How a kernel input that raised is recorded: a venue refusal by its kind, any
    other error by its type."""
    kind = getattr(error, "kind", None)
    return {"refused": kind} if isinstance(kind, str) else {"error": type(error).__name__}


def candle_rows(frame: Any) -> list[list[float]]:
    """A frame of 1-minute candles in the bars layout as `history` rows."""
    values = frame.loc[:, CANDLE_COLUMNS].to_numpy().tolist()
    return [[int(row[0]), *map(float, row[1:])] for row in values]


def run_output(output: Mapping[str, Any]) -> dict[str, Any]:
    """A kernel session's whole run output (`finish()`) as a `finished` line keeps it."""
    return plain(output)


@dataclass
class SessionRecord:
    """One session run's record (module docstring): its header and its lines in order."""

    header: dict[str, Any]
    lines: list[dict[str, Any]] = field(default_factory=list)

    def dump(self, path: os.PathLike[str] | str) -> None:
        """Write the record to `path` as strict JSON lines. Raises ReplayError for a value
        JSON cannot hold (an infinite number)."""
        with open(path, "w", encoding="utf-8") as stream:
            for line in (self.header, *self.lines):
                try:
                    text = json.dumps(plain(line), allow_nan=False)
                except ValueError as error:
                    raise ReplayError(f"a record keeps JSON values only: {error}") from error
                stream.write(text + "\n")

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
            value = output_of(line)
            if value is not None:
                found.append((index, line["strategy"], line.get("call", line["kind"]), value))
        return found

    def kinds(self) -> set[str]:
        """The kinds of line the record holds."""
        return {line["kind"] for line in self.lines}

    def decisions_from(self, strategy: str) -> int:
        """The instant (UTC nanoseconds) `strategy`'s session decided from: the close of
        the first whole slice it watched. Raises ReplayError for a record with none."""
        for line in self.lines:
            if (line["strategy"], line["kind"]) == (strategy, "drive") and line["decide_from_ns"]:
                return line["decide_from_ns"]
        raise ReplayError(f"the record holds no decision of {strategy}")

    def bars(self, strategy: str) -> pd.DataFrame:
        """The 1-minute bars `strategy` decided on, as a backtest takes them (`data=`,
        indexed by their opens in IST): the warm-up's candles before today, then for each
        of today's slices the last bar its live series took (a catch-up candle, a bar
        built from ticks, or the broker candle that replaced it or filled a slice marked
        missing, such as the seed slice the session started in). A slice the series
        never took a bar of is not in it, as a backtest runs a minute it has no bar of."""
        rows: dict[int, list[float]] = {}
        for line in self.lines:
            if line["strategy"] != strategy:
                continue
            if line["kind"] == "history":
                rows.update((row[0], row) for row in line["base"])
            elif line["kind"] == "series" and line["call"] == "take":
                (bar,) = line["args"]
                rows[bar["ts_open_ns"]] = [bar["ts_open_ns"], *(bar[name] for name in _PRICES)]
        return _bars_frame([rows[opened] for opened in sorted(rows)])

    def kernel_bars(self, strategy: str) -> pd.DataFrame:
        """The 1-minute bars `strategy`'s kernel session was given (`add_bar`): the
        warm-up's candles before today, then today's catch-up candles and the bars the
        session built from ticks, indexed by their opens in IST. A slice released with no
        bar (the seed slice the session started in, a minute with no trade) is not in
        it."""
        rows = []
        for line in self.lines:
            if line["strategy"] != strategy:
                continue
            if line["kind"] == "history":
                rows += line["base"]
            elif line["kind"] == "bar":
                rows.append(line["args"][1])
        return _bars_frame(rows)


def _bars_frame(rows: list[list[float]]) -> pd.DataFrame:
    """`history` rows as a backtest's `data=`: indexed by their opens in IST."""
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    opens = pd.to_datetime(frame.pop("ts"), utc=True).dt.tz_convert("Asia/Kolkata")
    frame.index = pd.DatetimeIndex(opens, name="time")
    return frame


def output_of(line: Mapping[str, Any]) -> Any:
    """What a line says the kernel gave back, or None for a line that is not an output:
    the commands, the run output at the end, each drive's outcome and each input's
    result."""
    kind = line["kind"]
    if kind == "commands":
        return line["commands"]
    if kind == "finished":
        return line["output"]
    if kind == "end":
        return line["live_state"]
    if kind == "drive":
        return {"raised": line["raised"]}
    if "call" in line and kind != "series":
        return {name: line[name] for name in ("result", "refused", "error") if name in line}
    return None


__all__ = [
    "CANDLE_COLUMNS",
    "DISTANCE_TAG",
    "FORMAT",
    "INPUTS",
    "SERIES_CALLS",
    "VERSION",
    "ReplayError",
    "SessionRecord",
    "candle_rows",
    "failure_of",
    "output_of",
    "plain",
    "run_output",
    "with_distances",
]
