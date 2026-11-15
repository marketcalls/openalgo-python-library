"""The session logs of a live session (08-live-trading section 8): a daily log file and a
structured event stream, one of each per IST trading day.

`configure_session_logs(folder, day, now=...)` opens, under `folder`:

- `<day>.log`: every record of the `openalgo.backtest.live` logger at INFO and above,
  one line each, stamped by the session's clock in IST:
  `2026-11-16 09:15:00.250 IST WARNING alert feed_dead: LIVE (orb): no tick for 30 s`.
- `<day>.events.jsonl`: one JSON object per line, written by `SessionLogs.event`: the
  event's time in UTC nanoseconds (`ts_ns`), its `kind`, then its fields.

Both files are appended to, so a session restarted on the same day continues them. While
the logs are open the logger's level is lowered to INFO if it was higher (else INFO
records never reach a handler), and restored by `close`; records still propagate to the
handlers the program configured.

The event stream never holds an API key: a field whose name ends like a key, token,
secret or password, at any depth, is written as "[redacted]" with a warning naming the
field. Values are written as JSON: an enum by its value, a date by its ISO text, a
dataclass and a mapping as an object, a sequence as a list, a path as its text, and a
number that is not finite as null; anything else is written as its repr with a warning,
so a log write never stops a live session.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import logging
import math
import numbers
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Any, TextIO

from .._calendar_convert import IST, DateLike, from_day, to_day
from ..errors import ConfigError, LiveError
from ._types import LOGGER_NAME, Now

#: The least level the daily log file holds.
LOG_FILE_LEVEL = logging.INFO
#: What a field named like a secret is written as.
REDACTED = "[redacted]"
#: The field of every event line that holds its time, UTC nanoseconds.
TIME_FIELD = "ts_ns"

# How a field's name ends when its value may be a credential, lower case, compared
# after dropping "-" and "_": api_key, apikey, access-token, client_secret, ...
_SECRET_ENDINGS = ("key", "token", "secret", "password", "passwd", "authorization", "totp")
# The handler's name marks it as the session file handler, so a second configuration of
# the same logger is refused without module state.
_HANDLER_NAME = "openalgo.backtest.live.session_logs"
_LINE_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_NANOS_PER_MICRO = 1_000
_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)

logger = logging.getLogger(LOGGER_NAME)


class SessionLogs:
    """The open session logs of one IST day. Made by `configure_session_logs`; also a
    context manager that closes them.

    `log_path` is the daily log file and `events_path` the event stream. `event` is
    called from the session loop's thread.
    """

    def __init__(
        self,
        *,
        log_path: Path,
        events_path: Path,
        handler: logging.Handler,
        events: TextIO,
        now: Now,
        restore_level: int | None,
    ) -> None:
        self._log_path = log_path
        self._events_path = events_path
        self._handler = handler
        self._events = events
        self._now = now
        self._restore_level = restore_level
        self._closed = False

    @property
    def log_path(self) -> Path:
        """The daily log file, `<folder>/<YYYY-MM-DD>.log`."""
        return self._log_path

    @property
    def events_path(self) -> Path:
        """The event stream, `<folder>/<YYYY-MM-DD>.events.jsonl`."""
        return self._events_path

    def event(self, kind: str, **fields: Any) -> None:
        """Append one event line: `{"ts_ns": <now>, "kind": kind, **fields}`, written to
        disk before it returns. A field named like a key or token is written as
        "[redacted]".

        Raises ConfigError for an empty `kind` or a field named `ts_ns`, and LiveError
        once the logs are closed or when the line cannot be written.
        """
        if self._closed:
            raise LiveError(f"session event {kind!r} after the session logs were closed")
        if not isinstance(kind, str) or not kind:
            raise ConfigError(f"an event kind must be a non-empty string, got {kind!r}")
        if TIME_FIELD in fields:
            raise ConfigError(
                f"event field {TIME_FIELD!r} is the event's time, written by the stream: "
                f"name the field differently"
            )
        line = {TIME_FIELD: self._now(), "kind": kind, **_clean(fields, kind, path="")}
        try:
            self._events.write(json.dumps(line, ensure_ascii=False) + "\n")
            self._events.flush()
        except OSError as error:
            raise LiveError(
                f"cannot write the session event stream {self._events_path}: {error}"
            ) from error

    def close(self) -> None:
        """Detach the log file from the logger, restore its level and close both files.
        Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        logger.info("session logs closed")
        logger.removeHandler(self._handler)
        self._handler.close()
        if self._restore_level is not None and logger.level == LOG_FILE_LEVEL:
            logger.setLevel(self._restore_level)
        self._events.close()

    def __enter__(self) -> SessionLogs:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def configure_session_logs(folder: str | Path, day: DateLike, *, now: Now) -> SessionLogs:
    """Open the session logs of IST day `day` under `folder` (created if missing): attach
    the daily log file to the `openalgo.backtest.live` logger and open the event stream.
    `now` gives the present in UTC nanoseconds, for the log lines and the events.

    Raises ConfigError for a `day` that is not a date, and LiveError when the logs of
    another session are attached to the logger, or the folder or a file cannot be opened.
    """
    date = from_day(to_day(day))
    if any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
        raise LiveError(
            "the session logs of another session are open in this process: close them "
            "before opening new ones (one live session per process)"
        )
    root = Path(folder)
    log_path = root / f"{date.isoformat()}.log"
    events_path = root / f"{date.isoformat()}.events.jsonl"
    try:
        root.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    except OSError as error:
        raise LiveError(
            f"cannot open the session log {log_path}: {error}. Give a folder you can write to"
        ) from None
    try:
        events = events_path.open("a", encoding="utf-8", newline="\n")
    except OSError as error:
        handler.close()
        raise LiveError(
            f"cannot open the session event stream {events_path}: {error}. Give a folder "
            f"you can write to"
        ) from None
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(LOG_FILE_LEVEL)
    handler.setFormatter(_IstFormatter(now))
    restore_level = None
    if logger.getEffectiveLevel() > LOG_FILE_LEVEL:
        restore_level = logger.level
        logger.setLevel(LOG_FILE_LEVEL)
    logger.addHandler(handler)
    logger.info("session logs of %s opened (events in %s)", date.isoformat(), events_path.name)
    return SessionLogs(
        log_path=log_path,
        events_path=events_path,
        handler=handler,
        events=events,
        now=now,
        restore_level=restore_level,
    )


class _IstFormatter(logging.Formatter):
    """Log lines stamped by the session's clock in IST to the millisecond, so the file
    and the event stream share one clock."""

    def __init__(self, now: Now) -> None:
        super().__init__(_LINE_FORMAT)
        self._now = now

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        stamp = _EPOCH + dt.timedelta(microseconds=self._now() // _NANOS_PER_MICRO)
        return stamp.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " IST"


def _clean(fields: Mapping[Any, Any], kind: str, *, path: str) -> dict[str, Any]:
    """The fields as a JSON object, a secret's value replaced, at any depth."""
    cleaned = {}
    for name, value in fields.items():
        key = str(name)
        where = f"{path}.{key}" if path else key
        if _secret(key):
            logger.warning(
                "session event %s: field %s looks like a key or token, written as %s",
                kind,
                where,
                REDACTED,
            )
            cleaned[key] = REDACTED
        else:
            cleaned[key] = _jsonable(value, kind, where)
    return cleaned


def _secret(name: str) -> bool:
    plain = name.lower().replace("_", "").replace("-", "")
    return plain.endswith(_SECRET_ENDINGS)


def _jsonable(value: Any, kind: str, where: str) -> Any:
    """`value` as JSON can hold it, by the rules of the module docstring."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, enum.Enum):
        return _jsonable(value.value, kind, where)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        return _clean(fields, kind, path=where)
    if isinstance(value, Mapping):
        return _clean(value, kind, path=where)
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(item, kind, f"{where}[{index}]") for index, item in enumerate(value)]
    logger.warning(
        "session event %s: field %s is a %s, written as its repr",
        kind,
        where,
        type(value).__name__,
    )
    return repr(value)


__all__ = ["LOG_FILE_LEVEL", "REDACTED", "TIME_FIELD", "SessionLogs", "configure_session_logs"]
