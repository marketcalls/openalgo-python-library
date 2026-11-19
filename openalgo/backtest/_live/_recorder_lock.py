"""The lock of the tick recorder (private to `recorder`): one process writes a cache's
recorded ticks at a time, so that a finalize never takes a raw file that another
recording is still appending to.

It is an exclusive lock the operating system holds on `v2/ticks/recorder.lock` in the
cache, taken without waiting with the calls the live session's locks use
(`_preflight_lock`). The system releases it when the process ends, however it ends, so a
crashed recording leaves no stale lock; the file is kept.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType

from ..data.cache import CACHE_ENV
from ..errors import ConfigError, LiveError
from ._preflight_lock import _try_lock, _unlock

#: The name of the recorder's lock file in the folder of recorded ticks.
RECORDER_LOCK_NAME = "recorder.lock"


class RecorderLock:
    """The recorder's lock on the folder of recorded ticks `root`, held inside a `with`
    block.

    Raises ConfigError when the lock file cannot be made or locked, and LiveError when
    another holder has it (another recording or finalize, in this process or another).
    """

    def __init__(self, root: Path) -> None:
        self.path = root / RECORDER_LOCK_NAME
        self._fd: int | None = None

    def __enter__(self) -> RecorderLock:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        except OSError as exc:
            raise ConfigError(self._unusable(exc)) from exc
        try:
            locked = _try_lock(fd)
        except OSError as exc:
            os.close(fd)
            raise ConfigError(self._unusable(exc)) from exc
        if not locked:
            os.close(fd)
            raise LiveError(
                f"another recording or finalize writes the ticks in {self.path.parent}: it "
                f"holds {self.path}. Wait for it to stop (Ctrl+C in its window stops a "
                "recording), or record into another cache_dir="
            )
        self._fd = fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        finally:
            os.close(fd)

    def _unusable(self, error: OSError) -> str:
        return (
            f"the recorder's lock file {self.path} cannot be made or locked ({error}): pass "
            f"cache_dir= or set {CACHE_ENV} to a folder on a local disk you can write"
        )


__all__ = ["RECORDER_LOCK_NAME", "RecorderLock"]
