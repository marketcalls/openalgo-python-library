"""The account lock of a live session (08-live-trading section 4): one engine process per
OpenAlgo account, so that all its strategies share one rate budget, one account-level
daily loss limit and one view of the funds.

The lock is an exclusive lock the operating system holds on `engine.lock` in the
account's folder, beside its journal: a byte-range lock on Windows (msvcrt), a lock of
the whole file elsewhere (fcntl.flock, which also refuses a second holder in the same
process). The system releases it when the process ends, however it ends, so a crashed
session leaves no stale lock behind. The file itself is kept: removing it would let
another process lock a new file while this one still holds the old one.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path
from types import TracebackType

from ..errors import JournalError, LiveError, PreflightError
from ._journal_file import LIVE_DIR_VARIABLE, _check_folder_name, default_journal_path

#: The name of the lock file in an account's folder.
LOCK_FILE_NAME = "engine.lock"

if sys.platform == "win32":
    import msvcrt

    # The errors of a byte-range lock held through another handle: the C runtime reports
    # EACCES, or EDEADLOCK once it has given up retrying.
    _HELD_ERRNOS = frozenset({errno.EACCES, errno.EDEADLOCK})

    def _try_lock(fd: int) -> bool:
        """Lock the file's first byte without waiting; False when another holds it."""
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in _HELD_ERRNOS:
                return False
            raise
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        """Lock the whole file without waiting; False when another holds it."""
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def lock_path(account: str, folder: str | os.PathLike[str] | None = None) -> Path:
    """Where an account's lock file lives: `<folder>/<account>/engine.lock`, with the
    folder of the journals (`~/.openalgo/live`, or `OPENALGO_LIVE_DIR`) when `folder` is
    None, so the lock sits beside the account's journal.

    Raises PreflightError, by the journal's own rules, when `account` cannot be a folder
    name on Windows, Linux and macOS alike, or when `OPENALGO_LIVE_DIR` is relative.
    """
    try:
        if folder is None:
            return default_journal_path(account).with_name(LOCK_FILE_NAME)
        _check_folder_name(account)
    except JournalError as error:
        raise PreflightError(str(error), (str(error),)) from error
    return Path(folder) / account / LOCK_FILE_NAME


class AccountLock:
    """The exclusive lock of one OpenAlgo account, held inside a `with` block:

        with AccountLock("my_account"):
            ...  # the live session

    `folder` is the folder holding the accounts' folders (see `lock_path`). Entering
    creates the folders and the file as needed and takes the lock without waiting;
    leaving releases it, as does the end of the process.

    Raises PreflightError when the account name or folder is refused, when the lock file
    cannot be created, or when another holder has the lock (another `bt.live` session of
    the account); LiveError when this lock is entered again while it is held.
    """

    def __init__(self, account: str, *, folder: str | os.PathLike[str] | None = None) -> None:
        self._account = account
        self._path = lock_path(account, folder)
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        """The lock file."""
        return self._path

    @property
    def held(self) -> bool:
        """True inside the `with` block."""
        return self._fd is not None

    def __enter__(self) -> AccountLock:
        if self._fd is not None:
            raise LiveError(
                f"the lock of account {self._account!r} is held by this AccountLock already: "
                "enter it once"
            )
        fd = self._open()
        try:
            locked = _try_lock(fd)
        except OSError as error:
            os.close(fd)
            raise _refusal(
                f"cannot lock {self._path} ({error}): put the live folder on a local disk "
                f"with the {LIVE_DIR_VARIABLE} environment variable"
            ) from error
        if not locked:
            os.close(fd)
            raise _refusal(
                f"another bt.live session runs for account {self._account!r}: it holds "
                f"{self._path}. Stop it first (Ctrl+C in its window); one process trades an "
                "account, so that its strategies share one rate budget, one daily loss limit "
                "and one view of the funds"
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

    def _open(self) -> int:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            return os.open(self._path, os.O_RDWR | os.O_CREAT)
        except OSError as error:
            raise _refusal(
                f"cannot create the account lock at {self._path} ({error}): choose a folder "
                f"you can write, for example with the {LIVE_DIR_VARIABLE} environment variable"
            ) from error


def _refusal(message: str) -> PreflightError:
    return PreflightError(message, (message,))


__all__ = ["LOCK_FILE_NAME", "AccountLock", "lock_path"]
