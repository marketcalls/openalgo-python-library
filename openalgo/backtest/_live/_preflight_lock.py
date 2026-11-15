"""The locks of a live session (08-live-trading section 4): one engine process per OpenAlgo
account, so that all its strategies share one rate budget, one account-level daily loss
limit and one view of the funds, and one session per OpenAlgo server, since a server is
logged in to one broker account: two sessions on it under two account names would trade
that one account twice over.

Each lock is an exclusive lock the operating system holds on a file: `engine.lock` in the
account's folder, beside its journal, and `server-<digest>.lock` in the folder holding
the accounts' folders, named by a digest of the server's key (`ServerAddress.key`). On
Windows it is a byte-range lock (msvcrt), elsewhere a lock of the whole file
(fcntl.flock, which also refuses a second holder in the same process). The system
releases it when the process ends, however it ends, so a crashed session leaves no stale
lock behind. The files are kept: removing one would let another process lock a new file
while this one still holds the old one.
"""

from __future__ import annotations

import errno
import hashlib
import os
import sys
from pathlib import Path
from types import TracebackType

from ..errors import JournalError, LiveError, PreflightError
from ._journal_file import LIVE_DIR_VARIABLE, _check_folder_name, default_journal_path
from ._preflight_url import ServerAddress

#: The name of the lock file in an account's folder.
LOCK_FILE_NAME = "engine.lock"
#: The name of a server's lock file: this prefix, a digest of the server's key, ".lock".
SERVER_LOCK_PREFIX = "server-"
# Hex digits of the key's SHA-256 digest in the file name: enough that two servers never
# share one, short enough to read.
_DIGEST_CHARS = 16

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
    """Where an account's lock file lives: `<root>/<account>/engine.lock`, beside the
    account's journal (see `live_root` for the root).

    Raises PreflightError as `live_root` does.
    """
    return live_root(account, folder) / account / LOCK_FILE_NAME


def server_lock_path(
    server: ServerAddress, account: str, folder: str | os.PathLike[str] | None = None
) -> Path:
    """Where the lock file of an OpenAlgo server lives: `<root>/server-<digest>.lock`, the
    digest that of the server's key (see `live_root` for the root, of which `account`
    only checks the name).

    Raises PreflightError as `live_root` does.
    """
    digest = hashlib.sha256(server.key.encode()).hexdigest()[:_DIGEST_CHARS]
    return live_root(account, folder) / f"{SERVER_LOCK_PREFIX}{digest}.lock"


def live_root(account: str, folder: str | os.PathLike[str] | None = None) -> Path:
    """The folder holding the accounts' folders: `folder` (a leading `~` expanded), or
    when None the folder of the journals (`~/.openalgo/live`, or `OPENALGO_LIVE_DIR`).

    Raises PreflightError, by the journal's own rules, when `account` cannot be a folder
    name on Windows, Linux and macOS alike, or when the folder (or `OPENALGO_LIVE_DIR`) is
    relative: a relative one would move with the folder the session is started from.
    """
    try:
        if folder is None:
            return default_journal_path(account).parent.parent
        _check_folder_name(account)
    except JournalError as error:
        raise _refusal(str(error)) from error
    root = Path(folder).expanduser()
    if not root.is_absolute():
        raise _refusal(
            f"folder must name an absolute folder, not {os.fspath(folder)!r}: a relative one "
            "would move with the folder the session is started from"
        )
    return root


class _LockFile:
    """An exclusive OS lock on one file, taken without waiting."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def take(self, held_elsewhere: str) -> None:
        """Take the lock. Raises PreflightError with `held_elsewhere` when another holder
        has it, and saying what to do when the file cannot be made or locked."""
        fd = self._open()
        try:
            locked = _try_lock(fd)
        except OSError as error:
            os.close(fd)
            raise _refusal(
                f"cannot lock {self.path} ({error}): put the live folder on a local disk "
                f"with the {LIVE_DIR_VARIABLE} environment variable"
            ) from error
        if not locked:
            os.close(fd)
            raise _refusal(held_elsewhere)
        self._fd = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        finally:
            os.close(fd)

    def _open(self) -> int:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return os.open(self.path, os.O_RDWR | os.O_CREAT)
        except OSError as error:
            raise _refusal(
                f"cannot create the lock file {self.path} ({error}): choose a folder you can "
                f"write, for example with the {LIVE_DIR_VARIABLE} environment variable"
            ) from error


class AccountLock:
    """The locks of one live session, held inside a `with` block: the OpenAlgo account's,
    and the OpenAlgo server's.

        with AccountLock("my_account", server_url=client.base_url):
            ...  # the live session

    `server_url` is the URL of the server the session trades through (the client's
    `base_url`, or its host): its scheme, host and port name the server, every loopback
    host (localhost, 127.0.0.1, ::1) as one. `folder` is the folder holding the accounts'
    folders (see `live_root`). Entering creates the folders and files as needed and takes
    both locks without waiting; leaving releases them, as does the end of the process.

    Raises ConfigError when `server_url` is not an http or https URL with a host;
    PreflightError when the account name or folder is refused, a lock file cannot be
    created, or another holder has a lock (another `bt.live` session of the account, or
    of another account on the same server); LiveError when this lock is entered again
    while it is held.
    """

    def __init__(
        self, account: str, *, server_url: str, folder: str | os.PathLike[str] | None = None
    ) -> None:
        self._account = account
        self._server = ServerAddress.of(server_url)
        self._account_file = _LockFile(lock_path(account, folder))
        self._server_file = _LockFile(server_lock_path(self._server, account, folder))

    @property
    def path(self) -> Path:
        """The account's lock file."""
        return self._account_file.path

    @property
    def server_path(self) -> Path:
        """The server's lock file."""
        return self._server_file.path

    @property
    def held(self) -> bool:
        """True inside the `with` block."""
        return self._account_file.held

    def __enter__(self) -> AccountLock:
        if self.held:
            raise LiveError(
                f"the lock of account {self._account!r} is held by this AccountLock already: "
                "enter it once"
            )
        self._account_file.take(
            f"another bt.live session runs for account {self._account!r}: it holds "
            f"{self.path}. Stop it first (Ctrl+C in its window); one process trades an "
            "account, so that its strategies share one rate budget, one daily loss limit "
            "and one view of the funds"
        )
        try:
            self._server_file.take(
                f"another bt.live session already trades through this OpenAlgo server "
                f"({self._server.key}), under another account name: it holds "
                f"{self.server_path}. Stop it first (Ctrl+C in its window); a server is "
                "logged in to one broker account, so one session trades through it"
            )
        except BaseException:
            self._account_file.release()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._server_file.release()
        finally:
            self._account_file.release()


def _refusal(message: str) -> PreflightError:
    return PreflightError(message, (message,))


__all__ = [
    "LOCK_FILE_NAME",
    "SERVER_LOCK_PREFIX",
    "AccountLock",
    "live_root",
    "lock_path",
    "server_lock_path",
]
