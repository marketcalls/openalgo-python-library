"""The locks of a live session (08-live-trading section 4): one `bt.live` process per
OpenAlgo account, and one session per OpenAlgo server. Where the lock files live (beside
the account's journal, by the journal's folder rules), a second holder refused, the locks
free again once their holder leaves or dies, and the names and folders refused.

Most tests hold a lock twice in one process: each AccountLock opens its files on its own,
and the operating system refuses the second lock as it would one of another process. Two
tests run another Python process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from openalgo.backtest._live.preflight import LOCK_FILE_NAME, AccountLock
from openalgo.backtest.errors import ConfigError, LiveError, PreflightError

ACCOUNT = "trader_one"
#: The API URL of an OpenAlgo server on this computer, as the SDK builds it.
SERVER = "http://127.0.0.1:5000/api/v1/"
#: Another server, on another computer.
OTHER_SERVER = "https://algo.example.com/api/v1/"
#: The root of the checkout, so another process imports this copy of openalgo.
ROOT = Path(__file__).resolve().parents[3]
#: Seconds another process may take to start, import openalgo and try the lock.
CHILD_TIMEOUT_S = 10
#: What the other process runs: it tries the lock once and says what happened. With
#: "crash" it dies holding the lock, with no chance to release it.
CHILD = """
import os, sys
from openalgo.backtest._live.preflight import AccountLock
from openalgo.backtest.errors import PreflightError
account, server, folder, then = sys.argv[1:]
try:
    with AccountLock(account, server_url=server, folder=folder):
        sys.stdout.write("held")
        sys.stdout.flush()
        if then == "crash":
            os._exit(3)
except PreflightError:
    sys.stdout.write("refused")
"""


@pytest.fixture(autouse=True)
def no_live_dir(monkeypatch):
    """Each test sets OPENALGO_LIVE_DIR itself, or runs without it."""
    monkeypatch.delenv("OPENALGO_LIVE_DIR", raising=False)


def account_lock(folder, account: str = ACCOUNT, server: str = SERVER) -> AccountLock:
    return AccountLock(account, server_url=server, folder=folder)


def try_in_another_process(folder: Path, *, then: str = "release") -> tuple[str, int]:
    """What another Python process gets when it tries the account's lock under `folder`
    ("held" or "refused"), and its exit code; with `then="crash"` it dies holding it."""
    finished = subprocess.run(
        [sys.executable, "-c", CHILD, ACCOUNT, SERVER, str(folder), then],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
    )
    return finished.stdout, finished.returncode


# ----- other processes ---------------------------------------------------------------


def test_another_process_is_refused_while_the_lock_is_held_and_gets_it_once_released(tmp_path):
    with account_lock(tmp_path):
        assert try_in_another_process(tmp_path) == ("refused", 0)

    assert try_in_another_process(tmp_path) == ("held", 0)


def test_a_holder_that_dies_without_releasing_frees_the_lock(tmp_path):
    assert try_in_another_process(tmp_path, then="crash") == ("held", 3)

    with account_lock(tmp_path) as lock:
        assert lock.held


# ----- where the files are -----------------------------------------------------------


def test_the_lock_file_is_engine_lock_in_the_accounts_folder_of_the_folder_given(tmp_path):
    lock = account_lock(tmp_path)

    assert LOCK_FILE_NAME == "engine.lock"
    assert lock.path == tmp_path / ACCOUNT / "engine.lock"


def test_the_servers_lock_file_is_in_the_folder_holding_the_accounts(tmp_path):
    lock = account_lock(tmp_path)

    assert lock.server_path.parent == tmp_path
    assert lock.server_path.name.startswith("server-")
    assert lock.server_path.suffix == ".lock"


def test_without_a_folder_the_lock_follows_the_live_dir_variable_like_the_journal(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", str(tmp_path))

    lock = account_lock(None)

    assert (lock.path, lock.server_path.parent) == (tmp_path / ACCOUNT / "engine.lock", tmp_path)


def test_without_a_folder_or_the_variable_the_lock_lives_under_the_home_folder(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    lock = account_lock(None)

    assert lock.path == tmp_path / ".openalgo" / "live" / ACCOUNT / "engine.lock"


def test_a_folder_starting_with_the_home_folder_mark_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    lock = account_lock("~/live")

    assert lock.path == tmp_path / "live" / ACCOUNT / "engine.lock"


# ----- holding the account's lock ----------------------------------------------------


def test_entering_creates_the_folders_and_holds_the_locks(tmp_path):
    lock = account_lock(tmp_path / "live")

    with lock:
        assert lock.held
        assert lock.path.is_file() and lock.server_path.is_file()
    assert not lock.held


def test_a_second_holder_of_the_accounts_lock_is_refused(tmp_path):
    with account_lock(tmp_path), pytest.raises(PreflightError) as raised:
        account_lock(tmp_path).__enter__()

    message = str(raised.value)
    assert message.startswith(f"another bt.live session runs for account {ACCOUNT!r}")
    assert str(tmp_path / ACCOUNT / "engine.lock") in message
    assert raised.value.failures == (message,)


def test_a_refused_holder_holds_nothing(tmp_path):
    second = account_lock(tmp_path)

    with account_lock(tmp_path), pytest.raises(PreflightError):
        second.__enter__()

    assert not second.held


def test_the_lock_is_free_again_once_its_holder_has_left(tmp_path):
    with account_lock(tmp_path):
        pass

    with account_lock(tmp_path) as again:
        assert again.held


def test_the_lock_is_released_when_its_block_raises(tmp_path):
    with pytest.raises(RuntimeError), account_lock(tmp_path):
        raise RuntimeError("the session failed")

    with account_lock(tmp_path) as again:
        assert again.held


def test_the_lock_files_are_kept_after_the_lock_is_released(tmp_path):
    lock = account_lock(tmp_path)

    with lock:
        pass

    assert lock.path.is_file() and lock.server_path.is_file()


def test_entering_a_lock_this_lock_holds_already_raises(tmp_path):
    lock = account_lock(tmp_path)

    with lock, pytest.raises(LiveError, match="held by this AccountLock already"):
        lock.__enter__()


# ----- one session per server --------------------------------------------------------


def test_two_accounts_on_two_servers_hold_their_locks_at_the_same_time(tmp_path):
    with (
        account_lock(tmp_path, "trader_one", SERVER) as one,
        account_lock(tmp_path, "trader_two", OTHER_SERVER) as two,
    ):
        assert one.held and two.held


def test_a_second_account_on_the_same_server_is_refused(tmp_path):
    with account_lock(tmp_path, "trader_one"), pytest.raises(PreflightError) as raised:
        account_lock(tmp_path, "trader_two").__enter__()

    assert str(raised.value).startswith(
        "another bt.live session already trades through this OpenAlgo server "
        "(http://localhost:5000), under another account name"
    )


@pytest.mark.parametrize(
    "second",
    [
        "http://localhost:5000/api/v1/",
        "http://LOCALHOST:5000",
        "http://[::1]:5000/api/v1/",
        "http://127.0.0.1:5000/other/path/",
    ],
)
def test_the_loopback_names_of_one_server_are_one_server(tmp_path, second):
    with account_lock(tmp_path, "trader_one"), pytest.raises(PreflightError):
        account_lock(tmp_path, "trader_two", second).__enter__()


@pytest.mark.parametrize(
    "second", ["http://127.0.0.1:5001/api/v1/", "https://127.0.0.1:5000/api/v1/"]
)
def test_another_port_or_scheme_is_another_server(tmp_path, second):
    with (
        account_lock(tmp_path, "trader_one"),
        account_lock(tmp_path, "trader_two", second) as two,
    ):
        assert two.held


def test_an_account_refused_on_its_server_lets_go_of_its_own_lock(tmp_path):
    with account_lock(tmp_path, "trader_one"):
        with pytest.raises(PreflightError):
            account_lock(tmp_path, "trader_two").__enter__()

        with account_lock(tmp_path, "trader_two", OTHER_SERVER) as two:
            assert two.held


@pytest.mark.parametrize("server", ["127.0.0.1:5000/api/v1/", "ftp://host/", "", "http://"])
def test_a_server_url_that_is_not_http_with_a_host_is_refused(tmp_path, server):
    with pytest.raises(ConfigError, match="must start with http:// or https://"):
        account_lock(tmp_path, server=server)


# ----- names and folders refused -----------------------------------------------------


@pytest.mark.parametrize(
    "account", ["", "  ", "a/b", "a\\b", "a:b", "CON", "nul.txt", "COM¹", "end."]
)
def test_an_account_name_that_cannot_be_a_folder_is_refused(account, tmp_path):
    with pytest.raises(PreflightError) as raised:
        account_lock(tmp_path, account)

    assert raised.value.failures == (str(raised.value),)


def test_an_account_name_is_refused_without_a_folder_too(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", str(tmp_path))

    with pytest.raises(PreflightError, match="device name on Windows"):
        account_lock(None, "AUX")


def test_a_relative_live_dir_is_refused(monkeypatch):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", "relative/live")

    with pytest.raises(PreflightError, match="must name an absolute folder"):
        account_lock(None)


@pytest.mark.parametrize("folder", ["relative_live", Path("relative") / "live", "."])
def test_a_relative_folder_is_refused(folder):
    with pytest.raises(PreflightError, match="folder must name an absolute folder"):
        account_lock(folder)


def test_a_folder_that_cannot_be_made_is_refused_on_entering(tmp_path):
    taken = tmp_path / "a_file"
    taken.write_text("not a folder", encoding="utf-8")
    lock = account_lock(taken)

    with pytest.raises(PreflightError, match="cannot create the lock file"):
        lock.__enter__()

    assert not lock.held
