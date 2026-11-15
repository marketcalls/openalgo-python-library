"""The account lock of a live session (08-live-trading section 4): one `bt.live` process
per OpenAlgo account. Where the lock file lives (beside the account's journal, by the
journal's folder rules), a second holder refused, the lock free again once its holder
leaves, and the names and folders refused.

The tests hold the lock twice in one process: each AccountLock opens the file on its own,
and the operating system refuses the second lock as it would one of another process.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live.preflight import LOCK_FILE_NAME, AccountLock
from openalgo.backtest.errors import LiveError, PreflightError

ACCOUNT = "trader_one"


@pytest.fixture(autouse=True)
def no_live_dir(monkeypatch):
    """Each test sets OPENALGO_LIVE_DIR itself, or runs without it."""
    monkeypatch.delenv("OPENALGO_LIVE_DIR", raising=False)


def test_the_lock_file_is_engine_lock_in_the_accounts_folder_of_the_folder_given(tmp_path):
    lock = AccountLock(ACCOUNT, folder=tmp_path)

    assert LOCK_FILE_NAME == "engine.lock"
    assert lock.path == tmp_path / ACCOUNT / "engine.lock"


def test_without_a_folder_the_lock_follows_the_live_dir_variable_like_the_journal(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", str(tmp_path))

    assert AccountLock(ACCOUNT).path == tmp_path / ACCOUNT / "engine.lock"


def test_without_a_folder_or_the_variable_the_lock_lives_under_the_home_folder(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    lock = AccountLock(ACCOUNT)

    assert lock.path == tmp_path / ".openalgo" / "live" / ACCOUNT / "engine.lock"


def test_entering_creates_the_folders_and_holds_the_lock(tmp_path):
    lock = AccountLock(ACCOUNT, folder=tmp_path / "live")

    with lock:
        assert lock.held
        assert lock.path.is_file()
    assert not lock.held


def test_a_second_holder_of_the_accounts_lock_is_refused(tmp_path):
    with AccountLock(ACCOUNT, folder=tmp_path), pytest.raises(PreflightError) as raised:
        AccountLock(ACCOUNT, folder=tmp_path).__enter__()

    message = str(raised.value)
    assert message.startswith(f"another bt.live session runs for account {ACCOUNT!r}")
    assert str(tmp_path / ACCOUNT / "engine.lock") in message
    assert raised.value.failures == (message,)


def test_a_refused_holder_holds_nothing(tmp_path):
    second = AccountLock(ACCOUNT, folder=tmp_path)

    with AccountLock(ACCOUNT, folder=tmp_path), pytest.raises(PreflightError):
        second.__enter__()

    assert not second.held


def test_the_lock_is_free_again_once_its_holder_has_left(tmp_path):
    with AccountLock(ACCOUNT, folder=tmp_path):
        pass

    with AccountLock(ACCOUNT, folder=tmp_path) as again:
        assert again.held


def test_the_lock_is_released_when_its_block_raises(tmp_path):
    with pytest.raises(RuntimeError), AccountLock(ACCOUNT, folder=tmp_path):
        raise RuntimeError("the session failed")

    with AccountLock(ACCOUNT, folder=tmp_path) as again:
        assert again.held


def test_the_lock_file_is_kept_after_the_lock_is_released(tmp_path):
    lock = AccountLock(ACCOUNT, folder=tmp_path)

    with lock:
        pass

    assert lock.path.is_file()


def test_two_accounts_hold_their_locks_at_the_same_time(tmp_path):
    with (
        AccountLock("trader_one", folder=tmp_path) as one,
        AccountLock("trader_two", folder=tmp_path) as two,
    ):
        assert one.held and two.held


def test_entering_a_lock_this_lock_holds_already_raises(tmp_path):
    lock = AccountLock(ACCOUNT, folder=tmp_path)

    with lock, pytest.raises(LiveError, match="held by this AccountLock already"):
        lock.__enter__()


@pytest.mark.parametrize(
    "account", ["", "  ", "a/b", "a\\b", "a:b", "CON", "nul.txt", "COM¹", "end."]
)
def test_an_account_name_that_cannot_be_a_folder_is_refused(account, tmp_path):
    with pytest.raises(PreflightError) as raised:
        AccountLock(account, folder=tmp_path)

    assert raised.value.failures == (str(raised.value),)


def test_an_account_name_is_refused_without_a_folder_too(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", str(tmp_path))

    with pytest.raises(PreflightError, match="device name on Windows"):
        AccountLock("AUX")


def test_a_relative_live_dir_is_refused(monkeypatch):
    monkeypatch.setenv("OPENALGO_LIVE_DIR", "relative/live")

    with pytest.raises(PreflightError, match="must name an absolute folder"):
        AccountLock(ACCOUNT)


def test_a_folder_that_cannot_be_made_is_refused_on_entering(tmp_path):
    taken = tmp_path / "a_file"
    taken.write_text("not a folder", encoding="utf-8")
    lock = AccountLock(ACCOUNT, folder=taken)

    with pytest.raises(PreflightError, match="cannot create the account lock"):
        lock.__enter__()

    assert not lock.held
