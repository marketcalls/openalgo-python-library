"""The command line of `benchmark/backtest_speed.py`: what it refuses before measuring."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "benchmark" / "backtest_speed.py"
# The exit status argparse gives a command line it refuses.
USAGE_ERROR = 2


def _script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The script as a module (registered while the test runs: its dataclasses look
    their module up by name)."""
    spec = importlib.util.spec_from_file_location("backtest_speed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_an_unknown_row_group_is_refused_so_check_cannot_pass_on_no_rows(monkeypatch, capsys):
    # "condition" is a misspelling of the group "conditions".
    with pytest.raises(SystemExit) as stopped:
        _script(monkeypatch).main(["--rows", "condition", "--check"])
    assert stopped.value.code == USAGE_ERROR
    assert "unknown row group 'condition'" in capsys.readouterr().err
