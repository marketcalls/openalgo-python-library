"""Packaging rules for the backtest kernel (FR-CORE-09, FR-CORE-10)."""

import subprocess
import sys


def test_native_submodule_is_bound():
    from openalgo.backtest._native import native

    assert isinstance(native.engine_version(), str)
    assert native.engine_version().count(".") == 2


def test_import_openalgo_does_not_import_backtest():
    code = (
        "import sys, openalgo; "
        "bad = [m for m in sys.modules if m.startswith('openalgo.backtest')]; "
        "assert not bad, bad; "
        "assert 'openstatz' not in sys.modules and 'matplotlib' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_submodules_load_lazily():
    import openalgo.backtest as bt

    assert "errors" in dir(bt)
    assert issubclass(bt.errors.NoDataError, bt.errors.DataError)
    assert issubclass(bt.errors.DataError, bt.errors.BacktestError)
