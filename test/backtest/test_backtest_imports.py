"""What importing the package loads: `import openalgo` loads no backtest module, and
`openalgo.backtest` with its run functions loads no indicator-check, timeframe or
optimisation module until a strategy uses them."""

import subprocess
import sys

DEFERRED = (
    "openalgo.indicators.causality",
    "openalgo.backtest._indicators",
    "openalgo.backtest._timeframes",
    "openalgo.backtest._optimize",
    "openalgo.backtest._optimize_result",
    "openalgo.backtest._sweep_pool",
    "openalgo.backtest._objective",
    "concurrent.futures.process",
)


def loaded_after(code: str) -> list[str]:
    script = f"import sys; {code}; print(','.join(m for m in {DEFERRED!r} if m in sys.modules))"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return [name for name in done.stdout.strip().split(",") if name]


def test_import_openalgo_loads_no_backtest_module():
    code = (
        "import openalgo; assert not [m for m in sys.modules if m.startswith('openalgo.backtest')]"
    )
    assert loaded_after(code) == []


def test_the_run_functions_load_no_indicator_or_optimisation_module():
    code = "import openalgo.backtest as bt; bt.backtest; bt.optimize; bt.Strategy; bt.Result"
    assert loaded_after(code) == []
