"""What importing the package loads: `import openalgo` loads no backtest module, and
`openalgo.backtest` with its run functions loads no indicator-check, timeframe,
optimisation or robustness module until a strategy or a report uses them, and no
module of the live layer until `bt.live`, `bt.record` or `bt.read_ticks` runs."""

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
    "openalgo.backtest._robust",
    "concurrent.futures.process",
)
ROBUST = "openalgo.backtest._robust"
# A backtest of a strategy that buys at its first bar, on three daily bars of own data.
RUN = "\n".join(
    [
        "import pandas as pd; import openalgo.backtest as bt",
        "class Buy(bt.Strategy):",
        "    symbol, exchange, interval = 'SBIN', 'NSE', 'D'",
        "    def on_bar(self, bar):",
        "        self.buy(qty=1) if self.position.is_flat else None",
        "days = ['2026-11-02 09:15', '2026-11-03 09:15', '2026-11-04 09:15']",
        "index = pd.DatetimeIndex(days).tz_localize('Asia/Kolkata')",
        "prices = {'open': [100.0, 101.0, 102.0], 'close': [100.0, 101.0, 102.0]}",
        "bars = pd.DataFrame({**prices, 'high': 103.0, 'low': 99.0, 'volume': 1e3}, index=index)",
        "result = bt.backtest(Buy, data=bars, capital=100000, costs=None, slippage=0)",
        "",
    ]
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


def test_the_live_verbs_and_their_result_load_no_live_module_until_they_run():
    code = (
        "import openalgo.backtest as bt; bt.live; bt.record; bt.read_ticks; bt.SessionResult; "
        "assert not [m for m in sys.modules if m.startswith('openalgo.backtest._live')]"
    )
    assert loaded_after(code) == []


def test_a_backtest_and_its_metrics_load_no_robustness_module():
    assert ROBUST not in loaded_after(RUN + "result.metrics")


def test_the_summary_loads_the_robustness_module_for_its_quick_checks():
    assert ROBUST in loaded_after(RUN + "result.summary()")
