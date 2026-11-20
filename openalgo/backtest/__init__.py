"""openalgo.backtest: backtesting and live trading engine for Indian markets.

Status: beta. `bt.backtest` runs one `bt.Strategy` class on OpenAlgo history or on your
own bars, with Indian charges, slippage, session calendars and contract terms applied by
default, and returns a `Result` with its tables, metrics and HTML tearsheet; given a list
of strategies it runs them together on one account and returns a `PortfolioResult`.
`bt.optimize` sweeps a strategy's parameters over a grid; `bt.walk_forward` re-fits them
fold by fold through time; `bt.robustness` (and `result.robustness()`) runs the
robustness and overfitting checks. `bt.live` trades the
same strategy classes through OpenAlgo for today, in the mode set inside OpenAlgo (sandbox
or live), and returns a `SessionResult`; `bt.record` records the feed's ticks for later
replay, and `bt.read_ticks` reads a day of them back as a table. A part this version
does not run raises NotImplementedError saying what to do
instead: `bt.rebalance` among them.

    import openalgo.backtest as bt

    class Demo(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "15m"

        def on_bar(self, bar):
            if self.position.is_flat:
                self.buy(qty=10)

    result = bt.backtest(Demo, data=frame, capital=100_000, costs=None, slippage=0)

`import openalgo` does not import this package. Submodules load on first use and no
network call is made on import; the live layer loads only when `bt.live`, `bt.record` or
`bt.read_ticks` runs.
"""

from importlib import import_module as _import_module

_SUBMODULES = (
    "calendar",
    "costs",
    "data",
    "errors",
    "metrics",
    "portfolio_result",
    "reference",
    "report",
    "resample",
    "result",
    "run",
    "session_result",
    "size",
    "slippage",
    "strategy",
    "units",
)

# Names exported from submodules, loaded on first use.
_EXPORTS = {
    "strategy": ("Strategy",),
    "result": ("Result",),
    "portfolio_result": ("PortfolioResult",),
    "session_result": ("SessionResult", "RecordSummary"),
    "_optimize_result": ("OptimizeResult",),
    "_read_ticks": ("read_ticks",),
    "_robust": ("RobustnessReport", "CheckResult", "WalkForwardResult"),
    "run": (
        "backtest",
        "live",
        "record",
        "optimize",
        "robustness",
        "walk_forward",
        "rebalance",
        "from_signals",
        "from_arrays",
    ),
    "units": (
        "points",
        "pct",
        "rupees",
        "ticks",
        "atr",
        "trail",
        "trail_points",
        "trail_pct",
        "trail_atr",
        "breakeven",
        "lock",
        "param",
    ),
    "_clock": ("time_between", "weekday"),
}
_ATTRIBUTES = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = list(_SUBMODULES) + list(_ATTRIBUTES)


def __getattr__(name):
    if name in _SUBMODULES:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _ATTRIBUTES:
        value = getattr(_import_module(f"{__name__}.{_ATTRIBUTES[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_SUBMODULES) + list(_ATTRIBUTES))
