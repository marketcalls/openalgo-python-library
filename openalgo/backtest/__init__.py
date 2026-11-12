"""openalgo.backtest: backtesting and live trading engine for Indian markets.

Status: beta. `bt.backtest` runs one `bt.Strategy` class on OpenAlgo history or on your
own bars, with Indian charges, slippage, session calendars and contract terms applied by
default, and returns a `Result` with its tables, metrics and HTML tearsheet;
`bt.optimize` sweeps a strategy's parameters over a grid. A part this version does not
run raises NotImplementedError saying what to do instead: `bt.live`, `bt.walk_forward`,
`bt.rebalance` and a run of several strategies among them.

    import openalgo.backtest as bt

    class Demo(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "15m"

        def on_bar(self, bar):
            if self.position.is_flat:
                self.buy(qty=10)

    result = bt.backtest(Demo, data=frame, capital=100_000, costs=None, slippage=0)

`import openalgo` does not import this package. Submodules load on first use and no
network call is made on import.
"""

from importlib import import_module as _import_module

_SUBMODULES = (
    "calendar",
    "costs",
    "data",
    "errors",
    "metrics",
    "reference",
    "report",
    "resample",
    "result",
    "run",
    "size",
    "slippage",
    "strategy",
    "units",
)

# Names exported from submodules, loaded on first use.
_EXPORTS = {
    "strategy": ("Strategy",),
    "result": ("Result",),
    "_optimize_result": ("OptimizeResult",),
    "run": (
        "backtest",
        "live",
        "optimize",
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
