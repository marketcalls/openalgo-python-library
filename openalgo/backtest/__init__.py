"""openalgo.backtest: backtesting, paper and live trading engine for Indian markets.

Status: foundations only. This package currently provides the Indian cost model,
session calendars, resampling, reference tables and the data layer. The event kernel,
the Strategy class and the run functions (backtest, paper, live) are not implemented
yet.

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
)

__all__ = list(_SUBMODULES)


def __getattr__(name):
    if name in _SUBMODULES:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_SUBMODULES))
