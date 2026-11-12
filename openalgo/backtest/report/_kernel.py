"""Calls into the native metrics kernel (`openalgo._oaindicators.backtest.metrics_*`)."""

from collections.abc import Callable
from typing import Any

from .._native import native
from ..errors import ConfigError


def kernel(name: str) -> Callable[..., Any]:
    """The native metrics function `metrics_<name>`, with a clear error when the
    installed extension predates it."""
    try:
        return getattr(native, f"metrics_{name}")
    except AttributeError:
        raise ImportError(
            f"the installed openalgo native module has no backtest.metrics_{name}: it is "
            "older than this Python package. Reinstall openalgo, or rebuild it from source "
            "with `maturin develop`."
        ) from None


def call(name: str, *args: Any) -> Any:
    """Call `metrics_<name>`; input the kernel refuses becomes a ConfigError whose
    message names the row and says what to change."""
    # Looked up first: an extension older than the function may lack the error type too,
    # and reading it in the except clause would then replace the reinstall advice.
    function = kernel(name)
    try:
        return function(*args)
    except native.NativeMetricsError as exc:
        raise ConfigError(str(exc)) from None


__all__ = ["call", "kernel"]
