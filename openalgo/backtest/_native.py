"""Access to the Rust kernel bindings (`openalgo._oaindicators.backtest`).

Usage inside the package:

    from ._native import native
    native.engine_version()
"""

from openalgo import _oaindicators as _ext

try:
    native = _ext.backtest
except AttributeError as exc:  # an extension built before the backtest kernel existed
    raise ImportError(
        "The installed openalgo native module has no backtest kernel. "
        "Reinstall openalgo, or rebuild it from source with `maturin develop`."
    ) from exc

__all__ = ["native"]
