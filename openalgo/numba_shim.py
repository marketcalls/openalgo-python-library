"""Shim that upgrades legacy @jit decorators to modern njit fast paths.

This lets existing code using `from numba import jit` automatically gain
caching and nogil without touching every file.  It also adds an easy
way to request parallel loops simply by passing ``parallel=True``.

When numba is not installed the shim provides transparent no-op
decorators so that the indicator code still works (interpreted, slower).
"""

try:
    from numba import njit, prange  # noqa: F401 -- re-export for callers

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def _noop_decorator(*args, **kwargs):
        """Return the function unchanged when numba is absent."""
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

    njit = _noop_decorator

    # prange is just range when not using numba
    prange = range  # type: ignore[assignment]


def jit(*args, **kwargs):  # type: ignore[override]
    """Drop-in replacement for numba.jit with better defaults.

    Usage in code remains the same:
    >>> from numba import jit
    >>> @jit(nopython=True)
    ... def foo(x): ...

    This shim ensures:
    - nopython=True  (stay in compiled mode)
    - cache=True     (persist compiled kernels to disk)
    - nogil=True     (release GIL so threads can run concurrently)

    fastmath is deliberately OFF to preserve IEEE 754 NaN semantics.
    np.isnan() silently returns wrong results under fastmath=True.

    When numba is not installed the decorator is a no-op.
    """
    if not HAS_NUMBA:
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

    kwargs.pop("nopython", None)
    kwargs.setdefault("cache", True)
    kwargs.setdefault("nogil", True)
    return njit(*args, **kwargs)
