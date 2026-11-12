"""The data provider protocol, the provider registry and entry-point plugins.

A provider is any object with a `name`, a set of `capabilities`, a declared
`timestamp_convention` and a `bars()` method (`DataProvider`). Third-party
packages add providers by declaring an entry point in the group
`openalgo.backtest.providers`:

    [project.entry-points."openalgo.backtest.providers"]
    myvendor = "my_package.provider:MyVendorProvider"

The entry point may resolve to a provider instance, a provider class or any
callable that returns a provider. Entry points are read on first use of the
registry, never on import.
"""

import warnings
from dataclasses import dataclass, field
from importlib import metadata as _metadata
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .._calendar_convert import DateLike
from .._layout import BAR_COLUMNS
from ..errors import ConfigError, DataError
from .files import Files
from .normalise import VALUE_COLUMNS
from .openalgo_provider import OpenAlgoProvider
from .own import Frame

ENTRY_POINT_GROUP = "openalgo.backtest.providers"

# The providers that ship with the engine. A plugin cannot take one of these names.
_BUILTIN_PROVIDERS = {"openalgo": OpenAlgoProvider, "frame": Frame, "files": Files}


@runtime_checkable
class DataProvider(Protocol):
    """What every data source implements: a name, its capabilities, the timestamp
    convention it declares (never guessed) and `bars`.

    `bars` returns a pandas DataFrame in the canonical layout of
    `openalgo.backtest._layout` for the inclusive IST date range `start` to `end`.
    """

    name: str
    capabilities: set[str]  # {"bars", "ticks", "oi", "symbol_master", "calendar",
    #                          "corporate_actions", "expiries"}
    timestamp_convention: str  # for example "utc_open", "ist_date", declared, never guessed

    def bars(
        self, symbol: str, exchange: str, interval: str, start: DateLike, end: DateLike
    ) -> pd.DataFrame:
        """Canonical bars of one instrument for the inclusive IST dates `start` to
        `end`. Raises NoDataError when the source has nothing for the range."""
        ...

    # optional, by capability: ticks, symbol_master, calendar, corporate_actions, expiries


@dataclass
class _Registry:
    """The process-wide provider registry: the one piece of mutable module state in
    the data layer. It changes only through the functions of this module.

    providers: name to a provider instance, a provider class or a factory.
    load_errors: entry points that failed to load, name to reason.
    discovered: whether the entry-point group has been read.
    """

    providers: dict[str, object] = field(default_factory=lambda: dict(_BUILTIN_PROVIDERS))
    load_errors: dict[str, str] = field(default_factory=dict)
    discovered: bool = False


_REGISTRY = _Registry()


def validate_provider(provider: object) -> DataProvider:
    """Return `provider` if it satisfies the protocol.

    Raises ConfigError naming the first thing that is wrong: no `name`, a
    `capabilities` set that lacks "bars", no declared `timestamp_convention` or no
    callable `bars`.
    """
    kind = type(provider).__name__
    name = getattr(provider, "name", None)
    if not isinstance(name, str) or not name:
        raise ConfigError(f"data provider {kind} must have a non-empty string attribute 'name'")
    capabilities = getattr(provider, "capabilities", None)
    if not isinstance(capabilities, (set, frozenset)) or not all(
        isinstance(item, str) for item in capabilities
    ):
        raise ConfigError(f"data provider {name!r}: 'capabilities' must be a set of strings")
    if "bars" not in capabilities:
        raise ConfigError(f"data provider {name!r} does not declare the 'bars' capability")
    convention = getattr(provider, "timestamp_convention", None)
    if not isinstance(convention, str) or not convention:
        raise ConfigError(
            f"data provider {name!r} must declare 'timestamp_convention' (for example "
            "'utc_open' or 'ist_date'); the engine never guesses it"
        )
    if not callable(getattr(provider, "bars", None)):
        raise ConfigError(
            f"data provider {name!r} has no bars(symbol, exchange, interval, start, end)"
        )
    return provider


def validate_layout(bars: object, what: str = "bars") -> pd.DataFrame:
    """Return `bars` if it is exactly in the canonical layout, as a provider's
    `bars()` must return it.

    Checks the column names and order, int64 time columns, float64 value columns, a
    RangeIndex from 0, rows sorted by `ts_open` without duplicates and every
    `ts_close` after its `ts_open`.

    Raises DataError naming the first rule that is broken; `what` names the frame.
    """
    if not isinstance(bars, pd.DataFrame):
        raise DataError(f"{what}: expected a pandas DataFrame, got {type(bars).__name__}")
    if list(bars.columns) != list(BAR_COLUMNS):
        raise DataError(
            f"{what}: columns must be exactly {list(BAR_COLUMNS)}, got {list(bars.columns)}"
        )
    for name in BAR_COLUMNS:
        wanted = "float64" if name in VALUE_COLUMNS else "int64"
        if str(bars[name].dtype) != wanted:
            unit = "" if name in VALUE_COLUMNS else " UTC nanoseconds"
            raise DataError(f"{what}: column {name} must be {wanted}{unit}, is {bars[name].dtype}")
    if not isinstance(bars.index, pd.RangeIndex) or (len(bars) and bars.index[0] != 0):
        raise DataError(f"{what}: the index must be a RangeIndex starting at 0")
    ts_open = bars["ts_open"].to_numpy()
    if len(ts_open) > 1 and np.any(ts_open[1:] <= ts_open[:-1]):
        raise DataError(f"{what}: rows must be sorted by ts_open with no duplicates")
    if np.any(bars["ts_close"].to_numpy() <= ts_open):
        raise DataError(f"{what}: every ts_close must be after its ts_open")
    return bars


def _is_factory(candidate: object) -> bool:
    """A class or a plain callable builds a provider; an object with `bars` is one."""
    return isinstance(candidate, type) or (callable(candidate) and not hasattr(candidate, "bars"))


def register_provider(name: str, provider: object, *, replace: bool = False) -> None:
    """Register a provider instance, class or factory under `name`.

    A class or factory is called by `get_provider(name, **options)`; an instance is
    returned as it is.

    Raises ConfigError for an empty name, a name already taken (unless `replace`)
    or an instance that does not satisfy the protocol.
    """
    if not isinstance(name, str) or not name:
        raise ConfigError("provider name must be a non-empty string")
    if name in _REGISTRY.providers and not replace:
        raise ConfigError(
            f"a data provider named {name!r} is already registered; pass replace=True to "
            "override it"
        )
    if not _is_factory(provider):
        validate_provider(provider)
    _REGISTRY.providers[name] = provider


def unregister_provider(name: str) -> None:
    """Remove a provider from the registry. A name that is not registered is ignored."""
    _REGISTRY.providers.pop(name, None)


def available_providers() -> list[str]:
    """Names of every registered provider, built-in and discovered, sorted."""
    discover_providers()
    return sorted(_REGISTRY.providers)


def get_provider(name: str, **options: object) -> DataProvider:
    """Return the provider registered as `name`, building it with `options` if it
    was registered as a class or factory.

    Raises ConfigError for an unknown name (saying so when its entry point failed
    to load), for options given to an instance and for a result that does not
    satisfy the protocol.
    """
    if name not in _REGISTRY.providers:
        discover_providers()
    if name not in _REGISTRY.providers:
        failure = _REGISTRY.load_errors.get(name)
        detail = f" Its entry point failed to load: {failure}." if failure else ""
        raise ConfigError(
            f"unknown data provider {name!r}. Registered: "
            f"{', '.join(sorted(_REGISTRY.providers))}.{detail} Third-party providers register "
            f"through the entry-point group {ENTRY_POINT_GROUP!r} or register_provider()."
        )
    candidate = _REGISTRY.providers[name]
    if _is_factory(candidate):
        try:
            candidate = candidate(**options)
        except TypeError as exc:
            raise ConfigError(f"data provider {name!r} could not be created: {exc}") from exc
    elif options:
        raise ConfigError(
            f"data provider {name!r} is registered as an instance and takes no options "
            f"(got {sorted(options)})"
        )
    return validate_provider(candidate)


def discover_providers(*, reload: bool = False) -> list[str]:
    """Load third-party providers from the entry-point group.

    Runs once per process unless `reload=True`. An entry point that fails to load is
    not registered; the failure is kept (see `provider_load_errors`) and issued as a
    RuntimeWarning, so a broken plugin is never silently missing. A plugin cannot
    replace a built-in provider. Returns the names that were added.
    """
    if _REGISTRY.discovered and not reload:
        return []
    _REGISTRY.discovered = True
    added = []
    for entry in _metadata.entry_points(group=ENTRY_POINT_GROUP):
        failure = _load_entry_point(entry)
        if failure is None:
            _REGISTRY.load_errors.pop(entry.name, None)
            added.append(entry.name)
        else:
            _REGISTRY.load_errors[entry.name] = failure
            warnings.warn(
                f"data provider entry point {entry.name!r} in {ENTRY_POINT_GROUP!r} was not "
                f"registered: {failure}",
                RuntimeWarning,
                stacklevel=2,
            )
    return added


def _load_entry_point(entry: _metadata.EntryPoint) -> str | None:
    """Register one entry point. Returns None on success, else why it was skipped."""
    if entry.name in _BUILTIN_PROVIDERS:
        return "the name is reserved for a built-in provider"
    try:
        loaded = entry.load()
        if not _is_factory(loaded):
            validate_provider(loaded)
    except Exception as exc:
        # A plugin can fail in any way while importing. Whatever it raised is
        # reported under its name instead of breaking every other provider.
        return f"{type(exc).__name__}: {exc}"
    _REGISTRY.providers[entry.name] = loaded
    return None


def provider_load_errors() -> dict[str, str]:
    """Entry points that failed to load, as {name: reason}."""
    return dict(_REGISTRY.load_errors)


__all__ = [
    "DataProvider",
    "ENTRY_POINT_GROUP",
    "available_providers",
    "discover_providers",
    "get_provider",
    "provider_load_errors",
    "register_provider",
    "unregister_provider",
    "validate_provider",
]
