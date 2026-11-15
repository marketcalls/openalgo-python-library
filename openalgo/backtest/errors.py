"""Exception types shared by the backtest package.

Every error the engine raises on purpose derives from BacktestError, so callers can
catch one type. Messages say what was wrong and what to do about it. The warnings a run
gives when orders were rejected or skipped (OrdersNotPlacedWarning) and when the strategy
waited inside the run for its indicators (WarmUpWarning) are here too, so they can be
filtered by their class.
"""


class BacktestError(Exception):
    """Base class for all errors raised by openalgo.backtest."""


class ConfigError(BacktestError, ValueError):
    """An argument or combination of arguments is invalid."""


class IndicatorOutsideInitError(ConfigError):
    """`self.I` was called after `init` (in `on_bar`, a timer, a callback or `on_stop`).
    Indicators are registered in `init`: the history before `start` is loaded for them
    and the strategy waits for their warm-up before its first decision, both settled
    before the first bar, so an indicator registered later would be neither."""


class DataError(BacktestError):
    """Market data is missing, malformed or fails a quality check."""


class DataQualityError(DataError):
    """Market data failed a quality check at error severity.

    `report` holds the quality report that describes every failed check.
    """

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


class NoDataError(DataError):
    """A source returned no rows for the requested symbol, interval and range."""


class SymbolNotFoundError(DataError):
    """The OpenAlgo server does not know the symbol on the exchange: it is not in the
    master contract the server downloaded from the broker (a contract that has expired,
    a misspelt symbol, or master contracts not downloaded yet).

    `symbol` and `exchange` name what was asked for, as OpenAlgo writes them.
    """

    def __init__(self, message, symbol=None, exchange=None):
        super().__init__(message)
        self.symbol = symbol
        self.exchange = exchange


class ServerError(DataError):
    """The OpenAlgo server could not answer a history request: it could not be reached,
    it kept answering that requests were too frequent (HTTP 429), or it answered with an
    error, such as a broker login that has expired. Check that OpenAlgo is running and
    logged in to the broker, then run again.

    `rate_limited` is True when the server kept answering HTTP 429: raise `pacing=` or
    try again later.
    """

    def __init__(self, message, rate_limited=False):
        super().__init__(message)
        self.rate_limited = rate_limited


class TimestampError(DataError):
    """Timestamps do not match the declared convention or the session calendar."""


class MissingDependencyError(BacktestError, ImportError):
    """An optional package needed for the requested feature is not installed."""


class CalendarError(BacktestError):
    """A session calendar is missing or inconsistent for a requested date."""


class ReferenceDataError(BacktestError):
    """A reference table has no row covering the requested instrument and date."""


class CostError(BacktestError, ValueError):
    """A cost segment, preset or schedule is unknown or inconsistent."""


class LiveError(BacktestError):
    """A live session (`bt.live`) cannot start or go on; the message says what to do."""


class PreflightError(LiveError):
    """A check before a live session failed. `failures` holds the message of every
    failed check, each saying what to fix."""

    def __init__(self, message, failures=()):
        super().__init__(message)
        self.failures = tuple(failures)


class ModeChangedError(LiveError):
    """OpenAlgo's analyzer switch changed during a live session (or semi-auto mode
    appeared): the session sends nothing more, and is started again in the mode the
    user wants. The mode is chosen inside OpenAlgo, never in code."""


class JournalError(LiveError):
    """The live session's journal cannot be opened, read or written."""


class OrdersNotPlacedWarning(UserWarning):
    """Orders of a run were not placed: rejected (margin, the product rule, a quantity
    off the lot size, ...) or skipped before they were placed (a sized quantity of less
    than one share or lot; the kernel gives a sizing function's 0 the same reason, so it
    is counted too). `bt.backtest` issues it once, at the end of such a run, with how
    many and the most common reason; the details of every reason are in
    `result.warnings`. Orders a rule of the run refused as it intends (the MIS entry
    window, a daily loss limit, a cooldown, max_positions, a size cap) do not raise it.
    A warning, not an error: the result is still returned. Silence it with
    `warnings.simplefilter("ignore", OrdersNotPlacedWarning)`."""


class WarmUpWarning(UserWarning):
    """The strategy waited inside the run for its indicators to warm up, so it made no
    decision for a part of the run (or none at all). `bt.backtest` issues it once, at the
    end of such a run, with the first decision and the strategy bars each symbol waited;
    why, and what to do, is in `result.warnings`. A warning, not an error: the result is
    still returned. Silence it with `warnings.simplefilter("ignore", WarmUpWarning)`."""


__all__ = [
    "BacktestError",
    "CalendarError",
    "ConfigError",
    "CostError",
    "DataError",
    "DataQualityError",
    "IndicatorOutsideInitError",
    "JournalError",
    "LiveError",
    "MissingDependencyError",
    "ModeChangedError",
    "NoDataError",
    "OrdersNotPlacedWarning",
    "PreflightError",
    "ReferenceDataError",
    "ServerError",
    "SymbolNotFoundError",
    "TimestampError",
    "WarmUpWarning",
]
