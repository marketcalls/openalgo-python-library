"""Exception types shared by the backtest package.

Every error the engine raises on purpose derives from BacktestError, so callers can
catch one type. Messages say what was wrong and what to do about it.
"""


class BacktestError(Exception):
    """Base class for all errors raised by openalgo.backtest."""


class ConfigError(BacktestError, ValueError):
    """An argument or combination of arguments is invalid."""


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


__all__ = [
    "BacktestError",
    "CalendarError",
    "ConfigError",
    "CostError",
    "DataError",
    "DataQualityError",
    "MissingDependencyError",
    "NoDataError",
    "ReferenceDataError",
    "SymbolNotFoundError",
    "TimestampError",
]
