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


class NoDataError(DataError):
    """A source returned no rows for the requested symbol, interval and range."""


class TimestampError(DataError):
    """Timestamps do not match the declared convention or the session calendar."""


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
    "NoDataError",
    "ReferenceDataError",
    "TimestampError",
]
