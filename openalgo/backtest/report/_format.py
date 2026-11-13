"""Numbers, dates and durations as report text.

Every function returns HTML-safe text: digits, signs, letters and character entities
only, so the result can be placed in the page as it is. Rupee amounts use Indian
digit grouping (12,34,567.89).
A figure the data cannot define (NaN, None) reads "n/a".
"""

import html
import math

import pandas as pd

from ._definitions import DEFINITIONS

RUPEE = "&#8377;"
INFINITY = "&#8734;"
MISSING = "n/a"
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3_600
_SECONDS_PER_DAY = 86_400


def text(value: object) -> str:
    """Escape user text for HTML."""
    return html.escape(str(value), quote=True)


def is_missing(value: object) -> bool:
    """True for None, NaN and NaT."""
    return value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value))


def _special(value: float) -> str | None:
    if is_missing(value):
        return MISSING
    if math.isinf(value):
        return INFINITY if value > 0 else "-" + INFINITY
    return None


def indian_digits(value: float, decimals: int = 2) -> str:
    """A number with Indian grouping: 1234567.891 is "12,34,567.89"."""
    sign = "-" if value < 0 else ""
    whole, _, fraction = f"{abs(value):.{decimals}f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    grouped = ",".join([*groups, tail])
    return sign + grouped + (f".{fraction}" if fraction else "")


def rupees(value: float, decimals: int = 2) -> str:
    """A rupee amount: "&#8377;12,34,567.89", negative as "-&#8377;500.00"."""
    special = _special(value)
    if special is not None:
        return special
    sign = "-" if value < 0 else ""
    return f"{sign}{RUPEE}{indian_digits(abs(value), decimals)}"


def percent(value: float, decimals: int = 2) -> str:
    """A fraction as a percentage: 0.1234 is "12.34%"."""
    special = _special(value)
    return special if special is not None else f"{value * 100:.{decimals}f}%"


def number(value: float, decimals: int = 2) -> str:
    """A plain number with Indian grouping."""
    special = _special(value)
    return special if special is not None else indian_digits(value, decimals)


def duration(seconds: float) -> str:
    """Elapsed time in words: "45 min", "3 h 20 min", "2 d 4 h"."""
    special = _special(seconds)
    if special is not None:
        return special
    total = round(seconds)
    if total < _SECONDS_PER_HOUR:
        return f"{total // _SECONDS_PER_MINUTE} min"
    if total < _SECONDS_PER_DAY:
        hours, rest = divmod(total, _SECONDS_PER_HOUR)
        return f"{hours} h {rest // _SECONDS_PER_MINUTE} min"
    days, rest = divmod(total, _SECONDS_PER_DAY)
    return f"{days} d {rest // _SECONDS_PER_HOUR} h"


def moment(value: object) -> str:
    """An IST instant as "12 Nov 2026 09:15"."""
    return MISSING if is_missing(value) else pd.Timestamp(value).strftime("%d %b %Y %H:%M")


_NUMERIC_UNITS = {
    "fraction": percent,
    "rupees": rupees,
    "ratio": number,
    "sessions": number,
    "count": lambda value: indian_digits(value, 0),
    "days": lambda value: f"{number(value, 1)} d",
    "seconds": duration,
}


def figure(key: str, value: object) -> str:
    """A metric value formatted by the unit in `DEFINITIONS`."""
    unit = DEFINITIONS[key].unit
    if is_missing(value):
        return MISSING
    if unit in _NUMERIC_UNITS:
        return _NUMERIC_UNITS[unit](float(value))
    if unit == "timestamp":
        return moment(value)
    if unit == "flag":
        return "yes" if value else "no"
    return text(value) if value != "" else "none"


__all__ = [
    "INFINITY",
    "MISSING",
    "RUPEE",
    "duration",
    "figure",
    "indian_digits",
    "is_missing",
    "moment",
    "number",
    "percent",
    "rupees",
    "text",
]
