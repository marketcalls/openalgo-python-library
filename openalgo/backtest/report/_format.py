"""Numbers, dates and durations as report text.

Every function returns HTML-safe text: digits, signs, letters and character entities
only, so the result can be placed in the page as it is. Rupee amounts use Indian
digit grouping (12,34,567.89) and the lakh and crore units in compact form.
A figure the data cannot define (NaN, None) reads "n/a".
"""

import html
import math

import pandas as pd

from ._definitions import DEFINITIONS

RUPEE = "&#8377;"
INFINITY = "&#8734;"
MISSING = "n/a"
#: Indian compact units, largest first: (size in rupees, suffix).
COMPACT_UNITS = ((10_000_000, "Cr"), (100_000, "L"), (1_000, "K"))
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


def rupees_compact(value: float) -> str:
    """A short rupee amount for axes and tiles: "&#8377;12.5 L", "&#8377;1.2 Cr"."""
    special = _special(value)
    if special is not None:
        return special
    sign = "-" if value < 0 else ""
    for size, suffix in COMPACT_UNITS:
        if abs(value) >= size:
            scaled = abs(value) / size
            digits = 0 if scaled >= 100 or scaled == int(scaled) else 1
            return f"{sign}{RUPEE}{scaled:.{digits}f} {suffix}"
    return f"{sign}{RUPEE}{abs(value):.0f}"


def _tick_decimals(step: float) -> int:
    """Decimals that show every multiple of `step` exactly (up to four)."""
    for decimals in range(5):
        if abs(round(step, decimals) - step) <= 1e-9 * max(abs(step), 1.0):
            return decimals
    return 4


def _tick_step(ticks: list[float]) -> float:
    gaps = [abs(b - a) for a, b in zip(ticks, ticks[1:], strict=False) if b != a]
    return min(gaps) if gaps else 1.0


def rupee_ticks(ticks: list[float]) -> list[str]:
    """Axis labels for rupee ticks: one compact unit for the whole axis (crore, lakh,
    thousand or none, by the largest tick) and enough decimals to tell them apart."""
    biggest = max((abs(t) for t in ticks), default=0.0)
    size, suffix = next(((s, f" {x}") for s, x in COMPACT_UNITS if biggest >= s), (1, ""))
    decimals = _tick_decimals(_tick_step(ticks) / size)
    return [f"{'-' if t < 0 else ''}{RUPEE}{abs(t) / size:.{decimals}f}{suffix}" for t in ticks]


def percent_ticks(ticks: list[float]) -> list[str]:
    """Axis labels for ticks that are fractions, with enough decimals to tell them apart."""
    decimals = _tick_decimals(_tick_step(ticks) * 100)
    return [percent(t, decimals) for t in ticks]


def percent(value: float, decimals: int = 2, *, signed: bool = False) -> str:
    """A fraction as a percentage: 0.1234 is "12.34%" ("+12.34%" when signed)."""
    special = _special(value)
    if special is not None:
        return special
    shown = f"{value * 100:+.{decimals}f}" if signed else f"{value * 100:.{decimals}f}"
    return shown + "%"


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


def day(value: object) -> str:
    """A date as "12 Nov 2026"."""
    return MISSING if is_missing(value) else pd.Timestamp(value).strftime("%d %b %Y")


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
    "day",
    "duration",
    "figure",
    "indian_digits",
    "is_missing",
    "moment",
    "number",
    "percent",
    "percent_ticks",
    "rupee_ticks",
    "rupees",
    "rupees_compact",
    "text",
]
