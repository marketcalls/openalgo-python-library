"""Split and bonus detection from the bars themselves (FR-DATA-10, FR-MKT-10; PRD 06
section 5 and PRD 04 section 7).

OpenAlgo's sources supply no corporate-action data, and the history they return may not
be adjusted for splits and bonuses. A split or bonus of factor f (shares held after the
event per share held before it, see `corporate_actions`) divides the price by f
overnight, so an unadjusted series opens the ex-date at about the previous close divided
by f: a return that never happened. The check reads every boundary between two sessions
that have bars: the last close of the earlier session, the first open of the later one
(the ex-date) and the last close of that later session.

A boundary is a suspected unadjusted split or bonus when both:

1. the gap ratio (previous close / open) is within FACTOR_TOLERANCE of a usual factor
   (USUAL_FACTORS). The smallest usual factor, 4/3 for a 1:3 bonus, lies beyond the
   widest daily price band (WIDEST_PRICE_BAND), so no session inside its band and no
   circuit-limit day opens that far from the previous close. A share with derivatives
   has dynamic bands, so a genuine gap that large is rare but possible: it is declared
   with `CorporateAction.none`. Smaller bonuses (1:4, 1:5, 1:10) move the price no
   further than an ordinary session can and are not detected;
2. at least PERSISTENCE of the gap is still there at the ex-date session's close: a
   split moves the price level for good, a bad print at the open does not.

What the user declared (`CorporateAction`) decides what a boundary means:

| Declared on the boundary | Gap | Status | Severity |
|---|---|---|---|
| nothing | suspected | suspected | error |
| a split or bonus | within FACTOR_TOLERANCE of its factor | unadjusted | warning |
| a split or bonus | beyond the widest price band, at another factor | mismatch | error |
| a split or bonus | an ordinary move | adjusted | info |
| no corporate action | suspected | acknowledged | info |

A declaration applies to the first boundary whose later session is on or after its
ex-date, so it still finds its gap when the bars miss the ex-date session.

The check runs on shares on NSE and BSE at intraday and daily intervals. An index has
no corporate actions, and adjusting derivative contracts at a corporate action is
FR-MKT-14 (Phase 6), so a series on another exchange is listed under `not_run`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._calendar_convert import from_day, to_day
from .._layout import parse_interval
from ..errors import ConfigError
from ._sessions import ist_days
from ._severity import ERROR, INFO, WARNING
from .corporate_actions import NONE, CorporateAction, shares_text

if TYPE_CHECKING:
    from .quality import QualityReport

CHECK = "split_or_bonus"
UNADJUSTED_CHECK = "split_or_bonus_unadjusted"
ADJUSTED_CHECK = "split_or_bonus_adjusted"
ACKNOWLEDGED_CHECK = "price_move_acknowledged"

SUSPECTED = "suspected"
MISMATCH = "mismatch"
UNADJUSTED = "unadjusted"
ADJUSTED = "adjusted"
ACKNOWLEDGED = "acknowledged"

# The exchanges whose series are shares (and exchange-traded funds), which can split.
CASH_EXCHANGES = ("NSE", "BSE")
# The widest daily price band of a share on NSE and BSE as a fraction of the previous
# close: the bands are 2, 5, 10 and 20 percent, and shares with derivatives have dynamic
# bands instead (the exchanges' price band framework; not verified against the circular
# text). It places the smallest usual factor beyond ordinary moves.
WIDEST_PRICE_BAND = 0.20
# How far the gap ratio may be from a factor, as a fraction of the factor. The ex-date
# opens at the adjusted price moved by that day's opening move, rarely more than 5
# percent; the closest usual factors (3/2 and 5/3) are 11 percent apart, so the windows
# of two factors never overlap.
FACTOR_TOLERANCE = 0.05
# The share of the gap, in log terms, still there at the ex-date session's close. Half:
# a new price level keeps most of the gap whatever the day brings, while a bad print at
# the open is undone within the session.
PERSISTENCE = 0.5
# The usual factors on NSE and BSE and what each would be: splits between the face
# values 10, 5, 2 and 1 rupees, the commonly announced bonus ratios (new:held) whose
# factor lies beyond the widest price band, and consolidations (splits upward).
USUAL_FACTORS: tuple[tuple[float, str], ...] = (
    (4 / 3, "a 1:3 bonus"),
    (3 / 2, "a 1:2 bonus"),
    (5 / 3, "a 2:3 bonus"),
    (2.0, "a 1:1 bonus, or a split from face value 10 to 5 or 2 to 1"),
    (5 / 2, "a 3:2 bonus, or a split from face value 5 to 2"),
    (3.0, "a 2:1 bonus"),
    (4.0, "a 3:1 bonus"),
    (5.0, "a 4:1 bonus, or a split from face value 10 to 2 or 5 to 1"),
    (10.0, "a split from face value 10 to 1"),
    (1 / 2, "a consolidation from face value 5 to 10 or 1 to 2"),
    (2 / 5, "a consolidation from face value 2 to 5"),
    (1 / 5, "a consolidation from face value 2 to 10 or 1 to 5"),
    (1 / 10, "a consolidation from face value 1 to 10"),
)
_FACTORS = np.array([factor for factor, _ in USUAL_FACTORS])
ADVICE = (
    "Supply bars adjusted for it (data=), run on dates that do not span it, or declare "
    "what happened with corporate_actions=[...] of openalgo.backtest.data.CorporateAction: "
    "CorporateAction.split(ex_date, from_face_value, to_face_value) or "
    "CorporateAction.bonus(ex_date, bonus, held) for a split or bonus, or "
    "CorporateAction.none(ex_date) when the move is genuine"
)
# The issues of the check: (name, severity, statuses, message with {count} and {gaps}).
_ISSUES = (
    (
        CHECK,
        ERROR,
        (SUSPECTED, MISMATCH),
        "{count} suspected unadjusted split or bonus ({gaps}). " + ADVICE,
    ),
    (
        UNADJUSTED_CHECK,
        WARNING,
        (UNADJUSTED,),
        "the bars are not adjusted for {count} declared corporate actions ({gaps}). Adjusting "
        "for them is FR-MKT-11 (Phase 5): until then an indicator whose window spans an "
        "ex-date reads prices from both sides of it, and a backtest that holds a position or "
        "a GTT across one stops with NotImplementedError",
    ),
    (
        ADJUSTED_CHECK,
        INFO,
        (ADJUSTED,),
        "the bars are already adjusted for {count} declared corporate actions ({gaps})",
    ),
    (
        ACKNOWLEDGED_CHECK,
        INFO,
        (ACKNOWLEDGED,),
        "{count} gaps that look like a split or bonus are read as genuine price moves, as "
        "declared with CorporateAction.none ({gaps})",
    ),
)


@dataclass(frozen=True)
class PriceGap:
    """A boundary between two sessions that the split check reports. Prices are
    rupees, instants UTC nanoseconds.

    ex_date: ISO IST date of the later session, the first at the new price.
    ex_open_ns: the open of that session's first bar.
    previous_close, open, session_close: the last close before the ex-date, the first
        open on it and its last close.
    ratio: previous_close / open (2.0 when the price halved).
    factor: the declared factor (the product of the declarations on the boundary), or
        the usual factor the gap is near.
    status: "suspected", "mismatch", "unadjusted", "adjusted" or "acknowledged" (see
        the module docstring).
    reading: the declaration in words, or what the usual factor would be.
    """

    ex_date: str
    ex_open_ns: int
    previous_close: float
    open: float
    session_close: float
    ratio: float
    factor: float
    status: str
    reading: str

    def describe(self) -> str:
        """The gap and the evidence, in one clause for a message."""
        prices = (
            f"the previous close {self.previous_close:.2f} against the open {self.open:.2f} "
            f"is a ratio of {self.ratio:.3f}"
        )
        if self.status == SUSPECTED:
            return (
                f"{self.ex_date}: {prices}, near {shares_text(self.factor)} ({self.reading}), "
                f"and the session closed at {self.session_close:.2f}, keeping the new level"
            )
        if self.status == MISMATCH:
            return f"{self.ex_date}: {self.reading} is declared, but {prices}"
        if self.status == ACKNOWLEDGED:
            return f"{self.ex_date}: {prices}, near {shares_text(self.factor)}"
        return f"{self.ex_date}: {self.reading}; {prices}"


@dataclass(frozen=True)
class _Boundaries:
    """The session boundaries of a series: for each later session, its IST day number,
    the open of its first bar, the previous close, its first open and its last close."""

    first_day: int
    days: np.ndarray
    ex_open_ns: np.ndarray
    previous_close: np.ndarray
    open: np.ndarray
    session_close: np.ndarray

    @classmethod
    def of(cls, bars: pd.DataFrame) -> _Boundaries:
        """The boundaries of `bars`, read in time order (none for fewer than two
        sessions)."""
        order = np.argsort(bars["ts_open"].to_numpy(dtype="int64"), kind="stable")
        ts_open = bars["ts_open"].to_numpy(dtype="int64")[order]
        days = ist_days(ts_open)
        starts = np.flatnonzero(np.diff(days)) + 1
        ends = np.append(starts, len(days))[1:] - 1  # the bar before the next start
        closes = bars["close"].to_numpy(dtype="float64")[order]
        opens = bars["open"].to_numpy(dtype="float64")[order]
        first_day = int(days[0]) if len(days) else 0
        return cls(
            first_day,
            days[starts],
            ts_open[starts],
            closes[starts - 1],
            opens[starts],
            closes[ends],
        )


def _beyond_band(ratio: float) -> bool:
    """True when an open at previous close / ratio is further from the previous close
    than the widest daily price band allows."""
    return abs(1.0 / ratio - 1.0) > WIDEST_PRICE_BAND


def _declared_by_boundary(
    boundaries: _Boundaries, declared: Sequence[CorporateAction]
) -> dict[int, list[CorporateAction]]:
    """The declarations on each boundary (by index), each on the first boundary whose
    later session is on or after its ex-date. Raises ConfigError when "none" and a split
    or bonus fall on one boundary."""
    found: dict[int, list[CorporateAction]] = {}
    for action in declared:
        day = to_day(action.ex_date)
        index = int(np.searchsorted(boundaries.days, day, side="left"))
        if index == len(boundaries.days) or day <= boundaries.first_day:
            continue  # on or before the first session, or after the last: no boundary
        found.setdefault(index, []).append(action)
    for index, actions in found.items():
        kinds = {action.kind for action in actions}
        if NONE in kinds and len(kinds) > 1:
            raise ConfigError(
                f"the declarations {[a.to_dict() for a in actions]} fall on the session of "
                f"{from_day(int(boundaries.days[index]))}: one says no corporate action, "
                "another a split or bonus"
            )
    return found


def find_gaps(bars: pd.DataFrame, declared: Sequence[CorporateAction] = ()) -> list[PriceGap]:
    """Every boundary of `bars` the split check reports, in date order.

    bars: canonical bars of one share in time order. declared: the checked declarations
        of `corporate_actions.declared`.

    Raises ConfigError when "none" and a split or bonus fall on one boundary.
    """
    b = _Boundaries.of(bars)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = b.previous_close / b.open
        usable = (b.previous_close > 0) & (b.open > 0) & (b.session_close > 0)
        usable &= np.isfinite(ratio) & np.isfinite(b.session_close)
        distance = np.abs(ratio[:, None] / _FACTORS[None, :] - 1.0)
        nearest = np.argmin(distance, axis=1)
        near = distance[np.arange(len(ratio)), nearest] <= FACTOR_TOLERANCE
        kept = np.log(b.previous_close / b.session_close) / np.log(ratio)
    suspected = usable & near & (kept >= PERSISTENCE)
    by_boundary = _declared_by_boundary(b, declared)
    candidates = sorted(set(np.flatnonzero(suspected).tolist()) | set(by_boundary))
    gaps = []
    for index in candidates:
        if not usable[index]:
            continue  # a price that is not usable is reported by the price checks
        status, factor, reading = _status(
            float(ratio[index]),
            by_boundary.get(index, []),
            bool(suspected[index]),
            int(nearest[index]),
        )
        if status is None:
            continue
        gaps.append(
            PriceGap(
                ex_date=from_day(int(b.days[index])).isoformat(),
                ex_open_ns=int(b.ex_open_ns[index]),
                previous_close=float(b.previous_close[index]),
                open=float(b.open[index]),
                session_close=float(b.session_close[index]),
                ratio=float(ratio[index]),
                factor=factor,
                status=status,
                reading=reading,
            )
        )
    return gaps


def _status(
    ratio: float, actions: list[CorporateAction], suspected: bool, nearest: int
) -> tuple[str | None, float, str]:
    """(status or None when there is nothing to report, factor, reading) of a boundary
    by the table of the module docstring."""
    if actions and actions[0].kind != NONE:
        factor = math.prod(action.factor for action in actions)
        reading = " and ".join(action.describe() for action in actions)
        if abs(ratio / factor - 1.0) <= FACTOR_TOLERANCE:
            return UNADJUSTED, factor, reading
        return (MISMATCH if _beyond_band(ratio) else ADJUSTED), factor, reading
    if not suspected:
        return None, 1.0, ""
    factor, reading = USUAL_FACTORS[nearest]
    return (ACKNOWLEDGED if actions else SUSPECTED), factor, reading


def check_splits(
    bars: pd.DataFrame,
    report: QualityReport,
    *,
    exchange: str | None,
    interval: str | None,
    declared: Sequence[CorporateAction] = (),
) -> None:
    """Run the split check on `bars` and record its findings in `report`: the issues by
    the table of the module docstring, every reported gap in `report.price_gaps`, or the
    reason it did not run in `report.not_run`.

    Raises ConfigError for declarations on a series that is not a share on NSE or BSE,
    or that conflict (see `find_gaps`).
    """
    require_share(exchange, declared)
    reason = _not_applicable(exchange, interval)
    if reason is not None:
        report.not_run.append(f"{CHECK}: {reason}")
        return
    gaps = find_gaps(bars, declared)
    report.price_gaps.extend(gaps)
    _record(report, gaps)


def require_share(exchange: str | None, declared: Sequence[CorporateAction]) -> None:
    """Raise ConfigError when corporate actions are declared for a series that is not a
    share on NSE or BSE (an index or a derivative contract)."""
    if declared and exchange not in CASH_EXCHANGES:
        raise ConfigError(
            f"corporate_actions apply to shares on {' and '.join(CASH_EXCHANGES)}; this "
            f"series is on {exchange}: leave them out"
        )


def _not_applicable(exchange: str | None, interval: str | None) -> str | None:
    """Why the check cannot run on a series, or None when it can."""
    if exchange is None or interval is None:
        return "the exchange or interval of the series is unknown"
    if exchange not in CASH_EXCHANGES:
        return (
            f"not checked on {exchange}: shares on NSE and BSE split, an index does not, and "
            "adjusting derivative contracts at a corporate action is FR-MKT-14 (Phase 6)"
        )
    if parse_interval(interval)[0] not in ("intraday", "D"):
        return (
            f"not checked for interval {interval!r}: a split inside a weekly or longer bar "
            "cannot be seen (load_bars checks the bars it builds them from)"
        )
    return None


def _record(report: QualityReport, gaps: list[PriceGap]) -> None:
    """Add the issues of the reported gaps to `report`, one per check of `_ISSUES`."""
    for check, severity, statuses, message in _ISSUES:
        chosen = [gap for gap in gaps if gap.status in statuses]
        text = "; ".join(gap.describe() for gap in chosen)
        days = [gap.ex_date for gap in chosen]
        report.add(check, severity, len(chosen), message.format(count=len(chosen), gaps=text), days)


__all__ = [
    "ACKNOWLEDGED_CHECK",
    "ADJUSTED_CHECK",
    "CASH_EXCHANGES",
    "CHECK",
    "FACTOR_TOLERANCE",
    "PERSISTENCE",
    "UNADJUSTED",
    "UNADJUSTED_CHECK",
    "USUAL_FACTORS",
    "WIDEST_PRICE_BAND",
    "PriceGap",
    "check_splits",
    "find_gaps",
    "require_share",
]
