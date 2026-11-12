"""Where each row's bar opens and closes, checked against the session calendar.

Intraday: a stamp is the instant its bar opens (label="open") or closes
(label="close"). `place_intraday` turns stamps into bar opens on the session lattice
and `check_intraday` runs the calendar's validator (`Calendar.validate_intraday`): a
5 hour 30 minute offset, bars stamped at their close and bars off the lattice are a
TimestampError, and the calendar's `finding` chooses the source-specific advice. Two
checks need the declared interval or label, so the calendar cannot make them and
they are made here: bars coarser than the declared interval, and bars declared
close-labelled that are stamped at their open.

Daily and longer: `place_rows` maps each stamp to its IST date (D) or to the period
that holds it (W, M, Q, Y), with the session bounds from the calendar, and
`check_rows` rejects rows that cannot be told apart or are dated a day off.

Nothing is ever shifted.
"""

import numpy as np
import pandas as pd

from .._calendar_convert import format_ist, from_day
from .._calendar_types import PLACEMENTS
from .._layout import IST_OFFSET_NANOS
from ..errors import TimestampError
from ._sessions import Sessions, day_start_ns, format_span, ist_days

IN_SESSION = PLACEMENTS.index("in_session")
# Bars that all sit on a grid coarser than the declared interval are that coarser
# interval when they fill more than this share of its slots; a thinly traded series
# that lands on such a grid by chance fills far less.
_COARSE_GRID_FILL = 0.5
# pandas period frequencies of the calendar intervals; a week runs Monday to Sunday.
PERIOD_FREQUENCY = {"W": "W-SUN", "M": "M", "Q": "Q", "Y": "Y"}
# Stamps of daily rows accepted without a note: 00:00 UTC of the IST date (the
# OpenAlgo convention) and IST midnight, as nanoseconds after IST midnight.
_USUAL_STAMP_TIMES = (IST_OFFSET_NANOS, 0)


def place_intraday(
    stamps_ns: np.ndarray, step_ns: int, label: str, sessions: Sessions
) -> np.ndarray:
    """Bar opens (UTC nanoseconds) of rows stamped at their open or their close.

    A bar that closes at C opened at the lattice point before C. That is C minus one
    interval, except for the last bar of a session whose length is not a whole number
    of intervals (15:15 to 15:30 for 30 minute bars on NSE).
    """
    if label != "close":
        return stamps_ns
    opens, closes, has = sessions.bounds(stamps_ns - 1)
    closes_in_session = has & (stamps_ns > opens) & (stamps_ns <= closes)
    lattice_open = opens + ((stamps_ns - opens - 1) // step_ns) * step_ns
    return np.where(closes_in_session, lattice_open, stamps_ns - step_ns)


def bar_closes(
    ts_open: np.ndarray, in_session: np.ndarray, step_ns: int, sessions: Sessions
) -> np.ndarray:
    """Bar closes: one interval after the open, cut at the session close for bars in
    the session (the short last bar of a session)."""
    _, session_closes, _ = sessions.bounds(ts_open)
    one_step = ts_open + step_ns
    return np.where(in_session, np.minimum(one_step, session_closes), one_step)


def check_intraday(
    ts_open: np.ndarray,
    step_ns: int,
    label: str,
    sessions: Sessions,
    *,
    tag: str,
    interval: str,
    hints: dict[str, str],
) -> dict:
    """Validate sorted bar opens against the calendar; returns its check report (the
    dict of `Calendar.check_intraday`, with `placement` one code per bar).

    tag names the series in messages; hints are appended to an error, keyed by the
    calendar's `finding` or "other". Bars on a closed day are counted, not an error.

    Raises TimestampError for what the calendar rejects, for bars coarser than the
    declared interval and for label="close" bars that are stamped at their open.
    """
    native, options = sessions.native, {"exchange": sessions.exchange, "interval": interval}
    try:
        counts = native.validate_intraday(ts_open, allow_closed_days=True, **options)
    except TimestampError as exc:
        finding = native.check_intraday(ts_open, **options)["finding"] or "other"
        hint = hints.get(finding, hints.get("other", ""))
        raise TimestampError(f"{tag}: {exc}. {hint}".rstrip(" .") + ".") from exc
    in_session = counts["placement"] == IN_SESSION
    opens, session_closes, has = sessions.bounds(ts_open)
    _raise_if_coarser(
        ts_open[in_session], opens[in_session], session_closes[in_session], step_ns, tag
    )
    if label == "close":
        _raise_if_open_labelled(ts_open, opens, session_closes, has, step_ns, tag)
    return counts


def _raise_if_coarser(
    ts_open: np.ndarray, opens: np.ndarray, session_closes: np.ndarray, step_ns: int, tag: str
) -> None:
    """Raise when the bars in the session are those of a coarser interval.

    Such bars pass every lattice check (09:15, 09:20, ... are whole minutes), but each
    would be closed one declared interval after its open while its prices run to the
    end of the coarser bar: look-ahead.
    """
    offsets = ts_open - opens
    grid_ns = int(np.gcd.reduce(offsets)) if len(offsets) else 0
    if grid_ns < 2 * step_ns or grid_ns % step_ns:
        return
    _, first_of_day = np.unique(opens, return_index=True)
    slots = int(np.ceil((session_closes - opens)[first_of_day] / grid_ns).sum())
    if len(offsets) <= _COARSE_GRID_FILL * slots:
        return
    raise TimestampError(
        f"{tag}: the bars are {format_span(grid_ns)} apart ({len(offsets)} of {slots} "
        f"slots of a {format_span(grid_ns)} grid are filled), coarser than the declared "
        f"interval of {format_span(step_ns)}. Each bar would be closed "
        f"{format_span(step_ns)} after it opens while its prices run for "
        f"{format_span(grid_ns)}. Declare the interval the data really has."
    )


def _raise_if_open_labelled(
    ts_open: np.ndarray,
    opens: np.ndarray,
    session_closes: np.ndarray,
    has: np.ndarray,
    step_ns: int,
    tag: str,
) -> None:
    """Raise when data declared as close-labelled looks open-labelled.

    Open-labelled bars read as close-labelled all move one bar earlier: most days
    then have a bar one interval before the session open and no day has a bar in the
    last slot of the session.
    """
    one_before_open = has & (ts_open == opens - step_ns)
    last_slot = opens + ((session_closes - opens - 1) // step_ns) * step_ns
    days = len(np.unique(opens[has]))
    days_before = len(np.unique(opens[one_before_open]))
    days_with_last_slot = len(np.unique(opens[has & (ts_open == last_slot)]))
    if 2 * days_before > days and days_with_last_slot == 0:
        example = format_ist(int(ts_open[one_before_open][0]))
        raise TimestampError(
            f"{tag}: label='close' was declared, but the bars look stamped at their "
            f"open: on {days_before} of {days} session days a bar would open one interval "
            f"before the session open (for example {example}) and no day has a bar in the "
            "last slot of the session. Declare label='open'. Nothing was shifted."
        )


# --- daily and longer ----------------------------------------------------------------


def place_rows(
    stamps_ns: np.ndarray,
    unit: str,
    canonical_input: bool,
    sessions: Sessions,
    tag: str,
    notes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(ts_open, ts_close, first_day, last_day, has_session) of daily or longer rows.

    A D row is its IST date's session; a W, M, Q or Y row runs from the open of the
    first session of its period to the close of the last one. A row without a session
    gets zeros and has_session False. Unless `canonical_input`, the stamps must share
    one time of day; an unusual one adds a sentence to `notes`.

    Raises TimestampError for rows stamped at different times of day.
    """
    days = ist_days(stamps_ns)
    if not canonical_input and len(stamps_ns):
        _check_stamp_convention(stamps_ns - day_start_ns(days), tag, notes)
    if unit == "D":
        opens, closes, has = sessions.bounds_of_days(days)
        return opens, closes, days, days, has
    periods = pd.PeriodIndex(days.astype("datetime64[D]"), freq=PERIOD_FREQUENCY[unit])
    first = periods.start_time.to_numpy().astype("datetime64[D]").astype("int64")
    last = periods.end_time.to_numpy().astype("datetime64[D]").astype("int64")
    opens, closes, has = sessions.period_bounds(first, last)
    # A period is measured by its sessions: January 2024 runs from the 2nd when the
    # 1st is a holiday.
    first = np.where(has, ist_days(opens), first)
    last = np.where(has, ist_days(closes), last)
    return opens, closes, first, last, has


def check_rows(
    first_day: np.ndarray,
    has_session: np.ndarray,
    unit: str,
    sessions: Sessions,
    *,
    tag: str,
    regular_only: bool,
) -> np.ndarray:
    """Which daily or longer rows fall on a date without a session (to be dropped).

    Raises TimestampError for two rows on one date or in one period, for daily rows
    that fit the calendar better moved by one day, and, with `regular_only` False,
    for any row on a closed date (it has no session open or close).
    """
    if len(first_day) > 1 and np.any(first_day[1:] == first_day[:-1]):
        what = "on the same IST date" if unit == "D" else "in the same period"
        raise TimestampError(
            f"{tag}: several rows fall {what}. The data is finer than the declared "
            f"interval {unit!r}, or mixes conventions."
        )
    closed = ~has_session
    if not closed.any():
        return closed
    if unit == "D":
        _raise_if_dated_wrong(first_day, closed, sessions, tag)
    if not regular_only:
        raise TimestampError(
            f"{tag}: {int(closed.sum())} rows fall on dates the {sessions.exchange} "
            f"calendar marks closed (for example {from_day(first_day[closed][0])}), so they "
            "have no session open or close. Add the session to the calendar or load with "
            "regular_session_only=True."
        )
    return closed


def _check_stamp_convention(time_of_day_ns: np.ndarray, tag: str, notes: list[str]) -> None:
    kinds = np.unique(time_of_day_ns)
    if len(kinds) > 1:
        shown = ", ".join(format_span(kind) for kind in kinds[:3])
        raise TimestampError(
            f"{tag}: rows mix timestamp conventions: they are stamped at {len(kinds)} "
            f"different times of day ({shown} after IST midnight). A daily or longer series "
            "must use one convention (OpenAlgo uses 00:00 UTC of the IST session date). "
            "Nothing was changed."
        )
    if int(kinds[0]) not in _USUAL_STAMP_TIMES:
        notes.append(
            f"rows are stamped {format_span(kinds[0])} after IST midnight, not at 00:00 UTC "
            "of the session date or at IST midnight; each was mapped to its IST calendar date"
        )


def _raise_if_dated_wrong(
    days: np.ndarray, closed: np.ndarray, sessions: Sessions, tag: str
) -> None:
    """Raise when moving every daily row by one day makes more rows fit the calendar."""
    as_delivered = len(days) - int(closed.sum())
    for shift, word, direction in ((1, "late", "back"), (-1, "early", "forward")):
        fit = int(sessions.bounds_of_days(days - shift)[2].sum())
        if fit > as_delivered:
            example = from_day(days[closed][0])
            raise TimestampError(
                f"{tag}: daily rows look dated one day {word}. As delivered, "
                f"{as_delivered} of {len(days)} rows fall on {sessions.exchange} session days "
                f"(for example {example} is closed); moved one day {direction}, {fit} fit. "
                "Likely cause: a midnight stamp was converted between UTC and IST. Nothing "
                "was shifted."
            )
