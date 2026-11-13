"""Which sessions a source answered with no rows the cache records as fetched.

A session the calendar has but the source does not (a holiday the calendar does not
know, such as 2018-12-25 in years before its holiday lists begin) would otherwise be
unsettled for ever and asked for again on every load, as a range of its own. Recording
it as fetched with no rows settles it like any other session (`cache.is_settled`).

Two kinds of empty answer are not recorded, so they are asked for again:

- Any answer from Historify ("db"). It is filled by downloads, so a day it lacks now
  may be there after the next download.
- A session before the oldest session with rows (cached or fetched now). That is
  where the source's history starts; asking again costs one request per load and keeps
  the short-history warning saying why.
"""

import datetime as dt
from collections.abc import Iterable, Sequence

from .._calendar_convert import to_day


def empty_sessions_to_record(
    source: str,
    answered: Sequence[tuple[str, str]],
    sessions: Iterable[dt.date],
    days_with_rows: set[int],
) -> list[dt.date]:
    """The `sessions` that were answered with no rows and are recorded as fetched.

    source: "api" (a broker) or "db" (Historify). answered: the (first, last) IST dates,
    as ISO strings, of the windows the source answered (`FetchReport.answered`).
    sessions: the IST session dates the load fetched. days_with_rows: day numbers (days
    since 1970-01-01) of every date that has rows, in the cache or in this fetch.
    """
    if source != "api" or not days_with_rows:
        return []
    oldest_with_rows = min(days_with_rows)
    windows = [
        (dt.date.fromisoformat(first), dt.date.fromisoformat(last)) for first, last in answered
    ]
    return [
        day
        for day in sessions
        if oldest_with_rows < to_day(day)
        and to_day(day) not in days_with_rows
        and any(first <= day <= last for first, last in windows)
    ]


__all__ = ["empty_sessions_to_record"]
