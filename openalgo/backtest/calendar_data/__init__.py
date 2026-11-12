"""Built-in holiday and special-session data for `openalgo.backtest.calendar`.

One JSON file per calendar year, `holidays_<year>.json`, read by `builtin_documents`
and loaded by `openalgo.backtest.calendar.Calendar` when it is created with
`builtin=True`. Times are IST clock times. Every entry names the sources it came from;
nothing here is inferred without being marked.

File layout (format 1):

    {
      "format": 1,
      "year": 2026,
      "timezone": "Asia/Kolkata",
      "coverage": {"NSE": "sourced", "BSE": "unverified"},
      "notes": ["free text about how the year was put together"],
      "sources": {"source-id": "citation: the document and its URL"},
      "days": [
        {
          "date": "2026-03-03",
          "name": "Holi",
          "closed": ["NSE", "BSE", "NFO", "BFO", "CDS", "BCD"],
          "sessions": {"MCX": ["17:00", null]},
          "sources": ["source-id"],
          "unverified": ["BSE", "MCX"],
          "hours_unverified": ["MCX"],
          "note": "free text"
        }
      ]
    }

- `coverage` lists the exchanges whose holiday list for the whole year is in the file.
  "sourced": the list was read from an exchange document (an NSE circular, a BSE
  notice). "unverified": it comes from a data vendor, a broker page or the OpenAlgo
  server, or is assumed equal to another exchange's list. An exchange that is not
  listed has no holiday list for that year: the calendar reports the year as
  template-only (weekday rule) even when the file holds single dated entries for it,
  such as a Muhurat session.
- `closed` are the exchanges with no session on the date.
- `sessions` are the exchanges that trade with explicit times, [open, close]. A close of
  null means the exchange's regular close for that date, resolved when the calendar is
  asked (for MCX it depends on United States daylight saving time).
- `unverified` (optional) are the exchanges whose open or closed status on this date is
  not stated by an exchange document. `hours_unverified` (optional) are the exchanges
  whose session times on this date are not.

To extend the data, edit or add a file here and cite the source of every date in
`sources`. Do not add a date without one.
"""

import json
from importlib import resources

_FILE_PREFIX = "holidays_"
_FILE_SUFFIX = ".json"


def builtin_documents() -> list[dict]:
    """The built-in holiday files as parsed documents, ordered by year."""
    files = sorted(
        (entry for entry in resources.files(__name__).iterdir() if _is_holiday_file(entry.name)),
        key=lambda entry: entry.name,
    )
    return [json.loads(entry.read_text(encoding="utf-8")) for entry in files]


def _is_holiday_file(name: str) -> bool:
    return name.startswith(_FILE_PREFIX) and name.endswith(_FILE_SUFFIX)


__all__ = ["builtin_documents"]
