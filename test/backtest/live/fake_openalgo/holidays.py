"""The market holidays the fake OpenAlgo answers `market/holidays` with: the rows of the 2026
calendar OpenAlgo ships (database/market_calendar_db.py, `shipped_holidays_2026`, from the
NSE and MCX circulars for 2026), in the shape its service answers them
(services/market_calendar_service.py): `date`, `description`, `holiday_type`,
`closed_exchanges` and `open_exchanges` (each `exchange`, `start_time` and `end_time`,
epoch milliseconds). A year without rows answers an empty list, as the server does.
"""

from __future__ import annotations

from typing import Any

#: Every exchange but MCX, which closes them on most holidays and keeps its evening.
_EQUITY = ["NSE", "BSE", "NFO", "BFO", "CDS", "BCD"]
#: Every exchange, closed on the days MCX closes too.
_ALL = [*_EQUITY, "MCX"]
#: The Muhurat session of Sunday 2026-11-08, 14:20 to 15:30 IST (equity derivatives to
#: 15:40), as epoch milliseconds.
_MUHURAT_OPEN = 1794127800000
_MUHURAT_CLOSE = 1794132000000
_MUHURAT_DERIVATIVES_CLOSE = 1794132600000


def _evening(start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    """MCX's evening session (17:00 to 23:55 IST) of a holiday of the other exchanges."""
    return [{"exchange": "MCX", "start_time": start_ms, "end_time": end_ms}]


def _row(date: str, description: str, closed: list[str], opened: list[dict[str, Any]]):
    return {
        "date": date,
        "description": description,
        "holiday_type": "TRADING_HOLIDAY",
        "closed_exchanges": list(closed),
        "open_exchanges": list(opened),
    }


_MUHURAT = {
    "date": "2026-11-08",
    "description": "Diwali Laxmi Pujan (Muhurat Trading)",
    "holiday_type": "SPECIAL_SESSION",
    "closed_exchanges": [],
    "open_exchanges": [
        {
            "exchange": exchange,
            "start_time": _MUHURAT_OPEN,
            "end_time": _MUHURAT_DERIVATIVES_CLOSE
            if exchange in ("NFO", "BFO")
            else _MUHURAT_CLOSE,
        }
        for exchange in ("NSE", "BSE", "NFO", "BFO", "CDS", "BCD", "MCX")
    ],
}

#: OpenAlgo's 2026 rows, in date order.
HOLIDAYS_2026: tuple[dict[str, Any], ...] = (
    _row("2026-01-15", "Municipal Corporation Election - Maharashtra", _EQUITY,
         _evening(1768476600000, 1768501500000)),
    _row("2026-01-26", "Republic Day", _ALL, []),
    _row("2026-03-03", "Holi", _EQUITY, _evening(1772537400000, 1772562300000)),
    _row("2026-03-26", "Shri Ram Navami", _EQUITY, _evening(1774524600000, 1774549500000)),
    _row("2026-03-31", "Shri Mahavir Jayanti", _EQUITY, _evening(1774956600000, 1774981500000)),
    _row("2026-04-03", "Good Friday", _ALL, []),
    _row("2026-04-14", "Dr. Baba Saheb Ambedkar Jayanti", _EQUITY,
         _evening(1776166200000, 1776191100000)),
    _row("2026-05-01", "Maharashtra Day", _EQUITY, _evening(1777635000000, 1777659900000)),
    _row("2026-05-28", "Bakri Id", _EQUITY, _evening(1779967800000, 1779992700000)),
    _row("2026-06-26", "Muharram", _EQUITY, _evening(1782473400000, 1782498300000)),
    _row("2026-09-14", "Ganesh Chaturthi", _EQUITY, _evening(1789385400000, 1789410300000)),
    _row("2026-10-02", "Mahatma Gandhi Jayanti", _ALL, []),
    _row("2026-10-20", "Dussehra", _EQUITY, _evening(1792495800000, 1792520700000)),
    _MUHURAT,
    _row("2026-11-10", "Diwali Balipratipada", _EQUITY, _evening(1794310200000, 1794335100000)),
    _row("2026-11-24", "Prakash Gurpurb Sri Guru Nanak Dev", _EQUITY,
         _evening(1795519800000, 1795544700000)),
    _row("2026-12-25", "Christmas", _ALL, []),
)  # fmt: skip


def holidays_answer(year: int, rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """The server's answer to `market/holidays` for `year` with `rows` (those of the year)."""
    kept = [dict(row) for row in rows if row["date"].startswith(f"{year}-")]
    return {"status": "success", "year": year, "timezone": "Asia/Kolkata", "data": kept}


__all__ = ["HOLIDAYS_2026", "holidays_answer"]
