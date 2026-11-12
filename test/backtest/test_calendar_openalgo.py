"""Loading the calendar from the shape OpenAlgo `holidays(year)` returns.

The responses here are written by hand in the server's shape
(`services/market_calendar_service.py`, `database/market_calendar_db.py`); no server is
contacted. A fake client stands in for `openalgo.api`.
"""

import datetime as dt

import pytest
from calendar_fixtures.builders import at_ist, epoch_ms

from openalgo.backtest.calendar import Calendar
from openalgo.backtest.errors import CalendarError, ConfigError

HOLI = dt.date(2026, 3, 3)  # Tuesday, before the United States switches to daylight time
MAHARASHTRA_DAY = dt.date(2026, 5, 1)  # Friday, United States on daylight time
MUHURAT = dt.date(2026, 11, 8)  # Sunday
SETTLEMENT = dt.date(2026, 2, 19)  # Thursday
EQUITY_AND_CURRENCY = ["NSE", "BSE", "NFO", "BFO", "CDS", "BCD"]


def window(exchange: str, day: dt.date, open_: tuple[int, int], close: tuple[int, int]) -> dict:
    return {
        "exchange": exchange,
        "start_time": epoch_ms(day, *open_),
        "end_time": epoch_ms(day, *close),
    }


def holiday_row(day: dt.date, description: str, kind: str, closed=(), opened=()) -> dict:
    return {
        "date": day.isoformat(),
        "description": description,
        "holiday_type": kind,
        "closed_exchanges": list(closed),
        "open_exchanges": list(opened),
    }


def holidays_2026() -> dict:
    """A `holidays(2026)` response with one row of each kind the server stores."""
    mcx_evening = lambda day: [window("MCX", day, (17, 0), (23, 55))]  # noqa: E731
    rows = [
        holiday_row(SETTLEMENT, "Chhatrapati Shivaji Maharaj Jayanti", "SETTLEMENT_HOLIDAY"),
        holiday_row(HOLI, "Holi", "TRADING_HOLIDAY", EQUITY_AND_CURRENCY, mcx_evening(HOLI)),
        holiday_row(
            MAHARASHTRA_DAY,
            "Maharashtra Day",
            "TRADING_HOLIDAY",
            EQUITY_AND_CURRENCY,
            mcx_evening(MAHARASHTRA_DAY),
        ),
        holiday_row(
            MUHURAT,
            "Diwali Laxmi Pujan (Muhurat Trading)",
            "SPECIAL_SESSION",
            opened=[
                window("NSE", MUHURAT, (14, 20), (15, 30)),
                window("NFO", MUHURAT, (14, 20), (15, 40)),
                window("NCO", MUHURAT, (14, 20), (15, 30)),  # not an engine exchange
            ],
        ),
    ]
    return {"status": "success", "year": 2026, "timezone": "Asia/Kolkata", "data": rows}


@pytest.fixture
def loaded() -> Calendar:
    cal = Calendar(builtin=False)
    cal.add_holidays(holidays_2026(), source="test server", verified=True)
    return cal


def test_closed_exchanges_of_a_trading_holiday_are_closed(loaded):
    for exchange in EQUITY_AND_CURRENCY:
        assert loaded.session(exchange, HOLI) is None


def test_a_special_session_opens_the_listed_exchanges_only(loaded):
    assert loaded.session("NSE", MUHURAT) == (at_ist(MUHURAT, 14, 20), at_ist(MUHURAT, 15, 30))
    assert loaded.session("NFO", MUHURAT) == (at_ist(MUHURAT, 14, 20), at_ist(MUHURAT, 15, 40))
    assert loaded.session("BSE", MUHURAT) is None  # not listed: the Sunday rule applies


def test_a_settlement_holiday_leaves_trading_on_regular_hours(loaded):
    assert loaded.session("NSE", SETTLEMENT) == (
        at_ist(SETTLEMENT, 9, 15),
        at_ist(SETTLEMENT, 15, 30),
    )


def test_rows_loaded_as_verified_mark_the_year_sourced(loaded):
    assert (loaded.coverage("NSE", 2026), loaded.coverage("MCX", 2026)) == ("sourced", "sourced")
    assert loaded.coverage("NSE", 2025) == "template"


def test_rows_are_unverified_by_default():
    cal = Calendar(builtin=False)
    cal.add_holidays(holidays_2026())
    assert (cal.coverage("NSE", 2026), cal.coverage("NSE", HOLI)) == ("unverified", "unverified")


def test_only_the_requested_exchanges_are_loaded_and_marked():
    cal = Calendar(builtin=False)
    cal.add_holidays(holidays_2026(), exchanges=["NSE"])
    assert cal.session("NSE", HOLI) is None
    assert cal.is_open("BSE", HOLI)
    assert cal.coverage("BSE", 2026) == "template"


def test_the_summary_counts_what_was_loaded():
    summary = Calendar(builtin=False).add_holidays(holidays_2026())
    expected = {"year": 2026, "rows": 4, "closures": 12, "sessions": 4, "kept": 0}
    assert summary == {**expected, "coverage_marked": True, "issues": []}


def test_holiday_names_are_kept_for_the_holidays_frame(loaded):
    names = loaded.holidays("NSE", 2026)["name"].tolist()
    assert names == ["Holi", "Maharashtra Day", "Diwali Laxmi Pujan (Muhurat Trading)"]


def test_the_data_list_alone_is_accepted_when_the_year_is_clear():
    cal = Calendar(builtin=False)
    assert cal.add_holidays(holidays_2026()["data"])["year"] == 2026
    assert cal.session("NSE", HOLI) is None


def test_an_empty_year_marks_nothing_and_is_reported():
    cal = Calendar(builtin=False)
    summary = cal.add_holidays({"status": "success", "year": 2023, "data": []}, source="server")
    assert summary["coverage_marked"] is False
    assert cal.coverage("NSE", 2023) == "template"
    assert cal.issues == [
        "server: no holiday rows for 2023; the year stays as it was (template-only for "
        "exchanges without other data)"
    ]


def test_an_error_response_is_a_calendar_error():
    response = {"status": "error", "message": "Year must be between 2020 and 2050"}
    with pytest.raises(CalendarError, match="Year must be between 2020 and 2050"):
        Calendar(builtin=False).add_holidays(response, year=2019)


def test_a_row_dated_outside_the_year_is_a_calendar_error():
    rows = [holiday_row(dt.date(2025, 12, 25), "Christmas", "TRADING_HOLIDAY", closed=["NSE"])]
    with pytest.raises(CalendarError, match="2025-12-25 was supplied for year 2026"):
        Calendar(builtin=False).add_holidays(rows, year=2026)


def test_an_unknown_holiday_type_is_a_calendar_error():
    rows = [holiday_row(HOLI, "Holi", "HALF_DAY", closed=["NSE"])]
    with pytest.raises(CalendarError, match="unknown holiday_type 'HALF_DAY'"):
        Calendar(builtin=False).add_holidays(rows)


# Rows of the OpenAlgo server's 2025 seed (database/market_calendar_db.py, commit
# d8cc13f24) in the API shape, epoch values as stored. Every one is wrong: 2025-10-21
# was the Muhurat session, the 2025-11-01 windows are those of 2024-11-01, and Guru
# Nanak Jayanti fell on 2025-11-05, not 2025-11-14 (NSE/CMTR/65587).
SEED_2025 = {
    "status": "success",
    "year": 2025,
    "data": [
        {
            "date": "2025-10-21",
            "description": "Dussehra",
            "holiday_type": "TRADING_HOLIDAY",
            "closed_exchanges": EQUITY_AND_CURRENCY,
            "open_exchanges": [
                {"exchange": "MCX", "start_time": 1761091800000, "end_time": 1761145500000}
            ],
        },
        {
            "date": "2025-11-01",
            "description": "Diwali Laxmi Pujan (Muhurat Trading)",
            "holiday_type": "SPECIAL_SESSION",
            "closed_exchanges": [],
            "open_exchanges": [
                {"exchange": "NSE", "start_time": 1730469000000, "end_time": 1730473500000}
            ],
        },
        {
            "date": "2025-11-14",
            "description": "Guru Nanak Jayanti",
            "holiday_type": "TRADING_HOLIDAY",
            "closed_exchanges": EQUITY_AND_CURRENCY,
            "open_exchanges": [
                {"exchange": "MCX", "start_time": 1763152200000, "end_time": 1763205900000}
            ],
        },
    ],
}


class FakeClient:
    """Answers `holidays(year)` from memory and records the years asked for."""

    def __init__(self, responses: dict[int, dict]) -> None:
        self.responses = responses
        self.asked: list[int] = []

    def holidays(self, year: int) -> dict:
        self.asked.append(year)
        return self.responses.get(year, {"status": "success", "year": year, "data": []})


@pytest.fixture
def from_seed() -> Calendar:
    return Calendar.from_openalgo(FakeClient({2025: SEED_2025}), [2025], strict=False)


def test_unverified_server_rows_do_not_close_a_sourced_trading_day(from_seed):
    day = dt.date(2025, 11, 14)
    assert from_seed.session("NSE", day) == (at_ist(day, 9, 15), at_ist(day, 15, 30))


def test_unverified_server_rows_do_not_remove_a_sourced_special_session(from_seed):
    day = dt.date(2025, 10, 21)
    assert from_seed.session("NSE", day) == (at_ist(day, 13, 45), at_ist(day, 14, 45))


def test_unverified_server_rows_never_raise_coverage_to_sourced(from_seed):
    assert from_seed.coverage("BSE", 2025) == "unverified"
    assert from_seed.coverage("NSE", dt.date(2025, 11, 14)) == "sourced"  # the built-in list


def test_each_kept_disagreement_and_skipped_window_is_reported(from_seed):
    issues = " ".join(from_seed.issues)
    assert "closes NSE on 2025-11-14" in issues
    assert "closes NSE on 2025-10-21" in issues
    assert "the NSE window on 2025-11-01" in issues


def test_server_rows_fill_a_year_without_built_in_data():
    row = holiday_row(dt.date(2027, 1, 26), "Republic Day", "TRADING_HOLIDAY", closed=["NSE"])
    cal = Calendar.from_openalgo(
        FakeClient({2027: {"status": "success", "year": 2027, "data": [row]}}), [2027]
    )
    assert cal.session("NSE", dt.date(2027, 1, 26)) is None
    assert cal.coverage("NSE", 2027) == "unverified"


def test_verified_rows_replace_built_in_rules_for_their_dates():
    row = holiday_row(
        MUHURAT, "Muhurat", "SPECIAL_SESSION", opened=[window("NSE", MUHURAT, (18, 0), (19, 0))]
    )
    cal = Calendar(builtin=True)
    cal.add_holidays([row], verified=True)
    assert cal.session("NSE", MUHURAT) == (at_ist(MUHURAT, 18), at_ist(MUHURAT, 19))


def test_from_openalgo_asks_once_per_year_and_loads_each_response():
    client = FakeClient({2026: holidays_2026()})
    cal = Calendar.from_openalgo(client, [2025, 2026], builtin=False)
    assert client.asked == [2025, 2026]
    assert cal.session("NSE", HOLI) is None
    assert (cal.coverage("NSE", 2026), cal.coverage("NSE", 2025)) == ("unverified", "template")
    assert len(cal.issues) == 1  # 2025 came back empty


SEED_2025_WINDOW = SEED_2025["data"][2]  # the MCX window decodes to the early hours of the 15th


def test_a_window_outside_its_date_is_a_calendar_error_that_names_it():
    with pytest.raises(CalendarError) as raised:
        Calendar(builtin=False).add_holidays([SEED_2025_WINDOW])
    message = str(raised.value)
    assert "the MCX window on 2025-11-14 runs from" in message
    assert "strict=False" in message


def test_a_failed_load_changes_nothing():
    cal = Calendar(builtin=False)
    with pytest.raises(CalendarError):
        cal.add_holidays([SEED_2025_WINDOW])
    assert cal.is_open("NSE", "2025-11-14")
    assert cal.coverage("NSE", 2025) == "template"


def test_a_load_that_skipped_rows_is_unverified_even_when_asked_as_verified():
    cal = Calendar(builtin=False)
    summary = cal.add_holidays([SEED_2025_WINDOW], strict=False, verified=True)
    assert cal.session("NSE", "2025-11-14") is None  # the valid part of the row is applied
    assert cal.coverage("NSE", 2025) == "unverified"
    assert len(summary["issues"]) == 1 and cal.issues == summary["issues"]


def test_the_server_mcx_close_of_2355_yields_to_the_regular_close_by_default(loaded):
    # 1 May 2026: the United States is on daylight time, so MCX closes at 23:30.
    assert loaded.session("MCX", MAHARASHTRA_DAY) == (
        at_ist(MAHARASHTRA_DAY, 17),
        at_ist(MAHARASHTRA_DAY, 23, 30),
    )


def test_the_server_mcx_close_is_kept_when_asked():
    cal = Calendar(builtin=False)
    cal.add_holidays(holidays_2026(), mcx_close="server")
    assert cal.session("MCX", MAHARASHTRA_DAY)[1] == at_ist(MAHARASHTRA_DAY, 23, 55)


def test_an_unknown_mcx_close_mode_is_a_config_error():
    with pytest.raises(ConfigError, match="mcx_close"):
        Calendar(builtin=False).add_holidays(holidays_2026(), mcx_close="guess")
