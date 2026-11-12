"""The built-in holiday data, checked against the facts its sources state.

Every expected value below is copied from a cited document, not read back from the
data files: the NSE circulars and BSE notices listed in each file's `sources` (for
example NSE/CMTR/42877 for the 2020 holidays, NSE/CMTR/43290 for the session of
2020-02-01, NSE/CMTR/57285 for the moved Bakri Id holiday of 2023), and the vendor or
broker pages for the dates no exchange document covers, which must then be unverified.
"""

import datetime as dt
import json

import pytest
from calendar_fixtures.builders import at_ist

from openalgo.backtest.calendar import HOLIDAY_EXCHANGES, default_calendar
from openalgo.backtest.calendar_data import builtin_documents

CLOSED = None

# (exchange, date, session as ((open hour, minute), (close hour, minute)) or CLOSED)
SESSIONS = [
    # 2019: NSE/CMTR/39612, 40536, NSE/MSD/42319, NSE/CMTR/42483, NSE/FAOP/39613, BSE 20181207-13.
    ("NSE", "2019-03-04", CLOSED),
    ("NFO", "2019-03-04", CLOSED),
    ("BSE", "2019-04-17", CLOSED),
    ("NSE", "2019-04-29", CLOSED),
    ("NSE", "2019-10-21", CLOSED),
    ("CDS", "2019-10-21", CLOSED),
    ("NSE", "2019-10-27", ((18, 15), (19, 15))),  # Muhurat, a Sunday
    ("NSE", "2019-10-28", CLOSED),
    # 2020: NSE/CMTR/42877, NSE/FAOP/42878, NSE/CMTR/43290, NSE/FAOP/43291.
    ("NSE", "2020-02-01", ((9, 15), (15, 30))),  # Union Budget, a Saturday
    ("NFO", "2020-02-01", ((9, 15), (15, 30))),
    ("NSE", "2020-02-21", CLOSED),
    ("NFO", "2020-12-25", CLOSED),
    ("NSE", "2020-11-14", ((18, 15), (19, 15))),  # Muhurat, a Saturday
    ("MCX", "2020-04-02", CLOSED),  # vendor table: no evening session (lockdown)
    # 2021: NSE/CMTR/46623, NSE/CMTR/50050, BSE 20201210-7.
    ("BSE", "2021-01-26", CLOSED),
    ("NSE", "2021-11-04", ((18, 15), (19, 15))),  # Muhurat on a trading holiday
    ("NSE", "2021-11-05", CLOSED),
    ("MCX", "2021-03-11", ((17, 0), (23, 55))),  # evening only, before the US switch
    # 2022 and 2023: NSE/CMTR/50560, NSE/CMTR/54757 and 57285, BSE 20230627-2.
    ("NSE", "2022-10-26", CLOSED),
    ("NSE", "2023-06-28", ((9, 15), (15, 30))),  # the original Bakri Id date
    ("NSE", "2023-06-29", CLOSED),
    ("BSE", "2023-06-29", CLOSED),
    ("NSE", "2023-11-12", ((18, 15), (19, 15))),  # Muhurat, a Sunday
    # 2024: NSE/CMTR/59722, 60338, 61518, 64960.
    ("NSE", "2024-01-22", CLOSED),
    ("NSE", "2024-05-20", CLOSED),
    ("NSE", "2024-11-20", CLOSED),
    ("NSE", "2024-11-01", ((18, 0), (19, 0))),  # Muhurat
    # 2025: NSE/CMTR/65587.
    ("NSE", "2025-10-21", ((13, 45), (14, 45))),  # Muhurat on a trading holiday
    ("NSE", "2025-10-22", CLOSED),
    ("NSE", "2025-11-05", CLOSED),
    ("NSE", "2025-11-14", ((9, 15), (15, 30))),  # a normal day (the server's 2025 seed is wrong)
    ("MCX", "2025-04-18", CLOSED),
    ("MCX", "2025-02-26", ((17, 0), (23, 55))),
    # 2026: NSE/CMTR/71775, NSE/FAOP/71777, NSE/FAOP/72352.
    ("NSE", "2026-03-03", CLOSED),
    ("NFO", "2026-03-26", CLOSED),
    ("NSE", "2026-11-10", CLOSED),
    ("NFO", "2026-02-01", ((9, 15), (15, 30))),  # Union Budget, a Sunday
    ("NSE", "2026-11-08", ((14, 20), (15, 30))),  # Muhurat, a Sunday
    ("MCX", "2026-05-01", ((17, 0), (23, 30))),
]


@pytest.mark.parametrize(("exchange", "day", "expected"), SESSIONS)
def test_sessions_match_the_cited_sources(exchange, day, expected):
    date = dt.date.fromisoformat(day)
    if expected is CLOSED:
        assert default_calendar().session(exchange, date) is None
    else:
        open_, close = expected
        assert default_calendar().session(exchange, date) == (
            at_ist(date, *open_),
            at_ist(date, *close),
        )


@pytest.mark.parametrize(
    ("exchange", "years", "level"),
    [
        ("NSE", range(2019, 2027), "sourced"),  # NSE circulars for every year
        ("NFO", range(2019, 2027), "sourced"),  # NSE F&O circulars for every year
        ("BSE", (2019, 2020, 2021, 2023), "sourced"),  # BSE notices read
        ("BSE", (2022, 2024, 2025, 2026), "unverified"),  # assumed or vendor only
        ("BFO", range(2019, 2027), "unverified"),
        ("MCX", (2020, 2021, 2023, 2025, 2026), "unverified"),  # mcxindia.com unread
        ("MCX", (2019, 2022, 2024), "template"),
        ("CDS", (2025, 2026), "unverified"),
        ("CDS", (2019, 2024), "template"),
        ("NSE", (2018, 2027), "template"),
        ("NSE_INDEX", (2026,), "sourced"),
    ],
)
def test_year_coverage_is_sourced_only_from_exchange_documents(exchange, years, level):
    assert {default_calendar().coverage(exchange, year) for year in years} == {level}


@pytest.mark.parametrize(
    ("year", "sessions"),
    [
        (2019, 261 - 17 + 1),  # 17 closures, the Muhurat Sunday
        (2020, 262 - 12 + 2),  # 12 closures, the Budget and Muhurat Saturdays
        (2021, 261 - 13),  # 13 full closures; the Muhurat weekday is a session
        (2022, 260 - 12),
        (2023, 260 - 15 + 1),  # 15 closures, the Muhurat Sunday
        (2024, 262 - 16 + 3),  # 16 closures, 3 Saturday sessions
        (2025, 261 - 13 + 1),  # 13 closures, the Union Budget Saturday
        (2026, 261 - 16 + 2),  # 16 closures, the Budget and Muhurat Sundays
    ],
)
def test_nse_sessions_per_year_are_weekdays_less_closures_plus_special_sessions(year, sessions):
    assert len(default_calendar().sessions("NSE", f"{year}-01-01", f"{year}-12-31")) == sessions


@pytest.mark.parametrize(
    ("exchange", "day", "level"),
    [
        ("NSE", "2019-04-29", "sourced"),  # NSE/CMTR/40536
        ("BSE", "2019-04-29", "unverified"),  # the BSE notice was not found
        ("NFO", "2023-06-29", "unverified"),  # the F&O change was not read
        ("NSE", "2026-01-15", "unverified"),  # not in NSE/CMTR/71775
        ("NSE", "2026-11-08", "sourced"),  # announced in NSE/CMTR/71775
        ("MCX", "2020-04-02", "unverified"),  # a vendor table
    ],
)
def test_single_entries_carry_their_own_level(exchange, day, level):
    assert default_calendar().coverage(exchange, dt.date.fromisoformat(day)) == level


@pytest.mark.parametrize(
    ("exchange", "day", "unverified"),
    [
        ("NSE", "2019-10-27", False),  # NSE/CMTR/42483 states the times
        ("NFO", "2019-10-27", True),  # a broker bulletin
        ("NFO", "2020-02-01", False),  # NSE/FAOP/43291
        ("BSE", "2020-02-01", True),  # assumed equal to NSE
        ("NSE", "2026-11-08", True),  # the server and a broker page
        ("MCX", "2026-03-03", True),  # the MCX regular close is unverified
    ],
)
def test_session_times_are_flagged_unless_an_exchange_document_states_them(
    exchange, day, unverified
):
    assert default_calendar().session_info(exchange, day).hours_unverified is unverified


def test_holidays_frame_names_the_2026_nse_closures_and_special_sessions():
    frame = default_calendar().holidays("NSE", 2026)
    closures = frame[frame["open"].isna()]
    special = frame[frame["open"].notna()]
    assert len(closures) == 16
    assert special["date"].tolist() == [dt.date(2026, 2, 1), dt.date(2026, 11, 8)]
    assert frame.loc[frame["date"] == dt.date(2026, 3, 3), "name"].item() == "Holi"


def test_sources_cite_the_exchange_documents_of_a_year():
    sources = default_calendar().sources(2020)
    assert "CMTR42877" in sources["nse-cmtr-42877"]
    assert default_calendar().sources(2030) == {}


DOCUMENTS = builtin_documents()
# Figure dash, en dash, em dash and horizontal bar (U+2012 to U+2015).
NON_HYPHEN_DASHES = [chr(code) for code in range(0x2012, 0x2016)]
EXCHANGE_DOCUMENT_PREFIXES = ("nse-cmtr-", "nse-faop-", "nse-msd-", "bse-")


def test_one_document_per_year_from_2019_to_2026():
    assert [document["year"] for document in DOCUMENTS] == list(range(2019, 2027))


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda document: str(document["year"]))
def test_every_entry_cites_a_source_listed_in_its_file(document):
    for entry in document["days"]:
        assert entry["sources"], entry["date"]
        assert set(entry["sources"]) <= set(document["sources"]), entry["date"]


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda document: str(document["year"]))
def test_a_sourced_year_cites_an_exchange_document(document):
    if "sourced" in document["coverage"].values():
        assert any(source.startswith(EXCHANGE_DOCUMENT_PREFIXES) for source in document["sources"])


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda document: str(document["year"]))
def test_entries_are_dated_inside_their_year_and_name_known_exchanges(document):
    for entry in document["days"]:
        assert dt.date.fromisoformat(entry["date"]).year == document["year"]
        named = set(entry["closed"]) | set(entry["sessions"])
        assert named <= set(HOLIDAY_EXCHANGES), entry["date"]
        assert not set(entry["closed"]) & set(entry["sessions"]), entry["date"]
        assert set(entry.get("unverified", ())) <= named, entry["date"]
        assert set(entry.get("hours_unverified", ())) <= set(entry["sessions"]), entry["date"]
    assert set(document["coverage"]) <= set(HOLIDAY_EXCHANGES)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda document: str(document["year"]))
def test_documents_are_plain_ascii_without_dashes_other_than_hyphens(document):
    text = json.dumps(document, ensure_ascii=False)
    assert text.isascii()
    assert not any(dash in text for dash in NON_HYPHEN_DASHES)
