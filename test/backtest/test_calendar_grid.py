"""The union-calendar grid (FR-MKT-02; 03-architecture.md section 4.2).

The grid is what the kernel iterates: one slice per base bar of every exchange in the
run, on each exchange's bar lattice (its regular open plus whole intervals), with the
close of every exchange's bar at that slice.
"""

import datetime as dt

import numpy as np
import pytest
from calendar_fixtures.builders import at_ist, nse_session, opens_between

from openalgo.backtest.calendar import NO_BAR, Calendar, grid_mask
from openalgo.backtest.errors import CalendarError, ConfigError

MONDAY = dt.date(2025, 6, 2)  # MCX closes 23:30 (United States daylight saving)
HOLI = dt.date(2026, 3, 3)
SPECIAL_SUNDAY = dt.date(2026, 11, 8)


@pytest.fixture
def cal() -> Calendar:
    return Calendar(builtin=False)


def test_one_exchange_has_one_slice_per_base_bar_of_its_session(cal):
    grid = cal.union_grid("NSE", MONDAY, MONDAY, base="1m")
    np.testing.assert_array_equal(grid["ts"].to_numpy(), nse_session(MONDAY))
    assert grid["date"].dt.date.unique().tolist() == [MONDAY]


def test_grid_frame_names_its_exchanges_base_and_close_columns(cal):
    grid = cal.union_grid(["NSE", "MCX", "NSE"], MONDAY, MONDAY, base="15m")
    columns = ["ts", "date", "active", "session_open", "session_close", "close_NSE", "close_MCX"]
    assert list(grid.columns) == columns
    assert grid.attrs == {"exchanges": ["NSE", "MCX"], "base": "15m"}


def test_the_first_and_last_bar_of_a_session_are_marked(cal):
    grid = cal.union_grid("NSE", MONDAY, MONDAY, base="1h")
    assert grid["ts"].tolist() == [
        at_ist(MONDAY, hour, 15) for hour in range(9, 16)
    ]  # 09:15 to 15:15
    assert grid_mask(grid, "NSE", "session_open").tolist() == [True] + [False] * 6
    assert grid_mask(grid, "NSE", "session_close").tolist() == [False] * 6 + [True]


def test_the_last_bar_closes_at_the_session_close(cal):
    grid = cal.union_grid("NSE", MONDAY, MONDAY, base="1h")
    assert grid["close_NSE"].tolist() == [at_ist(MONDAY, hour, 15) for hour in range(10, 16)] + [
        at_ist(MONDAY, 15, 30)
    ]


def test_an_exchange_without_a_bar_at_a_slice_has_no_close_there(cal):
    grid = cal.union_grid(["NSE", "CDS"], MONDAY, MONDAY, base="1h")
    first_cds = grid["ts"] == at_ist(MONDAY, 9, 0)
    assert grid.loc[first_cds, "close_NSE"].item() == NO_BAR
    assert grid.loc[first_cds, "close_CDS"].item() == at_ist(MONDAY, 10, 0)


def test_daily_bars_of_two_exchanges_overlap_and_keep_their_own_closes(cal):
    grid = cal.union_grid(["MCX", "NSE"], MONDAY, MONDAY, base="D")
    assert grid["ts"].tolist() == [at_ist(MONDAY, 9, 0), at_ist(MONDAY, 9, 15)]
    assert grid["close_MCX"].tolist() == [at_ist(MONDAY, 23, 30), NO_BAR]
    assert grid["close_NSE"].tolist() == [NO_BAR, at_ist(MONDAY, 15, 30)]


def test_two_exchanges_share_slices_where_both_trade(cal):
    grid = cal.union_grid(["NSE", "MCX"], MONDAY, MONDAY, base="1m")
    np.testing.assert_array_equal(grid["ts"].to_numpy(), opens_between(MONDAY, (9, 0), (23, 30)))
    np.testing.assert_array_equal(grid["ts"][grid_mask(grid, "NSE")], nse_session(MONDAY))
    assert grid_mask(grid, "MCX").all()


def test_nse_closed_while_mcx_is_open_leaves_only_mcx_slices(cal):
    cal.add_closure("NSE", HOLI)
    cal.add_session("MCX", HOLI, "17:00")
    grid = cal.union_grid(["NSE", "MCX"], HOLI, HOLI, base="1m")
    np.testing.assert_array_equal(grid["ts"].to_numpy(), opens_between(HOLI, (17, 0), (23, 55)))
    assert not grid_mask(grid, "NSE").any()


def test_exchanges_with_different_opens_keep_their_own_lattice(cal):
    grid = cal.union_grid(["NSE", "CDS"], MONDAY, MONDAY, base="1h")
    nse = [at_ist(MONDAY, hour, 15) for hour in range(9, 16)]
    cds = [at_ist(MONDAY, hour, 0) for hour in range(9, 17)]
    assert grid["ts"][grid_mask(grid, "NSE")].tolist() == nse
    assert grid["ts"][grid_mask(grid, "CDS")].tolist() == cds
    assert grid["ts"].tolist() == sorted(nse + cds)


def test_the_daily_base_has_one_slice_per_session_at_its_open(cal):
    cal.add_closure("NSE", HOLI)
    monday = dt.date(2026, 3, 2)
    grid = cal.union_grid(["NSE", "MCX"], monday, HOLI, base="D")
    assert grid["ts"].tolist() == [at_ist(monday, 9, 0), at_ist(monday, 9, 15), at_ist(HOLI, 9, 0)]
    assert grid_mask(grid, "NSE").tolist() == [False, True, False]
    assert grid["session_open"].tolist() == grid["active"].tolist()


def test_a_special_session_is_gridded_on_the_regular_lattice(cal):
    cal.add_session("NSE", SPECIAL_SUNDAY, "14:20", "15:30")
    grid = cal.union_grid("NSE", SPECIAL_SUNDAY, SPECIAL_SUNDAY, base="15m")
    # 09:15 plus whole quarter hours: the first bar opens at 14:15, before the session.
    expected = [at_ist(SPECIAL_SUNDAY, 14, 15) + k * 15 * 60 * 10**9 for k in range(5)]
    assert grid["ts"].tolist() == expected


def test_a_range_without_sessions_gives_an_empty_grid(cal):
    grid = cal.union_grid("NSE", "2025-06-07", "2025-06-08", base="1m")  # a weekend
    assert len(grid) == 0
    assert grid["ts"].dtype == np.int64


def test_grid_mask_rejects_an_exchange_outside_the_grid(cal):
    grid = cal.union_grid("NSE", MONDAY, MONDAY, base="D")
    with pytest.raises(ConfigError, match="not part of this grid"):
        grid_mask(grid, "MCX")


@pytest.mark.parametrize("base", ["W", "M"])
def test_a_base_that_is_neither_intraday_nor_daily_is_a_config_error(cal, base):
    with pytest.raises(ConfigError, match="cannot be a base interval"):
        cal.union_grid("NSE", MONDAY, MONDAY, base=base)


def test_a_grid_above_the_size_limit_is_a_calendar_error(cal):
    with pytest.raises(CalendarError, match="grid"):
        cal.union_grid("MCX", "1990-01-01", "2040-12-31", base="1s")
