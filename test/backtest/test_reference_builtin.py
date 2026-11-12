"""The figures shipped in the built-in reference tables.

Every expected value below is copied from the exchange document named beside it (the
same document the CSV row cites), not computed by the code under test. Dates are chosen
on both sides of each change so that an off-by-one in a validity range fails a test.
"""

import re
import warnings

import pytest
from test_reference_helpers import FakeCalendar, day, write_file

from openalgo.backtest import reference as ref
from openalgo.backtest.errors import ConfigError, ReferenceDataError
from openalgo.backtest.reference import ReferenceDataWarning, ReferenceTables


@pytest.fixture(scope="module")
def tables():
    return ReferenceTables()


def _quiet(call):
    """Run a lookup with reference-data warnings silenced (the row status is tested elsewhere)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)
        return call()


def _dates(*texts):
    return [day(text) for text in texts]


# ---- lot sizes ------------------------------------------------------------------------

LOTS = [
    # NIFTY. NSE/FAOP/39012: 75 unchanged in 2018.
    ("NIFTY", "NFO", "2019-01-01", "2019-01-31", 75),
    # NSE/FAOP/47854: June 2021 and July weekly contracts keep 75; July monthly and
    # August weekly contracts get 50; long-dated contracts convert after 24 June 2021.
    ("NIFTY", "NFO", "2021-05-10", "2021-06-24", 75),
    ("NIFTY", "NFO", "2021-05-10", "2021-07-22", 75),
    ("NIFTY", "NFO", "2021-05-10", "2021-07-29", 50),
    ("NIFTY", "NFO", "2021-06-10", "2021-08-05", 50),
    ("NIFTY", "NFO", "2021-06-24", "2021-12-30", 75),
    ("NIFTY", "NFO", "2021-06-25", "2021-12-30", 50),
    # NSE/FAOP/61415: every contract trades at 25 from trade date 26 April 2024.
    ("NIFTY", "NFO", "2024-04-25", "2024-04-25", 50),
    ("NIFTY", "NFO", "2024-04-25", "2024-05-30", 50),
    ("NIFTY", "NFO", "2024-04-26", "2024-05-30", 25),
    # NSE/FAOP/64625 annexure: last weekly on 25 is 19 Dec 2024, first on 75 is 2 Jan
    # 2025; last monthly on 25 is 30 Jan 2025, first on 75 is 27 Feb 2025; quarterly and
    # half-yearly contracts convert at the close of 26 Dec 2024.
    ("NIFTY", "NFO", "2024-12-02", "2024-12-19", 25),
    ("NIFTY", "NFO", "2024-12-02", "2025-01-02", 75),
    ("NIFTY", "NFO", "2024-12-02", "2025-01-30", 25),
    ("NIFTY", "NFO", "2024-12-02", "2025-02-27", 75),
    ("NIFTY", "NFO", "2024-12-26", "2025-03-27", 25),
    ("NIFTY", "NFO", "2024-12-27", "2025-03-27", 75),
    # NSE/FAOP/70616 annexure: last weekly on 75 is 23 Dec 2025, first on 65 is 6 Jan
    # 2026; last monthly on 75 is 30 Dec 2025, first on 65 is 27 Jan 2026; quarterly and
    # half-yearly contracts convert at the close of 30 Dec 2025.
    ("NIFTY", "NFO", "2025-11-03", "2025-12-23", 75),
    ("NIFTY", "NFO", "2025-11-03", "2025-12-30", 75),
    ("NIFTY", "NFO", "2025-11-03", "2026-01-06", 65),
    ("NIFTY", "NFO", "2025-11-03", "2026-01-27", 65),
    ("NIFTY", "NFO", "2025-12-30", "2026-03-31", 75),
    ("NIFTY", "NFO", "2025-12-31", "2026-03-31", 65),
    # fo_mktlots.csv of 2026-11-12.
    ("NIFTY", "NFO", "2026-11-11", "2026-11-17", 65),
    # BANKNIFTY. NSE/FAOP/39012: 20 from the January 2019 expiry.
    ("BANKNIFTY", "NFO", "2019-01-01", "2019-01-31", 20),
    # NSE/FAOP/44039: May and June 2020 keep 20; July 2020 and beyond get 25 from 4 May 2020.
    ("BANKNIFTY", "NFO", "2020-05-04", "2020-06-25", 20),
    ("BANKNIFTY", "NFO", "2020-05-04", "2020-07-30", 25),
    # NSE/FAOP/56233: June 2023 keeps 25; July 2023 monthly gets 15; long-dated
    # contracts convert on 30 June 2023.
    ("BANKNIFTY", "NFO", "2023-05-02", "2023-06-28", 25),
    ("BANKNIFTY", "NFO", "2023-05-02", "2023-07-20", 25),
    ("BANKNIFTY", "NFO", "2023-05-02", "2023-07-27", 15),
    ("BANKNIFTY", "NFO", "2023-06-28", "2023-09-28", 25),
    ("BANKNIFTY", "NFO", "2023-06-30", "2023-09-28", 15),
    # NSE/FAOP/64625 annexure: last monthly on 15 is 29 Jan 2025, first on 30 is 26 Feb
    # 2025; quarterly contracts convert at the close of 24 Dec 2024.
    ("BANKNIFTY", "NFO", "2024-12-02", "2025-01-29", 15),
    ("BANKNIFTY", "NFO", "2024-12-02", "2025-02-26", 30),
    ("BANKNIFTY", "NFO", "2024-12-24", "2025-03-26", 15),
    ("BANKNIFTY", "NFO", "2024-12-26", "2025-03-26", 30),
    # NSE/FAOP/67372: April, May, June 2025 keep 30; July 2025 gets 35; quarterly
    # contracts trade at 35 from 25 April 2025.
    ("BANKNIFTY", "NFO", "2025-04-25", "2025-06-26", 30),
    ("BANKNIFTY", "NFO", "2025-04-25", "2025-07-31", 35),
    ("BANKNIFTY", "NFO", "2025-04-24", "2025-09-25", 30),
    ("BANKNIFTY", "NFO", "2025-04-25", "2025-09-25", 35),
    # NSE/FAOP/70616 annexure.
    ("BANKNIFTY", "NFO", "2025-11-03", "2025-12-30", 35),
    ("BANKNIFTY", "NFO", "2025-11-03", "2026-01-27", 30),
    ("BANKNIFTY", "NFO", "2025-12-30", "2026-03-31", 35),
    ("BANKNIFTY", "NFO", "2025-12-31", "2026-03-31", 30),
    # FINNIFTY. NSE/FAOP/46603: 40 at launch on 11 January 2021.
    ("FINNIFTY", "NFO", "2021-01-11", "2021-01-28", 40),
    # NSE/FAOP/61415: monthly to 25 Jun 2024 and weekly to 23 Jul 2024 keep 40; the
    # 30 Jul 2024 monthly and weekly contracts from 6 Aug 2024 get 25.
    ("FINNIFTY", "NFO", "2024-05-02", "2024-06-25", 40),
    ("FINNIFTY", "NFO", "2024-05-02", "2024-07-23", 40),
    ("FINNIFTY", "NFO", "2024-05-02", "2024-07-30", 25),
    ("FINNIFTY", "NFO", "2024-06-24", "2024-08-06", 25),
    # NSE/FAOP/64625 and NSE/FAOP/70616 annexures.
    ("FINNIFTY", "NFO", "2024-12-02", "2025-01-28", 25),
    ("FINNIFTY", "NFO", "2024-12-02", "2025-02-25", 65),
    ("FINNIFTY", "NFO", "2025-11-03", "2025-12-30", 65),
    ("FINNIFTY", "NFO", "2025-11-03", "2026-01-27", 60),
    # MIDCPNIFTY. NSE/FAOP/50968: 75 at launch on 24 January 2022.
    ("MIDCPNIFTY", "NFO", "2022-01-24", "2022-01-25", 75),
    # NSE/FAOP/61415: weekly to 22 Jul 2024 keeps 75; the 29 Jul 2024 monthly gets 50.
    ("MIDCPNIFTY", "NFO", "2024-05-02", "2024-07-22", 75),
    ("MIDCPNIFTY", "NFO", "2024-05-02", "2024-07-29", 50),
    # NSE/FAOP/64625: last monthly on 50 is 27 Jan 2025, first on 120 is 24 Feb 2025.
    ("MIDCPNIFTY", "NFO", "2024-12-02", "2025-01-27", 50),
    ("MIDCPNIFTY", "NFO", "2024-12-02", "2025-02-24", 120),
    # NSE/FAOP/67372: June 2025 keeps 120; July 2025 gets 140.
    ("MIDCPNIFTY", "NFO", "2025-04-25", "2025-06-26", 120),
    ("MIDCPNIFTY", "NFO", "2025-04-25", "2025-07-31", 140),
    # NSE/FAOP/70616 annexure.
    ("MIDCPNIFTY", "NFO", "2025-11-03", "2025-12-30", 140),
    ("MIDCPNIFTY", "NFO", "2025-11-03", "2026-01-27", 120),
    # NIFTYNXT50. NSE/FAOP/61629: 10 at launch on 24 April 2024. NSE/FAOP/64625: last
    # monthly on 10 is 31 Jan 2025, first on 25 is 28 Feb 2025.
    ("NIFTYNXT50", "NFO", "2024-04-24", "2024-04-26", 10),
    ("NIFTYNXT50", "NFO", "2024-12-02", "2025-01-31", 10),
    ("NIFTYNXT50", "NFO", "2024-12-02", "2025-02-28", 25),
    # SENSEX. BSE notice 20230327-65: 10 from 15 May 2023.
    ("SENSEX", "BFO", "2023-05-15", "2023-05-19", 10),
    # BSE notice 20241021-13: weekly contracts up to 3 Jan 2025 and the January 2025
    # monthly keep 10; weekly contracts of 10 Jan 2025 and beyond get 20; long-dated
    # contracts convert at the close of 27 Dec 2024. BSE notice 20241128-70 then
    # re-dated 10 Jan to 7 Jan and 31 Jan to 28 Jan from 1 Jan 2025.
    ("SENSEX", "BFO", "2024-12-02", "2025-01-03", 10),
    ("SENSEX", "BFO", "2024-12-02", "2025-01-10", 20),
    ("SENSEX", "BFO", "2024-12-02", "2025-01-31", 10),
    ("SENSEX", "BFO", "2025-01-02", "2025-01-07", 20),
    ("SENSEX", "BFO", "2025-01-02", "2025-01-28", 10),
    ("SENSEX", "BFO", "2024-12-27", "2025-03-28", 10),
    ("SENSEX", "BFO", "2024-12-30", "2025-03-28", 20),
    # BSE notice 20260930-58: 25 from 30 Oct 2026 for January 2027 and later expiries;
    # November and December 2026 keep 20.
    ("SENSEX", "BFO", "2026-10-29", "2027-01-28", 20),
    ("SENSEX", "BFO", "2026-10-30", "2027-01-28", 25),
    ("SENSEX", "BFO", "2026-11-11", "2026-12-31", 20),
    # BANKEX. BSE notice 20230327-65: 15 from 15 May 2023. BSE notice 20241021-13:
    # the January 2025 monthly keeps 15; later contracts get 30.
    ("BANKEX", "BFO", "2023-05-15", "2023-05-19", 15),
    ("BANKEX", "BFO", "2024-12-02", "2025-01-27", 15),
    ("BANKEX", "BFO", "2024-12-02", "2025-02-24", 30),
    ("BANKEX", "BFO", "2026-11-11", "2026-11-26", 30),
]


@pytest.mark.parametrize("underlying,exchange,on_date,expiry,lot_size", LOTS)
def test_lot_size_of_a_contract_on_a_trade_date(
    tables, underlying, exchange, on_date, expiry, lot_size
):
    assert tables.lot_size(underlying, exchange, on_date, expiry=expiry) == lot_size


@pytest.mark.parametrize(
    "underlying,exchange,on_date",
    [
        ("NIFTY", "NFO", "2018-12-31"),  # the tables start on 2019-01-01
        ("FINNIFTY", "NFO", "2021-01-08"),  # listed from 2021-01-11
        ("MIDCPNIFTY", "NFO", "2022-01-21"),  # listed from 2022-01-24
        ("NIFTYNXT50", "NFO", "2024-04-23"),  # listed from 2024-04-24
        ("SENSEX", "BFO", "2023-05-12"),  # relaunched on 2023-05-15
        ("RELIANCE", "NFO", "2024-06-03"),  # stock lots are not built in
    ],
)
def test_lot_size_outside_the_built_in_coverage_raises(tables, underlying, exchange, on_date):
    with pytest.raises(ReferenceDataError, match="lot_sizes"):
        tables.lot_size(underlying, exchange, on_date)


def test_banknifty_july_2020_weekly_contracts_stayed_at_20():
    # Bhavcopies of 2020-06-26 and 2020-06-30: open interest of every July 2020 weekly
    # contract is a multiple of 20, of the July monthly contract a multiple of 25.
    fresh = ReferenceTables()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fresh.lot_size("BANKNIFTY", "NFO", "2020-06-26", expiry="2020-07-16") == 20
        assert fresh.lot_size("BANKNIFTY", "NFO", "2020-06-26", expiry="2020-07-30") == 25


def test_unverified_built_in_row_warns_when_used(tables):
    # No MCX document was read for the 2024 CRUDEOIL contracts; the row is unverified.
    with pytest.warns(ReferenceDataWarning, match="mcx_contracts.CRUDEOIL MCX.*is unverified"):
        assert tables.lot_size("CRUDEOIL", "MCX", "2024-11-04", expiry="2024-11-19") == 100


@pytest.mark.parametrize("on_date", ["2021-05-10", "2024-12-02", "2025-11-03"])
def test_nifty_lot_size_without_an_expiry_is_refused_during_a_transition(tables, on_date):
    with pytest.raises(ConfigError, match="depends on the contract"):
        tables.lot_size("NIFTY", "NFO", on_date)


# ---- freeze quantities ----------------------------------------------------------------

FREEZE = [
    # NSE monthly circulars "Quantity Freeze Limits for Indices": value, then the day
    # before and the day of each change.
    ("NIFTY", "2019-01-01", 7500),  # NSE/FAOP/39809
    ("NIFTY", "2019-03-31", 7500),
    ("NIFTY", "2019-04-01", 5000),  # NSE/FAOP/40595
    ("NIFTY", "2020-04-01", 10000),  # NSE/FAOP/44022
    ("NIFTY", "2020-05-04", 7500),  # NSE/FAOP/44274
    ("NIFTY", "2021-08-01", 5000),
    ("NIFTY", "2021-08-02", 2800),  # NSE/FAOP/49129
    ("NIFTY", "2022-11-01", 1800),  # NSE/FAOP/54248
    ("NIFTY", "2026-10-04", 1800),
    ("NIFTY", "2026-10-05", 3510),  # NSE/FAOP/76693
    ("NIFTY", "2026-11-01", 3510),
    ("NIFTY", "2026-11-02", 1800),  # NSE/FAOP/77189
    ("BANKNIFTY", "2019-01-01", 2500),  # NSE/FAOP/39809
    ("BANKNIFTY", "2021-08-02", 1200),  # NSE/FAOP/49129
    ("BANKNIFTY", "2022-11-01", 900),  # NSE/FAOP/54248
    ("BANKNIFTY", "2025-05-02", 600),  # NSE/FAOP/67792
    ("BANKNIFTY", "2025-09-01", 900),  # NSE/FAOP/69913
    ("BANKNIFTY", "2025-11-03", 600),  # NSE/FAOP/71087
    ("BANKNIFTY", "2026-10-05", 1440),  # NSE/FAOP/76693
    ("BANKNIFTY", "2026-11-02", 900),  # NSE/FAOP/77189
    ("FINNIFTY", "2021-02-01", 5000),  # NSE/FAOP/47155
    ("FINNIFTY", "2025-12-01", 1200),  # NSE/FAOP/71540
    ("FINNIFTY", "2026-11-02", 1800),  # NSE/FAOP/77189
    ("MIDCPNIFTY", "2022-01-24", 5500),  # NSE/FAOP/50968 (launch)
    ("MIDCPNIFTY", "2023-09-01", 4200),  # NSE/FAOP/58176
    ("MIDCPNIFTY", "2024-07-01", 2800),  # NSE/FAOP/62670
    ("NIFTYNXT50", "2024-04-24", 600),  # NSE/FAOP/61629 (launch)
    ("NIFTYNXT50", "2026-10-05", 1125),  # NSE/FAOP/76693
]


@pytest.mark.parametrize("underlying,on_date,freeze_qty", FREEZE)
def test_nse_index_freeze_quantity_on_a_date(tables, underlying, on_date, freeze_qty):
    assert tables.freeze_qty(underlying, "NFO", on_date) == freeze_qty


@pytest.mark.parametrize(
    "underlying,on_date,kind,limit",
    [
        ("SENSEX", "2023-05-29", "option", 500),  # BSE notice 20230526-54
        ("SENSEX", "2023-06-21", "option", 500),
        ("SENSEX", "2023-06-22", "option", 1000),  # BSE notice 20230621-48
        ("SENSEX", "2023-06-22", "future", 500),
        ("BANKEX", "2023-12-03", "option", 600),  # BSE notice 20230526-54
        ("BANKEX", "2023-12-04", "option", 900),  # BSE notice 20231201-16
        ("BANKEX", "2023-12-04", "future", 600),
    ],
)
def test_bse_single_order_limit_differs_between_futures_and_options(
    tables, underlying, on_date, kind, limit
):
    assert _quiet(lambda: tables.freeze_qty(underlying, "BFO", on_date, kind=kind)) == limit


def test_bse_limit_without_a_kind_is_refused_when_futures_and_options_differ(tables):
    with pytest.raises(ConfigError, match="differs between futures .500. and options .1000."):
        _quiet(lambda: tables.freeze_qty("SENSEX", "BFO", "2024-06-03"))


def test_bse_limit_before_the_first_notice_is_not_covered(tables):
    with pytest.raises(ReferenceDataError, match="freeze_qty"):
        tables.freeze_qty("SENSEX", "BFO", "2023-05-26")


# ---- strike steps ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "underlying,exchange,on_date,strike_step",
    [
        ("NIFTY", "NFO", "2019-01-01", 50),  # NSE/FAOP/37689
        ("BANKNIFTY", "NFO", "2019-01-01", 100),  # NSE/FAOP/37689
        ("FINNIFTY", "NFO", "2022-10-18", 100),  # NSE/FAOP/46603
        ("FINNIFTY", "NFO", "2022-10-19", 50),  # NSE/FAOP/53728
        ("MIDCPNIFTY", "NFO", "2023-07-25", 50),  # NSE/FAOP/50968
        ("MIDCPNIFTY", "NFO", "2023-07-26", 25),  # NSE/FAOP/57319
        ("NIFTYNXT50", "NFO", "2024-04-24", 100),  # NSE/FAOP/61629
        ("SENSEX", "BFO", "2026-11-12", 100),  # BSE contract specifications page
        ("BANKEX", "BFO", "2026-11-12", 100),
    ],
)
def test_strike_step_of_the_near_contracts(tables, underlying, exchange, on_date, strike_step):
    assert tables.strike_step(underlying, exchange, on_date) == strike_step


# ---- tick sizes -----------------------------------------------------------------------

TICKS = [
    # NSE shares. NSE/CMTR/40962: five paise. NSE/CMTR/62174: 0.01 below 250 from
    # 10 June 2024. NSE/CMTR/67133: slabs from 15 April 2025 ("> 1,000" starts the 0.10
    # slab, so exactly 1,000 is still 0.05).
    ("NSE", "equity", "2024-06-09", 100.0, 0.05),
    ("NSE", "equity", "2024-06-10", 249.99, 0.01),
    ("NSE", "equity", "2024-06-10", 250.0, 0.05),
    ("NSE", "equity", "2025-04-14", 3000.0, 0.05),
    ("NSE", "equity", "2025-04-15", 249.99, 0.01),
    ("NSE", "equity", "2025-04-15", 250.0, 0.05),
    ("NSE", "equity", "2025-04-15", 1000.0, 0.05),
    ("NSE", "equity", "2025-04-15", 1000.05, 0.10),
    ("NSE", "equity", "2025-04-15", 5000.0, 0.10),
    ("NSE", "equity", "2025-04-15", 5000.5, 0.50),
    ("NSE", "equity", "2025-04-15", 10000.0, 0.50),
    ("NSE", "equity", "2025-04-15", 10001.0, 1.00),
    ("NSE", "equity", "2025-04-15", 20000.0, 1.00),
    ("NSE", "equity", "2025-04-15", 20005.0, 5.00),
    # BSE shares. Notice 20160517-25: 0.01 below 15. Notice 20230215-39: 0.01 for a
    # close of 100 or less from 1 March 2023.
    ("BSE", "equity", "2023-02-28", 14.99, 0.01),
    ("BSE", "equity", "2023-02-28", 15.0, 0.05),
    ("BSE", "equity", "2023-03-01", 100.0, 0.01),
    ("BSE", "equity", "2023-03-01", 100.05, 0.05),
    # NSE index futures. NSE/FAOP/67135: by index level from 15 April 2025.
    ("NFO", "index_future", "2025-04-14", 24000.0, 0.05),
    ("NFO", "index_future", "2025-04-15", 15000.0, 0.05),
    ("NFO", "index_future", "2025-04-15", 24000.0, 0.10),
    ("NFO", "index_future", "2025-04-15", 30000.0, 0.10),
    ("NFO", "index_future", "2025-04-15", 52000.0, 0.20),
    # NSE stock futures. NSE/FAOP/62175: as the underlying share from 8 July 2024.
    ("NFO", "stock_future", "2024-07-07", 100.0, 0.05),
    ("NFO", "stock_future", "2024-07-08", 100.0, 0.01),
    ("NFO", "stock_future", "2025-04-15", 3000.0, 0.10),
    # NSE options. NSE/FAOP/67135: index options stay at 0.05. NSE/FAOP/70382: stock
    # options below 250 move to 0.01 from 3 November 2025.
    ("NFO", "index_option", "2019-01-01", None, 0.05),
    ("NFO", "index_option", "2026-11-11", None, 0.05),
    ("NFO", "stock_option", "2025-11-02", 100.0, 0.05),
    ("NFO", "stock_option", "2025-11-03", 100.0, 0.01),
    ("NFO", "stock_option", "2025-11-03", 250.0, 0.05),
    # BSE index derivatives. BSE contract specifications page read on 2026-11-12.
    ("BFO", "index_option", "2026-11-12", None, 0.05),
    ("BFO", "index_future", "2026-11-12", None, 0.05),
]


@pytest.mark.parametrize("exchange,segment,on_date,reference_price,tick", TICKS)
def test_tick_size_of_a_segment_for_a_reference_price(
    tables, exchange, segment, on_date, reference_price, tick
):
    assert tables.tick_size(exchange, segment, on_date, reference_price) == tick


def test_bse_index_tick_before_the_page_was_read_is_flagged_unverified(tables):
    with pytest.warns(ReferenceDataWarning, match="tick_sizes.BFO index_option.*is unverified"):
        assert tables.tick_size("BFO", "index_option", "2024-06-03") == 0.05


# ---- settlement -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "exchange,segment,on_date,settlement",
    [
        ("NFO", "index_future", "2019-01-01", "cash"),  # NSE/FAOP/37689 item 1.11
        ("NFO", "index_option", "2026-11-11", "cash"),
        ("NFO", "stock_future", "2019-11-01", "physical"),  # SEBI circular of 2018-12-31
        ("NFO", "stock_option", "2024-06-03", "physical"),
        ("BFO", "index_option", "2023-05-15", "cash"),  # BSE contract specifications page
    ],
)
def test_settlement_of_a_derivative_segment(tables, exchange, segment, on_date, settlement):
    assert (
        tables.find("settlement", (exchange, segment), on_date).values["settlement"] == settlement
    )


def test_stock_derivative_settlement_before_november_2019_is_not_covered(tables):
    # Stocks moved to physical settlement in batches until the October 2019 expiry.
    with pytest.raises(ReferenceDataError, match="settlement"):
        tables.find("settlement", ("NFO", "stock_future"), "2019-10-31")


# ---- expiry rules ---------------------------------------------------------------------

WEEKDAYS = [
    ("NIFTY", "NFO", "weekly", "2019-02-10", "NONE"),  # NSE/FAOP/39894: weekly from 11 Feb 2019
    ("NIFTY", "NFO", "weekly", "2019-02-11", "THU"),
    ("NIFTY", "NFO", "weekly", "2025-08-31", "THU"),  # NSE/FAOP/68589
    ("NIFTY", "NFO", "weekly", "2025-09-01", "TUE"),
    ("NIFTY", "NFO", "monthly", "2025-08-31", "THU"),
    ("NIFTY", "NFO", "monthly", "2025-09-01", "TUE"),
    ("BANKNIFTY", "NFO", "weekly", "2023-09-03", "THU"),  # NSE/FAOP/57540
    ("BANKNIFTY", "NFO", "weekly", "2023-09-04", "WED"),
    ("BANKNIFTY", "NFO", "weekly", "2024-11-13", "WED"),  # NSE/FAOP/64506
    ("BANKNIFTY", "NFO", "weekly", "2024-11-14", "NONE"),
    ("BANKNIFTY", "NFO", "monthly", "2024-02-29", "THU"),  # NSE/FAOP/60011
    ("BANKNIFTY", "NFO", "monthly", "2024-03-01", "WED"),
    ("BANKNIFTY", "NFO", "monthly", "2024-12-31", "WED"),  # NSE/FAOP/65336
    ("BANKNIFTY", "NFO", "monthly", "2025-01-01", "THU"),
    ("BANKNIFTY", "NFO", "monthly", "2025-09-01", "TUE"),
    ("FINNIFTY", "NFO", "weekly", "2021-10-14", "THU"),  # NSE/FAOP/49595
    ("FINNIFTY", "NFO", "weekly", "2021-10-15", "TUE"),
    ("FINNIFTY", "NFO", "weekly", "2024-11-19", "TUE"),  # NSE/FAOP/64506
    ("FINNIFTY", "NFO", "weekly", "2024-11-20", "NONE"),
    ("FINNIFTY", "NFO", "monthly", "2024-12-31", "TUE"),  # NSE/FAOP/65336
    ("FINNIFTY", "NFO", "monthly", "2025-01-01", "THU"),
    ("MIDCPNIFTY", "NFO", "weekly", "2023-04-25", "TUE"),  # NSE/FAOP/56128
    ("MIDCPNIFTY", "NFO", "weekly", "2023-04-26", "WED"),
    ("MIDCPNIFTY", "NFO", "weekly", "2023-08-16", "WED"),  # NSE/FAOP/57538
    ("MIDCPNIFTY", "NFO", "weekly", "2023-08-17", "MON"),
    ("MIDCPNIFTY", "NFO", "weekly", "2024-11-18", "MON"),  # NSE/FAOP/64506
    ("MIDCPNIFTY", "NFO", "weekly", "2024-11-19", "NONE"),
    ("MIDCPNIFTY", "NFO", "monthly", "2025-01-01", "THU"),  # NSE/FAOP/65336
    ("NIFTYNXT50", "NFO", "weekly", "2024-04-24", "NONE"),  # NSE/FAOP/61629: monthly only
    ("NIFTYNXT50", "NFO", "monthly", "2024-12-31", "FRI"),
    ("NIFTYNXT50", "NFO", "monthly", "2025-01-01", "THU"),  # NSE/FAOP/65336
    ("NIFTYNXT50", "NFO", "monthly", "2025-09-01", "TUE"),
    ("SENSEX", "BFO", "weekly", "2023-05-15", "FRI"),  # BSE notice 20230327-65
    ("SENSEX", "BFO", "weekly", "2025-01-03", "FRI"),  # BSE notice 20241128-70
    ("SENSEX", "BFO", "weekly", "2025-01-04", "TUE"),
    ("SENSEX", "BFO", "weekly", "2025-08-31", "TUE"),  # BSE notice 20250623-59
    ("SENSEX", "BFO", "weekly", "2025-09-01", "THU"),
    ("SENSEX", "BFO", "monthly", "2024-12-31", "FRI"),
    ("SENSEX", "BFO", "monthly", "2025-01-01", "TUE"),
    ("BANKEX", "BFO", "weekly", "2023-10-13", "FRI"),  # BSE notice 20230830-8
    ("BANKEX", "BFO", "weekly", "2023-10-14", "MON"),
    ("BANKEX", "BFO", "weekly", "2024-11-18", "MON"),  # BSE notice 20241003-42
    ("BANKEX", "BFO", "weekly", "2024-11-19", "NONE"),
    ("BANKEX", "BFO", "monthly", "2025-01-01", "TUE"),  # BSE notice 20241128-70
    ("BANKEX", "BFO", "monthly", "2025-09-01", "THU"),  # BSE notice 20250623-59
]


@pytest.mark.parametrize("underlying,exchange,series,on_date,weekday", WEEKDAYS)
def test_expiry_weekday_in_force_on_a_date(tables, underlying, exchange, series, on_date, weekday):
    assert tables.expiry_rule(underlying, exchange, on_date, series) == weekday


EXPIRIES = [
    # NSE/FAOP/39894 lists the first NIFTY weekly expiries; 21 March 2019 was a holiday.
    (
        "NIFTY",
        "NFO",
        "weekly",
        "2019-02-11",
        "2019-04-11",
        ["2019-03-21"],
        [
            "2019-02-14",
            "2019-02-21",
            "2019-03-07",
            "2019-03-14",
            "2019-03-20",
            "2019-04-04",
            "2019-04-11",
        ],
    ),
    # NSE/FAOP/49595: the FINNIFTY contract of 21 Oct 2021 moved to Tuesday 19 Oct.
    ("FINNIFTY", "NFO", "weekly", "2021-10-11", "2021-10-22", [], ["2021-10-14", "2021-10-19"]),
    # NSE/FAOP/57538: MIDCPNIFTY expired on Wednesday 16 Aug 2023, then Monday 21 Aug.
    ("MIDCPNIFTY", "NFO", "weekly", "2023-08-14", "2023-08-22", [], ["2023-08-16", "2023-08-21"]),
    # NSE/FAOP/57540: first Wednesday BANKNIFTY weekly expiry on 6 Sep 2023.
    ("BANKNIFTY", "NFO", "weekly", "2023-09-01", "2023-09-08", [], ["2023-09-06"]),
    # NSE/FAOP/60011: first Wednesday monthly expiry 27 Mar 2024; 26 Dec 2024 became
    # 24 Dec 2024 because 25 Dec is a holiday.
    ("BANKNIFTY", "NFO", "monthly", "2024-02-01", "2024-03-31", [], ["2024-02-29", "2024-03-27"]),
    ("BANKNIFTY", "NFO", "monthly", "2024-12-01", "2024-12-31", ["2024-12-25"], ["2024-12-24"]),
    # NSE/FAOP/64506: last weekly expiries.
    ("BANKNIFTY", "NFO", "weekly", "2024-11-11", "2024-12-31", [], ["2024-11-13"]),
    ("MIDCPNIFTY", "NFO", "weekly", "2024-11-14", "2024-12-31", [], ["2024-11-18"]),
    ("FINNIFTY", "NFO", "weekly", "2024-11-14", "2024-12-31", [], ["2024-11-19"]),
    # NSE/FAOP/65336 annexure: every January 2025 monthly expiry became 30 Jan 2025.
    ("BANKNIFTY", "NFO", "monthly", "2025-01-01", "2025-01-31", [], ["2025-01-30"]),
    ("FINNIFTY", "NFO", "monthly", "2025-01-01", "2025-01-31", [], ["2025-01-30"]),
    ("MIDCPNIFTY", "NFO", "monthly", "2025-01-01", "2025-01-31", [], ["2025-01-30"]),
    ("NIFTYNXT50", "NFO", "monthly", "2025-01-01", "2025-01-31", [], ["2025-01-30"]),
    # NSE/FAOP/68747 annexure: 28 Aug 2025 unchanged, first Tuesday weekly 2 Sep 2025,
    # monthly 30 Sep 2025, 30 Dec 2025 and 31 Mar 2026.
    ("NIFTY", "NFO", "all", "2025-08-25", "2025-09-05", [], ["2025-08-28", "2025-09-02"]),
    (
        "NIFTY",
        "NFO",
        "monthly",
        "2025-09-01",
        "2025-12-31",
        [],
        ["2025-09-30", "2025-10-28", "2025-11-25", "2025-12-30"],
    ),
    # 31 March 2026 became a holiday after the March contracts were listed: the final
    # date is 30 March (bhavcopy of 2025-12-29; contract_dates.csv).
    ("NIFTY", "NFO", "monthly", "2026-03-01", "2026-03-31", [], ["2026-03-30"]),
    # BSE notice 20230830-8 and the bhavcopy of 2023-10-16: BANKEX expired on Friday
    # 13 Oct 2023 and then on Monday 16 Oct 2023.
    ("BANKEX", "BFO", "weekly", "2023-10-09", "2023-10-17", [], ["2023-10-13", "2023-10-16"]),
    # BSE notice 20241128-70: 3 Jan 2025 stayed; 10 Jan became 7 Jan; monthly 28 Jan.
    (
        "SENSEX",
        "BFO",
        "all",
        "2025-01-01",
        "2025-01-31",
        [],
        ["2025-01-03", "2025-01-07", "2025-01-14", "2025-01-21", "2025-01-28"],
    ),
    ("BANKEX", "BFO", "monthly", "2025-01-01", "2025-01-31", [], ["2025-01-28"]),
    # BSE notice 20250623-59: 19 and 26 Aug 2025 unchanged; 2 Sep became 4 Sep; the
    # September monthly became 25 Sep 2025.
    (
        "SENSEX",
        "BFO",
        "all",
        "2025-08-18",
        "2025-09-30",
        [],
        ["2025-08-19", "2025-08-26", "2025-09-04", "2025-09-11", "2025-09-18", "2025-09-25"],
    ),
]


@pytest.mark.parametrize("underlying,exchange,series,start,end,holidays,expected", EXPIRIES)
def test_expiry_dates_match_the_exchange_circulars(
    tables, underlying, exchange, series, start, end, holidays, expected
):
    calendar = FakeCalendar(holidays=_dates(*holidays))
    found = tables.expiries(underlying, exchange, start, end, calendar, series=series)
    assert found == _dates(*expected)


EXPLICIT = [
    # The calendar passed here treats every weekday as a session, as a calendar that
    # counts a Muhurat session would; the explicit rows must still give the exchange's date.
    ("NIFTY", "NFO", "2021-11-01", "2021-11-05", "2021-11-03"),  # bhavcopy of 2021-11-03
    ("BANKNIFTY", "NFO", "2021-11-01", "2021-11-05", "2021-11-03"),
    ("SENSEX", "BFO", "2024-10-28", "2024-11-01", "2024-10-31"),  # bhavcopy of 2024-10-31
    ("NIFTY", "NFO", "2025-10-15", "2025-10-21", "2025-10-20"),  # bhavcopy of 2025-10-20
    ("MIDCPNIFTY", "NFO", "2024-01-16", "2024-01-22", "2024-01-20"),  # NSE/MSD/60340
    ("BANKEX", "BFO", "2024-01-16", "2024-01-22", "2024-01-20"),  # BSE notice 20240119-45
    ("MIDCPNIFTY", "NFO", "2024-05-14", "2024-05-20", "2024-05-17"),  # bhavcopy of 2024-05-17
    ("BANKEX", "BFO", "2024-05-14", "2024-05-20", "2024-05-17"),
    ("SENSEX", "BFO", "2026-01-09", "2026-01-15", "2026-01-14"),  # BSE notice 20260112-8
]


@pytest.mark.parametrize("underlying,exchange,start,end,expiry", EXPLICIT)
def test_expiry_set_by_the_exchange_around_a_special_session(
    tables, underlying, exchange, start, end, expiry
):
    assert tables.expiries(underlying, exchange, start, end, FakeCalendar()) == [day(expiry)]


def test_expiries_before_an_index_was_listed_are_not_covered(tables):
    with pytest.raises(ReferenceDataError, match="expiry_rules"):
        tables.expiries("FINNIFTY", "NFO", "2020-12-01", "2021-01-31", FakeCalendar())


# ---- instruments ------------------------------------------------------------------------


def test_index_option_instrument_from_the_built_in_tables():
    option = ref.default_tables().instrument("NIFTY26DEC2424000CE", "NFO", "2024-12-02")
    assert (option.lot_size, option.tick_size, option.freeze_qty) == (25, 0.05, 1800)
    assert (option.multiplier, option.settlement) == (1.0, "cash")
    assert {status for _, _, status in option.provenance} == {"verified", "assumed"}


def test_index_future_tick_follows_the_index_level_slab():
    future = ref.default_tables().instrument(
        "BANKNIFTY29MAY25FUT", "NFO", "2025-05-02", reference_price=55000.0
    )
    assert (future.lot_size, future.tick_size, future.freeze_qty) == (30, 0.20, 600)


def test_bse_option_uses_the_option_order_limit(tables):
    option = _quiet(lambda: tables.instrument("SENSEX28JAN2578000CE", "BFO", "2025-01-06"))
    assert (option.lot_size, option.freeze_qty, option.settlement) == (10, 1000, "cash")


@pytest.mark.parametrize(
    "symbol,on_date,lot_size,multiplier,contract_multiplier,tick,freeze_qty,settlement",
    [
        # MCX/TRD/479/2024: 100 barrels quoted per barrel, tick Re 1, maximum 10,000 barrels.
        ("CRUDEOIL19MAR25FUT", "2025-02-03", 100, 1.0, 100, 1.0, 10000, "cash"),
        ("CRUDEOILM19MAR25FUT", "2025-02-03", 10, 1.0, 10, 1.0, 10000, "cash"),
        # MCX/TRD/497/2024: 1,250 mmBtu quoted per mmBtu, tick 0.10, maximum 60,000 mmBtu.
        ("NATURALGAS26MAR25FUT", "2025-02-03", 1250, 1.0, 1250, 0.1, 60000, "cash"),
        # MCX/TRD/456/2025: 100 grams quoted per 10 grams, maximum 10 kg.
        ("GOLDM05DEC25FUT", "2025-11-03", 100, 0.1, 10, 1.0, 10000, "physical"),
        ("GOLDGUINEA31DEC25FUT", "2025-11-03", 8, 0.125, 1, 1.0, 10000, "physical"),
        # MCX/TRD/655/2025: 5 MT quoted per kg, tick 0.05, maximum 100 MT.
        ("ZINC29MAY26FUT", "2026-02-02", 5, 1000.0, 5000, 0.05, 100, "physical"),
        ("COPPER29MAY26FUT", "2026-02-02", 2500, 1.0, 2500, 0.05, 70000, "physical"),
        ("NICKEL20MAY26FUT", "2026-02-02", 250, 1.0, 250, 0.1, 24000, "physical"),
    ],
)
def test_mcx_future_terms_from_the_contract_specification(
    tables, symbol, on_date, lot_size, multiplier, contract_multiplier, tick, freeze_qty, settlement
):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # these rows are verified: no warning is expected
        future = tables.instrument(symbol, "MCX", on_date)
    assert (future.lot_size, future.multiplier, future.contract_multiplier) == (
        lot_size,
        multiplier,
        contract_multiplier,
    )
    assert (future.tick_size, future.freeze_qty, future.settlement) == (
        tick,
        freeze_qty,
        settlement,
    )


def test_mcx_gold_has_a_contract_multiplier_of_100_but_is_unverified(tables):
    with pytest.warns(ReferenceDataWarning, match="mcx_contracts.GOLD MCX.*is unverified"):
        gold = tables.instrument("GOLD05DEC24FUT", "MCX", "2024-11-04")
    assert (gold.lot_size, gold.multiplier, gold.contract_multiplier) == (1, 100.0, 100)


def test_mcx_contract_outside_the_circular_that_was_read_is_flagged_unverified(tables):
    with pytest.warns(ReferenceDataWarning, match="mcx_contracts.CRUDEOIL MCX.*is unverified"):
        assert tables.instrument("CRUDEOIL19NOV24FUT", "MCX", "2024-11-04").lot_size == 100


def test_mcx_nickel_lot_without_a_contract_is_refused_while_two_specifications_trade(tables):
    with pytest.raises(ConfigError, match="depends on the contract"):
        _quiet(lambda: tables.lot_size("NICKEL", "MCX", "2025-08-20"))


def test_stock_future_needs_a_user_file_or_a_snapshot():
    with pytest.raises(ReferenceDataError, match="write_symbol_master_snapshot"):
        ref.default_tables().instrument("RELIANCE28NOV24FUT", "NFO", "2024-11-04")


# ---- precedence, as-of and version -------------------------------------------------------


def test_user_file_overrides_a_built_in_row_for_its_dates_only(tmp_path):
    path = write_file(
        tmp_path,
        "lots.csv",
        "underlying,exchange,valid_from,valid_to,lot_size\nNIFTY,NFO,2024-06-01,2024-06-30,30\n",
    )
    custom = ref.load_tables(user_files=[path])
    assert custom.lot_size("NIFTY", "NFO", "2024-06-03") == 30
    assert custom.lot_size("NIFTY", "NFO", "2024-07-01") == 25
    assert ref.default_tables().lot_size("NIFTY", "NFO", "2024-06-03") == 25


def test_open_ended_built_in_row_used_after_its_as_of_date_warns_once():
    fresh = ReferenceTables()
    with pytest.warns(ReferenceDataWarning, match="was last checked on 2026-11-12"):
        assert fresh.lot_size("NIFTY", "NFO", "2027-02-01", expiry="2027-02-23") == 65
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fresh.lot_size("NIFTY", "NFO", "2027-02-02", expiry="2027-02-23")


def test_built_in_row_used_up_to_its_as_of_date_does_not_warn():
    fresh = ReferenceTables()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fresh.lot_size("NIFTY", "NFO", "2026-11-12", expiry="2026-11-17") == 65


def test_reference_version_names_the_check_date_and_a_digest_of_the_files():
    assert re.fullmatch(r"builtin:2026-11-12:[0-9a-f]{12}", ref.default_tables().version)


def test_reference_version_changes_when_a_user_file_is_added(tmp_path):
    path = write_file(
        tmp_path,
        "lots.csv",
        "underlying,exchange,valid_from,valid_to,lot_size\nRELIANCE,NFO,2024-01-01,,250\n",
    )
    version = ref.load_tables(user_files=[path]).version
    assert version.startswith(ref.default_tables().version + ";user:")


# ---- coverage ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table,key,valid_from",
    [
        ("lot_sizes", "NIFTY NFO", "2019-01-01"),
        ("lot_sizes", "BANKNIFTY NFO", "2019-01-01"),
        ("lot_sizes", "FINNIFTY NFO", "2021-01-11"),
        ("lot_sizes", "MIDCPNIFTY NFO", "2022-01-24"),
        ("lot_sizes", "NIFTYNXT50 NFO", "2024-04-24"),
        ("lot_sizes", "SENSEX BFO", "2023-05-15"),
        ("lot_sizes", "BANKEX BFO", "2023-05-15"),
        ("freeze_qty", "NIFTY NFO", "2019-01-01"),
        ("freeze_qty", "BANKNIFTY NFO", "2019-01-01"),
        ("expiry_rules", "NIFTY NFO weekly", "2019-01-01"),
        ("expiry_rules", "NIFTY NFO monthly", "2019-01-01"),
        ("expiry_rules", "BANKNIFTY NFO weekly", "2019-01-01"),
        ("strike_steps", "NIFTY NFO", "2019-01-01"),
        ("tick_sizes", "NSE equity", "2019-01-01"),
    ],
)
def test_coverage_of_a_key_is_open_ended_and_unbroken_from_its_start(table, key, valid_from):
    coverage = ref.default_tables().coverage()
    entry = next(item for item in coverage if (item["table"], item["key"]) == (table, key))
    assert (entry["valid_from"], entry["valid_to"], entry["gaps"]) == (valid_from, None, [])


def test_coverage_reports_verified_and_unverified_row_counts(tables):
    entry = next(item for item in tables.coverage() if item["key"] == "BANKNIFTY NFO")
    assert (entry["table"], entry["rows"], entry["verified"], entry["unverified"]) == (
        "lot_sizes",
        14,
        14,
        0,
    )
    assert entry["as_of"] == "2026-11-12"


def test_broker_units_report_what_the_server_source_shows_per_broker(tables):
    rows = tables.broker_units("zerodha")
    assert [row["segment"] for row in rows] == ["cds_bcd", "equity", "mcx", "nfo_bfo"]
    assert {row["status"] for row in rows} == {"unverified"}
    others = [row["live_status"] for row in rows if row["segment"] != "mcx"]
    assert set(others) == {"not confirmed live"}


def _zerodha_mcx(tables) -> dict:
    (row,) = [row for row in tables.broker_units("zerodha") if row["segment"] == "mcx"]
    return row


def test_zerodha_mcx_lot_sizes_were_seen_in_units_on_a_live_server(tables):
    row = _zerodha_mcx(tables)
    assert row["live_status"] == (
        "lotsize confirmed live on 2026-11-12; order quantity conversion not confirmed"
    )
    # The order side is still unconfirmed, so the row as a whole stays unverified.
    assert row["status"] == "unverified"


# Lot sizes a live OpenAlgo server's Zerodha master contract gave on 2026-11-12.
LIVE_ZERODHA_MCX_LOTS = {
    "CRUDEOIL": 100,
    "CRUDEOILM": 10,
    "GOLDM": 100,
    "NATURALGAS": 1250,
    "SILVERM": 5,
    "COPPER": 2500,
    "ZINC": 5,
}


@pytest.mark.parametrize("root,lot", LIVE_ZERODHA_MCX_LOTS.items())
def test_the_live_zerodha_mcx_lot_is_noted_and_equals_the_reference_tables(tables, root, lot):
    assert re.search(rf"\b{root} {lot}\b", _zerodha_mcx(tables)["note"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)
        assert tables.lot_size(root, "MCX", "2026-11-12") == lot


# ---- the shared default tables ------------------------------------------------------------------


def test_default_tables_answer_from_the_built_in_tables():
    tables = ref.default_tables()
    # Values as in the tables above: NSE/FAOP/77189, NSE/FAOP/37689, NSE/FAOP/67135.
    assert tables.freeze_qty("NIFTY", "NFO", "2026-11-02") == 1800
    assert tables.strike_step("NIFTY", "NFO", "2024-06-03") == 50
    assert tables.tick_size("NFO", "index_option", "2024-06-03") == 0.05
    assert tables.expiries("NIFTY", "NFO", "2025-09-01", "2025-09-05", FakeCalendar()) == _dates(
        "2025-09-02"
    )


@pytest.mark.parametrize(
    "table,as_of", [("lot_sizes", "2026-11-12"), ("mcx_contracts", "2026-03-31"), ("symbols", None)]
)
def test_each_built_in_table_reports_the_date_it_was_last_checked(tables, table, as_of):
    expected = None if as_of is None else day(as_of)
    assert tables.table_as_of(table) == expected


def test_as_of_warning_is_given_on_the_first_use_past_the_date_even_after_earlier_uses():
    fresh = ReferenceTables()
    fresh.lot_size("NIFTY", "NFO", "2026-11-10", expiry="2027-02-23")  # inside the as-of date
    with pytest.warns(ReferenceDataWarning, match="was last checked on 2026-11-12"):
        fresh.lot_size("NIFTY", "NFO", "2027-02-01", expiry="2027-02-23")


# ---- exchange traded funds -------------------------------------------------------------------


def _with_etfs(tmp_path):
    path = write_file(
        tmp_path,
        "etfs.csv",
        "underlying,exchange,valid_from,kind\nNIFTYBEES,NSE,2019-01-01,etf\nNIFTYBEES,BSE,2019-01-01,etf\n",
    )
    return ReferenceTables(user_files=[path])


def test_etf_tick_on_nse_is_the_unverified_one_paisa_row(tmp_path):
    # NSE/CMTR/67133 excludes ETFs from the share rule; no document read states their tick.
    with pytest.warns(ReferenceDataWarning, match="tick_sizes.NSE etf.*is unverified"):
        etf = _with_etfs(tmp_path).instrument("NIFTYBEES", "NSE", "2025-06-02")
    assert etf.tick_size == 0.01


def test_etf_tick_on_bse_is_not_covered(tmp_path):
    tables = _with_etfs(tmp_path)
    etf = tables.instrument("NIFTYBEES", "BSE", "2025-06-02")
    assert etf.tick_size is None
    with pytest.raises(ReferenceDataError, match="ETF"):
        tables.round_price(275.12, etf, "2025-06-02")
