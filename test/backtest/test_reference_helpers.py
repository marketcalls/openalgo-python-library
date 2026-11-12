"""Shared builders for the reference tests. This module holds no tests of its own.

The CSV texts below are small made-up tables for exercising the mechanics. They are
not the reference data shipped in the package (test_reference_builtin.py covers that).
"""

import datetime as dt

from openalgo.backtest.reference import ReferenceTables


class FakeCalendar:
    """A session calendar: weekdays are sessions except the listed holidays.

    `coverage` is what `coverage(exchange, day)` reports for every date, except the
    years listed in `template_years` ("template") and `unverified_years` ("unverified").
    """

    def __init__(self, holidays=(), template_years=(), unverified_years=(), coverage="sourced"):
        self.holidays = set(holidays)
        self.template_years = set(template_years)
        self.unverified_years = set(unverified_years)
        self.default_coverage = coverage
        self.calls = []

    def session(self, exchange, day):
        self.calls.append((exchange, day))
        if day.weekday() >= 5 or day in self.holidays:
            return None
        return (0, 1)

    def coverage(self, exchange, day):
        if day.year in self.template_years:
            return "template"
        if day.year in self.unverified_years:
            return "unverified"
        return self.default_coverage


LOTS = """\
# A lot revision in the style of an exchange circular: contracts already listed keep
# the old lot of 25 until they expire, new contracts get 75.
underlying,exchange,valid_from,valid_to,expiry_from,expiry_to,lot_size,source,status,note
NIFTY,NFO,2021-01-01,2024-04-25,,,50,test,user,
NIFTY,NFO,2024-04-26,2024-12-26,,2024-12-26,25,test,user,
NIFTY,NFO,2024-11-20,,2025-01-02,2025-01-23,75,test,user,
NIFTY,NFO,2024-04-26,2025-01-30,2025-01-30,2025-01-30,25,test,user,
NIFTY,NFO,2024-11-20,,2025-02-01,,75,test,user,
RELIANCE,NFO,2024-01-01,,,,250,test,user,
"""

UNDERLYINGS = """\
underlying,exchange,valid_from,valid_to,kind,source,status,note
NIFTY,NFO,2019-01-01,,index,test,user,
BANKNIFTY,NFO,2019-01-01,,index,test,user,
RELIANCE,NFO,2019-01-01,,stock,test,user,
RELIANCE,NSE,2019-01-01,,stock,test,user,
RELIANCE,BSE,2019-01-01,,stock,test,user,
NIFTYBEES,NSE,2019-01-01,,etf,test,user,
"""

FREEZE = """\
underlying,exchange,valid_from,valid_to,freeze_qty_futures,freeze_qty_options,source,status,note
NIFTY,NFO,2021-01-01,2021-12-31,2800,2800,test,user,
NIFTY,NFO,2022-01-01,,1800,1800,test,user,
"""

TICKS = """\
exchange,segment,valid_from,valid_to,price_from,price_to,tick_size,basis,source,status,note
NFO,index_option,2019-01-01,,,,0.05,,test,user,
NFO,index_future,2019-01-01,,,,0.05,,test,user,
NSE,equity,2019-01-01,2025-04-14,,,0.05,,test,user,
NSE,equity,2025-04-15,,0,250,0.01,reference close,test,user,
NSE,equity,2025-04-15,,250,1000,0.05,reference close,test,user,
NSE,equity,2025-04-15,,1000,5000,0.10,reference close,test,user,
NFO,stock_future,2025-04-15,,0,250,0.01,underlying close,test,user,
NFO,stock_future,2025-04-15,,250,,0.05,underlying close,test,user,
NFO,stock_option,2025-11-03,,0,250,0.01,underlying close,test,user,
NFO,stock_option,2025-11-03,,250,,0.05,underlying close,test,user,
"""

SETTLEMENT = """\
exchange,segment,valid_from,valid_to,settlement,source,status,note
NFO,index_future,2019-01-01,,cash,test,user,
NFO,index_option,2019-01-01,,cash,test,user,
NFO,stock_future,2019-01-01,,physical,test,user,
NFO,stock_option,2019-01-01,,physical,test,user,
"""

RULES = """\
underlying,exchange,series,valid_from,valid_to,weekday,source,status,note
NIFTY,NFO,weekly,2019-01-01,2019-02-10,NONE,test,user,
NIFTY,NFO,weekly,2019-02-11,2025-08-31,THU,test,user,
NIFTY,NFO,weekly,2025-09-01,,TUE,test,user,
NIFTY,NFO,monthly,2019-01-01,2025-08-31,THU,test,user,
NIFTY,NFO,monthly,2025-09-01,,TUE,test,user,
BANKNIFTY,NFO,weekly,2023-09-04,2024-11-13,WED,test,user,
BANKNIFTY,NFO,weekly,2024-11-14,,NONE,test,user,
BANKNIFTY,NFO,monthly,2023-09-01,2024-02-29,THU,test,user,
BANKNIFTY,NFO,monthly,2024-03-01,2024-12-31,WED,test,user,
BANKNIFTY,NFO,monthly,2025-01-01,,THU,test,user,
RELIANCE,NFO,weekly,2019-01-01,,NONE,test,user,
RELIANCE,NFO,monthly,2019-01-01,2025-08-31,THU,test,user,
RELIANCE,NFO,monthly,2025-09-01,,TUE,test,user,
"""

MCX = """\
underlying,exchange,valid_from,valid_to,expiry_from,expiry_to,lot_size,multiplier,tick_size,freeze_qty,settlement,trading_unit,quote_unit,source,status,note
GOLD,MCX,2019-01-01,,,,1,100,1,10,physical,1 kg,10 g,test,user,
CRUDEOIL,MCX,2019-01-01,,,,100,1,1,10000,cash,100 barrels,1 barrel,test,user,
NATURALGAS,MCX,2019-01-01,,,,1250,1,0.1,,cash,1250 mmBtu,1 mmBtu,test,user,
NICKEL,MCX,2019-09-01,2025-08-31,2019-12-01,2025-08-31,1500,1,0.1,,physical,1500 kg,1 kg,test,user,
NICKEL,MCX,2025-08-18,,2025-09-01,,250,1,0.1,24000,physical,250 kg,1 kg,test,user,
"""

ALL_TABLES = {
    "lots.csv": LOTS,
    "underlyings.csv": UNDERLYINGS,
    "freeze.csv": FREEZE,
    "ticks.csv": TICKS,
    "settlement.csv": SETTLEMENT,
    "rules.csv": RULES,
    "mcx.csv": MCX,
}


def write_file(folder, name, text):
    """Write `text` to `folder/name` byte for byte and return the path as a string."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(text.encode("utf-8"))
    return str(path)


def build_tables(folder, extra=None, snapshots=None, only=None, calendar=None):
    """Reference tables made of the texts above only (no built-in rows).

    `extra` maps further file names to CSV text; `only` restricts the standard texts to
    the named files. Contracts are checked against `calendar`, by default a FakeCalendar
    with no holidays.
    """
    texts = {name: text for name, text in ALL_TABLES.items() if only is None or name in only}
    texts.update(extra or {})
    files = [write_file(folder, name, text) for name, text in texts.items()]
    calendar = FakeCalendar() if calendar is None else calendar
    return ReferenceTables(user_files=files, snapshots=snapshots, builtin=False, calendar=calendar)


def day(text):
    """An ISO date string as a date, to keep expected values short."""
    return dt.date.fromisoformat(text)
