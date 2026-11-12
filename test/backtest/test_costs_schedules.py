"""Cost schedules: the built-in versions, lookup by trade date, how far rates are
verified, and user-supplied versions.

Dates and rates come from the reference note `indian-trading-costs.md` and the
sources recorded in each built-in version.
"""

import copy
import datetime as dt
import warnings

import numpy as np
import pytest

from openalgo.backtest import costs
from openalgo.backtest.errors import CostError

# The last day of the old F&O STT rates and the first day of the new ones.
LAST_OLD_DAY = "2026-03-31"
FIRST_NEW_DAY = "2026-04-01"
# The day of the reference snapshot: the current rates are verified through it.
SNAPSHOT = "2026-11-12"
SNAPSHOT_DATE = dt.date(2026, 11, 12)

ROW_FIELDS = {
    "stt_buy_rate",
    "stt_sell_rate",
    "exchange_txn_rate",
    "ipft_rate",
    "sebi_rate",
    "stamp_buy_rate",
    "stt_exercise_rate",
    "verified_through",
    "calculator_tax_base",
    "calculator_tax_unit",
    "notes",
}


def quietly(function, *args, **kwargs):
    """Call a costing function whose rates may be unverified, without the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", costs.UnverifiedScheduleWarning)
        return function(*args, **kwargs)


def user_schedule(identifier, valid_from, valid_to=None, intraday_stt_rate=0.001, **extra):
    """A schedule with one row: NSE intraday, with a recognisable STT rate."""
    return {
        "id": identifier,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "gst_rate": 0.18,
        "rates": {
            "NSE_EQ_INTRADAY": {
                "stt_buy_rate": 0.0,
                "stt_sell_rate": intraday_stt_rate,
                "exchange_txn_rate": 0.00003,
                "sebi_rate": 0.000001,
                "stamp_buy_rate": 0.00003,
            }
        },
        **extra,
    }


# Each version before 2024-10-01 starts the day a rate changed and has its GST verified
# through its last day: (id, first day, last day).
EARLIER_VERSIONS = [
    ("IN-2020-07-01", dt.date(2020, 7, 1), dt.date(2020, 12, 31)),
    ("IN-2021-01-01", dt.date(2021, 1, 1), dt.date(2021, 2, 28)),
    ("IN-2021-03-01", dt.date(2021, 3, 1), dt.date(2021, 3, 31)),
    ("IN-2021-04-01", dt.date(2021, 4, 1), dt.date(2021, 4, 30)),
    ("IN-2021-05-01", dt.date(2021, 5, 1), dt.date(2022, 5, 1)),
    ("IN-2022-05-02", dt.date(2022, 5, 2), dt.date(2022, 7, 17)),
    ("IN-2022-07-18", dt.date(2022, 7, 18), dt.date(2022, 11, 30)),
    ("IN-2022-12-01", dt.date(2022, 12, 1), dt.date(2023, 3, 31)),
    ("IN-2023-04-01", dt.date(2023, 4, 1), dt.date(2023, 10, 31)),
    ("IN-2023-11-01", dt.date(2023, 11, 1), dt.date(2024, 5, 12)),
    ("IN-2024-05-13", dt.date(2024, 5, 13), dt.date(2024, 9, 30)),
]


def test_the_built_in_versions_and_how_far_each_is_verified():
    listed = [
        (s["id"], s["valid_from"], s["valid_to"], s["verified_through"]) for s in costs.schedules()
    ]
    assert listed == [
        *((vid, first, last, last) for vid, first, last in EARLIER_VERSIONS),
        ("IN-2024-10-01", dt.date(2024, 10, 1), dt.date(2026, 3, 31), dt.date(2026, 3, 31)),
        ("IN-2026-04-01", dt.date(2026, 4, 1), None, SNAPSHOT_DATE),
    ]


def test_a_trade_in_2022_is_priced_by_the_version_in_force_then():
    # NSE delivery buy of Rs 1 crore on 2022-01-03 (IN-2021-05-01): exchange Rs 3.45 per lakh
    # plus IPFT Rs 0.01 per crore = 345.01, SEBI Rs 10, STT Rs 10,000, stamp Rs 1,500.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cost = costs.trade_cost("BUY", 10_000_000, 1, "2022-01-03", segment="NSE_EQ_DELIVERY")
    assert (cost.exchange_txn, cost.sebi_fee, cost.stt, cost.stamp_duty) == pytest.approx(
        (345.01, 10.0, 10_000.0, 1_500.0)
    )
    assert (cost.schedules, cost.verified) == (("IN-2021-05-01",), True)


@pytest.mark.parametrize("day, gst", [("2022-07-15", 62.1018), ("2022-07-18", 63.9018)])
def test_gst_falls_on_the_sebi_fee_from_2022_07_18(day, gst):
    # NSE delivery buy of Rs 1 crore (no brokerage): 18 percent of the exchange line
    # 345.01, and from 2022-07-18 of the SEBI fee of Rs 10 too (SEBI circular
    # SEBI/HO/GSD/TAD/CIR/P/2022/0097; the fee was exempt before).
    cost = costs.trade_cost("BUY", 10_000_000, 1, day, segment="NSE_EQ_DELIVERY")
    assert cost.gst == pytest.approx(gst)


def test_each_built_in_version_says_whether_gst_falls_on_the_sebi_fee():
    flags = {s["id"]: s["gst_on_sebi_fee"] for s in costs.schedules()}
    assert [vid for vid, taxed in flags.items() if not taxed] == [
        vid for vid, first, _ in EARLIER_VERSIONS if first < dt.date(2022, 7, 18)
    ]


@pytest.mark.parametrize("taxed, gst", [(True, 4.158), (False, 4.14)])
def test_a_user_version_charges_gst_on_the_sebi_fee_unless_it_says_not(taxed, gst):
    # An intraday sell of Rs 1,00,000: brokerage Rs 20, exchange 3.00, SEBI fee 0.10.
    schedule = user_schedule("user-2019", "2019-04-01", "2020-06-30", gst_on_sebi_fee=taxed)
    model = costs.CostModel(schedules=[schedule])
    cost = quietly(model.trade, "SELL", 1000, 100, "2019-06-03", "NSE_EQ_INTRADAY")
    assert cost.gst == pytest.approx(gst)


def test_a_user_version_charges_gst_on_the_sebi_fee_by_default():
    model = costs.CostModel(schedules=[user_schedule("user-2019", "2019-04-01", "2020-06-30")])
    assert model.schedules()[0]["gst_on_sebi_fee"] is True


def test_an_unverified_row_of_an_earlier_version_warns_with_its_reason():
    with pytest.warns(costs.UnverifiedScheduleWarning, match="IN-2021-05-01.*group ZP"):
        cost = costs.trade_cost("BUY", 100, 10, "2022-01-03", segment="BSE_EQ_DELIVERY_X")
    assert cost.verified is False


def test_every_built_in_version_records_its_source_and_caveats():
    for schedule in costs.schedules():
        assert schedule["source"] and schedule["notes"], schedule["id"]


def test_every_built_in_version_has_a_full_row_for_every_segment():
    for schedule in costs.schedules():
        assert set(schedule["rates"]) == set(costs.SEGMENTS)
        assert all(set(row) == ROW_FIELDS for row in schedule["rates"].values())


def test_schedule_rates_are_fractions_named_with_a_rate_suffix():
    # Section 1: delivery STT 0.1 percent, GST 18 percent.
    current = costs.schedules()[-1]
    assert current["gst_rate"] == 0.18
    assert current["rates"]["NSE_EQ_DELIVERY"]["stt_sell_rate"] == 0.001


# The rows of the last two versions that no primary source confirms: BSE groups X (ZP
# pays ten times the row) and SS (no longer on the BSE fee page).
UNCONFIRMED_SINCE_2024_10_01 = [
    "BSE_EQ_DELIVERY_X",
    "BSE_EQ_INTRADAY_X",
    "BSE_EQ_DELIVERY_SS",
    "BSE_EQ_INTRADAY_SS",
]


@pytest.mark.parametrize(
    "version, last_day, also_unconfirmed",
    [
        ("IN-2024-10-01", dt.date(2026, 3, 31), []),
        ("IN-2026-04-01", SNAPSHOT_DATE, ["BFO_OPT_OTHER"]),
    ],
)
def test_the_last_two_versions_leave_unverified_only_the_rows_no_source_confirms(
    version, last_day, also_unconfirmed
):
    rows = next(s for s in costs.schedules() if s["id"] == version)["rates"]
    unverified = {code for code, row in rows.items() if row["verified_through"] is None}
    assert unverified == {*UNCONFIRMED_SINCE_2024_10_01, *also_unconfirmed}
    assert all(row["notes"] for code, row in rows.items() if code in unverified)
    assert all(
        row["verified_through"] == last_day for code, row in rows.items() if code not in unverified
    )


def test_listed_schedules_are_copies():
    listed = costs.schedules()
    listed[-1]["rates"]["NFO_FUT"]["stt_sell_rate"] = 0.5
    assert costs.schedules()[-1]["rates"]["NFO_FUT"]["stt_sell_rate"] == 0.0005


# F&O STT on a Rs 1 crore sell: (segment, before 2026-04-01, from 2026-04-01).
FNO_STT = [
    ("NFO_FUT", 2000.0, 5000.0),
    ("BFO_FUT", 2000.0, 5000.0),
    ("NFO_OPT", 10000.0, 15000.0),
    ("BFO_OPT", 10000.0, 15000.0),
]


@pytest.mark.parametrize("segment, before, after", FNO_STT)
def test_a_trade_is_charged_at_the_rates_in_force_on_its_trade_date(segment, before, after):
    model = costs.CostModel()
    old = quietly(model.trade, "SELL", 10_000_000, 1, LAST_OLD_DAY, segment)
    new = model.trade("SELL", 10_000_000, 1, FIRST_NEW_DAY, segment)
    assert (old.stt, new.stt) == pytest.approx((before, after))


def test_exercise_stt_changed_with_the_fno_stt_rates():
    # 0.125 percent before, 0.15 percent from 2026-04-01, on Rs 1,00,000 of intrinsic value.
    model = costs.CostModel()
    before = model.exercise_stt(100, 1000, LAST_OLD_DAY)
    assert (before, model.exercise_stt(100, 1000, FIRST_NEW_DAY)) == pytest.approx((125.0, 150.0))


def test_a_trade_on_the_last_verified_day_is_verified_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cost = costs.trade_cost("BUY", 100, 10, SNAPSHOT, segment="NSE_EQ_DELIVERY")
    assert cost.verified is True


@pytest.mark.parametrize("date", ["2026-11-13", "2031-01-01"])
def test_a_trade_after_the_last_verified_day_is_charged_but_flagged(date):
    # The rates are those of the snapshot (delivery STT 0.1 percent: Rs 1 on Rs 1,000),
    # but nobody has checked them for this date.
    with pytest.warns(costs.UnverifiedScheduleWarning, match="verified only up to"):
        cost = costs.trade_cost("BUY", 100, 10, date, segment="NSE_EQ_DELIVERY")
    assert (cost.stt, cost.verified, cost.backfilled) == (pytest.approx(1.0), False, False)


def test_a_trade_in_the_older_version_is_priced_by_it_and_verified():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cost = costs.trade_cost("BUY", 100, 10, "2025-06-02", segment="NSE_EQ_DELIVERY")
    assert (cost.schedules, cost.verified) == (("IN-2024-10-01",), True)


def test_an_unconfirmed_row_of_the_older_version_warns_with_its_reason():
    with pytest.warns(costs.UnverifiedScheduleWarning, match="IN-2024-10-01.*group ZP"):
        cost = costs.trade_cost("BUY", 100, 10, "2025-06-02", segment="BSE_EQ_DELIVERY_X")
    assert (cost.schedules, cost.verified) == (("IN-2024-10-01",), False)


def test_kapas_pays_the_uniform_mcx_charge_from_2024_10_01():
    # MCX/F&A/631/2024: Rs 2.10 per lakh for every futures contract from 2024-10-01, so a
    # Rs 1 crore KAPAS sale pays 210 (the Rs 2.60 per lakh of 2024-09-30 was its own rate).
    before = costs.trade_cost("SELL", 10_000_000, 1, "2024-09-30", segment="MCX_FUT_KAPAS")
    after = costs.trade_cost("SELL", 10_000_000, 1, "2024-10-01", segment="MCX_FUT_KAPAS")
    assert (before.exchange_txn, after.exchange_txn) == pytest.approx((260.0, 210.0))
    assert after.verified is True


def test_the_unconfirmed_bse_options_rate_is_flagged_with_its_notice_and_the_gap():
    # Rs 500 per crore of premium, from BSE notice 20240927-37 of 2024-09-27; no source
    # covers the dates from 2026-04-01.
    with pytest.warns(costs.UnverifiedScheduleWarning, match="20240927-37.*2026-04-01"):
        cost = costs.trade_cost("SELL", 10_000_000, 1, SNAPSHOT, segment="BFO_OPT_OTHER")
    assert (cost.exchange_txn, cost.verified) == (pytest.approx(500.0), False)


def test_sensex_and_bankex_options_pay_3250_rupees_per_crore():
    cost = costs.trade_cost("SELL", 10_000_000, 1, SNAPSHOT, segment="BFO_OPT")
    assert (cost.exchange_txn, cost.verified) == (pytest.approx(3250.0), True)


def test_a_trade_before_the_first_version_raises_saying_from_when_charges_are_available():
    with pytest.raises(CostError, match="2020-06-30: charges are available from 2020-07-01"):
        costs.trade_cost("BUY", 100, 10, "2020-06-30", segment="NSE_EQ_DELIVERY")


def test_the_dp_charge_before_the_first_version_raises():
    with pytest.raises(CostError, match="no cost schedule covers"):
        costs.dp_charge("2020-01-01")


def test_backfill_charges_early_trades_at_the_first_version_rates():
    model = costs.CostModel(backfill=True)
    with pytest.warns(costs.UnverifiedScheduleWarning, match="backfill"):
        early = model.trade("SELL", 10_000_000, 1, "2019-06-03", "NFO_FUT")
    # The first version's futures STT: 0.01 percent (NSE/FATAX/56235, rate up to 2023-03-31).
    assert (early.stt, early.verified, early.backfilled) == (pytest.approx(1000.0), False, True)


def test_a_round_trip_across_the_boundary_charges_each_leg_at_its_own_rates():
    # Buy on the old day, sell Rs 1 crore on the new day: STT at 0.05 percent.
    cost = costs.round_trip_cost(
        10_000_000, 10_000_000, 1, LAST_OLD_DAY, sell_date=FIRST_NEW_DAY, segment="NFO_FUT"
    )
    assert cost.stt == pytest.approx(5000.0)
    assert (cost.schedules, cost.verified) == (("IN-2024-10-01", "IN-2026-04-01"), True)


def test_a_user_version_for_an_earlier_period_is_used_on_its_dates():
    # Rs 1,00,000 sold at the user version's 0.1 percent STT.
    model = costs.CostModel(schedules=[user_schedule("user-2019", "2019-04-01", "2020-06-30")])
    cost = quietly(model.trade, "SELL", 1000, 100, "2020-01-15", "NSE_EQ_INTRADAY")
    assert (cost.stt, cost.schedules) == (pytest.approx(100.0), ("user-2019",))


def test_user_versions_are_listed_with_the_built_in_ones():
    model = costs.CostModel(schedules=[user_schedule("user-2019", "2019-04-01", "2020-06-30")])
    listed = [s["id"] for s in model.schedules()]
    assert listed == ["user-2019"] + [s["id"] for s in costs.schedules()]


def test_a_user_version_is_unverified_unless_it_says_how_far_it_was_checked():
    plain = user_schedule("plain", "2018-04-01", "2018-12-31")
    checked = user_schedule("checked", "2019-01-01", "2020-06-30", verified_through="2019-06-30")
    model = costs.CostModel(schedules=[plain, checked])
    trade = ("SELL", 1000, 100)
    flags = [
        quietly(model.trade, *trade, date, "NSE_EQ_INTRADAY").verified
        for date in ("2018-06-01", "2019-06-30", "2019-07-01")
    ]
    assert flags == [False, True, False]


def test_a_user_row_defaults_its_optional_fields():
    model = costs.CostModel(schedules=[user_schedule("user-2019", "2019-04-01", "2020-06-30")])
    row = model.schedules()[0]["rates"]["NSE_EQ_INTRADAY"]
    defaults = ("ipft_rate", "stt_exercise_rate", "calculator_tax_base", "calculator_tax_unit")
    assert [row[name] for name in defaults] == [0.0, 0.0, "sides", "rupee"]


@pytest.mark.parametrize(
    "field, default",
    [("notes", ""), ("calculator_tax_base", "sides"), ("calculator_tax_unit", "rupee")],
)
def test_a_user_row_given_none_for_a_text_field_takes_its_default(field, default):
    # As for the schedule's own source and notes, None means not given (not the text "None").
    schedule = broken(lambda s: row(s).update({field: None}))
    model = costs.CostModel(schedules=[schedule])
    assert model.schedules()[0]["rates"]["NSE_EQ_INTRADAY"][field] == default


def test_a_segment_missing_from_a_user_version_raises_on_its_dates():
    model = costs.CostModel(schedules=[user_schedule("user-2019", "2019-04-01", "2020-06-30")])
    with pytest.raises(CostError, match="has no rates for segment NFO_FUT"):
        model.trade("SELL", 1000, 100, "2020-01-15", "NFO_FUT")


def test_a_later_user_version_takes_over_from_the_open_built_in_version():
    model = costs.CostModel(schedules=[user_schedule("user-2027", dt.date(2027, 4, 1))])
    ranges = [(s["id"], s["valid_to"]) for s in model.schedules()[-2:]]
    assert ranges == [("IN-2026-04-01", dt.date(2027, 3, 31)), ("user-2027", None)]


def test_user_versions_can_replace_the_built_in_ones():
    model = costs.CostModel(schedules=[user_schedule("only", "2020-01-01")], include_builtin=False)
    assert [s["id"] for s in model.schedules()] == ["only"]


def test_a_built_in_version_can_be_copied_edited_and_supplied():
    # The current version with a hypothetical NFO futures STT of 0.1 percent from 2027.
    edited = copy.deepcopy(costs.schedules()[-1])
    edited.update(id="what-if-2027", valid_from="2027-01-01")
    edited["rates"]["NFO_FUT"]["stt_sell_rate"] = 0.001
    model = costs.CostModel(schedules=[edited])
    cost = quietly(model.trade, "SELL", 10_000_000, 1, "2027-02-01", "NFO_FUT")
    assert cost.stt == pytest.approx(10000.0)


def test_a_date_outside_the_supported_range_never_reaches_the_kernel_as_a_panic():
    # A user schedule that ends, no built-in versions, and the largest numpy date.
    only = user_schedule("ends", "2020-01-01", "2027-01-01")
    model = costs.CostModel(schedules=[only], include_builtin=False)
    with pytest.raises(CostError, match="outside the supported range"):
        model.trade("SELL", 100, 1, np.datetime64(2**31 - 1, "D"), "NSE_EQ_INTRADAY")


def broken(change):
    """A valid user schedule with one thing wrong."""
    schedule = user_schedule("broken", "2019-04-01", "2020-06-30")
    change(schedule)
    return schedule


def row(schedule):
    return schedule["rates"]["NSE_EQ_INTRADAY"]


UNITS = "rates are fractions: 0.001 is 0.1 percent"

# (what is wrong, the schedule, text the message must contain)
MALFORMED = [
    ("a misspelt field", broken(lambda s: s.update(valid_until="2020-06-30")), "unknown field"),
    ("the old gst key", broken(lambda s: s.update(gst=0.18)), "unknown field 'gst'"),
    ("a missing field", broken(lambda s: s.pop("gst_rate")), "missing field 'gst_rate'"),
    ("a missing rate", broken(lambda s: row(s).pop("sebi_rate")), "missing rate 'sebi_rate'"),
    ("a misspelt rate", broken(lambda s: row(s).update(stamp=0.1)), "unknown field 'stamp'"),
    ("STT in percent units", broken(lambda s: row(s).update(stt_sell_rate=0.025)), UNITS),
    ("stamp duty in percent units", broken(lambda s: row(s).update(stamp_buy_rate=0.015)), UNITS),
    ("the SEBI fee in percent units", broken(lambda s: row(s).update(sebi_rate=0.0001)), UNITS),
    ("GST in percent units", broken(lambda s: s.update(gst_rate=18)), UNITS),
    (
        "a GST base that is not True or False",
        broken(lambda s: s.update(gst_on_sebi_fee="no")),
        "gst_on_sebi_fee must be True or False",
    ),
    (
        "a rate that is not a number",
        broken(lambda s: row(s).update(stt_sell_rate="0.1%")),
        "number",
    ),
    (
        "an unknown segment",
        broken(lambda s: s["rates"].update(NSE=dict(row(s)))),
        "unknown cost segment",
    ),
    (
        "an unknown tax base",
        broken(lambda s: row(s).update(calculator_tax_base="both")),
        "sides or turnover",
    ),
    (
        "one segment given twice in two spellings",
        broken(lambda s: s["rates"].update(nse_eq_intraday=dict(row(s), stt_sell_rate=0.0005))),
        "'NSE_EQ_INTRADAY' and 'nse_eq_intraday' are both segment NSE_EQ_INTRADAY",
    ),
    ("no rates", broken(lambda s: s.update(rates={})), "no segment rates"),
    ("an end before the start", broken(lambda s: s.update(valid_to="2019-01-01")), "cannot end"),
    ("a day-first date", broken(lambda s: s.update(valid_from="01-04-2023")), "ISO date"),
    ("a date out of range", broken(lambda s: s.update(valid_to="9999-12-31")), "supported range"),
    ("an overlap with a built-in version", user_schedule("clash", "2025-01-01"), "same dates"),
    ("not a dict", "IN-2026-04-01", "must be a dict"),
]


@pytest.mark.parametrize("problem, schedule, text", MALFORMED, ids=[case[0] for case in MALFORMED])
def test_a_malformed_user_schedule_raises(problem, schedule, text):
    with pytest.raises(CostError, match=text):
        costs.CostModel(schedules=[schedule])


def test_replacing_the_built_in_versions_needs_at_least_one_schedule():
    with pytest.raises(CostError, match="needs at least one schedule"):
        costs.CostModel(include_builtin=False)
