"""The cost wrapper: one trade, a round trip, segments, presets, DP charge, exercise STT.

Expected figures come from the reference note `indian-trading-costs.md` (section and
row named in each test) or are worked out by hand in the test's comment. None is
produced by running the engine.
"""

import datetime as dt
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from openalgo.backtest import costs
from openalgo.backtest.errors import BacktestError, CostError

# The day of the reference snapshot: the last day the current rates are verified for.
SNAPSHOT = "2026-11-12"

# Reference note section 6: the worked example, NSE selected.
# (segment, buy, sell, brokerage, stt, exchange_txn, gst, sebi_fee, stamp_duty, total)
WORKED_EXAMPLE = [
    ("NSE_EQ_INTRADAY", 1000, 1100, 40.0, 110.0, 25.79, 11.99, 0.84, 12.0, 200.62),
    ("NSE_EQ_DELIVERY", 1000, 1100, 0.0, 840.0, 25.79, 4.79, 0.84, 60.0, 931.42),
    ("NFO_FUT", 1000, 1100, 40.0, 220.0, 15.37, 10.12, 0.84, 8.0, 294.33),
    ("NFO_OPT", 100, 110, 40.0, 66.0, 29.85, 12.59, 0.08, 1.0, 149.52),
]


@pytest.mark.parametrize("row", WORKED_EXAMPLE, ids=lambda row: row[0])
def test_calculator_mode_reproduces_the_worked_example_to_the_paisa(row):
    segment, buy, sell, brokerage, stt, exchange_txn, gst, sebi_fee, stamp_duty, total = row
    cost = costs.round_trip_cost(buy, sell, 400, SNAPSHOT, segment=segment, rounding="calculator")
    assert cost.as_dict() == {
        "brokerage": brokerage,
        "stt": stt,
        "exchange_txn": exchange_txn,
        "sebi_fee": sebi_fee,
        "stamp_duty": stamp_duty,
        "gst": gst,
        "total": total,
    }


def test_trade_cost_itemises_a_delivery_buy():
    # Section 1 rates on a buy worth Rs 4,00,000: no brokerage, STT 0.1 percent = 400,
    # exchange 0.00307 percent = 12.28, SEBI Rs 10 per crore = 0.40, stamp duty 0.015
    # percent = 60, GST 0.18 x (12.28 + 0.40) = 2.2824.
    cost = costs.trade_cost("BUY", 1000, 400, SNAPSHOT, exchange="NSE", product="CNC")
    assert cost.as_dict() == pytest.approx(
        {
            "brokerage": 0.0,
            "stt": 400.0,
            "exchange_txn": 12.28,
            "sebi_fee": 0.40,
            "stamp_duty": 60.0,
            "gst": 2.2824,
            "total": 474.9624,
        }
    )


def test_trade_cost_itemises_an_intraday_sell():
    # A sell worth Rs 4,40,000: brokerage capped at 20, STT 0.025 percent = 110,
    # exchange 0.00307 percent = 13.508, SEBI 0.44, no stamp duty, GST 0.18 x 33.948.
    cost = costs.trade_cost("SELL", 1100, 400, SNAPSHOT, exchange="NSE", product="MIS")
    assert cost.as_dict() == pytest.approx(
        {
            "brokerage": 20.0,
            "stt": 110.0,
            "exchange_txn": 13.508,
            "sebi_fee": 0.44,
            "stamp_duty": 0.0,
            "gst": 6.11064,
            "total": 150.05864,
        }
    )


def test_the_breakdown_names_its_segment_rounding_schedule_and_status():
    cost = costs.trade_cost("BUY", 1000, 400, SNAPSHOT, exchange="NSE", product="CNC")
    described = (cost.segment, cost.rounding, cost.schedules, cost.verified, cost.backfilled)
    assert described == ("NSE_EQ_DELIVERY", "contract_note", ("IN-2026-04-01",), True, False)


@pytest.mark.parametrize("side", ["SELL", "sell", " Sell ", -1, -1.0, np.int64(-1)])
def test_a_sell_is_the_text_sell_or_the_signed_number_minus_one(side):
    # Intraday STT of 0.025 percent falls on the sell side only.
    cost = costs.trade_cost(side, 1100, 400, SNAPSHOT, segment="NSE_EQ_INTRADAY")
    assert cost.stt == pytest.approx(110.0)


@pytest.mark.parametrize("side", ["BUY", "buy", 1, 1.0, np.int32(1)])
def test_a_buy_is_the_text_buy_or_the_signed_number_plus_one(side):
    cost = costs.trade_cost(side, 1100, 400, SNAPSHOT, segment="NSE_EQ_INTRADAY")
    assert cost.stt == 0.0


@pytest.mark.parametrize(
    "side", ["HOLD", "", 0, 2, -400.0, 2.5, None, True, float("nan"), np.uint8(1), np.uint8(0)]
)
def test_any_other_side_raises(side):
    with pytest.raises(CostError, match="side"):
        costs.trade_cost(side, 1100, 400, SNAPSHOT, segment="NSE_EQ_INTRADAY")


IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


@pytest.mark.parametrize(
    "date",
    [
        "2026-04-01",
        dt.date(2026, 4, 1),
        dt.datetime(2026, 4, 1, 9, 15),
        dt.datetime(2026, 4, 1, 9, 15, tzinfo=IST),
        # 20:00 UTC on 31 March is 01:30 IST on 1 April.
        dt.datetime(2026, 3, 31, 20, 0, tzinfo=dt.UTC),
        pd.Timestamp("2026-04-01 15:29"),
        pd.Timestamp("2026-03-31 20:00", tz="UTC"),
        np.datetime64("2026-04-01"),
        np.datetime64("2026-04-01T09:15:00"),
    ],
    ids=repr,
)
def test_a_trade_date_may_be_given_in_any_date_form_and_is_read_as_ist(date):
    # Futures STT is 0.05 percent from 2026-04-01: Rs 5,000 on Rs 1 crore sold.
    cost = costs.trade_cost("SELL", 10_000_000, 1, date, segment="NFO_FUT")
    assert cost.stt == pytest.approx(5000.0)


@pytest.mark.parametrize(
    "date",
    [None, 20261112, "01-04-2026", "04/01/2026", "20260401", "2026-02-30", "soon", pd.NaT],
    ids=repr,
)
def test_a_date_that_is_not_an_iso_date_raises(date):
    with pytest.raises(CostError, match="date"):
        costs.trade_cost("BUY", 100, 1, date, segment="NSE_EQ_DELIVERY")


@pytest.mark.parametrize(
    "date",
    [
        np.datetime64(2**31 - 1, "D"),
        np.datetime64(-(2**31) + 5, "D"),
        dt.date(1, 1, 1),
        "9999-12-31",
    ],
    ids=repr,
)
def test_a_date_outside_the_supported_range_raises_a_cost_error(date):
    model = costs.CostModel(backfill=True)
    with pytest.raises(CostError, match="outside the supported range"):
        model.trade("BUY", 1, 1, date, "NSE_EQ_DELIVERY")


# (exchange, product, instrument, extra arguments, segment)
SEGMENT_FOR = [
    ("NSE", "CNC", "EQ", {}, "NSE_EQ_DELIVERY"),
    ("NSE", "MIS", "EQ", {}, "NSE_EQ_INTRADAY"),
    ("nse", "cnc", "eq", {}, "NSE_EQ_DELIVERY"),
    ("BSE", "CNC", "EQ", {"bse_group": "A"}, "BSE_EQ_DELIVERY"),
    ("BSE", "MIS", "EQ", {"bse_group": "B"}, "BSE_EQ_INTRADAY"),
    ("BSE", "CNC", "EQ", {"bse_group": "MT"}, "BSE_EQ_DELIVERY_M"),
    ("BSE", "CNC", "EQ", {"bse_group": "Z"}, "BSE_EQ_DELIVERY_X"),
    ("BSE", "MIS", "EQ", {"bse_group": "st"}, "BSE_EQ_INTRADAY_SS"),
    ("NFO", "NRML", "FUT", {}, "NFO_FUT"),
    ("NFO", "MIS", "OPT", {}, "NFO_OPT"),
    ("NFO", "NRML", "CE", {}, "NFO_OPT"),
    ("BFO", "NRML", "FUT", {}, "BFO_FUT"),
    ("BFO", "NRML", "PE", {"underlying": "SENSEX"}, "BFO_OPT"),
    ("BFO", "NRML", "CE", {"underlying": "BANKEX"}, "BFO_OPT"),
    ("BFO", "NRML", "OPT", {"underlying": "SENSEX50"}, "BFO_OPT_OTHER"),
    ("BFO", "MIS", "OPT", {"underlying": "RELIANCE"}, "BFO_OPT_OTHER"),
    ("MCX", "NRML", "FUT", {"underlying": "CRUDEOIL"}, "MCX_FUT"),
    ("MCX", "MIS", "FUT", {"underlying": "MENTHAOIL"}, "MCX_FUT_AGRI"),
    ("MCX", "NRML", "FUT", {"underlying": "KAPAS"}, "MCX_FUT_KAPAS"),
]


@pytest.mark.parametrize("exchange, product, instrument, extra, segment", SEGMENT_FOR)
def test_segment_for_resolves_openalgo_codes(exchange, product, instrument, extra, segment):
    assert costs.segment_for(exchange, product, instrument, **extra) == segment


# (exchange, product, instrument, extra arguments, text the message must contain)
NO_SEGMENT = [
    ("CDS", "NRML", "FUT", {}, "not costed yet"),
    ("BCD", "NRML", "OPT", {}, "not costed yet"),
    ("MCX", "NRML", "OPT", {"underlying": "GOLD"}, "MCX options"),
    ("NSE_INDEX", "MIS", "EQ", {}, "index exchange"),
    ("NYSE", "CNC", "EQ", {}, "unknown exchange"),
    ("NSE", "BO", "EQ", {}, "unknown product"),
    ("NSE", "CNC", "SWAP", {}, "unknown instrument type"),
    ("NSE", "NRML", "EQ", {}, "CNC (delivery) or MIS (intraday)"),
    ("NFO", "CNC", "FUT", {}, "NRML or MIS"),
    ("NSE", "NRML", "FUT", {}, "trade on NFO"),
    ("BSE", "CNC", "EQ", {}, "needs its scrip group"),
    ("BSE", "MIS", "EQ", {}, "instrument master"),
    ("BSE", "CNC", "EQ", {"bse_group": "QQ"}, "unknown BSE scrip group"),
    ("BFO", "NRML", "OPT", {}, "pass the underlying"),
    ("MCX", "NRML", "FUT", {}, "pass the commodity"),
    ("MCX", "NRML", "FUT", {"underlying": "UNOBTAINIUM"}, "unknown MCX commodity"),
]


@pytest.mark.parametrize("exchange, product, instrument, extra, text", NO_SEGMENT)
def test_segment_for_raises_instead_of_guessing(exchange, product, instrument, extra, text):
    with pytest.raises(CostError, match=re.escape(text)):
        costs.segment_for(exchange, product, instrument, **extra)


def test_a_bse_trade_without_its_scrip_group_is_not_charged_as_group_a():
    # A Z-group delivery buy of Rs 1,00,000 owes an exchange charge of Rs 10,000 per
    # crore: Rs 100, with GST of 0.18 x (100 + 0.10 SEBI) = 18.018. Without the group
    # there is no charge to assume (group A would be Rs 3.75).
    with pytest.raises(CostError, match="scrip group"):
        costs.trade_cost("BUY", 100_000, 1, SNAPSHOT, exchange="BSE", product="CNC")
    cost = costs.trade_cost(
        "BUY", 100_000, 1, SNAPSHOT, exchange="BSE", product="CNC", bse_group="Z"
    )
    assert (cost.exchange_txn, cost.gst) == pytest.approx((100.0, 18.018))


@pytest.mark.parametrize("segment", ["NSE", "NFO-OPT", "NSE_EQ", "", "CDS_FUT", "MCX_OPT", 7])
def test_an_unknown_or_unsupported_segment_code_raises(segment):
    with pytest.raises(CostError):
        costs.trade_cost("BUY", 100, 1, SNAPSHOT, segment=segment)


def test_a_segment_code_may_be_given_in_lower_case():
    cost = costs.trade_cost("BUY", 100, 1, SNAPSHOT, segment=" nse_eq_delivery ")
    assert cost.segment == "NSE_EQ_DELIVERY"


def test_passing_both_a_segment_and_an_exchange_raises():
    with pytest.raises(CostError, match="not both"):
        costs.trade_cost("BUY", 1, 1, SNAPSHOT, segment="NFO_FUT", exchange="NFO", product="MIS")


def test_passing_neither_a_segment_nor_an_exchange_raises():
    with pytest.raises(CostError, match="say what is being traded"):
        costs.trade_cost("BUY", 1, 1, SNAPSHOT)


def test_the_segment_table_lists_every_segment_once():
    table = costs.segments()
    assert [row["code"] for row in table] == list(costs.SEGMENTS)
    assert [row["id"] for row in table] == list(range(len(costs.SEGMENTS)))
    assert len(set(costs.SEGMENTS)) == 21


def test_components_are_listed_in_output_order():
    expected = ("brokerage", "stt", "exchange_txn", "sebi_fee", "stamp_duty", "gst")
    assert expected == costs.COMPONENTS


def test_bse_exchange_charge_is_not_the_nse_charge():
    # Section 1: NSE equity 0.00307 percent, BSE equity 0.00375 percent.
    nse = costs.trade_cost("BUY", 10_000_000, 1, SNAPSHOT, segment="NSE_EQ_DELIVERY")
    bse = costs.trade_cost("BUY", 10_000_000, 1, SNAPSHOT, segment="BSE_EQ_DELIVERY")
    assert (nse.exchange_txn, bse.exchange_txn) == pytest.approx((307.0, 375.0))


# Brokerage on a Rs 2.5 lakh NSE intraday order under each preset.
# 0.05 percent of 2.5 lakh is Rs 125; the custom intraday rule is a flat Rs 5.
PRESET_BROKERAGE = [
    ("zerodha", 20.0),
    ("zero_brokerage", 0.0),
    (costs.zerodha(), 20.0),
    (costs.zero_brokerage(), 0.0),
    (costs.flat(10), 10.0),
    (costs.percent(0.03, 20), 20.0),
    (costs.percent(0.05, 50), 50.0),
    (costs.percent(0.05), 125.0),
    (costs.custom(delivery=("percent", 0.1, 30), intraday=5, futures=None, options=20), 5.0),
    (costs.custom(intraday=("flat", 5)), 5.0),
    (costs.custom(intraday=("percent", 0.01)), 25.0),
]


@pytest.mark.parametrize("brokerage, expected", PRESET_BROKERAGE, ids=lambda value: str(value))
def test_brokerage_follows_the_preset(brokerage, expected):
    cost = costs.trade_cost(
        "BUY", 250_000, 1, SNAPSHOT, segment="NSE_EQ_INTRADAY", brokerage=brokerage
    )
    assert cost.brokerage == pytest.approx(expected)


def test_percent_takes_percent_units():
    # percent(0.03, 20) is Zerodha's "0.03 percent or Rs 20": Rs 15 on Rs 50,000.
    cost = costs.trade_cost(
        "BUY", 50_000, 1, SNAPSHOT, segment="NSE_EQ_DELIVERY", brokerage=costs.percent(0.03, 20)
    )
    assert cost.brokerage == pytest.approx(15.0)


def test_a_preset_reports_its_rules_as_rates():
    # The kernel stores 0.03 percent as the fraction 0.0003.
    assert costs.percent(0.03, 20).rules["intraday"] == ("rate", pytest.approx(0.0003), 20.0)
    assert costs.zerodha().rules["delivery"] == ("zero",)
    assert costs.zerodha().rules["options"] == ("flat", 20.0)


def test_statutory_components_are_the_same_under_every_preset():
    zerodha = costs.trade_cost("SELL", 250_000, 1, SNAPSHOT, segment="NFO_OPT")
    flat = costs.trade_cost(
        "SELL", 250_000, 1, SNAPSHOT, segment="NFO_OPT", brokerage=costs.flat(5)
    )
    statutory = ("stt", "exchange_txn", "sebi_fee", "stamp_duty")
    assert [getattr(flat, name) for name in statutory] == [
        getattr(zerodha, name) for name in statutory
    ]


@pytest.mark.parametrize("name", ["upstox", "Zerodha", "", "flat"])
def test_an_unknown_preset_name_raises(name):
    with pytest.raises(CostError, match="unknown brokerage preset"):
        costs.CostModel(brokerage=name)


# (what is wrong, how it is built, text the message must contain)
BAD_PRESETS = [
    ("a fraction passed as a percentage", lambda: costs.percent(0.0003, 20), "percent units"),
    ("a percentage above 5", lambda: costs.percent(30), "not plausible"),
    ("a fraction in a custom rule", lambda: costs.custom(intraday=("percent", 0.0003)), "percent"),
    ("a negative flat amount", lambda: costs.flat(-1), "zero or more"),
    ("a flat amount that is not a number", lambda: costs.flat("20"), "must be a number"),
    ("an amount too large for a float", lambda: costs.flat(10**400), "too large"),
    ("a NaN amount", lambda: costs.flat(float("nan")), "zero or more"),
    ("a negative cap", lambda: costs.percent(0.03, -5), "cap"),
    # A cap that is not a number of rupees must not read as no cap, in custom() as in percent().
    ("a NaN cap in a custom rule", lambda: costs.custom(intraday=("percent", 0.03, np.nan)), "cap"),
    (
        "a negative infinite cap in a custom rule",
        lambda: costs.custom(intraday=("percent", 0.03, -np.inf)),
        "cap",
    ),
    ("a negative DP fee", lambda: costs.zero_brokerage(dp_fee=-1), "DP fee"),
    ("an unknown rule kind", lambda: costs.custom(intraday=("tiered", 1, 2)), "cannot read"),
    ("a rule given as text", lambda: costs.custom(options="free"), "cannot read"),
    ("a number where a preset is expected", lambda: costs.CostModel(brokerage=20), "preset name"),
]


@pytest.mark.parametrize("problem, build, text", BAD_PRESETS, ids=[row[0] for row in BAD_PRESETS])
def test_an_invalid_preset_raises(problem, build, text):
    with pytest.raises(CostError, match=text):
        build()


def test_an_unknown_rounding_mode_raises():
    with pytest.raises(CostError, match="contract_note or calculator"):
        costs.CostModel(rounding="nearest")


def test_dp_charge_is_15_34_per_scrip_per_sell_day():
    # Section 4: Rs 3.5 CDSL + Rs 9.5 Zerodha + Rs 2.34 GST.
    charge = costs.dp_charge(SNAPSHOT)
    assert (charge.fee, charge.gst, charge.total) == (13.0, 2.34, pytest.approx(15.34))
    assert (charge.fee_source, charge.schedule, charge.verified) == (
        "zerodha_preset",
        "IN-2026-04-01",
        True,
    )


# (preset, DP fee before GST, where the fee comes from)
DP_FEES = [
    (costs.zerodha(), 13.0, "zerodha_preset"),
    (costs.flat(10), 13.0, "zerodha_stand_in"),
    (costs.percent(0.03, 20), 13.0, "zerodha_stand_in"),
    (costs.zero_brokerage(), 13.0, "zerodha_stand_in"),
    (costs.custom(intraday=5), 13.0, "zerodha_stand_in"),
    (costs.flat(10, dp_fee=20), 20.0, "supplied"),
    (costs.zerodha().with_dp_fee(0), 0.0, "supplied"),
]


@pytest.mark.parametrize("brokerage, fee, source", DP_FEES, ids=lambda value: str(value))
def test_a_dp_charge_says_whose_fee_it_charged(brokerage, fee, source):
    charge = costs.dp_charge(SNAPSHOT, brokerage)
    assert (charge.fee, charge.fee_source) == (fee, source)
    assert (brokerage.dp_fee, brokerage.dp_fee_source) == (fee, source)


@pytest.mark.parametrize("date", ["2025-06-02", "2031-01-01"])
def test_dp_charge_warns_when_gst_is_not_verified_for_the_date(date):
    with pytest.warns(costs.UnverifiedScheduleWarning, match="GST"):
        charge = costs.dp_charge(date)
    assert charge.verified is False


def test_exercise_stt_is_15_bps_of_intrinsic_value():
    # Section 1: intrinsic Rs 50 on 75 units is Rs 3,750, and 0.15 percent is 5.625.
    assert costs.exercise_stt(50, 75, SNAPSHOT) == pytest.approx(5.625)
    assert costs.exercise_stt(50, 75, SNAPSHOT, "BFO_OPT") == pytest.approx(5.625)


def test_exercise_stt_is_rounded_to_the_rupee_in_calculator_mode():
    # Zerodha's STT article: Rs 3.75 "rounds to Rs 4".
    assert costs.exercise_stt(50, 50, SNAPSHOT, rounding="calculator") == 4.0


def test_exercise_stt_is_for_options_only():
    with pytest.raises(CostError, match="options only"):
        costs.exercise_stt(50, 75, SNAPSHOT, "NFO_FUT")


@pytest.mark.parametrize("segment", costs.SEGMENTS)
def test_zero_quantity_costs_zero(segment):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", costs.UnverifiedScheduleWarning)
        cost = costs.trade_cost("SELL", 1234.5, 0, SNAPSHOT, segment=segment)
    assert cost.total == 0.0


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "100", None, 10**400])
def test_a_price_or_quantity_that_is_not_a_valid_number_raises_a_cost_error(bad):
    with pytest.raises(CostError):
        costs.trade_cost("BUY", bad, 1, SNAPSHOT, segment="NSE_EQ_DELIVERY")
    with pytest.raises(CostError):
        costs.trade_cost("BUY", 1, bad, SNAPSHOT, segment="NSE_EQ_DELIVERY")


def test_a_traded_value_that_overflows_raises_instead_of_returning_nan():
    # Each of 1e200 is a valid number; their product is not.
    with pytest.raises(CostError, match="traded value"):
        costs.trade_cost("BUY", 1e200, 1e200, SNAPSHOT, segment="NFO_OPT")


def test_cost_error_is_a_backtest_error_and_a_value_error():
    with pytest.raises(BacktestError) as caught:
        costs.trade_cost("BUY", 1, 1, SNAPSHOT, segment="NOPE")
    assert isinstance(caught.value, ValueError)
