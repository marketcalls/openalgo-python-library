"""Costing many trades at once, and rounding them into contract notes.

Expected figures are worked out by hand from the rates of the reference note
`indian-trading-costs.md`; none is produced by running the engine.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from openalgo.backtest import costs
from openalgo.backtest.errors import CostError

SNAPSHOT = "2026-11-12"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
AMOUNTS = ["brokerage", "stt", "exchange_txn", "sebi_fee", "stamp_duty", "gst", "total"]
COLUMNS = ["date", "segment", *AMOUNTS, "schedule", "verified", "backfilled"]

# The two legs of the intraday worked example (section 6) as one-sided calculator
# trades, by hand:
#   buy 1000 x 400: brokerage 20, exchange 11.88 + IPFT 0.40, SEBI 0.40,
#     GST 0.18 x 32.68 = 5.88, stamp duty 12, total 50.56
#   sell 1100 x 400: brokerage 20, STT 110, exchange 13.07 + IPFT 0.44, SEBI 0.44,
#     GST 0.18 x 33.95 = 6.11, total 150.06
INTRADAY_LEGS = pd.DataFrame(
    [[20.0, 0.0, 12.28, 0.40, 12.0, 5.88, 50.56], [20.0, 110.0, 13.51, 0.44, 0.0, 6.11, 150.06]],
    columns=AMOUNTS,
)


def intraday_legs(**overrides):
    arguments = {
        "side": ["BUY", "SELL"],
        "price": [1000.0, 1100.0],
        "quantity": 400,
        "date": SNAPSHOT,
        "segment": "NSE_EQ_INTRADAY",
        "rounding": "calculator",
    }
    arguments.update(overrides)
    return costs.trade_costs(**arguments)


def assert_intraday_legs(frame):
    pd.testing.assert_frame_equal(frame[AMOUNTS], INTRADAY_LEGS, check_exact=True)


def test_trades_returns_one_row_per_trade_with_its_date_segment_amounts_and_status():
    frame = intraday_legs()
    assert list(frame.columns) == COLUMNS
    assert_intraday_legs(frame)
    described = frame[["segment", "schedule", "verified", "backfilled"]].iloc[0].tolist()
    assert described == ["NSE_EQ_INTRADAY", "IN-2026-04-01", True, False]
    assert list(frame["date"]) == [pd.Timestamp(SNAPSHOT)] * 2


def test_trades_can_return_the_same_columns_as_a_dict_of_arrays():
    arrays = intraday_legs(as_frame=False)
    assert list(arrays) == COLUMNS
    np.testing.assert_array_equal(arrays["total"], [50.56, 150.06])
    np.testing.assert_array_equal(arrays["verified"], [True, True])


def test_trades_keeps_the_index_of_a_price_series():
    prices = pd.Series([1000.0, 1100.0], index=["entry", "exit"])
    assert list(intraday_legs(price=prices).index) == ["entry", "exit"]


@pytest.mark.parametrize(
    "side",
    [("BUY", "SELL"), np.array(["buy", "sell"]), [1, -1], np.array([1.0, -1.0]), ["BUY", -1]],
    ids=["tuple of text", "array of text", "signed ints", "signed floats", "mixed"],
)
def test_sides_are_buy_and_sell_text_or_signed_ones(side):
    assert_intraday_legs(intraday_legs(side=side))


@pytest.mark.parametrize(
    "side",
    [
        np.array([1, 1], dtype=np.uint8),
        np.array([0, 1], dtype=np.uint8),
        np.array([True, False]),
        [1, 0],
        [400.0, -400.0],
        [2, -2],
        ["BUY", "HOLD"],
        [float("nan"), -1.0],
    ],
    ids=["uint8 ones", "uint8 kernel ids", "booleans", "zero", "quantities", "twos", "text", "nan"],
)
def test_any_other_side_array_raises_rather_than_being_read_as_buys(side):
    # An unsigned [1, 1] is two buys under one convention and two sells under the
    # kernel's ids, so it is refused.
    with pytest.raises(CostError, match="side"):
        intraday_legs(side=side)


@pytest.mark.parametrize(
    "date",
    [
        [dt.date(2026, 11, 12), dt.date(2026, 11, 12)],
        np.array(["2026-11-12", "2026-11-12"], dtype="datetime64[D]"),
        np.array(["2026-11-12T09:15", "2026-11-12T15:20"], dtype="datetime64[s]"),
        pd.DatetimeIndex(["2026-11-12 09:15", "2026-11-12 15:20"]),
        pd.DatetimeIndex(["2026-11-12 09:15", "2026-11-12 15:20"], tz=IST),
        # 20:00 UTC on 11 November is 01:30 IST on the 12th.
        pd.Series(pd.to_datetime(["2026-11-11 20:00", "2026-11-12 03:45"], utc=True)),
        ["2026-11-12", "2026-11-12"],
        pd.Series(["2026-11-12", "2026-11-12"]),
        [pd.Timestamp("2026-11-12 10:00"), "2026-11-12"],
    ],
    ids=[
        "dates",
        "datetime64[D]",
        "datetime64[s]",
        "naive index",
        "IST index",
        "UTC series",
        "ISO text",
        "series of ISO text",
        "mixed",
    ],
)
def test_dates_may_be_any_array_of_dates_and_are_read_as_ist(date):
    assert_intraday_legs(intraday_legs(date=date))


@pytest.mark.parametrize(
    "date",
    [["01-04-2026"], ["2026-04-01", "01-04-2026"], pd.Series(["04/01/2026"]), ["1 April 2026"]],
    ids=["day first", "one of two day first", "slashes", "words"],
)
def test_a_date_string_in_an_array_must_be_iso_as_for_a_single_date(date):
    # "01-04-2026" must not be read as 4 January (old futures STT of Rs 2,000 on
    # Rs 1 crore) or as 1 April (Rs 5,000): it is refused, as the scalar call refuses it.
    with pytest.raises(CostError, match="ISO date"):
        costs.trade_costs("SELL", 10_000_000, 1, date, "NFO_FUT")


@pytest.mark.parametrize(
    "segment",
    [["NSE_EQ_INTRADAY", "NSE_EQ_INTRADAY"], np.array([1, 1]), ["nse_eq_intraday"] * 2],
    ids=["codes", "ids", "lower case"],
)
def test_segments_may_be_codes_or_ids(segment):
    assert_intraday_legs(intraday_legs(segment=segment))


def test_each_trade_is_charged_at_the_rates_of_its_own_date():
    # Futures STT on Rs 1 crore sold: 0.02 percent before 2026-04-01, 0.05 percent after.
    with pytest.warns(costs.UnverifiedScheduleWarning):
        frame = costs.trade_costs("SELL", 10_000_000, 1, ["2026-03-31", "2026-04-01"], "NFO_FUT")
    assert list(frame["stt"]) == pytest.approx([2000.0, 5000.0])


def test_each_row_says_which_version_priced_it_and_whether_it_is_verified():
    # The older version, the last verified day, and a day after it.
    dates = ["2026-03-31", SNAPSHOT, "2026-11-13"]
    with pytest.warns(costs.UnverifiedScheduleWarning):
        frame = costs.trade_costs("SELL", 100, 1, dates, "NFO_FUT")
    assert list(frame["schedule"]) == ["IN-2024-10-01", "IN-2026-04-01", "IN-2026-04-01"]
    assert list(frame["verified"]) == [False, True, False]


def test_a_batch_warns_once_per_version_and_reason():
    dates = ["2025-06-02", "2025-06-03", "2026-11-13", "2026-11-14", SNAPSHOT]
    with pytest.warns(costs.UnverifiedScheduleWarning) as caught:
        costs.trade_costs("SELL", 100, 1, dates, "NFO_FUT")
    assert len(caught) == 2


def test_a_backfilled_row_is_flagged():
    model = costs.CostModel(backfill=True)
    with pytest.warns(costs.UnverifiedScheduleWarning, match="backfill"):
        arrays = model.trades("SELL", 100, 1, ["2019-06-03", SNAPSHOT], "NFO_FUT", as_frame=False)
    assert (list(arrays["backfilled"]), list(arrays["verified"])) == ([True, False], [False, True])


def test_an_empty_batch_returns_an_empty_frame():
    frame = costs.trade_costs([], [], [], [], [])
    assert (len(frame), list(frame.columns)) == (0, COLUMNS)


# (what is wrong, the argument to override, text the message must contain)
BAD_BATCHES = [
    ("lengths differ", {"price": [1000.0, 1100.0, 1200.0]}, "different lengths"),
    ("a negative price", {"price": [1000.0, -1.0]}, "row 1: price"),
    ("a missing price", {"price": [float("nan"), 1100.0]}, "row 0: price"),
    ("a negative quantity", {"quantity": [400, -400]}, "row 1: quantity"),
    ("a value that overflows", {"price": [1e200, 1.0], "quantity": 1e200}, "row 0: traded value"),
    ("a quantity too large for a float", {"quantity": [400, 10**400]}, "quantity must be numeric"),
    ("text for a price", {"price": ["a", "b"]}, "price must be numeric"),
    # Values a float cast would take silently (None as NaN, True as 1, "1100" as 1100)
    # are refused, as a single trade refuses them.
    ("None for a price", {"price": [None, 1100.0]}, "price must be numeric, got None"),
    ("numeric text for a price", {"price": ["1000", "1100"]}, "price must be numeric, got '1000'"),
    (
        "a boolean in a quantity list",
        {"quantity": [400, True]},
        "quantity must be numeric, got True",
    ),
    ("a boolean quantity array", {"quantity": np.array([True, True])}, "quantity must be numeric"),
    ("a boolean quantity", {"quantity": True}, "quantity must be numeric, got True"),
    ("an unknown segment code", {"segment": ["NSE_EQ_INTRADAY", "NSE"]}, "unknown cost segment"),
    ("a segment id past the table", {"segment": [1, 99]}, "segment ids must be between"),
    ("a missing date", {"date": [pd.Timestamp(SNAPSHOT), pd.NaT]}, "missing"),
    ("a missing date in an index", {"date": pd.DatetimeIndex([SNAPSHOT, None])}, "missing"),
    ("a date before the first version", {"date": ["2026-11-12", "2019-01-01"]}, "row 1: no cost"),
    (
        "a date outside the supported range",
        {"date": np.array([2**31 - 1, 0], dtype="datetime64[D]")},
        "outside the supported range",
    ),
    ("a two-dimensional price", {"price": [[1000.0, 1100.0]]}, "one-dimensional"),
]


@pytest.mark.parametrize("problem, override, text", BAD_BATCHES, ids=[b[0] for b in BAD_BATCHES])
def test_a_bad_batch_raises_a_cost_error_that_names_the_problem(problem, override, text):
    with pytest.raises(CostError, match=text):
        intraday_legs(**override)


def small_option_sales(count, rounding="contract_note"):
    """`count` sales of one lot of 75 at a premium of Rs 4, each owing STT of Rs 0.45."""
    return costs.trade_costs("SELL", 4.0, 75, [SNAPSHOT] * count, "NFO_OPT", rounding=rounding)


def test_contract_note_needs_only_the_frame_trades_returned():
    # Ten sales owe 10 x 0.45 = Rs 4.50 of STT, charged as Rs 5. Rounded trade by
    # trade (as the calculator does) each would owe nothing.
    note = costs.contract_note(small_option_sales(10))
    assert (len(note), note["note"][0], note["fills"][0], note["stt"][0]) == (1, "NFO", 10, 5.0)


def test_contract_note_refuses_trades_costed_in_calculator_mode():
    # The same ten sales in calculator mode carry STT of 0 each: summing them would
    # charge nothing instead of Rs 5.
    with pytest.raises(CostError, match="needs unrounded accruals.*rounding='calculator'"):
        costs.contract_note(small_option_sales(10, rounding="calculator"))


def test_contract_note_refuses_a_frame_that_does_not_say_how_it_was_costed():
    trades = pd.DataFrame(small_option_sales(2).to_dict("list"))
    with pytest.raises(CostError, match="rounding=None"):
        costs.contract_note(trades)


def test_contract_note_charges_flat_brokerage_per_trade():
    assert costs.contract_note(small_option_sales(10))["brokerage"][0] == 200.0


def test_a_stamp_duty_of_exactly_one_rupee_fifty_is_charged_two_rupees():
    # NSE delivery buy of 100 shares at Rs 100: 0.015 percent of Rs 10,000 is Rs 1.50
    # exactly (the float product is 1.4999999999999998), and 50 paise rounds up.
    trades = costs.trade_costs("BUY", 100.0, 100, SNAPSHOT, "NSE_EQ_DELIVERY")
    assert costs.contract_note(trades)["stamp_duty"][0] == 2.0


@pytest.mark.parametrize(
    "value, stamp_duty", [(50_000, 8.0), (70_000, 11.0), (90_000, 14.0), (150_000, 23.0)]
)
def test_delivery_stamp_duty_ties_round_up(value, stamp_duty):
    # 0.015 percent of the value is 7.50, 10.50, 13.50 and 22.50.
    trades = costs.trade_costs("BUY", float(value), 1, SNAPSHOT, "NSE_EQ_DELIVERY")
    assert costs.contract_note(trades)["stamp_duty"][0] == stamp_duty


def test_contract_note_of_the_worked_example_matches_the_calculator_total():
    # Section 6, intraday: a day holding only the example's two trades is charged 200.62.
    trades = costs.trade_costs(["BUY", "SELL"], [1000, 1100], 400, SNAPSHOT, "NSE_EQ_INTRADAY")
    assert costs.contract_note(trades)["total"][0] == 200.62


def test_contract_note_groups_by_trade_date_and_exchange_segment():
    dates = ["2026-11-12", "2026-11-12", "2026-11-12", "2026-11-12", "2026-11-13"]
    segments = ["NSE_EQ_DELIVERY", "NSE_EQ_INTRADAY", "NFO_FUT", "MCX_FUT", "NSE_EQ_DELIVERY"]
    with pytest.warns(costs.UnverifiedScheduleWarning):
        note = costs.contract_note(costs.trade_costs("BUY", 500.0, 100, dates, segments))
    days = note["date"].dt.strftime("%Y-%m-%d")
    assert list(zip(days, note["note"], note["fills"], strict=True)) == [
        ("2026-11-12", "NSE_EQ", 2),
        ("2026-11-12", "NFO", 1),
        ("2026-11-12", "MCX", 1),
        ("2026-11-13", "NSE_EQ", 1),
    ]


def test_contract_note_adjustment_is_charged_minus_accrued():
    # STT accrued 3 x 0.45 = 1.35 is charged as Rs 1: the tax line moves by -0.35.
    trades = small_option_sales(3)
    note = costs.contract_note(trades).iloc[0]
    assert note["stt"] - trades["stt"].sum() == pytest.approx(-0.35)
    assert note["accrued_total"] == pytest.approx(trades["total"].sum())
    assert note["adjustment"] == pytest.approx(note["total"] - trades["total"].sum())


def test_contract_note_reports_the_rule_it_assumed():
    attrs = costs.contract_note(small_option_sales(2)).attrs
    assert attrs == {"rule": costs.CONTRACT_NOTE_RULE, "rule_verified": False}
    assert attrs["rule"].startswith("assumed, not verified")


def test_contract_note_of_no_trades_is_empty():
    assert len(costs.contract_note(costs.trade_costs([], [], [], [], []))) == 0


@pytest.mark.parametrize(
    "not_trades",
    [pd.DataFrame({"total": [1.0]}), {"stt": [1.0]}, None],
    ids=["a frame without the columns", "a dict", "nothing"],
)
def test_contract_note_takes_only_a_trades_frame(not_trades):
    with pytest.raises(CostError, match="takes the DataFrame returned by trades"):
        costs.contract_note(not_trades)


def test_contract_note_rejects_a_negative_amount():
    trades = small_option_sales(2)
    trades.loc[1, "stt"] = -1.0
    with pytest.raises(CostError, match="row 1: stt"):
        costs.contract_note(trades)
