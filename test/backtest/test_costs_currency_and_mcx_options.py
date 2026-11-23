"""NSE currency futures and options and MCX options in `bt.costs` (items C-2 and C-3 of
2.4.1): the maintainer's calculator screenshots of 2026-11-23, in calculator rounding
under brokerage="zerodha", and what a contract note charges. Every figure is worked by
hand in the test's comment; the kernel's own tests carry the per-crore rows.
"""

import pytest

from openalgo.backtest import costs
from openalgo.backtest.errors import CostError

DAY = "2026-11-23"


def calculator(buy, sell, quantity, segment):
    return costs.round_trip_cost(
        buy, sell, quantity, DAY, segment=segment, brokerage="zerodha", rounding="calculator"
    )


def test_a_usdinr_future_reproduces_the_calculator():
    # One lot of USD 1,000 (1,000 units), buy 88.3300, sell 88.3525: turnover 1,76,682.50;
    # brokerage 20 + 20; exchange 0.6184 + IPFT 0.0883 shown as 0.71; SEBI 0.18; GST 18
    # percent of 40 + 0.62 + 0.18 = 7.34 (the calculator leaves the IPFT out of GST and of
    # the total); stamp 0.0883 rounded to Rs 0; total 40 + 0.62 + 0.18 + 7.34 = 48.14.
    cost = calculator(88.33, 88.3525, 1_000, "CDS_FUT")
    assert cost.as_dict() == {
        "brokerage": 40.0,
        "stt": 0.0,
        "exchange_txn": 0.71,
        "sebi_fee": 0.18,
        "stamp_duty": 0.0,
        "gst": 7.34,
        "total": 48.14,
    }


def test_a_usdinr_option_reproduces_the_calculator():
    # The same prices as premiums (strike 88.50): exchange 54.948 + IPFT 3.534 = 58.48;
    # GST 18 percent of 40 + 58.48 + 0.18 = 17.76; total 116.42.
    cost = calculator(88.33, 88.3525, 1_000, "CDS_OPT")
    assert cost.as_dict() == {
        "brokerage": 40.0,
        "stt": 0.0,
        "exchange_txn": 58.48,
        "sebi_fee": 0.18,
        "stamp_duty": 0.0,
        "gst": 17.76,
        "total": 116.42,
    }


def test_a_copper_option_reproduces_the_calculator():
    # One lot of 2,500 units, premiums 100 and 110 (strike 900): turnover 5,25,000;
    # brokerage 40; exchange 0.0418 percent 219.45; CTT 0.05 percent of 2,75,000 = 137.50;
    # SEBI 0.53; GST 18 percent of 40 + 219.45 + 0.53 = 46.80; stamp 0.003 percent of
    # 2,50,000 = 7.50, rounded to Rs 8; total 452.28.
    cost = calculator(100, 110, 2_500, "MCX_OPT")
    assert cost.as_dict() == {
        "brokerage": 40.0,
        "stt": 137.5,
        "exchange_txn": 219.45,
        "sebi_fee": 0.53,
        "stamp_duty": 8.0,
        "gst": 46.8,
        "total": 452.28,
    }


def test_a_usdinr_round_trip_on_one_contract_note_charges_what_the_exchange_bills():
    # Accrued per fill, then rounded once on the day's CDS note: brokerage 40.00; exchange
    # with IPFT 0.0004 percent of 1,76,682.50 = 0.70673, 0.71; SEBI 0.18; stamp 0.08833,
    # Rs 0; GST 18 percent of 40 + 0.70673 + 0.17668 = 7.35901, 7.36: 48.25.
    trades = costs.trade_costs(["BUY", "SELL"], [88.33, 88.3525], 1_000, DAY, "CDS_FUT")
    note = costs.contract_note(trades)
    assert list(note["note"]) == ["CDS"]
    row = note.iloc[0]
    assert (row["exchange_txn"], row["gst"], row["stamp_duty"], row["total"]) == (
        0.71,
        7.36,
        0.0,
        48.25,
    )


@pytest.mark.parametrize(
    ("exchange", "product", "instrument", "underlying", "segment"),
    [
        ("CDS", "NRML", "FUT", None, "CDS_FUT"),
        ("CDS", "MIS", "OPT", None, "CDS_OPT"),
        ("MCX", "NRML", "OPT", "COPPER", "MCX_OPT"),
    ],
)
def test_the_new_segments_are_found_from_the_exchange_and_product(
    exchange, product, instrument, underlying, segment
):
    assert costs.segment_for(exchange, product, instrument, underlying=underlying) == segment


@pytest.mark.parametrize("code", ["BCD_FUT", "BCD_OPT"])
def test_bse_currency_derivatives_are_not_costed_and_named(code):
    with pytest.raises(CostError, match="BSE currency derivatives"):
        costs.trade_cost("BUY", 88.33, 1_000, DAY, segment=code)


def test_the_bse_currency_exchange_is_not_costed_and_named():
    with pytest.raises(CostError, match=r"exchange BCD \(BSE currency derivatives\)"):
        costs.segment_for("BCD", "NRML", "FUT")


def test_an_mcx_option_on_an_agri_commodity_is_not_costed_and_named():
    with pytest.raises(CostError, match="MCX options on the agri commodity CARDAMOM"):
        costs.segment_for("MCX", "NRML", "OPT", underlying="CARDAMOM")


def test_the_exercise_of_an_mcx_option_is_refused():
    with pytest.raises(CostError, match="the exercise of an MCX option"):
        costs.exercise_stt(5.0, 2_500, DAY, segment="MCX_OPT")
