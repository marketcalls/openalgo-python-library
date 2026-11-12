//! Reference note sections 5 and 6: the calculation rules and the worked example,
//! reproduced to the paisa in calculator mode.

use crate::model::Side;

use super::super::Segment;
use super::{accrual_model, assert_close, assert_paise, calculator_model, snapshot_day, SameDay};

/// One column of a worked example: a round trip and the calculator's figures.
struct Example {
    segment: Segment,
    buy: f64,
    sell: f64,
    quantity: f64,
    brokerage: f64,
    stt: f64,
    exchange_txn: f64,
    gst: f64,
    sebi_fee: f64,
    stamp_duty: f64,
    total: f64,
}

#[track_caller]
fn assert_example(example: &Example) {
    let cost = calculator_model()
        .round_trip_on(
            example.segment,
            example.buy,
            example.sell,
            example.quantity,
            snapshot_day(),
        )
        .unwrap();
    assert_paise(cost.brokerage, example.brokerage);
    assert_paise(cost.stt, example.stt);
    assert_paise(cost.exchange_txn, example.exchange_txn);
    assert_paise(cost.gst, example.gst);
    assert_paise(cost.sebi_fee, example.sebi_fee);
    assert_paise(cost.stamp_duty, example.stamp_duty);
    assert_paise(cost.total(), example.total);
}

/// Section 6, the screenshot-verified example with NSE selected.
const NSE_EXAMPLES: [Example; 4] = [
    Example {
        segment: Segment::NseEqIntraday,
        buy: 1_000.0,
        sell: 1_100.0,
        quantity: 400.0,
        brokerage: 40.0,
        stt: 110.0,
        exchange_txn: 25.79,
        gst: 11.99,
        sebi_fee: 0.84,
        stamp_duty: 12.0,
        total: 200.62,
    },
    Example {
        segment: Segment::NseEqDelivery,
        buy: 1_000.0,
        sell: 1_100.0,
        quantity: 400.0,
        brokerage: 0.0,
        stt: 840.0,
        exchange_txn: 25.79,
        gst: 4.79,
        sebi_fee: 0.84,
        stamp_duty: 60.0,
        total: 931.42,
    },
    Example {
        segment: Segment::NfoFut,
        buy: 1_000.0,
        sell: 1_100.0,
        quantity: 400.0,
        brokerage: 40.0,
        stt: 220.0,
        exchange_txn: 15.37,
        gst: 10.12,
        sebi_fee: 0.84,
        stamp_duty: 8.0,
        total: 294.33,
    },
    Example {
        segment: Segment::NfoOpt,
        buy: 100.0,
        sell: 110.0,
        quantity: 400.0,
        brokerage: 40.0,
        stt: 66.0,
        exchange_txn: 29.85,
        gst: 12.59,
        sebi_fee: 0.08,
        stamp_duty: 1.0,
        total: 149.52,
    },
];

/// The same four trades with BSE selected. Figures from the reference helper
/// `indian_trading_costs.py` called with `exchange="BSE"`.
const BSE_EXAMPLES: [Example; 4] = [
    Example {
        segment: Segment::BseEqIntraday,
        buy: 1_000.0,
        sell: 1_100.0,
        quantity: 400.0,
        brokerage: 40.0,
        stt: 110.0,
        exchange_txn: 31.50,
        gst: 13.02,
        sebi_fee: 0.84,
        stamp_duty: 12.0,
        total: 207.36,
    },
    Example {
        segment: Segment::BseEqDelivery,
        buy: 1_000.0,
        sell: 1_100.0,
        quantity: 400.0,
        brokerage: 0.0,
        stt: 840.0,
        exchange_txn: 31.50,
        gst: 5.82,
        sebi_fee: 0.84,
        stamp_duty: 60.0,
        total: 938.16,
    },
    Example {
        segment: Segment::BfoFut,
        buy: 1_000.0,
        sell: 1_100.0,
        quantity: 400.0,
        brokerage: 40.0,
        stt: 220.0,
        exchange_txn: 0.0,
        gst: 7.35,
        sebi_fee: 0.84,
        stamp_duty: 8.0,
        total: 276.19,
    },
    Example {
        segment: Segment::BfoOpt,
        buy: 100.0,
        sell: 110.0,
        quantity: 400.0,
        brokerage: 40.0,
        stt: 66.0,
        exchange_txn: 27.30,
        gst: 12.13,
        sebi_fee: 0.08,
        stamp_duty: 1.0,
        total: 146.51,
    },
];

#[test]
fn worked_example_intraday_equity_totals_200_62() {
    assert_example(&NSE_EXAMPLES[0]);
}

#[test]
fn worked_example_delivery_equity_totals_931_42() {
    assert_example(&NSE_EXAMPLES[1]);
}

#[test]
fn worked_example_futures_totals_294_33() {
    assert_example(&NSE_EXAMPLES[2]);
}

#[test]
fn worked_example_options_totals_149_52() {
    assert_example(&NSE_EXAMPLES[3]);
}

#[test]
fn worked_example_on_bse_and_bfo() {
    for example in &BSE_EXAMPLES {
        assert_example(example);
    }
}

#[test]
fn worked_example_net_pnl_and_points_to_breakeven() {
    // (example, net P&L, points to breakeven) from the last two rows of section 6.
    let rows = [
        (0, 39_799.38, 0.50),
        (1, 39_068.58, 2.33),
        (2, 39_705.67, 0.74),
        (3, 3_850.48, 0.37),
    ];
    for (index, net_pnl, breakeven) in rows {
        let example = &NSE_EXAMPLES[index];
        let gross = (example.sell - example.buy) * example.quantity;
        let per_unit_in_paise = (example.total / example.quantity * 100.0).round();
        assert_paise(gross - example.total, net_pnl);
        assert_paise(per_unit_in_paise / 100.0, breakeven);
    }
}

#[test]
fn percent_brokerage_is_capped_per_executed_order() {
    // Section 5: each leg is min(value x 0.0003, 20). A buy worth Rs 10,000 pays
    // Rs 3 and a sell worth Rs 1,00,000 pays Rs 20.
    let cost = calculator_model()
        .round_trip_on(
            Segment::NseEqIntraday,
            100.0,
            1_000.0,
            100.0,
            snapshot_day(),
        )
        .unwrap();
    assert_paise(cost.brokerage, 23.0);
}

#[test]
fn calculator_rounds_tax_to_the_rupee_and_the_rest_to_the_paisa() {
    // Section 5. Intraday, buy 123.45 and sell 124.55, 37 shares, by hand:
    //   buy value 4567.65, sell value 4608.35, turnover 9176.00
    //   brokerage 1.37 + 1.38 = 2.75
    //   STT 0.025 percent of the sell value = 1.152, rounded to Rs 1
    //   exchange 0.00297 percent = 0.27, IPFT 0.0001 percent = 0.01, sum 0.28
    //   SEBI 0.009, rounded 0.01
    //   GST 0.18 x (2.75 + 0.28 + 0.01) = 0.5472, rounded 0.55
    //   stamp duty 0.003 percent of the buy value = 0.137, rounded to Rs 0
    //   total 4.59
    let cost = calculator_model()
        .round_trip_on(Segment::NseEqIntraday, 123.45, 124.55, 37.0, snapshot_day())
        .unwrap();
    assert_paise(cost.brokerage, 2.75);
    assert_eq!(cost.stt, 1.0);
    assert_paise(cost.exchange_txn, 0.28);
    assert_paise(cost.sebi_fee, 0.01);
    assert_paise(cost.gst, 0.55);
    assert_eq!(cost.stamp_duty, 0.0);
    assert_paise(cost.total(), 4.59);
}

#[test]
fn calculator_buy_leg_alone() {
    // The intraday example's buy with no sell price, by hand: turnover 4,00,000;
    // brokerage 20; no STT; exchange 11.88 + IPFT 0.40 = 12.28; SEBI 0.40; GST
    // 0.18 x 32.68 = 5.8824, rounded 5.88; stamp duty 12; total 50.56.
    let cost = calculator_model()
        .fill_cost(
            Segment::NseEqIntraday,
            Side::Buy,
            1_000.0,
            400.0,
            snapshot_day(),
        )
        .unwrap();
    assert_paise(cost.brokerage, 20.0);
    assert_eq!(cost.stt, 0.0);
    assert_paise(cost.exchange_txn, 12.28);
    assert_paise(cost.sebi_fee, 0.40);
    assert_paise(cost.gst, 5.88);
    assert_paise(cost.stamp_duty, 12.0);
    assert_paise(cost.total(), 50.56);
}

#[test]
fn calculator_sell_leg_alone() {
    // The intraday example's sell with no buy price, by hand: turnover 4,40,000;
    // brokerage 20; STT 110; exchange 13.07 + IPFT 0.44 = 13.51; SEBI 0.44; GST
    // 0.18 x 33.95 = 6.111, rounded 6.11; no stamp duty; total 150.06.
    let cost = calculator_model()
        .fill_cost(
            Segment::NseEqIntraday,
            Side::Sell,
            1_100.0,
            400.0,
            snapshot_day(),
        )
        .unwrap();
    assert_paise(cost.brokerage, 20.0);
    assert_paise(cost.stt, 110.0);
    assert_paise(cost.exchange_txn, 13.51);
    assert_paise(cost.sebi_fee, 0.44);
    assert_paise(cost.gst, 6.11);
    assert_eq!(cost.stamp_duty, 0.0);
    assert_paise(cost.total(), 150.06);
}

#[test]
fn calculator_rounding_drops_the_tax_on_a_small_option_sale() {
    // The bias contract-note rounding removes: one lot of 75 sold at a premium of
    // Rs 4 owes STT of 0.15 percent of Rs 300 = Rs 0.45. The calculator rounds that
    // to nothing; the accrual keeps it.
    let day = snapshot_day();
    let rounded = calculator_model().fill_cost(Segment::NfoOpt, Side::Sell, 4.0, 75.0, day);
    let accrued = accrual_model().fill_cost(Segment::NfoOpt, Side::Sell, 4.0, 75.0, day);
    assert_eq!(rounded.unwrap().stt, 0.0);
    assert_close(accrued.unwrap().stt, 0.45);
}

#[test]
fn the_reported_total_in_calculator_mode_is_a_whole_number_of_paise() {
    let model = calculator_model();
    for example in &NSE_EXAMPLES {
        let cost = model
            .round_trip_on(
                example.segment,
                example.buy,
                example.sell,
                example.quantity,
                snapshot_day(),
            )
            .unwrap();
        assert_eq!(
            model.reported_total(&cost),
            example.total,
            "{:?}",
            example.segment
        );
    }
}

#[test]
fn the_reported_total_in_contract_note_mode_is_the_plain_sum() {
    // Section 1 rates on a delivery buy worth Rs 4,00,000: STT 400, exchange 12.28,
    // SEBI 0.40, stamp duty 60, GST 0.18 x 12.68 = 2.2824.
    let model = accrual_model();
    let cost = model
        .fill_cost(
            Segment::NseEqDelivery,
            Side::Buy,
            1_000.0,
            400.0,
            snapshot_day(),
        )
        .unwrap();
    assert_close(model.reported_total(&cost), 474.9624);
}

#[test]
fn a_fill_given_as_a_traded_value_costs_the_same_as_price_times_quantity() {
    // Rs 4,00,000 as 1000 x 400, or as the value itself: STT 400, stamp duty 60.
    let model = accrual_model();
    let by_value =
        model.fill_cost_value(Segment::NseEqDelivery, Side::Buy, 400_000.0, snapshot_day());
    let by_units = model.fill_cost(
        Segment::NseEqDelivery,
        Side::Buy,
        1_000.0,
        400.0,
        snapshot_day(),
    );
    assert_eq!(by_value, by_units);
    assert_close(by_value.unwrap().stt, 400.0);
}
