//! Reference note section 1, "Equity and F&O (NSE, BSE, NFO, BFO)": one test per
//! row of the rate table and of the BSE scrip-group table. Amounts are rupees on a
//! fill of Rs 1 crore unless a row says otherwise.

use crate::model::{Exchange, Side};

use super::super::Segment;
use super::{accrued, assert_close, current_rates, CRORE};

const BOTH_SIDES: [Side; 2] = [Side::Buy, Side::Sell];

#[test]
fn brokerage_row_delivery_is_free() {
    for segment in [
        Segment::NseEqDelivery,
        Segment::BseEqDelivery,
        Segment::BseEqDeliveryX,
    ] {
        for side in BOTH_SIDES {
            assert_eq!(
                accrued(segment, side, 5_000_000.0).brokerage,
                0.0,
                "{segment:?}"
            );
        }
    }
}

#[test]
fn brokerage_row_intraday_and_futures_pay_the_lower_of_3_bps_and_20_rupees() {
    // (order value, brokerage): 0.03 percent below Rs 66,667, Rs 20 from there on.
    let rows = [
        (10_000.0, 3.0),
        (50_000.0, 15.0),
        (66_700.0, 20.0),
        (500_000.0, 20.0),
    ];
    for segment in [
        Segment::NseEqIntraday,
        Segment::BseEqIntraday,
        Segment::NfoFut,
        Segment::BfoFut,
    ] {
        for side in BOTH_SIDES {
            for (value, brokerage) in rows {
                assert_close(accrued(segment, side, value).brokerage, brokerage);
            }
        }
    }
}

#[test]
fn brokerage_row_options_pay_a_flat_20_rupees_per_order() {
    for segment in [Segment::NfoOpt, Segment::BfoOpt] {
        for side in BOTH_SIDES {
            for premium_value in [100.0, 50_000.0, 5_000_000.0] {
                assert_eq!(accrued(segment, side, premium_value).brokerage, 20.0);
            }
        }
    }
}

#[test]
fn stt_row_rates_and_sides() {
    // (segment, STT on a buy, STT on a sell): delivery 0.1 percent both ways,
    // intraday 0.025 percent on sell, futures 0.05 percent on sell, options 0.15
    // percent of premium on sell.
    let rows = [
        (Segment::NseEqDelivery, 10_000.0, 10_000.0),
        (Segment::BseEqDelivery, 10_000.0, 10_000.0),
        (Segment::BseEqDeliverySs, 10_000.0, 10_000.0),
        (Segment::NseEqIntraday, 0.0, 2_500.0),
        (Segment::BseEqIntraday, 0.0, 2_500.0),
        (Segment::BseEqIntradayM, 0.0, 2_500.0),
        (Segment::NfoFut, 0.0, 5_000.0),
        (Segment::BfoFut, 0.0, 5_000.0),
        (Segment::NfoOpt, 0.0, 15_000.0),
        (Segment::BfoOpt, 0.0, 15_000.0),
    ];
    for (segment, on_buy, on_sell) in rows {
        assert_close(accrued(segment, Side::Buy, CRORE).stt, on_buy);
        assert_close(accrued(segment, Side::Sell, CRORE).stt, on_sell);
    }
}

#[test]
fn exchange_txn_nse_row_includes_the_ipft_levy() {
    // 0.00307 percent on equity, 0.00183 percent on futures, 0.03553 percent of
    // premium on options.
    let rows = [
        (Segment::NseEqDelivery, 307.0),
        (Segment::NseEqIntraday, 307.0),
        (Segment::NfoFut, 183.0),
        (Segment::NfoOpt, 3_553.0),
    ];
    for (segment, per_crore) in rows {
        for side in BOTH_SIDES {
            assert_close(accrued(segment, side, CRORE).exchange_txn, per_crore);
        }
    }
}

#[test]
fn exchange_txn_nse_is_split_into_transaction_charge_and_ipft() {
    // "Equity 0.00297% + 0.0001% IPFT, futures 0.00173% + 0.0001% IPFT, options
    // 0.03503% + 0.0005% IPFT."
    let rows = [
        (Segment::NseEqDelivery, 0.0000297, 0.000001),
        (Segment::NseEqIntraday, 0.0000297, 0.000001),
        (Segment::NfoFut, 0.0000173, 0.000001),
        (Segment::NfoOpt, 0.0003503, 0.000005),
    ];
    for (segment, exchange_txn, ipft) in rows {
        let rates = current_rates(segment);
        assert_eq!(
            (rates.exchange_txn_rate, rates.ipft_rate),
            (exchange_txn, ipft),
            "{segment:?}"
        );
    }
}

#[test]
fn exchange_txn_bse_row() {
    // 0.00375 percent on equity, nothing on futures, 0.0325 percent of premium on
    // options.
    let rows = [
        (Segment::BseEqDelivery, 375.0),
        (Segment::BseEqIntraday, 375.0),
        (Segment::BfoFut, 0.0),
        (Segment::BfoOpt, 3_250.0),
    ];
    for (segment, per_crore) in rows {
        for side in BOTH_SIDES {
            assert_close(accrued(segment, side, CRORE).exchange_txn, per_crore);
        }
    }
}

#[test]
fn only_nse_segments_carry_an_ipft_levy() {
    for segment in Segment::ALL {
        let on_nse = matches!(segment.exchange(), Exchange::Nse | Exchange::Nfo);
        assert_eq!(
            current_rates(segment).ipft_rate > 0.0,
            on_nse,
            "{segment:?}"
        );
    }
}

#[test]
fn bse_scrip_group_table() {
    // Rs 375, 275, 10,000 and 1,00,000 per crore, delivery and intraday alike.
    let rows = [
        (Segment::BseEqDelivery, Segment::BseEqIntraday, 375.0),
        (Segment::BseEqDeliveryM, Segment::BseEqIntradayM, 275.0),
        (Segment::BseEqDeliveryX, Segment::BseEqIntradayX, 10_000.0),
        (
            Segment::BseEqDeliverySs,
            Segment::BseEqIntradaySs,
            100_000.0,
        ),
    ];
    for (delivery, intraday, per_crore) in rows {
        for segment in [delivery, intraday] {
            for side in BOTH_SIDES {
                assert_close(accrued(segment, side, CRORE).exchange_txn, per_crore);
            }
        }
    }
}

#[test]
fn bse_scrip_group_changes_only_the_exchange_charge_and_its_gst() {
    let standard = accrued(Segment::BseEqDelivery, Side::Sell, CRORE);
    for segment in [
        Segment::BseEqDeliveryM,
        Segment::BseEqDeliveryX,
        Segment::BseEqDeliverySs,
    ] {
        let tier = accrued(segment, Side::Sell, CRORE);
        assert_eq!(
            (tier.brokerage, tier.stt, tier.sebi_fee, tier.stamp_duty),
            (
                standard.brokerage,
                standard.stt,
                standard.sebi_fee,
                standard.stamp_duty
            ),
            "{segment:?}"
        );
        assert_ne!(tier.exchange_txn, standard.exchange_txn);
    }
}

#[test]
fn sebi_fee_row_is_ten_rupees_per_crore() {
    let equity_and_fno = Segment::ALL
        .into_iter()
        .filter(|s| s.exchange() != Exchange::Mcx);
    for segment in equity_and_fno {
        for side in BOTH_SIDES {
            assert_close(accrued(segment, side, CRORE).sebi_fee, 10.0);
        }
    }
}

#[test]
fn stamp_duty_row_is_charged_on_buys_only() {
    // Delivery Rs 1500, intraday Rs 300, futures Rs 200, options Rs 300 per crore.
    let rows = [
        (Segment::NseEqDelivery, 1_500.0),
        (Segment::BseEqDelivery, 1_500.0),
        (Segment::BseEqDeliveryM, 1_500.0),
        (Segment::NseEqIntraday, 300.0),
        (Segment::BseEqIntraday, 300.0),
        (Segment::BseEqIntradayX, 300.0),
        (Segment::NfoFut, 200.0),
        (Segment::BfoFut, 200.0),
        (Segment::NfoOpt, 300.0),
        (Segment::BfoOpt, 300.0),
    ];
    for (segment, per_crore) in rows {
        assert_close(accrued(segment, Side::Buy, CRORE).stamp_duty, per_crore);
        assert_eq!(
            accrued(segment, Side::Sell, CRORE).stamp_duty,
            0.0,
            "{segment:?}"
        );
    }
}

#[test]
fn gst_row_is_18_percent_of_brokerage_exchange_charge_and_sebi_fee() {
    // On a Rs 1 crore sell, worked from the rows above. STT and stamp duty are not
    // in the base.
    //   NSE delivery: 0.18 x (0 + 307 + 10) = 57.06
    //   NSE intraday: 0.18 x (20 + 307 + 10) = 60.66
    //   NFO futures:  0.18 x (20 + 183 + 10) = 38.34
    //   NFO options:  0.18 x (20 + 3553 + 10) = 644.94
    //   BSE delivery: 0.18 x (0 + 375 + 10) = 69.30
    //   BFO futures:  0.18 x (20 + 0 + 10) = 5.40
    //   BFO options:  0.18 x (20 + 3250 + 10) = 590.40
    let rows = [
        (Segment::NseEqDelivery, 57.06),
        (Segment::NseEqIntraday, 60.66),
        (Segment::NfoFut, 38.34),
        (Segment::NfoOpt, 644.94),
        (Segment::BseEqDelivery, 69.30),
        (Segment::BfoFut, 5.40),
        (Segment::BfoOpt, 590.40),
    ];
    for (segment, gst) in rows {
        assert_close(accrued(segment, Side::Sell, CRORE).gst, gst);
    }
}

#[test]
fn options_are_charged_on_the_premium_value_passed_in() {
    // Section 5: "Options STT, exchange charge and stamp duty are on premium value,
    // not notional." A premium value of Rs 30,000: STT 0.15 percent is Rs 45, the
    // exchange charge 0.03553 percent is Rs 10.659, stamp duty 0.003 percent is
    // Rs 0.90.
    let sell = accrued(Segment::NfoOpt, Side::Sell, 30_000.0);
    assert_close(sell.stt, 45.0);
    assert_close(sell.exchange_txn, 10.659);
    assert_close(
        accrued(Segment::NfoOpt, Side::Buy, 30_000.0).stamp_duty,
        0.9,
    );
}

#[test]
fn bse_and_bfo_rates_are_not_the_nse_and_nfo_rates() {
    // Each exchange has its own transaction charge. Reusing the NSE and NFO rates
    // for BSE and BFO would overcharge BFO futures and undercharge BSE equity.
    let pairs = [
        (Segment::NseEqDelivery, Segment::BseEqDelivery),
        (Segment::NseEqIntraday, Segment::BseEqIntraday),
        (Segment::NfoFut, Segment::BfoFut),
        (Segment::NfoOpt, Segment::BfoOpt),
    ];
    for (nse, bse) in pairs {
        for side in BOTH_SIDES {
            let (n, b) = (accrued(nse, side, CRORE), accrued(bse, side, CRORE));
            assert_ne!(n.exchange_txn, b.exchange_txn, "{nse:?} and {bse:?}");
            assert_ne!(n.total(), b.total(), "{nse:?} and {bse:?}");
        }
    }
}

#[test]
fn statutory_lines_are_the_same_on_nse_and_bse() {
    let pairs = [
        (Segment::NseEqDelivery, Segment::BseEqDelivery),
        (Segment::NseEqIntraday, Segment::BseEqIntraday),
        (Segment::NfoFut, Segment::BfoFut),
        (Segment::NfoOpt, Segment::BfoOpt),
    ];
    for (nse, bse) in pairs {
        for side in BOTH_SIDES {
            let (n, b) = (accrued(nse, side, CRORE), accrued(bse, side, CRORE));
            assert_eq!(
                (n.stt, n.stamp_duty, n.sebi_fee),
                (b.stt, b.stamp_duty, b.sebi_fee)
            );
        }
    }
}

#[test]
fn bse_options_other_than_sensex_and_bankex_pay_500_rupees_per_crore() {
    // BSE notice of 2024-09-27 (Angel One, 2024-09-30): Rs 3,250 per crore of premium
    // for Sensex and Bankex options, Rs 500 for Sensex 50 and stock options.
    for side in BOTH_SIDES {
        assert_close(accrued(Segment::BfoOpt, side, CRORE).exchange_txn, 3_250.0);
        assert_close(
            accrued(Segment::BfoOptOther, side, CRORE).exchange_txn,
            500.0,
        );
    }
}

#[test]
fn both_bse_options_segments_share_every_other_charge() {
    // On a Rs 1 crore sell: STT 0.15 percent, SEBI Rs 10, brokerage Rs 20; GST for
    // the lower tier is 0.18 x (20 + 500 + 10) = 95.40.
    let other = accrued(Segment::BfoOptOther, Side::Sell, CRORE);
    let index = accrued(Segment::BfoOpt, Side::Sell, CRORE);
    assert_eq!(
        (other.stt, other.sebi_fee, other.brokerage),
        (index.stt, index.sebi_fee, index.brokerage)
    );
    assert_close(other.gst, 95.40);
}
