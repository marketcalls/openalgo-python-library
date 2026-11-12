//! Charges that are not part of a fill: the DP charge on a delivery sell (reference
//! note section 4) and STT on exercised options (section 1, STT row).

use crate::model::days_from_civil;

use super::super::{Brokerage, CostError, DpFeeSource, Segment};
use super::{
    accrual_model, assert_close, assert_paise, calculator_model, model_with, snapshot_day,
};

#[test]
fn dp_charge_is_15_34_per_scrip_per_sell_day() {
    // "Rs 15.34 per scrip per day on sell (Rs 3.5 CDSL + Rs 9.5 Zerodha + Rs 2.34
    // GST)".
    let charge = accrual_model().dp_charge(snapshot_day()).unwrap();
    assert_eq!((charge.fee, charge.gst), (13.0, 2.34));
    assert_paise(charge.total(), 15.34);
}

#[test]
fn dp_charge_is_the_same_in_both_rounding_modes() {
    let day = snapshot_day();
    assert_eq!(
        accrual_model().dp_charge(day),
        calculator_model().dp_charge(day)
    );
}

#[test]
fn dp_charge_names_the_source_of_its_fee() {
    let day = snapshot_day();
    let rows = [
        (Brokerage::zerodha(), DpFeeSource::ZerodhaPreset),
        (Brokerage::flat(10.0).unwrap(), DpFeeSource::ZerodhaStandIn),
        (
            Brokerage::flat(10.0).unwrap().with_dp_fee(20.0).unwrap(),
            DpFeeSource::Supplied,
        ),
    ];
    for (preset, source) in rows {
        assert_eq!(
            model_with(preset).dp_charge(day).unwrap().fee_source,
            source
        );
    }
}

#[test]
fn dp_charge_follows_the_fee_of_the_preset() {
    // Rs 20 plus 18 percent GST is Rs 23.60.
    let model = model_with(Brokerage::zerodha().with_dp_fee(20.0).unwrap());
    assert_paise(model.dp_charge(snapshot_day()).unwrap().total(), 23.60);
}

#[test]
fn dp_charge_can_be_set_to_zero() {
    let model = model_with(Brokerage::zero_brokerage().with_dp_fee(0.0).unwrap());
    assert_eq!(model.dp_charge(snapshot_day()).unwrap().total(), 0.0);
}

#[test]
fn exercise_stt_is_15_bps_of_intrinsic_value() {
    // "0.15% of intrinsic value on options bought and exercised". Intrinsic Rs 50 on
    // 75 units is Rs 3,750, so STT is Rs 5.625.
    for segment in [Segment::NfoOpt, Segment::BfoOpt, Segment::BfoOptOther] {
        let stt = accrual_model().exercise_stt(segment, 50.0, 75.0, snapshot_day());
        assert_close(stt.unwrap(), 5.625);
    }
}

#[test]
fn exercise_stt_was_12_5_bps_before_2026_04_01() {
    // Intrinsic Rs 100 on 1,000 units is Rs 1,00,000: Rs 125 before, Rs 150 after.
    let model = accrual_model();
    let rows = [
        (days_from_civil(2026, 3, 31), 125.0),
        (days_from_civil(2026, 4, 1), 150.0),
    ];
    for (day, stt) in rows {
        assert_close(
            model
                .exercise_stt(Segment::NfoOpt, 100.0, 1_000.0, day)
                .unwrap(),
            stt,
        );
    }
}

#[test]
fn exercise_stt_is_rounded_to_the_rupee_in_calculator_mode() {
    // Zerodha's STT article: one lot of 50, strike 17,300, settlement 17,350, so
    // intrinsic value Rs 2,500 and STT Rs 3.75, "which rounds to Rs 4".
    let day = snapshot_day();
    assert_close(
        accrual_model()
            .exercise_stt(Segment::NfoOpt, 50.0, 50.0, day)
            .unwrap(),
        3.75,
    );
    assert_eq!(
        calculator_model().exercise_stt(Segment::NfoOpt, 50.0, 50.0, day),
        Ok(4.0)
    );
}

#[test]
fn an_option_with_no_intrinsic_value_pays_no_exercise_stt() {
    assert_eq!(
        accrual_model().exercise_stt(Segment::NfoOpt, 0.0, 50.0, snapshot_day()),
        Ok(0.0)
    );
}

#[test]
fn exercise_stt_is_for_options_only() {
    for segment in [Segment::NfoFut, Segment::NseEqDelivery, Segment::McxFut] {
        assert_eq!(
            accrual_model().exercise_stt(segment, 50.0, 75.0, snapshot_day()),
            Err(CostError::NotAnOption(segment))
        );
    }
}

#[test]
fn exercise_stt_rejects_amounts_that_are_negative_not_finite_or_overflow() {
    let model = accrual_model();
    for (intrinsic, quantity) in [
        (-1.0, 1.0),
        (f64::NAN, 1.0),
        (f64::INFINITY, 1.0),
        (1e200, 1e200),
    ] {
        let result = model.exercise_stt(Segment::NfoOpt, intrinsic, quantity, snapshot_day());
        assert!(
            matches!(result, Err(CostError::InvalidInput { .. })),
            "{intrinsic} x {quantity}"
        );
    }
}
