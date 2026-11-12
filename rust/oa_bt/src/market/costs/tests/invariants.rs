//! Properties that hold for every segment, side, preset, rounding mode and
//! schedule version.

use crate::model::{days_from_civil, Day, Side};

use super::super::{
    Brokerage, CostBreakdown, CostError, CostModel, Rounding, Segment, COMPONENT_NAMES,
};
use super::{
    accrual_model, assert_close, builtin_book, calculator_model, model_with, snapshot_day, Lcg,
    SameDay,
};

const BOTH_SIDES: [Side; 2] = [Side::Buy, Side::Sell];

/// Every rounding mode under a spread of brokerage presets.
fn all_models() -> Vec<CostModel> {
    let presets = [
        Brokerage::zerodha(),
        Brokerage::zero_brokerage(),
        Brokerage::flat(20.0).unwrap(),
        Brokerage::percent(0.05, Some(25.0)).unwrap(),
        Brokerage::percent(0.1, None).unwrap(),
    ];
    let mut models = Vec::new();
    for rounding in [Rounding::ContractNote, Rounding::Calculator] {
        for preset in presets {
            models.push(CostModel::new(builtin_book(), preset, rounding));
        }
    }
    models
}

/// The first built-in day, a day in mid 2022, the first and last days of the 2024
/// version, and the snapshot day.
fn sample_days() -> [Day; 5] {
    [
        days_from_civil(2020, 7, 1),
        days_from_civil(2022, 6, 1),
        days_from_civil(2024, 10, 1),
        days_from_civil(2026, 3, 31),
        snapshot_day(),
    ]
}

/// Call `check` with random fills across every model, segment and side.
fn for_random_fills(mut check: impl FnMut(&CostModel, Segment, Side, f64, f64, Day)) {
    let days = sample_days();
    let mut rng = Lcg(42);
    for model in all_models() {
        for segment in Segment::ALL {
            for side in BOTH_SIDES {
                for _ in 0..25 {
                    let price = (rng.next() * 9_000.0 * rng.next()).max(0.05);
                    let quantity =
                        (rng.next() * 2_000.0).floor() + 0.37 * (rng.next() * 1.25).floor();
                    let day = days[(rng.next() * days.len() as f64) as usize];
                    check(&model, segment, side, price, quantity, day);
                }
            }
        }
    }
}

#[test]
fn components_are_never_negative() {
    for_random_fills(|model, segment, side, price, quantity, day| {
        let cost = model
            .fill_cost(segment, side, price, quantity, day)
            .unwrap();
        for (name, amount) in COMPONENT_NAMES.iter().zip(cost.components()) {
            assert!(
                amount >= 0.0 && amount.is_finite(),
                "{segment:?} {name} = {amount}"
            );
        }
    });
}

#[test]
fn total_is_the_sum_of_the_components() {
    for_random_fills(|model, segment, side, price, quantity, day| {
        let cost = model
            .fill_cost(segment, side, price, quantity, day)
            .unwrap();
        let sum: f64 = cost.components().iter().sum();
        assert_close(cost.total(), sum);
    });
}

#[test]
fn round_trip_components_are_never_negative() {
    for_random_fills(|model, segment, _, price, quantity, day| {
        let cost = model
            .round_trip_on(segment, price, price * 1.01, quantity, day)
            .unwrap();
        for amount in cost.components() {
            assert!(amount >= 0.0 && amount.is_finite(), "{segment:?} {cost:?}");
        }
    });
}

#[test]
fn zero_quantity_costs_zero() {
    for model in all_models() {
        for segment in Segment::ALL {
            for side in BOTH_SIDES {
                let cost = model.fill_cost(segment, side, 1_234.5, 0.0, snapshot_day());
                assert_eq!(cost, Ok(CostBreakdown::ZERO), "{segment:?} {side:?}");
            }
        }
    }
}

#[test]
fn zero_price_costs_zero() {
    for model in all_models() {
        for segment in Segment::ALL {
            for side in BOTH_SIDES {
                let cost = model.fill_cost(segment, side, 0.0, 100.0, snapshot_day());
                assert_eq!(cost, Ok(CostBreakdown::ZERO), "{segment:?} {side:?}");
            }
        }
    }
}

#[test]
fn a_round_trip_of_zero_quantity_costs_zero() {
    for model in all_models() {
        for segment in Segment::ALL {
            let cost = model.round_trip_on(segment, 100.0, 110.0, 0.0, snapshot_day());
            assert_eq!(cost, Ok(CostBreakdown::ZERO), "{segment:?}");
        }
    }
}

#[test]
fn exercise_stt_on_zero_quantity_is_zero() {
    for model in all_models() {
        assert_eq!(
            model.exercise_stt(Segment::NfoOpt, 50.0, 0.0, snapshot_day()),
            Ok(0.0)
        );
    }
}

#[test]
fn accrual_without_brokerage_is_linear_in_quantity() {
    // With no per-order brokerage, a fill costs twice what half of it costs.
    let model = model_with(Brokerage::zero_brokerage());
    for segment in Segment::ALL {
        for side in BOTH_SIDES {
            let whole = model
                .fill_cost(segment, side, 812.35, 600.0, snapshot_day())
                .unwrap();
            let half = model
                .fill_cost(segment, side, 812.35, 300.0, snapshot_day())
                .unwrap();
            assert_close(whole.total(), 2.0 * half.total());
        }
    }
}

#[test]
fn an_accrued_round_trip_is_the_sum_of_its_two_fills() {
    let model = accrual_model();
    let day = snapshot_day();
    for segment in Segment::ALL {
        let buy = model
            .fill_cost(segment, Side::Buy, 250.0, 120.0, day)
            .unwrap();
        let sell = model
            .fill_cost(segment, Side::Sell, 262.5, 120.0, day)
            .unwrap();
        let round_trip = model
            .round_trip_on(segment, 250.0, 262.5, 120.0, day)
            .unwrap();
        assert_close(round_trip.total(), buy.total() + sell.total());
    }
}

#[test]
fn a_calculator_fill_is_a_round_trip_with_the_other_price_at_zero() {
    let model = calculator_model();
    let day = snapshot_day();
    for segment in Segment::ALL {
        assert_eq!(
            model.fill_cost(segment, Side::Buy, 417.3, 55.0, day),
            model.round_trip_on(segment, 417.3, 0.0, 55.0, day)
        );
        assert_eq!(
            model.fill_cost(segment, Side::Sell, 417.3, 55.0, day),
            model.round_trip_on(segment, 0.0, 417.3, 55.0, day)
        );
    }
}

#[test]
fn calculator_rounding_moves_tax_by_at_most_half_a_rupee_and_the_rest_by_a_paisa() {
    let (calculator, accrual) = (calculator_model(), accrual_model());
    let mut rng = Lcg(7);
    for segment in Segment::ALL {
        for _ in 0..40 {
            let buy = 10.0 + rng.next() * 3_000.0;
            let sell = buy * (0.95 + rng.next() * 0.1);
            let quantity = 1.0 + (rng.next() * 900.0).floor();
            let a = accrual
                .round_trip_on(segment, buy, sell, quantity, snapshot_day())
                .unwrap();
            let c = calculator
                .round_trip_on(segment, buy, sell, quantity, snapshot_day())
                .unwrap();
            assert!(
                (a.stt - c.stt).abs() <= 0.506,
                "{segment:?} stt {a:?} {c:?}"
            );
            assert!(
                (a.stamp_duty - c.stamp_duty).abs() <= 0.506,
                "{segment:?} stamp"
            );
            assert!(
                (a.brokerage - c.brokerage).abs() <= 0.011,
                "{segment:?} brokerage"
            );
            assert!(
                (a.exchange_txn - c.exchange_txn).abs() <= 0.011,
                "{segment:?} exchange"
            );
            assert!((a.sebi_fee - c.sebi_fee).abs() <= 0.006, "{segment:?} sebi");
            assert!((a.gst - c.gst).abs() <= 0.011, "{segment:?} gst");
        }
    }
}

#[test]
fn negative_infinite_and_nan_prices_and_quantities_are_rejected() {
    let model = accrual_model();
    let day = snapshot_day();
    for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let price = model.fill_cost(Segment::NseEqDelivery, Side::Buy, bad, 10.0, day);
        let quantity = model.fill_cost(Segment::NseEqDelivery, Side::Buy, 10.0, bad, day);
        assert!(
            matches!(price, Err(CostError::InvalidInput { field: "price", .. })),
            "{bad}"
        );
        assert!(
            matches!(
                quantity,
                Err(CostError::InvalidInput {
                    field: "quantity",
                    ..
                })
            ),
            "{bad}"
        );
        assert!(model
            .round_trip_on(Segment::NseEqDelivery, bad, 10.0, 1.0, day)
            .is_err());
        assert!(model
            .round_trip_on(Segment::NseEqDelivery, 10.0, bad, 1.0, day)
            .is_err());
    }
}

#[test]
fn a_traded_value_that_overflows_is_an_error_not_nan() {
    // Each operand is a valid number; their product is not.
    let model = accrual_model();
    let day = snapshot_day();
    let fill = model.fill_cost(Segment::NfoOpt, Side::Buy, 1e200, 1e200, day);
    let round_trip = model.round_trip_on(Segment::NfoOpt, 1e200, 1e200, 1e200, day);
    let calculator = calculator_model().round_trip_on(Segment::NfoOpt, 1e200, 1e200, 1e200, day);
    for result in [fill, round_trip, calculator] {
        assert!(
            matches!(result, Err(CostError::InvalidInput { .. })),
            "{result:?}"
        );
    }
}

#[test]
fn every_cost_of_a_large_finite_trade_is_finite() {
    for model in all_models() {
        let cost =
            model.fill_cost_value(Segment::BseEqDeliverySs, Side::Buy, 1e300, snapshot_day());
        assert!(cost
            .unwrap()
            .components()
            .iter()
            .all(|amount| amount.is_finite()));
    }
}
