//! Brokerage presets: the broker's part of the cost.

use crate::model::Side;

use super::super::{
    Brokerage, BrokerageClass, BrokerageRule, CostError, DpFeeSource, NameKind, Segment,
};
use super::{accrual_model, assert_close, model_with, snapshot_day};

const ORDER_VALUE: f64 = 250_000.0;

/// Brokerage on one Rs 2.5 lakh order in every segment under a preset.
fn brokerage_everywhere(preset: Brokerage) -> Vec<(Segment, f64)> {
    let model = model_with(preset);
    Segment::ALL
        .into_iter()
        .map(|segment| {
            let cost = model.fill_cost_value(segment, Side::Buy, ORDER_VALUE, snapshot_day());
            (segment, cost.unwrap().brokerage)
        })
        .collect()
}

#[test]
fn zerodha_is_the_default_preset() {
    assert_eq!(Brokerage::default(), Brokerage::zerodha());
}

#[test]
fn zerodha_preset_rules() {
    // Section 1, brokerage row: delivery 0; intraday and futures 0.03 percent or
    // Rs 20, whichever is lower; options flat Rs 20.
    let capped = BrokerageRule::Rate {
        rate: 0.0003,
        cap: 20.0,
    };
    let rows = [
        (BrokerageClass::Delivery, BrokerageRule::Zero),
        (BrokerageClass::Intraday, capped),
        (BrokerageClass::Futures, capped),
        (BrokerageClass::Options, BrokerageRule::Flat(20.0)),
    ];
    for (class, rule) in rows {
        assert_eq!(*Brokerage::zerodha().rule(class), rule, "{class:?}");
    }
}

#[test]
fn flat_preset_charges_the_amount_in_every_segment() {
    for (segment, brokerage) in brokerage_everywhere(Brokerage::flat(10.0).unwrap()) {
        assert_eq!(brokerage, 10.0, "{segment:?}");
    }
}

#[test]
fn percent_takes_percent_units() {
    // percent(0.03, 20) is Zerodha's "0.03 percent or Rs 20": Rs 15 on an order of
    // Rs 50,000 and the Rs 20 cap on Rs 2.5 lakh.
    let model = model_with(Brokerage::percent(0.03, Some(20.0)).unwrap());
    let rows = [(50_000.0, 15.0), (ORDER_VALUE, 20.0)];
    for (value, brokerage) in rows {
        let cost = model.fill_cost_value(Segment::NseEqDelivery, Side::Buy, value, snapshot_day());
        assert_close(cost.unwrap().brokerage, brokerage);
    }
}

#[test]
fn percent_preset_without_a_cap_charges_the_full_percentage() {
    // 0.05 percent of Rs 2.5 lakh is Rs 125.
    for (_, brokerage) in brokerage_everywhere(Brokerage::percent(0.05, None).unwrap()) {
        assert_close(brokerage, 125.0);
    }
}

#[test]
fn a_fraction_passed_as_a_percentage_is_rejected() {
    // 0.0003 was meant as the fraction for 0.03 percent; as a percentage it would
    // be a brokerage of 0.0003 percent.
    let error = Brokerage::percent(0.0003, Some(20.0)).unwrap_err();
    assert!(matches!(error, CostError::InvalidBrokerage(_)));
    assert!(
        error.to_string().contains("percent() takes percent units"),
        "{error}"
    );
}

#[test]
fn a_percentage_passed_as_a_rate_is_rejected() {
    // 0.5 was meant as 0.5 percent; as a fraction it is 50 percent of the order.
    let half = BrokerageRule::Rate {
        rate: 0.5,
        cap: f64::INFINITY,
    };
    let error = Brokerage::custom(half, half, half, half).unwrap_err();
    assert!(error.to_string().contains("rates are fractions"), "{error}");
}

#[test]
fn implausible_percentages_are_rejected_at_both_ends() {
    for percent in [5.01, 30.0, 0.0009, -0.03, f64::NAN, f64::INFINITY] {
        assert!(Brokerage::percent(percent, None).is_err(), "{percent}");
    }
}

#[test]
fn plausible_percentages_and_zero_are_accepted() {
    for percent in [0.0, 0.001, 0.01, 0.03, 0.5, 2.5, 5.0] {
        assert!(Brokerage::percent(percent, None).is_ok(), "{percent}");
    }
}

#[test]
fn zero_brokerage_preset_charges_no_brokerage() {
    for (segment, brokerage) in brokerage_everywhere(Brokerage::zero_brokerage()) {
        assert_eq!(brokerage, 0.0, "{segment:?}");
    }
}

#[test]
fn custom_preset_applies_one_rule_per_brokerage_class() {
    // Delivery 0.1 percent capped at Rs 100 (Rs 250 uncapped), intraday flat Rs 5,
    // futures free, options flat Rs 15.
    let preset = Brokerage::custom(
        BrokerageRule::percent(0.1, Some(100.0)),
        BrokerageRule::Flat(5.0),
        BrokerageRule::Zero,
        BrokerageRule::Flat(15.0),
    )
    .unwrap();
    for (segment, brokerage) in brokerage_everywhere(preset) {
        let expected = match segment.brokerage_class() {
            BrokerageClass::Delivery => 100.0,
            BrokerageClass::Intraday => 5.0,
            BrokerageClass::Futures => 0.0,
            BrokerageClass::Options => 15.0,
        };
        assert_close(brokerage, expected);
    }
}

#[test]
fn statutory_components_are_the_same_under_every_preset() {
    let presets = [
        Brokerage::flat(10.0).unwrap(),
        Brokerage::percent(0.05, Some(50.0)).unwrap(),
        Brokerage::zero_brokerage(),
    ];
    let day = snapshot_day();
    for segment in Segment::ALL {
        for side in [Side::Buy, Side::Sell] {
            let zerodha = accrual_model()
                .fill_cost_value(segment, side, ORDER_VALUE, day)
                .unwrap();
            for preset in presets {
                let other = model_with(preset)
                    .fill_cost_value(segment, side, ORDER_VALUE, day)
                    .unwrap();
                assert_eq!(
                    (
                        other.stt,
                        other.exchange_txn,
                        other.sebi_fee,
                        other.stamp_duty
                    ),
                    (
                        zerodha.stt,
                        zerodha.exchange_txn,
                        zerodha.sebi_fee,
                        zerodha.stamp_duty
                    ),
                    "{segment:?} {side:?}"
                );
            }
        }
    }
}

#[test]
fn gst_follows_the_brokerage_of_the_preset() {
    // NSE delivery buy of Rs 2.5 lakh: exchange 0.00307 percent = 7.675, SEBI 0.25.
    // Flat Rs 10 brokerage: GST 0.18 x (10 + 7.675 + 0.25) = 3.2265.
    let model = model_with(Brokerage::flat(10.0).unwrap());
    let cost = model.fill_cost_value(
        Segment::NseEqDelivery,
        Side::Buy,
        ORDER_VALUE,
        snapshot_day(),
    );
    assert_close(cost.unwrap().gst, 3.2265);
}

#[test]
fn the_dp_fee_says_where_it_comes_from() {
    // Zerodha publishes its own; other presets borrow it until the caller gives one.
    let rows = [
        (Brokerage::zerodha(), 13.0, DpFeeSource::ZerodhaPreset),
        (
            Brokerage::zero_brokerage(),
            13.0,
            DpFeeSource::ZerodhaStandIn,
        ),
        (
            Brokerage::flat(10.0).unwrap(),
            13.0,
            DpFeeSource::ZerodhaStandIn,
        ),
        (
            Brokerage::percent(0.03, None).unwrap(),
            13.0,
            DpFeeSource::ZerodhaStandIn,
        ),
        (
            Brokerage::flat(10.0).unwrap().with_dp_fee(20.0).unwrap(),
            20.0,
            DpFeeSource::Supplied,
        ),
        (
            Brokerage::zerodha().with_dp_fee(13.0).unwrap(),
            13.0,
            DpFeeSource::Supplied,
        ),
    ];
    for (preset, fee, source) in rows {
        assert_eq!((preset.dp_fee(), preset.dp_fee_source()), (fee, source));
    }
}

#[test]
fn built_in_presets_are_found_by_name() {
    assert_eq!(Brokerage::from_name("zerodha"), Ok(Brokerage::zerodha()));
    assert_eq!(
        Brokerage::from_name("zero_brokerage"),
        Ok(Brokerage::zero_brokerage())
    );
}

#[test]
fn an_unknown_preset_name_is_an_error() {
    for name in ["upstox", "Zerodha", "", "flat", "zero"] {
        assert_eq!(
            Brokerage::from_name(name),
            Err(CostError::UnknownName {
                kind: NameKind::Preset,
                name: name.to_string()
            }),
            "{name:?}"
        );
    }
}

#[test]
fn negative_infinite_and_nan_amounts_are_rejected() {
    for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(Brokerage::flat(bad).is_err(), "flat {bad}");
        assert!(
            Brokerage::zerodha().with_dp_fee(bad).is_err(),
            "dp fee {bad}"
        );
    }
}

#[test]
fn a_negative_or_nan_cap_is_rejected() {
    for bad in [-1.0, f64::NAN] {
        assert!(Brokerage::percent(0.03, Some(bad)).is_err(), "cap {bad}");
    }
}
