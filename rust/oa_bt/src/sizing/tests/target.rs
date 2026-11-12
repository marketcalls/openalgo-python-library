//! Target positions (`self.target(...)`, FR-SIZE-08), each checked against a position
//! computed by hand. The context is that of `super::context`: equity 2,00,000, initial
//! capital 1,00,000, price 800, whole shares.

use super::*;

fn target(spec: TargetSize, ctx: &SizingContext) -> f64 {
    match target_units(spec, ctx) {
        Ok(units) => units,
        Err(error) => panic!("expected a target, got {error}"),
    }
}

#[test]
fn a_target_in_units_is_kept_as_given_and_signed() {
    // Like buy(qty=), a quantity given in units is never rounded here.
    assert_eq!(target(TargetSize::Units(-30.5), &context()), -30.5);
    assert_eq!(target(TargetSize::Units(0.0), &context()), 0.0);
}

#[test]
fn a_target_in_lots_is_lots_of_the_lot_size() {
    // -2 lots of 75 is a short of 150 units.
    assert_eq!(target(TargetSize::Lots(-2), &future()), -150.0);
}

#[test]
fn a_target_value_rounds_towards_zero_to_whole_shares() {
    // 10,000 / 800 = 12.5 shares: 12 long, or 12 short.
    assert_eq!(target(TargetSize::Rupees(10_000.0), &context()), 12.0);
    assert_eq!(target(TargetSize::Rupees(-10_000.0), &context()), -12.0);
}

#[test]
fn a_target_value_on_a_future_rounds_to_whole_lots() {
    // 2,00,000 / 800 = 250 units, 3.33 lots of 75: 3 lots, 225 units.
    assert_eq!(target(TargetSize::Rupees(200_000.0), &future()), 225.0);
}

#[test]
fn a_target_percent_is_of_the_sizing_equity() {
    // 40 percent of 2,00,000 is 80,000 / 800 = 100 shares; without reinvest, of the
    // initial 1,00,000: 50 shares.
    let percent = TargetSize::PercentOfEquity(40.0);
    assert_eq!(target(percent, &context()), 100.0);
    let initial = SizingContext {
        reinvest: false,
        ..context()
    };
    assert_eq!(target(percent, &initial), 50.0);
}

#[test]
fn a_fractional_target_value_is_not_rounded() {
    let fractional = SizingContext {
        fractional: true,
        ..context()
    };
    assert_eq!(target(TargetSize::Rupees(10_000.0), &fractional), 12.5);
}

#[test]
fn a_target_that_is_not_a_finite_number_is_refused() {
    let cases = [
        TargetSize::Units(f64::NAN),
        TargetSize::Rupees(f64::INFINITY),
        TargetSize::PercentOfEquity(f64::NAN),
    ];
    for spec in cases {
        let result = target_units(spec, &context());
        assert!(
            matches!(result, Err(SizingError::InvalidSizer { .. })),
            "{spec:?}"
        );
    }
}

#[test]
fn a_target_value_needs_a_sizing_price() {
    let unpriced = SizingContext {
        decision_price_rupees: 0.0,
        ..context()
    };
    let result = target_units(TargetSize::Rupees(10_000.0), &unpriced);
    assert!(matches!(result, Err(SizingError::InvalidSizer { .. })));
}
