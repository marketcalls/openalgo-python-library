//! Sizers, each checked against a quantity computed by hand in the comment above it.

use super::*;

/// Equity 2,00,000; initial capital 1,00,000; free cash 50,000; price 800; cash equity
/// in venue mode (whole shares); no costs, no freeze quantity.
fn context() -> SizingContext {
    SizingContext {
        equity_rupees: 200_000.0,
        initial_capital_rupees: 100_000.0,
        free_cash_rupees: 50_000.0,
        decision_price_rupees: 800.0,
        stop_distance_rupees: None,
        lot_size_units: 1,
        multiplier: 1.0,
        estimated_cost_rupees: 0.0,
        fractional: false,
        derivative: false,
        freeze_quantity_units: None,
        instruments: 1,
        cash_buffer: 0.0,
    }
}

/// A NIFTY-like future: lot 75, freeze 1,800 units.
fn future() -> SizingContext {
    SizingContext {
        lot_size_units: 75,
        derivative: true,
        freeze_quantity_units: Some(1_800),
        ..context()
    }
}

fn units(spec: SizerSpec, ctx: &SizingContext) -> f64 {
    match size(spec, ctx) {
        Ok(SizeOutcome::Units(units)) => units,
        other => panic!("expected units, got {other:?}"),
    }
}

#[test]
fn a_fixed_quantity_rounds_down_to_whole_shares_in_venue_mode() {
    assert_eq!(units(SizerSpec::Qty(10.7), &context()), 10.0);
}

#[test]
fn a_fixed_quantity_stays_fractional_in_fractional_mode() {
    let fractional = SizingContext {
        fractional: true,
        ..context()
    };
    assert_eq!(units(SizerSpec::Qty(10.7), &fractional), 10.7);
}

#[test]
fn lots_are_units_of_the_lot_size() {
    // 2 lots of 75 units.
    assert_eq!(units(SizerSpec::Lots(2), &future()), 150.0);
}

#[test]
fn a_value_sizer_divides_rupees_by_the_price() {
    // 10,000 / 800 = 12.5, rounded down to 12 shares.
    let value = SizerSpec::Value { rupees: 10_000.0 };
    assert_eq!(units(value, &context()), 12.0);
}

#[test]
fn a_percent_sizer_takes_its_percent_of_the_chosen_basis() {
    // 50 percent of equity 2,00,000 is 1,00,000 / 800 = 125; of the initial capital
    // 50,000 / 800 = 62.5 -> 62; of the free cash 25,000 / 800 = 31.25 -> 31.
    let cases = [
        (SizingBasis::Equity, 125.0),
        (SizingBasis::InitialCapital, 62.0),
        (SizingBasis::FreeCash, 31.0),
    ];
    for (basis, expected) in cases {
        let spec = SizerSpec::Percent {
            percent: 50.0,
            basis,
        };
        assert_eq!(units(spec, &context()), expected, "{basis:?}");
    }
}

fn fixed_risk(units: u32, hard_limit_units: Option<f64>) -> SizerSpec {
    SizerSpec::FixedRisk {
        risk_percent: 1.0,
        units,
        hard_limit_units,
    }
}

fn with_stop(ctx: SizingContext) -> SizingContext {
    SizingContext {
        stop_distance_rupees: Some(8.0),
        estimated_cost_rupees: 100.0,
        ..ctx
    }
}

#[test]
fn fixed_risk_divides_the_risk_budget_net_of_costs_by_the_loss_at_the_stop() {
    // 1 percent of 2,00,000 is 2,000, less 100 of costs is 1,900; a stop 8.00 away
    // loses 8 per share: 237.5 -> 237 shares.
    assert_eq!(units(fixed_risk(1, None), &with_stop(context())), 237.0);
}

#[test]
fn fixed_risk_rounds_down_to_whole_lots() {
    // 237.5 units of a 75-unit lot is 3.17 lots -> 3 lots, 225 units.
    assert_eq!(units(fixed_risk(1, None), &with_stop(future())), 225.0);
}

#[test]
fn fixed_risk_is_capped_by_the_hard_limit() {
    assert_eq!(
        units(fixed_risk(1, Some(200.0)), &with_stop(context())),
        200.0
    );
}

#[test]
fn fixed_risk_split_into_units_orders_one_tranche() {
    // 237.5 / 2 = 118.75 -> 118 shares.
    assert_eq!(units(fixed_risk(2, None), &with_stop(context())), 118.0);
}

#[test]
fn fixed_risk_without_a_stop_is_refused() {
    let result = size(fixed_risk(1, None), &context());
    assert!(matches!(result, Err(SizingError::InvalidSizer { .. })));
}

#[test]
fn fixed_risk_whose_costs_exceed_the_budget_is_skipped() {
    let costly = SizingContext {
        estimated_cost_rupees: 2_500.0,
        ..with_stop(context())
    };
    let result = size(fixed_risk(1, None), &costly);
    assert_eq!(result, Ok(SizeOutcome::Skipped(SkipReason::RoundedToZero)));
}

#[test]
fn atr_risk_puts_the_stop_a_multiple_of_the_atr_away() {
    // 1 percent of 2,00,000 = 2,000; stop 2 x 12.50 = 25.00 away: 80 shares.
    let spec = SizerSpec::AtrRisk {
        risk_percent: 1.0,
        multiple: 2.0,
        atr_rupees: 12.5,
    };
    assert_eq!(units(spec, &context()), 80.0);
}

#[test]
fn atr_risk_without_an_atr_is_refused() {
    let spec = SizerSpec::AtrRisk {
        risk_percent: 1.0,
        multiple: 2.0,
        atr_rupees: f64::NAN,
    };
    let result = size(spec, &context());
    assert!(matches!(result, Err(SizingError::InvalidSizer { .. })));
}

fn whole_capital() -> SizingContext {
    SizingContext {
        free_cash_rupees: 200_000.0,
        estimated_cost_rupees: 400.0,
        ..context()
    }
}

#[test]
fn the_engine_default_is_the_capital_net_of_costs_in_fractional_mode() {
    // (2,00,000 - 400) / 800 = 249.5 shares, kept fractional.
    let fractional = SizingContext {
        fractional: true,
        ..whole_capital()
    };
    assert_eq!(units(SizerSpec::EngineDefault, &fractional), 249.5);
}

#[test]
fn the_engine_default_keeps_a_two_percent_buffer_in_venue_mode() {
    // 2,00,000 x 0.98 = 1,96,000, less 400 = 1,95,600 / 800 = 244.5 -> 244 shares.
    let venue = SizingContext {
        cash_buffer: VENUE_CASH_BUFFER,
        ..whole_capital()
    };
    assert_eq!(units(SizerSpec::EngineDefault, &venue), 244.0);
}

#[test]
fn the_engine_default_never_exceeds_the_free_cash() {
    // Equity 2,00,000 but only 50,000 free: 50,000 / 800 = 62.5 -> 62 shares.
    assert_eq!(units(SizerSpec::EngineDefault, &context()), 62.0);
}

#[test]
fn the_engine_default_splits_the_capital_across_the_instruments() {
    // 2,00,000 over 4 instruments is 50,000 each: (50,000 - 400) / 800 = 62 shares.
    let four = SizingContext {
        instruments: 4,
        ..whole_capital()
    };
    assert_eq!(units(SizerSpec::EngineDefault, &four), 62.0);
}

#[test]
fn the_engine_default_for_a_derivative_is_one_lot() {
    assert_eq!(units(SizerSpec::EngineDefault, &future()), 75.0);
}

#[test]
fn a_sized_quantity_is_capped_at_the_freeze_quantity_in_whole_lots() {
    // 30 lots of 75 = 2,250 units, above the freeze quantity of 1,800 (24 lots).
    assert_eq!(units(SizerSpec::Lots(30), &future()), 1_800.0);
    // A freeze quantity of 1,000 is 13.3 lots: capped at 13 lots, 975 units.
    let odd_freeze = SizingContext {
        freeze_quantity_units: Some(1_000),
        ..future()
    };
    assert_eq!(units(SizerSpec::Lots(30), &odd_freeze), 975.0);
}

#[test]
fn a_freeze_quantity_below_one_lot_leaves_no_room() {
    let tiny = SizingContext {
        freeze_quantity_units: Some(50),
        ..future()
    };
    let result = size(SizerSpec::Lots(1), &tiny);
    assert_eq!(result, Ok(SizeOutcome::Skipped(SkipReason::Cap)));
}

#[test]
fn a_size_that_rounds_to_zero_is_skipped() {
    // 100 / 800 = 0.125 of a share.
    let tiny = size(SizerSpec::Value { rupees: 100.0 }, &context());
    assert_eq!(tiny, Ok(SizeOutcome::Skipped(SkipReason::RoundedToZero)));
}

#[test]
fn a_quantity_a_hair_below_a_lot_boundary_counts_as_on_it() {
    // Two lots of 75 computed as 149.99999999999997 units are two lots, not one.
    assert_eq!(
        round_down_to_venue(149.999_999_999_999_97, 75, false),
        150.0
    );
    assert_eq!(round_down_to_venue(149.0, 75, false), 75.0);
}
