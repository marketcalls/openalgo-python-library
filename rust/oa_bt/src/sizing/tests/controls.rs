//! The controls shared by every sizer (12-position-sizing 5, FR-SIZE-07): `reinvest`,
//! the caps, and a strategy's own sizing function, each checked against a quantity
//! computed by hand in the comment above it. The context is that of `super::context`:
//! equity 2,00,000, initial capital 1,00,000, free cash 50,000, price 800, whole shares.

use super::*;

fn not_reinvesting(ctx: SizingContext) -> SizingContext {
    SizingContext {
        reinvest: false,
        ..ctx
    }
}

fn capped(caps: SizeCaps) -> SizingContext {
    SizingContext { caps, ..context() }
}

#[test]
fn without_reinvest_a_percent_of_equity_is_taken_of_the_initial_capital() {
    // 50 percent of the initial 1,00,000 is 50,000 / 800 = 62.5 -> 62 shares.
    let spec = SizerSpec::Percent {
        percent: 50.0,
        basis: SizingBasis::Equity,
    };
    assert_eq!(units(spec, &not_reinvesting(context())), 62.0);
}

#[test]
fn without_reinvest_fixed_risk_risks_a_percent_of_the_initial_capital() {
    // 1 percent of 1,00,000 is 1,000, less 100 of costs is 900; 8.00 a share to the
    // stop: 112.5 -> 112 shares.
    let ctx = SizingContext {
        stop_distance_rupees: Some(8.0),
        estimated_cost_rupees: 100.0,
        ..not_reinvesting(context())
    };
    let spec = SizerSpec::FixedRisk {
        risk_percent: 1.0,
        units: 1,
        hard_limit_units: None,
    };
    assert_eq!(units(spec, &ctx), 112.0);
}

#[test]
fn without_reinvest_the_engine_default_spends_at_most_the_initial_capital() {
    // Free cash 2,00,000 but the initial capital is 1,00,000: (1,00,000 - 400) / 800 =
    // 124.5 -> 124 shares.
    let ctx = SizingContext {
        free_cash_rupees: 200_000.0,
        estimated_cost_rupees: 400.0,
        ..not_reinvesting(context())
    };
    assert_eq!(units(SizerSpec::EngineDefault, &ctx), 124.0);
}

#[test]
fn reinvest_does_not_change_a_percent_of_free_cash() {
    // 50 percent of the free 50,000 is 25,000 / 800 = 31.25 -> 31 shares either way.
    let spec = SizerSpec::Percent {
        percent: 50.0,
        basis: SizingBasis::FreeCash,
    };
    assert_eq!(units(spec, &not_reinvesting(context())), 31.0);
    assert_eq!(units(spec, &context()), 31.0);
}

/// 10,00,000 rupees of value at 800 asks for 1,250 shares.
const LARGE: SizerSpec = SizerSpec::Value {
    rupees: 1_000_000.0,
};

#[test]
fn each_position_cap_bounds_the_position_after_the_order() {
    // 80,000 / 800 = 100; 10 percent of 2,00,000 = 20,000 / 800 = 25; half the equity,
    // 1,00,000 / 800 = 125.
    let cases = [
        (
            SizeCaps {
                position_units: Some(300.0),
                ..SizeCaps::default()
            },
            300.0,
        ),
        (
            SizeCaps {
                position_rupees: Some(80_000.0),
                ..SizeCaps::default()
            },
            100.0,
        ),
        (
            SizeCaps {
                position_percent: Some(10.0),
                ..SizeCaps::default()
            },
            25.0,
        ),
        (
            SizeCaps {
                leverage: Some(0.5),
                ..SizeCaps::default()
            },
            125.0,
        ),
    ];
    for (caps, expected) in cases {
        assert_eq!(units(LARGE, &capped(caps)), expected, "{caps:?}");
    }
}

#[test]
fn a_position_cap_leaves_room_only_for_what_is_not_yet_held() {
    // A cap of 150 shares with 80 held leaves 70.
    let ctx = SizingContext {
        held_on_side_units: 80.0,
        ..capped(SizeCaps {
            position_units: Some(150.0),
            ..SizeCaps::default()
        })
    };
    assert_eq!(units(SizerSpec::Qty(100.0), &ctx), 70.0);
}

#[test]
fn a_cap_in_lots_rounds_the_room_down_to_whole_lots() {
    // 4 lots of 75 = 300 units, 1 lot held: 225 units of room, 3 lots.
    let ctx = SizingContext {
        held_on_side_units: 75.0,
        caps: SizeCaps {
            position_lots: Some(4),
            ..SizeCaps::default()
        },
        ..future()
    };
    assert_eq!(units(SizerSpec::Lots(10), &ctx), 225.0);
}

#[test]
fn an_opposite_position_adds_to_the_room_of_an_order_that_reverses_it() {
    // Short 100, buying: the position on the buy side after the order is the order less
    // 100, so a cap of 50 lets the order be 150.
    let ctx = SizingContext {
        held_on_side_units: -100.0,
        ..capped(SizeCaps {
            position_units: Some(50.0),
            ..SizeCaps::default()
        })
    };
    assert_eq!(units(SizerSpec::Qty(200.0), &ctx), 150.0);
}

#[test]
fn the_risk_cap_bounds_the_loss_of_the_order_at_its_stop() {
    // 8.00 a share to the stop: 2,000 rupees is 250 shares; 0.5 percent of 2,00,000 is
    // 1,000 rupees, 125 shares.
    let with_stop = |caps| SizingContext {
        stop_distance_rupees: Some(8.0),
        ..capped(caps)
    };
    let rupees = with_stop(SizeCaps {
        risk_rupees: Some(2_000.0),
        ..SizeCaps::default()
    });
    let percent = with_stop(SizeCaps {
        risk_percent: Some(0.5),
        ..SizeCaps::default()
    });
    assert_eq!(units(SizerSpec::Qty(500.0), &rupees), 250.0);
    assert_eq!(units(SizerSpec::Qty(500.0), &percent), 125.0);
}

#[test]
fn under_a_risk_cap_an_order_without_a_stop_is_skipped_for_the_cap() {
    let ctx = capped(SizeCaps {
        risk_rupees: Some(1_000.0),
        ..SizeCaps::default()
    });
    let result = size(SizerSpec::Qty(10.0), &ctx);
    assert_eq!(result, Ok(SizeOutcome::Skipped(SkipReason::Cap)));
}

#[test]
fn the_tightest_cap_wins() {
    // 300 shares, or half the equity (125 shares): 125.
    let ctx = capped(SizeCaps {
        position_units: Some(300.0),
        leverage: Some(0.5),
        ..SizeCaps::default()
    });
    assert_eq!(units(LARGE, &ctx), 125.0);
}

#[test]
fn a_cap_already_reached_skips_the_order_for_the_cap() {
    let ctx = SizingContext {
        held_on_side_units: 300.0,
        ..capped(SizeCaps {
            position_units: Some(300.0),
            ..SizeCaps::default()
        })
    };
    let result = size(SizerSpec::Qty(10.0), &ctx);
    assert_eq!(result, Ok(SizeOutcome::Skipped(SkipReason::Cap)));
}

#[test]
fn without_reinvest_a_percent_cap_is_of_the_initial_capital() {
    // 10 percent of 1,00,000 is 10,000 / 800 = 12.5 -> 12 shares.
    let ctx = not_reinvesting(capped(SizeCaps {
        position_percent: Some(10.0),
        ..SizeCaps::default()
    }));
    assert_eq!(units(LARGE, &ctx), 12.0);
}

#[test]
fn a_size_below_one_share_is_rounded_to_zero_whatever_the_caps() {
    // 100 / 800 is an eighth of a share.
    let ctx = capped(SizeCaps {
        position_units: Some(1.0),
        ..SizeCaps::default()
    });
    let result = size(SizerSpec::Value { rupees: 100.0 }, &ctx);
    assert_eq!(result, Ok(SizeOutcome::Skipped(SkipReason::RoundedToZero)));
}

#[test]
fn unusable_controls_are_refused_before_the_run() {
    let caps = [
        SizeCaps {
            position_units: Some(0.0),
            ..SizeCaps::default()
        },
        SizeCaps {
            position_lots: Some(0),
            ..SizeCaps::default()
        },
        SizeCaps {
            position_rupees: Some(f64::NAN),
            ..SizeCaps::default()
        },
        SizeCaps {
            position_percent: Some(-1.0),
            ..SizeCaps::default()
        },
        SizeCaps {
            leverage: Some(0.0),
            ..SizeCaps::default()
        },
        SizeCaps {
            risk_rupees: Some(-5.0),
            ..SizeCaps::default()
        },
        SizeCaps {
            risk_percent: Some(100.0),
            ..SizeCaps::default()
        },
    ];
    for caps in caps {
        let controls = SizingControls {
            caps,
            ..SizingControls::default()
        };
        assert!(controls.validate().is_err(), "{caps:?}");
    }
    assert_eq!(SizingControls::default().validate(), Ok(()));
    assert!(SizingControls::default().reinvest);
}

#[test]
fn a_sizing_function_quantity_rounds_down_to_the_venue_rule() {
    assert_eq!(units(SizerSpec::Function { units: 10.7 }, &context()), 10.0);
}

#[test]
fn a_sizing_function_that_returns_zero_skips_the_order() {
    let result = size(SizerSpec::Function { units: 0.0 }, &context());
    assert_eq!(result, Ok(SizeOutcome::Skipped(SkipReason::RoundedToZero)));
}

#[test]
fn a_sizing_function_quantity_is_capped_like_any_sizer() {
    let ctx = capped(SizeCaps {
        position_units: Some(100.0),
        ..SizeCaps::default()
    });
    assert_eq!(units(SizerSpec::Function { units: 500.0 }, &ctx), 100.0);
}

#[test]
fn a_sizing_function_quantity_that_is_negative_or_not_a_number_is_refused() {
    for given in [-1.0, f64::NAN, f64::INFINITY] {
        let result = size(SizerSpec::Function { units: given }, &context());
        assert!(
            matches!(result, Err(SizingError::InvalidSizer { .. })),
            "{given}"
        );
    }
}
