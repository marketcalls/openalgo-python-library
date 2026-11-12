//! The server's 35 golden risk vectors (`tests/fixtures/risk_vectors.json`) through the
//! leg state, with the options set to the live module's behaviour (levels not rounded).
//!
//! A vector is one last price against one position's state. Each is run two ways:
//!
//! - **Tick path** (`LegState::on_tick`): extreme, trail, stop, then target on the one
//!   price, the live module's order. Every vector the engine can represent matches.
//! - **Bar path** (step 3 then step 4 on a bar whose four prices are the last price):
//!   the levels in force are tested first and a level moved by the bar is in force from
//!   the next slice (03-architecture 4.2 step 4), so a vector whose breach needs the
//!   stop that very price moved breaches one slice later.
//!
//! Keys compared: `evaluated`, `breached`, `reason`, `current_sl`, `highest_price`,
//! `lowest_price`, `stop_moved`, `trail_armed`. `pnl` is the portfolio's, not the leg's
//! (the vectors' notes let a consumer assert only the keys it implements). The leg
//! tracks only the extreme on its own side; the other is carried as the state gave it.
//! Wire values that are not prices (zero, negative, null) are absent levels, as the
//! server's `is_price` reads them; the engine itself never makes such a level.

use crate::exec::fill::FillAt;
use crate::model::Side;
use crate::rms::error::RmsError;
use crate::rms::leg::ratchet::TrailPlan;
use crate::rms::leg::{
    BarPrices, LegContext, LegState, LevelInForce, LevelKind, TargetLevel, TargetState, TrailState,
};
use crate::rms::rules::LegRules;

use super::json::{parse, Json};

const FIXTURE: &str = include_str!("../../../../tests/fixtures/risk_vectors.json");

/// The arming threshold of a continuous trail when the state names none: OpenAlgo's
/// MIN_TRAIL_PROFIT (the vectors' notes).
const DEFAULT_TRAIL_TRIGGER: f64 = 1.0;

/// Vectors the engine refuses on both paths, and why.
const REFUSED: [(&str, &str); 2] = [
    (
        "side_is_case_insensitive",
        "a side is BUY or SELL; the binding rejects any other spelling, so a lowercase \
         side is never read as a long and never evaluated either",
    ),
    (
        "non_positive_entry_disables_trailing",
        "a leg opens only from a positive fill average (every fill price is positive), \
         so no leg exists to keep the absolute stop while the trail is disabled",
    ),
];

/// Vectors whose breach comes one slice later on bars: the stop this price moves is
/// tested from the next slice. (name, the stop in force from the next slice).
const DEFERRED_ON_BARS: [(&str, f64); 2] = [
    ("trail_arms_off_the_peak_not_the_current_price", 107.0),
    ("stepped_trail_never_passes_the_best_price_seen", 105.0),
];

/// What one evaluation gave, in the vectors' keys.
#[derive(Clone, Debug, PartialEq)]
struct Outcome {
    evaluated: bool,
    reason: Option<&'static str>,
    current_sl: Option<f64>,
    highest_price: Option<f64>,
    lowest_price: Option<f64>,
    stop_moved: bool,
    trail_armed: bool,
}

fn cases() -> Vec<Json> {
    match parse(FIXTURE) {
        Ok(document) => document.get("cases").list().to_vec(),
        Err(error) => panic!("the risk vectors parse: {error}"),
    }
}

fn name(case: &Json) -> &str {
    case.get("name").text().unwrap_or("")
}

/// A wire price: a positive finite number, else absent (the server's `is_price`).
fn wire_price(value: &Json) -> Option<f64> {
    value
        .number()
        .filter(|price| price.is_finite() && *price > 0.0)
}

/// The leg a vector's state describes, or the reason the engine refuses it.
fn seed(state: &Json) -> Result<LegState, RmsError> {
    let side_code = state.get("side").text().unwrap_or("");
    let side = Side::from_code(side_code).ok_or(RmsError::InvalidRule {
        rule: "side",
        reason: format!("{side_code:?} is not BUY or SELL"),
    })?;
    let context = LegContext {
        side,
        tick_rupees: 0.05,
        multiplier: 1.0,
        round_levels: false,
        lot_size_units: 1,
        fractional: true,
    };
    let entry = state.get("entry_price").number().unwrap_or(0.0);
    let held = state.get("quantity").number().unwrap_or(0.0);
    // The engine's own guard: a leg opens only from a positive average price.
    LegState::open(LegRules::default(), context, entry, held.max(1.0))?;
    let stop = wire_price(state.get("current_sl")).or(wire_price(state.get("initial_sl")));
    let extreme_key = if side == Side::Buy {
        "highest_price"
    } else {
        "lowest_price"
    };
    let extreme = wire_price(state.get(extreme_key)).unwrap_or(entry);
    let plan = trail_plan(state);
    Ok(LegState {
        rules: LegRules::default(),
        context,
        reference_rupees: entry,
        stop: stop.map(|level_rupees| LevelInForce {
            level_rupees,
            quantity_units: held,
        }),
        targets: wire_price(state.get("target"))
            .map(|level_rupees| TargetLevel {
                index: 0,
                level_rupees,
                quantity_units: held.max(1.0),
                state: TargetState::Working,
            })
            .into_iter()
            .collect(),
        remainder_units: 0.0,
        trail: plan.map(|_| TrailState {
            armed: false,
            favourable_extreme_rupees: extreme,
        }),
        breakeven: None,
        held_units: held,
        extreme_rupees: extreme,
        trail_plan: plan,
        anchor_rupees: wire_price(state.get("initial_sl")).or(stop),
        after_first_target_done: false,
    })
}

/// The trail of a state: continuous by default with the default trigger, stepped when
/// `trail_mode` says so; none unless enabled with a positive step.
fn trail_plan(state: &Json) -> Option<TrailPlan> {
    let step = state.get("trailing_step").number().unwrap_or(0.0);
    if state.get("trailing_enabled").boolean() != Some(true) || step <= 0.0 {
        return None;
    }
    let trigger = state
        .get("trail_trigger")
        .number()
        .unwrap_or(DEFAULT_TRAIL_TRIGGER);
    Some(match state.get("trail_mode").text() {
        Some("stepped") => TrailPlan::Stepped {
            trigger_points: trigger,
            step_points: step,
        },
        _ => TrailPlan::Continuous {
            arm_points: trigger,
            distance_points: step,
        },
    })
}

fn reason(hits: &[crate::rms::leg::LevelHit]) -> Option<&'static str> {
    hits.first().map(|hit| match hit.kind {
        LevelKind::Stop => "sl",
        LevelKind::Target(_) => "target",
    })
}

fn outcome(state: &Json, leg: &LegState, before: Option<f64>, evaluated: bool) -> Outcome {
    let current_sl = leg.stop.map(|stop| stop.level_rupees);
    let carried = |key: &str| wire_price(state.get(key));
    let (highest_price, lowest_price) = match leg.context.side {
        Side::Buy => (Some(leg.extreme_rupees), carried("lowest_price")),
        Side::Sell => (carried("highest_price"), Some(leg.extreme_rupees)),
    };
    Outcome {
        evaluated,
        reason: None,
        current_sl,
        highest_price,
        lowest_price,
        stop_moved: current_sl != before,
        trail_armed: leg.trail.is_some_and(|trail| trail.armed),
    }
}

/// The tick path: `on_tick`; an unusable price is refused and leaves the leg as it was.
fn on_tick(case: &Json) -> Result<Outcome, RmsError> {
    let state = case.get("state");
    let mut leg = seed(state)?;
    let before = leg.stop.map(|stop| stop.level_rupees);
    let price = case.get("ltp").number().unwrap_or(f64::NAN);
    let untouched = leg.clone();
    match leg.on_tick(price) {
        Ok(tick) => Ok(Outcome {
            reason: reason(&tick.hits),
            ..outcome(state, &leg, before, true)
        }),
        Err(RmsError::InvalidValue { .. }) => {
            assert_eq!(
                leg,
                untouched,
                "{}: a refused tick changes nothing",
                name(case)
            );
            Ok(outcome(state, &leg, before, false))
        }
        Err(other) => Err(other),
    }
}

/// The bar path: step 3 against the bar, then step 4. `None` for a price no bar can
/// carry (bars are validated when the data is loaded).
fn on_bar(case: &Json) -> Result<Option<Outcome>, RmsError> {
    let state = case.get("state");
    let mut leg = seed(state)?;
    let Some(price) = wire_price(case.get("ltp")) else {
        return Ok(None);
    };
    let before = leg.stop.map(|stop| stop.level_rupees);
    let bar = BarPrices {
        open: price,
        high: price,
        low: price,
        close: price,
    };
    let hits = leg.test_bar(&bar);
    leg.ratchet(&bar)?;
    Ok(Some(Outcome {
        reason: reason(&hits),
        ..outcome(state, &leg, before, true)
    }))
}

fn expected(case: &Json) -> Outcome {
    let want = case.get("expected");
    Outcome {
        evaluated: want.get("evaluated").boolean() == Some(true),
        reason: match want.get("reason").text() {
            Some("sl") => Some("sl"),
            Some("target") => Some("target"),
            _ => None,
        },
        current_sl: want.get("current_sl").number(),
        highest_price: want.get("highest_price").number(),
        lowest_price: want.get("lowest_price").number(),
        stop_moved: want.get("stop_moved").boolean() == Some(true),
        trail_armed: want.get("trail_armed").boolean() == Some(true),
    }
}

/// Equal within the vectors' tolerance.
fn same(got: &Outcome, want: &Outcome) -> bool {
    let close = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-6,
        (a, b) => a == b,
    };
    got.evaluated == want.evaluated
        && got.reason == want.reason
        && close(got.current_sl, want.current_sl)
        && close(got.highest_price, want.highest_price)
        && close(got.lowest_price, want.lowest_price)
        && got.stop_moved == want.stop_moved
        && got.trail_armed == want.trail_armed
}

fn is_refused(case: &Json) -> bool {
    REFUSED.iter().any(|(refused, _)| *refused == name(case))
}

#[test]
fn the_fixture_holds_the_35_vectors() {
    let names: Vec<String> = cases().iter().map(|c| name(c).to_string()).collect();
    assert_eq!(names.len(), 35);
    for (listed, _) in REFUSED {
        assert!(names.iter().any(|n| n == listed), "{listed}");
    }
    for (listed, _) in DEFERRED_ON_BARS {
        assert!(names.iter().any(|n| n == listed), "{listed}");
    }
}

#[test]
fn every_representable_vector_matches_on_the_tick_path() {
    let mut differences = Vec::new();
    for case in cases().iter().filter(|case| !is_refused(case)) {
        let want = expected(case);
        match on_tick(case) {
            Ok(got) if same(&got, &want) => {}
            got => differences.push(format!("{}: got {got:?}, want {want:?}", name(case))),
        }
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
}

#[test]
fn every_representable_vector_matches_on_the_bar_path_but_the_deferred_ones() {
    let mut differences = Vec::new();
    for case in cases().iter().filter(|case| !is_refused(case)) {
        if DEFERRED_ON_BARS
            .iter()
            .any(|(deferred, _)| *deferred == name(case))
        {
            continue;
        }
        let want = expected(case);
        match on_bar(case) {
            Ok(None) => {}
            Ok(Some(got)) if same(&got, &want) => {}
            got => differences.push(format!("{}: got {got:?}, want {want:?}", name(case))),
        }
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
}

#[test]
fn on_bars_a_stop_moved_by_the_bar_breaches_from_the_next_slice() {
    for (deferred, next_stop) in DEFERRED_ON_BARS {
        let Some(case) = cases().into_iter().find(|case| name(case) == deferred) else {
            panic!("{deferred} is in the fixture");
        };
        let got = on_bar(&case);
        let this_slice = got.map(|o| o.map(|o| (o.reason, o.current_sl)));
        assert_eq!(this_slice, Ok(Some((None, Some(next_stop)))), "{deferred}");
        // The same price on the next bar meets the moved stop.
        let next = seed(case.get("state")).and_then(|mut leg| {
            let price = case.get("ltp").number().unwrap_or(f64::NAN);
            let bar = BarPrices {
                open: price,
                high: price,
                low: price,
                close: price,
            };
            leg.ratchet(&bar)?;
            Ok(leg.test_bar(&bar))
        });
        let first = next.map(|hits| hits.first().map(|hit| (hit.kind, hit.at)));
        assert_eq!(
            first,
            Ok(Some((LevelKind::Stop, FillAt::Open))),
            "{deferred}"
        );
    }
}

#[test]
fn the_refused_vectors_are_refused_with_a_reason() {
    for case in cases().iter().filter(|case| is_refused(case)) {
        let refused = seed(case.get("state"));
        assert!(
            matches!(
                refused,
                Err(RmsError::InvalidRule { .. } | RmsError::InvalidValue { .. })
            ),
            "{}: {refused:?}",
            name(case)
        );
    }
}
