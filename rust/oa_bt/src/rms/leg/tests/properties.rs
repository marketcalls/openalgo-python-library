//! Properties of a leg over many generated runs: random rules, random bars, adds,
//! target fills and partial exits, driven the way the engine drives a leg (step 2
//! adds, step 3 hits, step 4 ratchets). After every call:
//!
//! - the stop never loosens and never disappears while the position is open;
//! - the stop covers the position, working targets never exceed it, and the remainder
//!   is the position less the working targets;
//! - every level sits on the tick grid.
//!
//! The generator is a fixed-seed xorshift, so a failure names a seed that reproduces.

use crate::exec::price::is_on_tick;
use crate::model::Side;
use crate::rms::error::RmsError;
use crate::rms::leg::{BarPrices, LegState, LevelHit, LevelKind};
use crate::rms::rules::{AfterFirstTarget, Distance, LegRules, TargetSpec};

use super::support::{breakeven, context, trail};

const RUNS: u64 = 400;
const BARS_PER_RUN: usize = 60;
const TICK: f64 = 0.05;

/// xorshift64: deterministic, good enough to spread test inputs.
struct Generator(u64);

impl Generator {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A whole number in `0..below`.
    fn below(&mut self, below: u64) -> u64 {
        self.next() % below
    }

    /// A price step: a whole number of ticks in `-span..=span`.
    fn ticks(&mut self, span: i64) -> f64 {
        let count = (self.below((2 * span + 1) as u64) as i64) - span;
        count as f64 * TICK
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

fn rules(gen: &mut Generator) -> LegRules {
    let distance = |gen: &mut Generator| Distance::Points(1.0 + gen.below(40) as f64 * 0.25);
    let mut rules = LegRules::default();
    if gen.chance(80) {
        rules.stop = Some(distance(gen));
    }
    match gen.below(3) {
        0 => rules.target = Some(distance(gen)),
        1 => {
            rules.targets = vec![
                TargetSpec {
                    distance: distance(gen),
                    fraction: 0.5,
                },
                TargetSpec {
                    distance: distance(gen),
                    fraction: 0.3,
                },
            ]
        }
        _ => {}
    }
    if gen.chance(60) {
        let step = if gen.chance(50) { 0.0 } else { 0.5 };
        rules.trail = Some(trail(1.0 + gen.below(12) as f64 * 0.5, step));
    }
    if gen.chance(40) {
        rules.breakeven = Some(breakeven(1.0 + gen.below(8) as f64, 0.25));
    }
    if gen.chance(30) {
        rules.after_first_target = AfterFirstTarget::Breakeven;
    }
    rules
}

fn next_bar(gen: &mut Generator, close: f64) -> BarPrices {
    let open = (close + gen.ticks(20)).max(10.0);
    let a = (open + gen.ticks(40)).max(5.0);
    let b = (open + gen.ticks(40)).max(5.0);
    let high = open.max(a).max(b);
    let low = open.min(a).min(b);
    let close = if gen.chance(50) { high } else { low };
    BarPrices {
        open,
        high,
        low,
        close,
    }
}

/// What a step must never do to the stop: loosen it or drop it.
fn check_stop(before: Option<f64>, leg: &LegState, seed: u64, problems: &mut Vec<String>) {
    let after = leg.stop.map(|stop| stop.level_rupees);
    let loosened = match (before, after, leg.context.side) {
        (Some(_), None, _) => true,
        (Some(b), Some(a), Side::Buy) => a < b - 1e-9,
        (Some(b), Some(a), Side::Sell) => a > b + 1e-9,
        _ => false,
    };
    if loosened {
        problems.push(format!(
            "seed {seed}: stop loosened from {before:?} to {after:?}"
        ));
    }
}

fn check_quantities(leg: &LegState, seed: u64, problems: &mut Vec<String>) {
    let held = leg.held_units();
    let working = leg.working_target_units();
    if leg.stop.is_some_and(|stop| stop.quantity_units != held) {
        problems.push(format!("seed {seed}: stop quantity differs from {held}"));
    }
    if working > held + 1e-9 || (leg.remainder_units - (held - working)).abs() > 1e-9 {
        problems.push(format!(
            "seed {seed}: working {working} and remainder {} against {held}",
            leg.remainder_units
        ));
    }
}

fn check_tick_grid(leg: &LegState, seed: u64, problems: &mut Vec<String>) {
    let mut levels: Vec<f64> = leg.targets.iter().map(|t| t.level_rupees).collect();
    levels.extend(leg.stop.map(|stop| stop.level_rupees));
    levels.extend(leg.breakeven.map(|state| state.level_rupees));
    for level in levels {
        if is_on_tick(level, TICK) != Ok(true) {
            problems.push(format!("seed {seed}: level {level} is off the tick grid"));
        }
    }
}

/// How often the generated runs reached each rule, so the properties are not vacuous.
#[derive(Debug, Default)]
struct Reached {
    stops: u32,
    targets: u32,
    armed: u32,
    adds: u32,
    partial_exits: u32,
}

/// Apply the hits of step 3 as the engine does. `Ok(false)` when the position closed.
fn apply_hits(
    leg: &mut LegState,
    hits: &[LevelHit],
    reached: &mut Reached,
) -> Result<bool, RmsError> {
    for hit in hits {
        let remaining = leg.held_units() - hit.quantity_units;
        let LevelKind::Target(index) = hit.kind else {
            reached.stops += 1;
            return Ok(false);
        };
        reached.targets += 1;
        if remaining <= 0.0 {
            return Ok(false);
        }
        leg.on_reduce(remaining, Some(index))?;
        leg.after_target_fill()?;
    }
    Ok(true)
}

/// Step 2 of a bar: sometimes an add at the open, sometimes a partial exit by signal.
fn step_two(
    gen: &mut Generator,
    leg: &mut LegState,
    open: f64,
    reached: &mut Reached,
) -> Result<(), RmsError> {
    if gen.chance(10) {
        let tranche = 1.0 + gen.below(50) as f64;
        let held = leg.held_units();
        let average = (leg.reference_rupees * held + open * tranche) / (held + tranche);
        reached.adds += 1;
        leg.on_add(average, tranche)
    } else if gen.chance(5) && leg.held_units() > 2.0 {
        reached.partial_exits += 1;
        leg.on_reduce((leg.held_units() / 2.0).floor(), None)
    } else {
        Ok(())
    }
}

/// One generated run, bar by bar until the position closes.
fn run(seed: u64, reached: &mut Reached) -> Result<Vec<String>, RmsError> {
    let mut gen = Generator(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let side = if gen.chance(50) {
        Side::Buy
    } else {
        Side::Sell
    };
    let mut close = 500.0 + gen.ticks(2_000);
    let held = 10.0 + gen.below(200) as f64;
    let mut leg = LegState::open(rules(&mut gen), context(side), close, held)?;
    let mut problems = Vec::new();
    for _ in 0..BARS_PER_RUN {
        let bar = next_bar(&mut gen, close);
        close = bar.close;
        let before = leg.stop.map(|stop| stop.level_rupees);
        step_two(&mut gen, &mut leg, bar.open, reached)?;
        check_stop(before, &leg, seed, &mut problems);
        let hits = leg.test_bar(&bar);
        if !apply_hits(&mut leg, &hits, reached)? {
            break;
        }
        let before = leg.stop.map(|stop| stop.level_rupees);
        let outcome = leg.ratchet(&bar)?;
        reached.armed += outcome.events.len() as u32;
        check_stop(before, &leg, seed, &mut problems);
        check_quantities(&leg, seed, &mut problems);
        check_tick_grid(&leg, seed, &mut problems);
    }
    Ok(problems)
}

#[test]
fn stops_never_loosen_quantities_stay_covered_and_levels_sit_on_the_tick_grid() {
    let mut reached = Reached::default();
    let mut problems = Vec::new();
    for seed in 1..=RUNS {
        match run(seed, &mut reached) {
            Ok(found) => problems.extend(found),
            Err(error) => problems.push(format!("seed {seed}: {error}")),
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    let every_rule = [
        reached.stops,
        reached.targets,
        reached.armed,
        reached.adds,
        reached.partial_exits,
    ];
    assert!(every_rule.iter().all(|&count| count > 20), "{reached:?}");
}
