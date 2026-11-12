//! The price-level rules of one leg (one position): its stop, targets, trail and
//! breakeven, and the places the slice sequence touches them.
//!
//! | Call | Step of 03-architecture 4.2 | What it does |
//! |---|---|---|
//! | `LegState::open` | 2, after an entry fill | Levels from the fill average; the favourable extreme seeded from it (SPEC-01) |
//! | `LegState::on_add` | 2, after an add | Stop and unfilled targets recomputed from the new average, in force for step 3 of the same slice (SPEC-08 to SPEC-12) |
//! | `LegState::test_bar` | 3 | Levels against the base bar: a level crossed at the open fills at the open; otherwise the stop beats a target inside the range; every target reached fills, nearest first (SPEC-28, SPEC-33). Legs opened in step 2 are tested (SPEC-02) |
//! | `LegState::ratchet` | 4 | Trail and breakeven from this bar's extremes, in force from the next slice; the entry bar's extreme counts (SPEC-01) |
//! | `LegState::on_reduce` | 2 or 3, after a partial exit | Stop quantity shrinks to the position |
//! | `LegState::shift` | 2, futures roll | Levels, extreme and reference shifted by the roll spread (SPEC-19, SPEC-20) |
//! | `LegState::on_tick` | tick mode | Best price, trail, stop then target on one price: the order of the live module and of the server's 35 risk vectors |
//!
//! Built: a leg with no rules (every call is a no-op that keeps the quantities). Not
//! built yet: every rule; `open` refuses rules with `RmsError::NotBuilt`.

use crate::exec::fill::FillAt;
use crate::model::Side;

use super::error::RmsError;
use super::rules::LegRules;
use super::vocabulary::RiskEvent;

/// The prices of one base bar, rupees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarPrices {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

/// A level in force and the quantity it would exit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelInForce {
    /// The level, rupees, on the tick grid unless `round_levels` is off.
    pub level_rupees: f64,
    /// Units the level exits when it is hit.
    pub quantity_units: f64,
}

/// Whether a target is still working.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetState {
    /// Waiting for its level.
    Working,
    /// Filled; never re-armed (12-position-sizing 3).
    Filled,
}

/// One target of a leg.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetLevel {
    /// Position in the order's `targets` (0 is the first).
    pub index: u16,
    /// The level, rupees.
    pub level_rupees: f64,
    /// Units it exits (rounded down to the venue quantity, SPEC-13).
    pub quantity_units: f64,
    /// Working or filled.
    pub state: TargetState,
}

/// A trail's state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrailState {
    /// Armed is latched (SPEC-12).
    pub armed: bool,
    /// Best price seen since the entry, rupees (highest for a long).
    pub favourable_extreme_rupees: f64,
}

/// A breakeven move's state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakevenState {
    /// Armed is latched.
    pub armed: bool,
    /// Price at which it arms, rupees.
    pub trigger_price_rupees: f64,
    /// The stop it sets, rupees.
    pub level_rupees: f64,
}

/// Which level was hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelKind {
    /// The stop (fixed, trailed or breakeven).
    Stop,
    /// A target, by index.
    Target(u16),
}

/// A level hit in step 3 or on a tick: the engine turns it into an exit fill.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelHit {
    /// Which level.
    pub kind: LevelKind,
    /// The level in force, rupees (reported in the risk event).
    pub level_rupees: f64,
    /// The price before slippage: the level, or the open on a gap.
    pub price_rupees: f64,
    /// Open or level.
    pub at: FillAt,
    /// Units to exit.
    pub quantity_units: f64,
}

/// Facts about the leg the rules need.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LegContext {
    /// `Buy` for a long leg, `Sell` for a short one.
    pub side: Side,
    /// Tick size, rupees.
    pub tick_rupees: f64,
    /// Rupees per one-rupee move per unit (for `Distance::Rupees`).
    pub multiplier: f64,
    /// Round levels to the tick (SPEC-03).
    pub round_levels: bool,
}

/// The levels of one leg.
#[derive(Clone, Debug, PartialEq)]
pub struct LegState {
    /// The rules of the opening order (SPEC-14).
    pub rules: LegRules,
    /// Side, tick and rounding.
    pub context: LegContext,
    /// The reference price of every level: the average price (shifted by a roll).
    pub reference_rupees: f64,
    /// The stop in force.
    pub stop: Option<LevelInForce>,
    /// The targets.
    pub targets: Vec<TargetLevel>,
    /// Units assigned to no target (they leave by the stop, the trail or a signal).
    pub remainder_units: f64,
    /// The trail.
    pub trail: Option<TrailState>,
    /// The breakeven move.
    pub breakeven: Option<BreakevenState>,
}

impl LegState {
    /// The leg a fill opened, with levels from its fill average (step 2).
    ///
    /// Errors: `NotBuilt` when any rule is set.
    pub fn open(
        rules: LegRules,
        context: LegContext,
        average_price_rupees: f64,
        quantity_units: f64,
    ) -> Result<LegState, RmsError> {
        if !rules.is_empty() {
            return Err(not_built());
        }
        Ok(LegState {
            rules,
            context,
            reference_rupees: average_price_rupees,
            stop: None,
            targets: Vec::new(),
            remainder_units: quantity_units,
            trail: None,
            breakeven: None,
        })
    }

    /// True when the leg has no rule, so every call below is a no-op.
    pub fn is_plain(&self) -> bool {
        self.rules.is_empty()
    }

    fn plain(&self) -> Result<(), RmsError> {
        if self.is_plain() {
            Ok(())
        } else {
            Err(not_built())
        }
    }

    /// Test the levels in force against a base bar (step 3).
    pub fn test_bar(&self, _bar: &BarPrices) -> Result<Vec<LevelHit>, RmsError> {
        self.plain().map(|()| Vec::new())
    }

    /// Update trail and breakeven from a bar's extremes (step 4); the events of any
    /// arming are returned.
    pub fn ratchet(&mut self, _bar: &BarPrices) -> Result<Vec<RiskEvent>, RmsError> {
        self.plain().map(|()| Vec::new())
    }

    /// An add filled: the new average price and the units the tranche added.
    pub fn on_add(
        &mut self,
        average_price_rupees: f64,
        tranche_units: f64,
    ) -> Result<(), RmsError> {
        self.plain()?;
        self.reference_rupees = average_price_rupees;
        self.remainder_units += tranche_units;
        Ok(())
    }

    /// The position was reduced to `remaining_units` (positive).
    pub fn on_reduce(&mut self, remaining_units: f64) -> Result<(), RmsError> {
        self.plain()?;
        self.remainder_units = remaining_units;
        Ok(())
    }

    /// The position was rolled: shift every level, the extreme and the reference by
    /// `spread_rupees` (next contract's open minus the near contract's open).
    pub fn shift(&mut self, spread_rupees: f64) -> Result<(), RmsError> {
        self.plain()?;
        self.reference_rupees += spread_rupees;
        Ok(())
    }

    /// One price event in tick mode: update, then test the stop, then the target.
    pub fn on_tick(&mut self, _last_price_rupees: f64) -> Result<Option<LevelHit>, RmsError> {
        self.plain().map(|()| None)
    }
}

fn not_built() -> RmsError {
    RmsError::NotBuilt {
        feature: "leg stops, targets, trailing stops and breakeven",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rms::rules::Distance;

    fn context() -> LegContext {
        LegContext {
            side: Side::Buy,
            tick_rupees: 0.05,
            multiplier: 1.0,
            round_levels: true,
        }
    }

    #[test]
    fn a_plain_leg_keeps_its_whole_quantity_in_the_remainder() {
        let mut leg = LegState::open(LegRules::default(), context(), 500.0, 100.0);
        let bar = BarPrices {
            open: 500.0,
            high: 501.0,
            low: 499.0,
            close: 500.5,
        };
        assert_eq!(leg.as_ref().map(|leg| leg.remainder_units), Ok(100.0));
        if let Ok(leg) = leg.as_mut() {
            assert_eq!(leg.test_bar(&bar), Ok(Vec::new()));
            assert_eq!(leg.on_add(501.0, 100.0), Ok(()));
            assert_eq!(leg.remainder_units, 200.0);
        }
    }

    #[test]
    fn rules_are_not_built_yet() {
        let rules = LegRules {
            stop: Some(Distance::Points(4.0)),
            ..LegRules::default()
        };
        let error = LegState::open(rules, context(), 800.5, 100.0);
        assert!(matches!(error, Err(RmsError::NotBuilt { .. })));
    }
}
