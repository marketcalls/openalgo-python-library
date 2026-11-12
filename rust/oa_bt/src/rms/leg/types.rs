//! The plain data a leg reports: the levels in force, the targets, the trail and
//! breakeven states, a level hit, and what a ratchet or a tick changed.

use crate::exec::fill::FillAt;
use crate::model::Side;

use super::super::vocabulary::RiskEvent;

/// The prices of one base bar, rupees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarPrices {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

impl BarPrices {
    /// The bar's best price for a leg on `side`: the high for a long, the low for a
    /// short.
    pub fn favourable(&self, side: Side) -> f64 {
        match side {
            Side::Buy => self.high,
            Side::Sell => self.low,
        }
    }

    /// The bar's worst price for a leg on `side`: the low for a long, the high for a
    /// short.
    pub fn adverse(&self, side: Side) -> f64 {
        match side {
            Side::Buy => self.low,
            Side::Sell => self.high,
        }
    }
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
    /// Position in the order's `targets` (0 is the first; a single `target=` is 0).
    pub index: u16,
    /// The level, rupees.
    pub level_rupees: f64,
    /// Units it exits, rounded down to the venue quantity (SPEC-13). A working target
    /// with no units (its share rounded to zero) is skipped when its level is reached.
    pub quantity_units: f64,
    /// Working or filled.
    pub state: TargetState,
}

impl TargetLevel {
    /// True while the target can still exit units.
    pub fn is_working(&self) -> bool {
        self.state == TargetState::Working && self.quantity_units > 0.0
    }
}

/// A trail's state, as reported in the position history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrailState {
    /// Armed is latched (SPEC-12).
    pub armed: bool,
    /// Best price seen since the entry, rupees (highest for a long, lowest for a short).
    pub favourable_extreme_rupees: f64,
}

/// A breakeven move's state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakevenState {
    /// Armed is latched.
    pub armed: bool,
    /// Price at which it arms, rupees: the reference plus the trigger distance on the
    /// favourable side. A threshold, not a level, so it is not rounded to the tick.
    pub trigger_price_rupees: f64,
    /// The stop it sets, rupees: the reference plus the offset on the favourable side.
    pub level_rupees: f64,
}

/// Which level was hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelKind {
    /// The stop (fixed, trailed or moved to breakeven).
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
    /// The price before slippage: the level, the open when the level was already
    /// crossed there, or the tick's price in tick mode.
    pub price_rupees: f64,
    /// Open or level.
    pub at: FillAt,
    /// Units to exit: the whole position for the stop, the target's units for a
    /// target.
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
    /// Lot size in units: target quantities round down to whole lots (SPEC-13).
    pub lot_size_units: u32,
    /// Fractional quantities are allowed (cash equity in fractional mode), so target
    /// quantities are not rounded.
    pub fractional: bool,
}

/// What a ratchet (step 4) or a tick changed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RatchetOutcome {
    /// `leg_trail_armed` and `leg_breakeven_armed`, at most once each per leg.
    pub events: Vec<RiskEvent>,
    /// True when the stop or an armed state changed, so the position history needs a
    /// row. A new favourable extreme alone is not a change of levels.
    pub levels_moved: bool,
}

/// What one price event in tick mode did: the moves of the ratchet, then the hits.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TickOutcome {
    /// Arming events and whether the levels moved.
    pub ratchet: RatchetOutcome,
    /// The stop, or the targets the price reached, nearest first.
    pub hits: Vec<LevelHit>,
}
