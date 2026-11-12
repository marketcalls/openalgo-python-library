//! GTT (good till triggered) orders: the one kind of order that rests across sessions.
//!
//! Every ordinary order is a DAY order. Anything that must survive the session on a
//! CNC or NRML position (a stop, a target, a resting entry) is a GTT: SINGLE for one
//! trigger, OCO for a stop with a target (04-execution 4.4). MIS cannot hold a GTT.
//!
//! A GTT has triggers and a child order. When a base bar reaches a trigger, or opens
//! beyond it, the GTT fires (`GttAction::Fire`) and places its child: a DAY LIMIT order
//! beyond the trigger by `GttPolicy` buffer. The child fills only if the price trades
//! through its limit after the trigger. When it has not filled through one base bar
//! (a gap beyond the limit), the engine raises `gtt_child_unfilled`, cancels the child
//! first and only after the cancel exits at the next base-bar open, so the leg never
//! has two exit owners (lead ruling). For an OCO, when one leg fires the other is
//! cancelled with it; with both triggers inside one base bar the stop fires.
//!
//! The four decisions on GTTs (all reading A) are the defaults of `GttPolicy`:
//! GTT-1 the child's limit buffer, GTT-2 re-creation after 360 days, GTT-3 at most one
//! modification per base bar and only for a move of at least one tick, GTT-4 the
//! engine is assumed running at every session open (so the follow-up exit of an
//! unfilled child happens at the next base-bar open).
//!
//! Built here: the types, the transition table and the child-order price. Not built
//! yet: firing, OCO cancellation and the follow-up exit in the slice sequence.

use crate::model::{Day, Side};

use super::error::ExecError;
use super::instrument::InstrumentId;
use super::order::OrderId;
use super::price::round_away_from_trader;

/// Identifier of a GTT, unique within a run, in creation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GttId(pub u64);

/// One trigger, or a stop and a target where the first to fire cancels the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GttKind {
    /// One trigger.
    Single,
    /// One cancels the other: a stop leg and a target leg.
    Oco,
}

/// What a GTT leg protects or opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GttLegRole {
    /// A protective stop on a carried position.
    Stop,
    /// A profit target on a carried position.
    Target,
    /// A resting entry (`validity="gtt"` on buy or sell).
    Entry,
}

/// One trigger and the limit price of the child order it places.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GttLeg {
    /// What the leg is for.
    pub role: GttLegRole,
    /// Trigger price in rupees, on the tick grid.
    pub trigger_rupees: f64,
    /// Limit price of the child order in rupees, on the tick grid (`child_limit`).
    pub child_limit_rupees: f64,
}

/// Where a GTT stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GttStatus {
    /// Resting: its triggers are tested on every base bar.
    Active = 0,
    /// A trigger fired and the child order was placed. Terminal for the GTT; the child
    /// lives on in the order book.
    Triggered = 1,
    /// Cancelled by the strategy or the engine (the other leg of an OCO, a flat
    /// position). Terminal.
    Canceled = 2,
    /// Reached `GttPolicy::max_age_days` and was replaced by a new GTT (GTT-2). Terminal.
    Expired = 3,
    /// Replaced by a modification: a modify is a full replacement, never a patch, and
    /// takes effect from the next base bar. Terminal.
    Replaced = 4,
}

impl GttStatus {
    /// The state name used in events and results.
    pub fn code(self) -> &'static str {
        match self {
            GttStatus::Active => "Active",
            GttStatus::Triggered => "Triggered",
            GttStatus::Canceled => "Canceled",
            GttStatus::Expired => "Expired",
            GttStatus::Replaced => "Replaced",
        }
    }
}

/// What happens to a GTT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GttAction {
    /// A base bar reached a trigger: place the child order.
    Fire = 0,
    /// Cancel the GTT.
    Cancel = 1,
    /// The GTT is too old: re-create it (GTT-2).
    Expire = 2,
    /// Replace the GTT with a modified one.
    Replace = 3,
}

impl GttAction {
    /// The action name used in messages.
    pub fn code(self) -> &'static str {
        match self {
            GttAction::Fire => "Fire",
            GttAction::Cancel => "Cancel",
            GttAction::Expire => "Expire",
            GttAction::Replace => "Replace",
        }
    }
}

/// Every legal GTT transition as (from, action, to). Anything not listed is illegal.
pub const GTT_TRANSITIONS: [(GttStatus, GttAction, GttStatus); 4] = [
    (GttStatus::Active, GttAction::Fire, GttStatus::Triggered),
    (GttStatus::Active, GttAction::Cancel, GttStatus::Canceled),
    (GttStatus::Active, GttAction::Expire, GttStatus::Expired),
    (GttStatus::Active, GttAction::Replace, GttStatus::Replaced),
];

/// The state a GTT moves to. Errors: `IllegalGttTransition`.
pub fn next_gtt_status(from: GttStatus, action: GttAction) -> Result<GttStatus, ExecError> {
    GTT_TRANSITIONS
        .iter()
        .find(|(state, taken, _)| *state == from && *taken == action)
        .map(|(_, _, to)| *to)
        .ok_or(ExecError::IllegalGttTransition { from, action })
}

/// The decided GTT rules (GTT-1 to GTT-4), as configurable values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GttPolicy {
    /// Child limit beyond the trigger for cash equity and futures, percent (GTT-1).
    pub buffer_cash_and_futures_percent: f64,
    /// Child limit beyond the trigger for options, percent, the same as the options
    /// stop-limit (GTT-1).
    pub buffer_options_percent: f64,
    /// A GTT older than this many calendar days is re-created at the next session open,
    /// in backtest and live alike (GTT-2).
    pub max_age_days: Day,
    /// A trailing or breakeven move modifies the GTT only when the level moves by at
    /// least this many ticks, at most once per base bar (GTT-3).
    pub min_modify_ticks: u32,
}

impl Default for GttPolicy {
    /// The values decided as GTT-1 to GTT-3.
    fn default() -> GttPolicy {
        GttPolicy {
            buffer_cash_and_futures_percent: 1.0,
            buffer_options_percent: 5.0,
            max_age_days: 360,
            min_modify_ticks: 1,
        }
    }
}

/// The limit price of a GTT child order: beyond the trigger by `buffer_percent` in the
/// direction of the child's side (above for a buy, below for a sell), rounded to the
/// tick away from the trader so the buffer is never smaller than asked.
///
/// Units: rupees and percent. Errors: `InvalidPrice` for an unusable trigger or a
/// negative buffer, `InvalidTick` for an unusable tick.
pub fn child_limit(
    trigger_rupees: f64,
    child_side: Side,
    buffer_percent: f64,
    tick_rupees: f64,
) -> Result<f64, ExecError> {
    if !buffer_percent.is_finite() || buffer_percent < 0.0 {
        return Err(ExecError::InvalidPrice {
            what: "GTT buffer percent",
            value: buffer_percent,
        });
    }
    let factor = 1.0 + child_side.sign() * buffer_percent / 100.0;
    round_away_from_trader(trigger_rupees * factor, tick_rupees, child_side)
}

/// A GTT in the book.
#[derive(Clone, Debug, PartialEq)]
pub struct Gtt {
    /// Its identifier.
    pub id: GttId,
    /// The instrument it trades.
    pub instrument: InstrumentId,
    /// Side of the child order.
    pub side: Side,
    /// Quantity of the child order in units; for a protective GTT it always equals the
    /// position quantity.
    pub quantity_units: f64,
    /// SINGLE or OCO.
    pub kind: GttKind,
    /// One leg for SINGLE, the stop then the target for OCO.
    pub legs: Vec<GttLeg>,
    /// Where it stands.
    pub status: GttStatus,
    /// IST date it was created, for the age rule (GTT-2).
    pub created_on: Day,
    /// Grid slice of its last modification, for the once-per-base-bar rule (GTT-3).
    pub last_modified_slice: Option<u32>,
    /// The child order, once the GTT has fired.
    pub child: Option<OrderId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_child_limit_lies_beyond_the_trigger_by_the_buffer() {
        // A long's protective stop sells: 1 percent below 1460.00 is 1445.40.
        assert_eq!(child_limit(1460.0, Side::Sell, 1.0, 0.05), Ok(1445.4));
        // 1 percent above 101.03 is 102.0403, away from the buyer: 102.05.
        assert_eq!(child_limit(101.03, Side::Buy, 1.0, 0.05), Ok(102.05));
        assert!(child_limit(100.0, Side::Buy, -1.0, 0.05).is_err());
    }

    #[test]
    fn only_an_active_gtt_moves() {
        assert_eq!(
            next_gtt_status(GttStatus::Active, GttAction::Fire),
            Ok(GttStatus::Triggered)
        );
        assert!(next_gtt_status(GttStatus::Triggered, GttAction::Cancel).is_err());
    }

    #[test]
    fn the_policy_defaults_are_the_decided_values() {
        let policy = GttPolicy::default();
        assert_eq!(policy.buffer_cash_and_futures_percent, 1.0);
        assert_eq!(policy.buffer_options_percent, 5.0);
        assert_eq!(policy.max_age_days, 360);
        assert_eq!(policy.min_modify_ticks, 1);
    }
}
