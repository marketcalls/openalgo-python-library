//! GTT (good till triggered) orders: the one kind of order that rests across sessions.
//!
//! Every ordinary order is a DAY order. Anything that must survive the session on a
//! CNC or NRML position (a stop, a target, a resting entry) is a GTT: SINGLE for one
//! trigger, OCO for a stop with a target (04-execution 4.4). MIS cannot hold a GTT.
//!
//! A GTT has triggers and a child order. When a base bar reaches a trigger, or opens
//! beyond it, the GTT fires (`GttAction::Fire`) and places its child: a LIMIT beyond the
//! trigger by the segment buffer (`child_limit`). The child fills, or does not fill
//! through its bar and is cancelled (`FillChild`, `CancelChild`); either way the GTT is
//! spent. For an OCO, the leg that does not fire is cancelled with the one that does;
//! with both triggers inside one base bar the stop fires. A modification is a full
//! replacement of the triggers and quantity under the same id (`Modify`).
//!
//! The four decisions on GTTs (all reading A) are the defaults of `GttPolicy`: GTT-1 the
//! child's limit buffer, GTT-2 re-creation after 360 days, GTT-3 at most one
//! modification per base bar and only for a move of at least one tick, GTT-4 the engine
//! is assumed running at every session open. GTT-6 (the child's rounding) is
//! `child_limit`.
//!
//! The book of a run's GTTs is `gtt_book`, their events `gtt_event`; the slice sequence
//! that fires and resolves them is in `engine::gtt_fire`.

use std::fmt;

use crate::model::{Day, InstrumentKind, Product, Side, TsNanos};

use super::error::ExecError;
use super::instrument::InstrumentId;
use super::matching::TriggerDirection;
use super::order::OrderId;
use super::price::round_away_from_trader;
use super::segment::OPTIONS_STOP_LIMIT_BUFFER_PERCENT;

/// Identifier of a GTT, unique within a run, in creation order. Shown as `g1`, `g2`, ...
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GttId(pub u64);

impl fmt::Display for GttId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}", self.0)
    }
}

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
    /// A strategy's own GTT (`validity="gtt"`): an entry, an add or a resting exit.
    Order,
}

impl GttLegRole {
    /// The name used in events ("stop", "target", "order").
    pub fn code(self) -> &'static str {
        match self {
            GttLegRole::Stop => "stop",
            GttLegRole::Target => "target",
            GttLegRole::Order => "order",
        }
    }
}

/// One trigger and the limit price of the child order it places.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GttLeg {
    /// What the leg is for.
    pub role: GttLegRole,
    /// Trigger price in rupees, on the tick grid.
    pub trigger_rupees: f64,
    /// From which side the price reaches the trigger.
    pub direction: TriggerDirection,
    /// Limit price of the child order in rupees, on the tick grid.
    pub child_limit_rupees: f64,
}

/// What a protective GTT holds of its position's levels (decision GTT-11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GttSlot {
    /// The target with this index, with a stop leg for the target's quantity when the
    /// position has a stop.
    Target(u16),
    /// The stop for the units no working target covers.
    Remainder,
}

/// Why a GTT exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GttPurpose {
    /// Placed by the engine to hold a CNC or NRML position's stop and target.
    Protect(GttSlot),
    /// Placed by the strategy with `validity="gtt"`; the order in the book it belongs to.
    Order(OrderId),
}

/// Where a GTT stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GttStatus {
    /// Resting: its triggers are tested on every base bar.
    Active = 0,
    /// A trigger fired and the child order was placed.
    Triggered = 1,
    /// The child order filled. Terminal.
    ChildFilled = 2,
    /// The child order did not fill through its bar and was cancelled; the GTT is
    /// spent. Terminal.
    ChildCanceled = 3,
    /// Cancelled by the strategy or the engine (an engine exit, a flat position).
    /// Terminal.
    Canceled = 4,
    /// Older than `GttPolicy::max_age_days`: replaced by a new GTT (GTT-2). Terminal.
    Expired = 5,
}

impl GttStatus {
    /// Every state, in code order.
    pub const ALL: [GttStatus; 6] = [
        GttStatus::Active,
        GttStatus::Triggered,
        GttStatus::ChildFilled,
        GttStatus::ChildCanceled,
        GttStatus::Canceled,
        GttStatus::Expired,
    ];

    /// The state name used in results.
    pub fn code(self) -> &'static str {
        match self {
            GttStatus::Active => "Active",
            GttStatus::Triggered => "Triggered",
            GttStatus::ChildFilled => "ChildFilled",
            GttStatus::ChildCanceled => "ChildCanceled",
            GttStatus::Canceled => "Canceled",
            GttStatus::Expired => "Expired",
        }
    }

    /// True when no action can move the GTT any more.
    pub fn is_terminal(self) -> bool {
        !matches!(self, GttStatus::Active | GttStatus::Triggered)
    }
}

/// What happens to a GTT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GttAction {
    /// Replace the triggers and the quantity (a trail, a breakeven, an add).
    Modify = 0,
    /// A base bar reached a trigger: place the child order.
    Fire = 1,
    /// The child order filled.
    FillChild = 2,
    /// The child order did not fill through its bar and was cancelled.
    CancelChild = 3,
    /// Cancel the GTT.
    Cancel = 4,
    /// The GTT is too old: it is replaced by a new one (GTT-2).
    Expire = 5,
}

impl GttAction {
    /// Every action, in code order.
    pub const ALL: [GttAction; 6] = [
        GttAction::Modify,
        GttAction::Fire,
        GttAction::FillChild,
        GttAction::CancelChild,
        GttAction::Cancel,
        GttAction::Expire,
    ];

    /// The action name used in messages.
    pub fn code(self) -> &'static str {
        match self {
            GttAction::Modify => "Modify",
            GttAction::Fire => "Fire",
            GttAction::FillChild => "FillChild",
            GttAction::CancelChild => "CancelChild",
            GttAction::Cancel => "Cancel",
            GttAction::Expire => "Expire",
        }
    }
}

use GttAction as A;
use GttStatus as S;

/// Every legal GTT transition as (from, action, to). Anything not listed is illegal.
pub const GTT_TRANSITIONS: [(GttStatus, GttAction, GttStatus); 6] = [
    (S::Active, A::Modify, S::Active),
    (S::Active, A::Fire, S::Triggered),
    (S::Active, A::Cancel, S::Canceled),
    (S::Active, A::Expire, S::Expired),
    (S::Triggered, A::FillChild, S::ChildFilled),
    (S::Triggered, A::CancelChild, S::ChildCanceled),
];

/// The state a GTT moves to. Errors: `IllegalGttTransition`.
pub fn next_gtt_status(from: GttStatus, action: GttAction) -> Result<GttStatus, ExecError> {
    GTT_TRANSITIONS
        .iter()
        .find(|(state, taken, _)| *state == from && *taken == action)
        .map(|(_, _, to)| *to)
        .ok_or(ExecError::IllegalGttTransition { from, action })
}

/// The decided GTT rules (GTT-1 to GTT-3), as values.
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

/// The child's buffer beyond the trigger for cash equity and futures, percent (GTT-1).
pub const CASH_AND_FUTURES_CHILD_BUFFER_PERCENT: f64 = 1.0;

/// Age in calendar days beyond which a GTT is re-created (GTT-2): under the broker's
/// usual one-year expiry.
pub const GTT_MAX_AGE_DAYS: Day = 360;

impl Default for GttPolicy {
    /// The values decided as GTT-1 to GTT-3.
    fn default() -> GttPolicy {
        GttPolicy {
            buffer_cash_and_futures_percent: CASH_AND_FUTURES_CHILD_BUFFER_PERCENT,
            buffer_options_percent: OPTIONS_STOP_LIMIT_BUFFER_PERCENT,
            max_age_days: GTT_MAX_AGE_DAYS,
            min_modify_ticks: 1,
        }
    }
}

impl GttPolicy {
    /// The child's buffer beyond the trigger for an instrument kind, percent (GTT-1).
    pub fn child_buffer_percent(&self, kind: InstrumentKind) -> f64 {
        match kind {
            InstrumentKind::Option => self.buffer_options_percent,
            _ => self.buffer_cash_and_futures_percent,
        }
    }

    /// True when a GTT created on `created_on` is older than the policy allows on
    /// `today` and must be re-created (GTT-2). Days are IST session dates.
    pub fn is_due_for_renewal(&self, created_on: Day, today: Day) -> bool {
        today - created_on > self.max_age_days
    }
}

/// The limit price of a GTT child order: beyond the trigger by `buffer_percent` in the
/// direction of the child's side (above for a buy, below for a sell), computed from the
/// trigger as rounded to the tick and rounded to the tick away from the trader, so the
/// buffer is never smaller than asked. A target's child sits on the far side of its
/// trigger like a stop's (decision GTT-6).
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
    /// The product of the position it protects or trades.
    pub product: Product,
    /// Side of the child order.
    pub side: Side,
    /// Quantity of the child order in units.
    pub quantity_units: f64,
    /// SINGLE or OCO.
    pub kind: GttKind,
    /// One leg for SINGLE, the stop then the target for OCO.
    pub legs: Vec<GttLeg>,
    /// Why it exists.
    pub purpose: GttPurpose,
    /// Where it stands.
    pub status: GttStatus,
    /// IST date it was created, for the age rule (GTT-2).
    pub created_on: Day,
    /// It is tested on base bars that open at or after this instant (UTC nanoseconds).
    pub eligible_from: TsNanos,
    /// Grid slice of its last modification, for the once-per-base-bar rule (GTT-3).
    pub last_modified_slice: Option<u32>,
    /// The leg that fired, once it has.
    pub fired: Option<GttLegRole>,
}

impl Gtt {
    /// The leg with a role, if the GTT has one.
    pub fn leg(&self, role: GttLegRole) -> Option<&GttLeg> {
        self.legs.iter().find(|leg| leg.role == role)
    }

    /// The trigger of a leg, rupees.
    pub fn trigger(&self, role: GttLegRole) -> Option<f64> {
        self.leg(role).map(|leg| leg.trigger_rupees)
    }

    /// True while its triggers are tested.
    pub fn is_active(&self) -> bool {
        self.status == GttStatus::Active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The legal transitions, written out from 04-execution 4.4: placed (Active), then
    /// modified any number of times, then cancelled, expired, or fired with its child
    /// filled or cancelled.
    const LEGAL: [(GttStatus, GttAction, GttStatus); 6] = [
        (S::Active, A::Modify, S::Active),
        (S::Active, A::Fire, S::Triggered),
        (S::Active, A::Cancel, S::Canceled),
        (S::Active, A::Expire, S::Expired),
        (S::Triggered, A::FillChild, S::ChildFilled),
        (S::Triggered, A::CancelChild, S::ChildCanceled),
    ];

    #[test]
    fn every_legal_transition_moves_the_gtt_as_documented() {
        for (from, action, to) in LEGAL {
            assert_eq!(
                next_gtt_status(from, action),
                Ok(to),
                "{} --{}-->",
                from.code(),
                action.code()
            );
        }
    }

    #[test]
    fn every_other_transition_is_illegal() {
        for from in S::ALL {
            for action in A::ALL {
                if LEGAL.iter().any(|(s, a, _)| *s == from && *a == action) {
                    continue;
                }
                assert_eq!(
                    next_gtt_status(from, action),
                    Err(ExecError::IllegalGttTransition { from, action }),
                    "{} --{}-->",
                    from.code(),
                    action.code()
                );
            }
        }
    }

    #[test]
    fn terminal_states_are_those_after_the_child_or_the_cancel() {
        let terminal: Vec<&str> = S::ALL
            .into_iter()
            .filter(|s| s.is_terminal())
            .map(GttStatus::code)
            .collect();
        assert_eq!(
            terminal,
            ["ChildFilled", "ChildCanceled", "Canceled", "Expired"]
        );
    }

    #[test]
    fn the_child_limit_lies_beyond_the_trigger_by_the_buffer() {
        // A long's protective stop sells: 1 percent below 1460.00 is 1445.40.
        assert_eq!(child_limit(1460.0, Side::Sell, 1.0, 0.05), Ok(1445.4));
        // 1 percent above 101.03 is 102.0403, away from the buyer: 102.05.
        assert_eq!(child_limit(101.03, Side::Buy, 1.0, 0.05), Ok(102.05));
        assert!(child_limit(100.0, Side::Buy, -1.0, 0.05).is_err());
    }

    #[test]
    fn the_child_limit_rounds_away_from_the_trader() {
        // Trace 06b: 1473.90 x 0.99 = 1459.161; down for a sell is 1459.15 (GTT-6).
        assert_eq!(child_limit(1473.9, Side::Sell, 1.0, 0.05), Ok(1459.15));
    }

    #[test]
    fn the_buffer_is_one_percent_for_cash_and_futures_and_five_for_options() {
        let policy = GttPolicy::default();
        assert_eq!(policy.child_buffer_percent(InstrumentKind::Equity), 1.0);
        assert_eq!(policy.child_buffer_percent(InstrumentKind::Future), 1.0);
        assert_eq!(policy.child_buffer_percent(InstrumentKind::Option), 5.0);
    }

    #[test]
    fn a_gtt_is_renewed_once_it_is_older_than_360_days() {
        let policy = GttPolicy::default();
        assert!(!policy.is_due_for_renewal(100, 460));
        assert!(policy.is_due_for_renewal(100, 461));
        assert_eq!(policy.min_modify_ticks, 1);
    }

    #[test]
    fn gtt_ids_print_as_in_results() {
        assert_eq!(GttId(2).to_string(), "g2");
    }
}
