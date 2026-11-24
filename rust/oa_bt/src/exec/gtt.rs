//! GTT (good till triggered) orders: the one kind of order that rests across sessions.
//!
//! Every ordinary order is a DAY order. Anything that must survive the session on a
//! CNC or NRML position (its stop, a resting entry) is a GTT with one trigger, a SINGLE
//! GTT (04-execution 4.4). A position's targets are held by the engine on the bars of
//! every session, as an MIS leg's are: no OCO GTT is used, since OpenAlgo's OCO behaves
//! differently from one broker to the next (design 20-positional-live P6). MIS cannot
//! hold a GTT.
//!
//! A GTT has a trigger and a child order. When a base bar reaches the trigger, or opens
//! beyond it, the GTT fires (`GttAction::Fire`) and places its child: a LIMIT beyond the
//! trigger by the segment buffer (`child_limit`). The child fills, or does not fill
//! through its bar and is cancelled (`FillChild`, `CancelChild`); either way the GTT is
//! spent. A modification is a full replacement of the trigger and quantity under the
//! same id (`Modify`).
//!
//! A live session's GTT is held by the broker, so it has the states of a request too
//! (design 20-positional-live 3.1, unit N1): `Requested` until the venue confirms the
//! place (`Active`) or refuses it (`Rejected`), `ModifyPending` and `CancelPending` while
//! a modification or cancellation waits for the venue's answer (the old terms rest until
//! then), and `VenueEnded` for a GTT that ended at the venue without the session asking
//! (expired, rejected after placement, cancelled outside the session). A backtest never
//! takes those states. `GTT_TRANSITIONS` lists every legal move.
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
use super::price::limit_beyond;
use super::segment::OPTIONS_STOP_LIMIT_BUFFER_PERCENT;

/// Identifier of a GTT, unique within a run, in creation order. Shown as `g1`, `g2`, ...
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GttId(pub u64);

impl fmt::Display for GttId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}", self.0)
    }
}

/// What a GTT's trigger is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GttLegRole {
    /// The stop of a carried position, placed by the engine.
    Stop,
    /// A strategy's own GTT (`validity="gtt"`): an entry, an add or a resting exit.
    Order,
}

impl GttLegRole {
    /// The name used in events ("stop", "order").
    pub fn code(self) -> &'static str {
        match self {
            GttLegRole::Stop => "stop",
            GttLegRole::Order => "order",
        }
    }
}

/// The trigger of a GTT and the limit price of the child order it places.
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

/// Why a GTT exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GttPurpose {
    /// Placed by the engine to hold a CNC or NRML position's stop, for all its open units.
    Protect,
    /// Placed by the strategy with `validity="gtt"`; the order in the book it belongs to.
    Order(OrderId),
}

/// Where a GTT stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GttStatus {
    /// Resting: its triggers are tested on every base bar (in a live session, at the
    /// venue, its place or last modification confirmed).
    Active = 0,
    /// A trigger fired and the child order was placed.
    Triggered = 1,
    /// The child order filled. Terminal.
    ChildFilled = 2,
    /// The child order did not fill through its bar and was cancelled; the GTT is
    /// spent. Terminal.
    ChildCanceled = 3,
    /// Cancelled by the strategy or the engine (an engine exit, a flat position); in a
    /// live session, its cancellation confirmed by the venue. Terminal.
    Canceled = 4,
    /// Older than `GttPolicy::max_age_days`: replaced by a new GTT (GTT-2). Terminal.
    Expired = 5,
    /// A live session sent its place and the venue has not answered: not in force yet.
    Requested = 6,
    /// A live session sent a modification and the venue has not answered: the old terms
    /// rest until it does.
    ModifyPending = 7,
    /// A live session sent a cancellation and the venue has not answered: it rests until
    /// the venue confirms it.
    CancelPending = 8,
    /// The venue refused its place. Terminal.
    Rejected = 9,
    /// It ended at the venue without the session asking: expired, rejected after its
    /// placement, or cancelled outside the session. Terminal.
    VenueEnded = 10,
}

impl GttStatus {
    /// Every state, in code order.
    pub const ALL: [GttStatus; 11] = [
        GttStatus::Active,
        GttStatus::Triggered,
        GttStatus::ChildFilled,
        GttStatus::ChildCanceled,
        GttStatus::Canceled,
        GttStatus::Expired,
        GttStatus::Requested,
        GttStatus::ModifyPending,
        GttStatus::CancelPending,
        GttStatus::Rejected,
        GttStatus::VenueEnded,
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
            GttStatus::Requested => "Requested",
            GttStatus::ModifyPending => "ModifyPending",
            GttStatus::CancelPending => "CancelPending",
            GttStatus::Rejected => "Rejected",
            GttStatus::VenueEnded => "VenueEnded",
        }
    }

    /// True while its trigger rests on known terms: active, or with a modification or
    /// cancellation waiting for the venue's answer (the confirmed terms rest until then).
    pub fn is_active(self) -> bool {
        matches!(
            self,
            GttStatus::Active | GttStatus::ModifyPending | GttStatus::CancelPending
        )
    }

    /// True while it is at the venue or asked for there: resting (`is_active`), or its
    /// place not answered yet.
    pub fn at_venue(self) -> bool {
        self.is_active() || self == GttStatus::Requested
    }

    /// True when no action can move the GTT any more.
    pub fn is_terminal(self) -> bool {
        !(self.at_venue() || self == GttStatus::Triggered)
    }
}

/// What happens to a GTT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GttAction {
    /// Replace the triggers and the quantity (a trail, a breakeven, an add).
    Modify = 0,
    /// A base bar reached a trigger (in a live session, the venue fired it): place the
    /// child order.
    Fire = 1,
    /// The child order filled.
    FillChild = 2,
    /// The child order did not fill through its bar and was cancelled.
    CancelChild = 3,
    /// Cancel the GTT.
    Cancel = 4,
    /// The GTT is too old: it is replaced by a new one (GTT-2).
    Expire = 5,
    /// The venue confirmed its place.
    Confirm = 6,
    /// The venue refused its place.
    Refuse = 7,
    /// A live session sends a modification.
    AskModify = 8,
    /// The venue confirmed the modification.
    ConfirmModify = 9,
    /// The venue refused the modification.
    RefuseModify = 10,
    /// A live session sends a cancellation.
    AskCancel = 11,
    /// The venue confirmed the cancellation.
    ConfirmCancel = 12,
    /// The venue refused the cancellation.
    RefuseCancel = 13,
    /// It ended at the venue without the session asking.
    VenueEnd = 14,
}

impl GttAction {
    /// Every action, in code order.
    pub const ALL: [GttAction; 15] = [
        GttAction::Modify,
        GttAction::Fire,
        GttAction::FillChild,
        GttAction::CancelChild,
        GttAction::Cancel,
        GttAction::Expire,
        GttAction::Confirm,
        GttAction::Refuse,
        GttAction::AskModify,
        GttAction::ConfirmModify,
        GttAction::RefuseModify,
        GttAction::AskCancel,
        GttAction::ConfirmCancel,
        GttAction::RefuseCancel,
        GttAction::VenueEnd,
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
            GttAction::Confirm => "Confirm",
            GttAction::Refuse => "Refuse",
            GttAction::AskModify => "AskModify",
            GttAction::ConfirmModify => "ConfirmModify",
            GttAction::RefuseModify => "RefuseModify",
            GttAction::AskCancel => "AskCancel",
            GttAction::ConfirmCancel => "ConfirmCancel",
            GttAction::RefuseCancel => "RefuseCancel",
            GttAction::VenueEnd => "VenueEnd",
        }
    }
}

use GttAction as A;
use GttStatus as S;

/// Every legal GTT transition as (from, action, to). Anything not listed is illegal.
/// The first six are a backtest's; the rest a live session's, whose venue confirms or
/// refuses each request, may fire a GTT while a modification or cancellation of it
/// waits, and may end it on its own while it rests (module docs).
pub const GTT_TRANSITIONS: [(GttStatus, GttAction, GttStatus); 19] = [
    (S::Active, A::Modify, S::Active),
    (S::Active, A::Fire, S::Triggered),
    (S::Active, A::Cancel, S::Canceled),
    (S::Active, A::Expire, S::Expired),
    (S::Triggered, A::FillChild, S::ChildFilled),
    (S::Triggered, A::CancelChild, S::ChildCanceled),
    (S::Requested, A::Confirm, S::Active),
    (S::Requested, A::Refuse, S::Rejected),
    (S::Active, A::AskModify, S::ModifyPending),
    (S::ModifyPending, A::ConfirmModify, S::Active),
    (S::ModifyPending, A::RefuseModify, S::Active),
    (S::Active, A::AskCancel, S::CancelPending),
    (S::CancelPending, A::ConfirmCancel, S::Canceled),
    (S::CancelPending, A::RefuseCancel, S::Active),
    (S::ModifyPending, A::Fire, S::Triggered),
    (S::CancelPending, A::Fire, S::Triggered),
    (S::Active, A::VenueEnd, S::VenueEnded),
    (S::ModifyPending, A::VenueEnd, S::VenueEnded),
    (S::CancelPending, A::VenueEnd, S::VenueEnded),
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
/// buffer is never smaller than asked, and never below one tick (`limit_beyond`;
/// decision GTT-6).
///
/// Units: rupees and percent. Errors: `InvalidPrice` for an unusable trigger or a
/// negative buffer, `InvalidTick` for an unusable tick.
pub fn child_limit(
    trigger_rupees: f64,
    child_side: Side,
    buffer_percent: f64,
    tick_rupees: f64,
) -> Result<f64, ExecError> {
    limit_beyond(trigger_rupees, child_side, buffer_percent, tick_rupees)
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
    /// Its trigger and the child's limit.
    pub leg: GttLeg,
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
}

impl Gtt {
    /// Its trigger, rupees, when it is for `role`.
    pub fn trigger(&self, role: GttLegRole) -> Option<f64> {
        (self.leg.role == role).then_some(self.leg.trigger_rupees)
    }

    /// True while its trigger rests (`GttStatus::is_active`): a backtest tests it on every
    /// base bar.
    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The legal transitions, written out from 04-execution 4.4: placed (Active), then
    /// modified any number of times, then cancelled, expired, or fired with its child
    /// filled or cancelled. Then from design 20-positional-live 3.1 for a live session:
    /// each request (place, modify, cancel) waits for the venue's answer, confirmed or
    /// refused; the venue may fire the GTT while a modify or cancel waits, and may end it
    /// on its own while it rests.
    const LEGAL: [(GttStatus, GttAction, GttStatus); 19] = [
        (S::Active, A::Modify, S::Active),
        (S::Active, A::Fire, S::Triggered),
        (S::Active, A::Cancel, S::Canceled),
        (S::Active, A::Expire, S::Expired),
        (S::Triggered, A::FillChild, S::ChildFilled),
        (S::Triggered, A::CancelChild, S::ChildCanceled),
        (S::Requested, A::Confirm, S::Active),
        (S::Requested, A::Refuse, S::Rejected),
        (S::Active, A::AskModify, S::ModifyPending),
        (S::ModifyPending, A::ConfirmModify, S::Active),
        (S::ModifyPending, A::RefuseModify, S::Active),
        (S::ModifyPending, A::Fire, S::Triggered),
        (S::ModifyPending, A::VenueEnd, S::VenueEnded),
        (S::Active, A::AskCancel, S::CancelPending),
        (S::CancelPending, A::ConfirmCancel, S::Canceled),
        (S::CancelPending, A::RefuseCancel, S::Active),
        (S::CancelPending, A::Fire, S::Triggered),
        (S::CancelPending, A::VenueEnd, S::VenueEnded),
        (S::Active, A::VenueEnd, S::VenueEnded),
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
            [
                "ChildFilled",
                "ChildCanceled",
                "Canceled",
                "Expired",
                "Rejected",
                "VenueEnded"
            ]
        );
    }

    #[test]
    fn a_gtt_rests_while_a_request_of_it_waits_and_is_at_the_venue_once_asked_for() {
        // (state, its trigger rests, at the venue or asked for there)
        let table = [
            (S::Requested, false, true),
            (S::Active, true, true),
            (S::ModifyPending, true, true),
            (S::CancelPending, true, true),
            (S::Triggered, false, false),
            (S::ChildFilled, false, false),
            (S::Canceled, false, false),
            (S::Rejected, false, false),
            (S::VenueEnded, false, false),
        ];
        for (state, rests, at_venue) in table {
            assert_eq!(
                (state.is_active(), state.at_venue()),
                (rests, at_venue),
                "{}",
                state.code()
            );
        }
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
    fn a_sell_child_at_one_tick_keeps_a_limit_of_one_tick() {
        // 5 percent below 0.05 rounds down to 0.00, which no order can carry.
        assert_eq!(child_limit(0.05, Side::Sell, 5.0, 0.05), Ok(0.05));
        assert_eq!(child_limit(0.01, Side::Sell, 5.0, 0.01), Ok(0.01));
        // Two ticks keep their rounding: 0.095 down to the tick is 0.05.
        assert_eq!(child_limit(0.1, Side::Sell, 5.0, 0.05), Ok(0.05));
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
