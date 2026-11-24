//! Where a GTT stands and what moves it (`gtt`): `GttStatus`, `GttAction` and the table
//! of every legal transition, a backtest's and a live session's (design
//! 20-positional-live 3.1, unit N1). Anything not listed is illegal.

use super::error::ExecError;

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
}
