//! The order state machine: its states, the actions that move an order, and the one
//! table of legal transitions.
//!
//! The same machine runs in backtest and live. A backtest never takes the `Accept`
//! action (an order is working once it passes the submission checks), so `Accepted`
//! appears only in live; the event stream of a backtest has no `Accepted` event
//! (decision SPEC-21).
//!
//! Lifecycle in a backtest:
//!
//! ```text
//!   Initialized --PassChecks--> Submitted --Fill--> Filled
//!        |                         |  |---Trigger--> Triggered --Fill--> Filled
//!        |                         |  |---Reject---> Rejected   (checks at the fill bar)
//!        |                         |  |---Cancel---> Canceled   (strategy or engine)
//!        |                         |  '---Expire---> Expired    (DAY validity ended)
//!        '--FailChecks--> Rejected (submission checks; Rejected is the only event)
//! ```

use super::error::ExecError;

/// Where an order stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OrderStatus {
    /// Created by a strategy; the submission checks have not run.
    Initialized = 0,
    /// Passed the submission checks: queued for an open or resting.
    Submitted = 1,
    /// Acknowledged by the venue (live only).
    Accepted = 2,
    /// A stop order whose trigger was reached; it now works as its limit or market.
    Triggered = 3,
    /// Part of the quantity has filled (partial fills are not modelled in Phase 1).
    PartiallyFilled = 4,
    /// The whole quantity has filled. Terminal.
    Filled = 5,
    /// Cancelled by the strategy or the engine. Terminal.
    Canceled = 6,
    /// Its validity ended unfilled (a DAY order at its session close). Terminal.
    Expired = 7,
    /// Refused by a check, at submission or at the fill bar. Terminal.
    Rejected = 8,
}

impl OrderStatus {
    /// Every state, in code order.
    pub const ALL: [OrderStatus; 9] = [
        OrderStatus::Initialized,
        OrderStatus::Submitted,
        OrderStatus::Accepted,
        OrderStatus::Triggered,
        OrderStatus::PartiallyFilled,
        OrderStatus::Filled,
        OrderStatus::Canceled,
        OrderStatus::Expired,
        OrderStatus::Rejected,
    ];

    /// The state name used in events and results.
    pub fn code(self) -> &'static str {
        match self {
            OrderStatus::Initialized => "Initialized",
            OrderStatus::Submitted => "Submitted",
            OrderStatus::Accepted => "Accepted",
            OrderStatus::Triggered => "Triggered",
            OrderStatus::PartiallyFilled => "PartiallyFilled",
            OrderStatus::Filled => "Filled",
            OrderStatus::Canceled => "Canceled",
            OrderStatus::Expired => "Expired",
            OrderStatus::Rejected => "Rejected",
        }
    }

    /// True when no action can move the order any more.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            OrderStatus::Filled
                | OrderStatus::Canceled
                | OrderStatus::Expired
                | OrderStatus::Rejected
        )
    }

    /// True while the order can still fill.
    pub fn is_working(self) -> bool {
        matches!(
            self,
            OrderStatus::Submitted
                | OrderStatus::Accepted
                | OrderStatus::Triggered
                | OrderStatus::PartiallyFilled
        )
    }
}

/// What happens to an order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OrderAction {
    /// The submission checks passed.
    PassChecks = 0,
    /// The submission checks refused the order.
    FailChecks = 1,
    /// The venue acknowledged the order (live only).
    Accept = 2,
    /// A check at the fill bar refused the order, or the venue rejected it.
    Reject = 3,
    /// A stop order's trigger was reached.
    Trigger = 4,
    /// Part of the remaining quantity filled.
    FillPart = 5,
    /// The remaining quantity filled.
    Fill = 6,
    /// The strategy or the engine cancelled the order.
    Cancel = 7,
    /// The order's validity ended.
    Expire = 8,
}

impl OrderAction {
    /// The action name used in messages.
    pub fn code(self) -> &'static str {
        match self {
            OrderAction::PassChecks => "PassChecks",
            OrderAction::FailChecks => "FailChecks",
            OrderAction::Accept => "Accept",
            OrderAction::Reject => "Reject",
            OrderAction::Trigger => "Trigger",
            OrderAction::FillPart => "FillPart",
            OrderAction::Fill => "Fill",
            OrderAction::Cancel => "Cancel",
            OrderAction::Expire => "Expire",
        }
    }
}

use OrderAction as A;
use OrderStatus as S;

/// Every legal transition as (from, action, to). Anything not listed is illegal.
pub const ORDER_TRANSITIONS: [(OrderStatus, OrderAction, OrderStatus); 25] = [
    (S::Initialized, A::PassChecks, S::Submitted),
    (S::Initialized, A::FailChecks, S::Rejected),
    (S::Submitted, A::Accept, S::Accepted),
    (S::Submitted, A::Reject, S::Rejected),
    (S::Submitted, A::Trigger, S::Triggered),
    (S::Submitted, A::FillPart, S::PartiallyFilled),
    (S::Submitted, A::Fill, S::Filled),
    (S::Submitted, A::Cancel, S::Canceled),
    (S::Submitted, A::Expire, S::Expired),
    (S::Accepted, A::Reject, S::Rejected),
    (S::Accepted, A::Trigger, S::Triggered),
    (S::Accepted, A::FillPart, S::PartiallyFilled),
    (S::Accepted, A::Fill, S::Filled),
    (S::Accepted, A::Cancel, S::Canceled),
    (S::Accepted, A::Expire, S::Expired),
    (S::Triggered, A::Reject, S::Rejected),
    (S::Triggered, A::FillPart, S::PartiallyFilled),
    (S::Triggered, A::Fill, S::Filled),
    (S::Triggered, A::Cancel, S::Canceled),
    (S::Triggered, A::Expire, S::Expired),
    (S::PartiallyFilled, A::FillPart, S::PartiallyFilled),
    (S::PartiallyFilled, A::Fill, S::Filled),
    (S::PartiallyFilled, A::Cancel, S::Canceled),
    (S::PartiallyFilled, A::Expire, S::Expired),
    // A venue may still reject the rest of a partly filled order (live only).
    (S::PartiallyFilled, A::Reject, S::Rejected),
];

/// The state an order moves to.
///
/// Errors: `IllegalOrderTransition` when the table has no row for (from, action).
pub fn next_status(from: OrderStatus, action: OrderAction) -> Result<OrderStatus, ExecError> {
    ORDER_TRANSITIONS
        .iter()
        .find(|(state, taken, _)| *state == from && *taken == action)
        .map(|(_, _, to)| *to)
        .ok_or(ExecError::IllegalOrderTransition { from, action })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backtest_order_is_submitted_then_filled() {
        let submitted = next_status(S::Initialized, A::PassChecks);
        assert_eq!(submitted, Ok(S::Submitted));
        assert_eq!(next_status(S::Submitted, A::Fill), Ok(S::Filled));
    }

    #[test]
    fn terminal_states_take_no_action() {
        for from in S::ALL.into_iter().filter(|state| state.is_terminal()) {
            assert!(ORDER_TRANSITIONS.iter().all(|(state, _, _)| *state != from));
        }
    }

    #[test]
    fn the_table_has_one_row_per_state_and_action() {
        for (index, (from, action, _)) in ORDER_TRANSITIONS.iter().enumerate() {
            let later = &ORDER_TRANSITIONS[index + 1..];
            assert!(!later.iter().any(|(s, a, _)| s == from && a == action));
        }
    }

    #[test]
    fn an_illegal_transition_is_an_error() {
        assert!(next_status(S::Filled, A::Cancel).is_err());
        assert!(next_status(S::Initialized, A::Fill).is_err());
    }

    #[test]
    fn working_states_are_exactly_the_non_terminal_ones_after_submission() {
        for state in S::ALL {
            let expected = state != S::Initialized && !state.is_terminal();
            assert_eq!(state.is_working(), expected, "{}", state.code());
        }
    }
}
