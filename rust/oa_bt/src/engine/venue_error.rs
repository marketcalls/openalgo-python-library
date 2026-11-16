//! Why a result from the venue of a live session was refused (`venue`, `venue_fills`).
//!
//! A refused result changes nothing, but for `WouldReverse`: a fill beyond the position
//! is recorded up to it before the rest is refused. Each refusal has a kind
//! (`VenueError::kind`, all of them in `VENUE_REFUSAL_KINDS`), so the live layer tells a
//! race it expects (a result for an order that has just ended) from a defect without
//! reading the message.

use std::fmt;

use crate::exec::{OrderId, OrderStatus};

/// What a refused result was waiting for (`VenueError::NotPending`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pending {
    /// A cancellation sent to the venue (`apply_cancel_refused`).
    Cancel,
    /// A modification sent to the venue (`apply_modify`, `apply_modify_refused`).
    Modify,
}

impl Pending {
    /// The word used in messages: "cancellation" or "modification".
    pub fn code(self) -> &'static str {
        match self {
            Pending::Cancel => "cancellation",
            Pending::Modify => "modification",
        }
    }
}

/// Why a result from the venue was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum VenueError {
    /// The run fills its own orders: a backtest (`Engine::new`) takes no venue results.
    NotExternal,
    /// The order has ended (filled, cancelled, expired or rejected).
    OrderEnded {
        /// The kernel order the result was for.
        order: OrderId,
        /// Its terminal state.
        status: OrderStatus,
    },
    /// The order was never sent: an exit still waiting for the cancels it sent.
    NotSent {
        /// The kernel order the result was for.
        order: OrderId,
    },
    /// A fill of more units than the order has left.
    TooManyUnits {
        /// The kernel order the fill was for.
        order: OrderId,
        /// Units of the refused fill.
        units: f64,
        /// Units the order has not filled yet.
        left_units: f64,
    },
    /// A fill whose units or price cannot be used.
    InvalidFill {
        /// What was wrong with it.
        reason: String,
    },
    /// No cancellation or modification of the order waits for the venue's answer.
    NotPending {
        /// The kernel order the result was for.
        order: OrderId,
        /// What the result answered.
        what: Pending,
    },
    /// A modification confirmed for fewer units than the order has filled.
    ModifyBelowFilled {
        /// The kernel order the result was for.
        order: OrderId,
        /// Units of the replacement.
        units: f64,
        /// Units the order has filled.
        filled_units: f64,
    },
    /// A fill of an order that closes the position (reduce-only, or one without
    /// `reverse=True`) beyond the open position, from a race at the broker: the units that
    /// closed the position were recorded, these were not. The broker's position now differs
    /// from the session's.
    WouldReverse {
        /// The kernel order the fill was for.
        order: OrderId,
        /// Units of the fill beyond the open position, not recorded.
        excess_units: f64,
    },
    /// The engine is paused at a bar close (step 7): the slice must complete first.
    AtBarClose,
    /// The session has finished: no result can be recorded any more.
    SessionOver,
}

/// The kind of an unknown order's refusal (`KernelError::UnknownOrder`), as the bindings
/// report it alongside `VenueError::kind`.
pub const UNKNOWN_ORDER_KIND: &str = "unknown_order";

/// Every kind of refusal a venue result can meet: the kinds of `VenueError` and
/// `UNKNOWN_ORDER_KIND`. The bindings export it, so the Python list cannot drift from it.
pub const VENUE_REFUSAL_KINDS: [&str; 11] = [
    "not_external",
    UNKNOWN_ORDER_KIND,
    "order_ended",
    "not_sent",
    "too_many_units",
    "invalid_fill",
    "not_pending",
    "modify_below_filled",
    "would_reverse",
    "at_bar_close",
    "session_over",
];

impl VenueError {
    /// The kind of refusal, as the bindings report it, one of `VENUE_REFUSAL_KINDS`.
    pub fn kind(&self) -> &'static str {
        match self {
            VenueError::NotExternal => "not_external",
            VenueError::OrderEnded { .. } => "order_ended",
            VenueError::NotSent { .. } => "not_sent",
            VenueError::TooManyUnits { .. } => "too_many_units",
            VenueError::InvalidFill { .. } => "invalid_fill",
            VenueError::NotPending { .. } => "not_pending",
            VenueError::ModifyBelowFilled { .. } => "modify_below_filled",
            VenueError::WouldReverse { .. } => "would_reverse",
            VenueError::AtBarClose => "at_bar_close",
            VenueError::SessionOver => "session_over",
        }
    }
}

impl fmt::Display for VenueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VenueError::NotExternal => write!(
                f,
                "this run fills its own orders (a backtest): fills, rejections and \
                 cancellations from a venue go to a live session (Engine::new_session)"
            ),
            VenueError::OrderEnded { order, status } => write!(
                f,
                "order {order} is {} and takes no more results from the venue",
                status.code()
            ),
            VenueError::NotSent { order } => write!(
                f,
                "order {order} has not been sent to the venue: it is an exit waiting for \
                 the orders it cancels to end"
            ),
            VenueError::TooManyUnits {
                order,
                units,
                left_units,
            } => write!(
                f,
                "a fill of {units} units for order {order}, which has {left_units} units \
                 left: pass the units newly filled, not the order's total"
            ),
            VenueError::InvalidFill { reason } => write!(f, "fill refused: {reason}"),
            VenueError::NotPending { order, what } => write!(
                f,
                "no {} of order {order} waits for the venue's answer",
                what.code()
            ),
            VenueError::ModifyBelowFilled {
                order,
                units,
                filled_units,
            } => write!(
                f,
                "the modification of order {order} to {units} units is below the \
                 {filled_units} units it has filled: the venue refuses it \
                 (apply_modify_refused)"
            ),
            VenueError::WouldReverse {
                order,
                excess_units,
            } => write!(
                f,
                "the fill of order {order} goes {excess_units} units beyond the open \
                 position: the units that closed it are recorded, these are not; the \
                 broker's position differs from the session's"
            ),
            VenueError::AtBarClose => write!(
                f,
                "the engine is paused at a bar close: call advance first, then pass the \
                 venue's results while it waits"
            ),
            VenueError::SessionOver => write!(
                f,
                "the session has finished: a result from the venue can no longer be \
                 recorded"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refused_fill_says_what_to_pass_instead() {
        let error = VenueError::TooManyUnits {
            order: OrderId(3),
            units: 61.0,
            left_units: 60.0,
        };
        assert_eq!(
            error.to_string(),
            "a fill of 61 units for order o3, which has 60 units left: pass the units \
             newly filled, not the order's total"
        );
    }

    #[test]
    fn every_refusal_has_its_own_kind() {
        let order = OrderId(1);
        let all = [
            VenueError::NotExternal,
            VenueError::OrderEnded {
                order,
                status: OrderStatus::Filled,
            },
            VenueError::NotSent { order },
            VenueError::TooManyUnits {
                order,
                units: 2.0,
                left_units: 1.0,
            },
            VenueError::InvalidFill {
                reason: String::new(),
            },
            VenueError::NotPending {
                order,
                what: Pending::Cancel,
            },
            VenueError::ModifyBelowFilled {
                order,
                units: 1.0,
                filled_units: 2.0,
            },
            VenueError::WouldReverse {
                order,
                excess_units: 1.0,
            },
            VenueError::AtBarClose,
            VenueError::SessionOver,
        ];
        let mut kinds: Vec<&str> = all.iter().map(VenueError::kind).collect();
        kinds.push(UNKNOWN_ORDER_KIND);
        kinds.sort_unstable();
        kinds.dedup();
        assert_eq!(kinds.len(), all.len() + 1);
        let mut listed = VENUE_REFUSAL_KINDS.to_vec();
        listed.sort_unstable();
        assert_eq!(kinds, listed, "VENUE_REFUSAL_KINDS lists every kind once");
    }
}
