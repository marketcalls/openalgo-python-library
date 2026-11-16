//! Why a result from the venue of a live session was refused (`venue`, `venue_fills`).

use std::fmt;

use crate::exec::{OrderId, OrderStatus};

/// Why a result from the venue was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum VenueError {
    /// The run fills its own orders: a backtest (`Engine::new`) takes no venue results.
    NotExternal,
    /// The order has ended (filled, cancelled, expired or rejected).
    OrderEnded { order: OrderId, status: OrderStatus },
    /// A fill of more units than the order has left.
    TooManyUnits {
        order: OrderId,
        units: f64,
        left_units: f64,
    },
    /// A fill whose units or price cannot be used.
    InvalidFill { reason: String },
    /// The engine is paused at a bar close (step 7): the slice must complete first.
    AtBarClose,
    /// Every slice of the session has run.
    SessionOver,
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
            VenueError::AtBarClose => write!(
                f,
                "the engine is paused at a bar close: call advance first, then pass the \
                 venue's results while it waits"
            ),
            VenueError::SessionOver => write!(
                f,
                "every slice of the session has run: a result from the venue can no \
                 longer be recorded"
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
}
