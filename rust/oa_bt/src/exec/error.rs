//! Errors of the order and execution layer.
//!
//! These are errors in what the kernel was given (an unusable instrument, a malformed
//! request) or in its own bookkeeping (an illegal state transition). A market rule that
//! refuses an order is not an error: it is a `Rejected` order event with a
//! `checks::RejectCode`.

use std::fmt;

use super::gtt::{GttAction, GttStatus};
use super::state::{OrderAction, OrderStatus};

/// What went wrong in the execution layer.
#[derive(Clone, Debug, PartialEq)]
pub enum ExecError {
    /// An instrument definition the kernel cannot trade with.
    InvalidInstrument { symbol: String, reason: String },
    /// A price that is negative, infinite, NaN or beyond `price::MAX_PRICE_RUPEES`.
    InvalidPrice { what: &'static str, value: f64 },
    /// A tick size that is not a positive whole number of price units.
    InvalidTick { tick: f64 },
    /// An order request that cannot be read, whatever the market state.
    InvalidRequest { reason: String },
    /// An order transition the state table does not allow (a kernel defect).
    IllegalOrderTransition {
        from: OrderStatus,
        action: OrderAction,
    },
    /// A GTT transition the state table does not allow (a kernel defect).
    IllegalGttTransition { from: GttStatus, action: GttAction },
    /// A part of the execution layer that is not built yet.
    NotBuilt { feature: &'static str },
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecError::InvalidInstrument { symbol, reason } => {
                write!(f, "instrument {symbol}: {reason}")
            }
            ExecError::InvalidPrice { what, value } => write!(
                f,
                "{what} {value} is not a usable price: give a finite number of rupees \
                 from 0 to 1e12"
            ),
            ExecError::InvalidTick { tick } => write!(
                f,
                "tick size {tick} is not usable: give a positive tick in rupees with at \
                 most four decimals (0.05, 0.0025)"
            ),
            ExecError::InvalidRequest { reason } => write!(f, "invalid order request: {reason}"),
            ExecError::IllegalOrderTransition { from, action } => write!(
                f,
                "kernel defect: an order in state {} cannot take the action {}",
                from.code(),
                action.code()
            ),
            ExecError::IllegalGttTransition { from, action } => write!(
                f,
                "kernel defect: a GTT in state {} cannot take the action {}",
                from.code(),
                action.code()
            ),
            ExecError::NotBuilt { feature } => {
                write!(
                    f,
                    "{feature} is not implemented yet (Phase 1 work in progress)"
                )
            }
        }
    }
}

impl std::error::Error for ExecError {}
