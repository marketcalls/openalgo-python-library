//! Condition and array strategies: decided entirely in Rust, with no Python call per
//! bar (14-strategy-api 4, `bt.from_arrays`, `bt.from_signals`).
//!
//! The Python layer evaluates the conditions once per run into boolean arrays, one
//! value per strategy bar, and hands them over with the order templates. The driver
//! then applies, at every strategy-bar close, exactly the `on_bar` of 14-strategy-api
//! 4.1, reading the position in step 7 after the fills of the slice (SPEC-29):
//!
//! ```text
//! if flat (no position and no working entry order):
//!     if entry:  submit entry_order        # long entry has priority over short
//!     elif short: submit short_order
//! elif long:
//!     if exit:   exit the position          # exit has priority over adding
//!     elif entry and adds < pyramiding: submit entry_order
//! elif short:
//!     if cover:  exit the position
//!     elif short and adds < pyramiding: submit short_order
//! ```
//!
//! Conditions are level-triggered. The equivalence with the scripted `on_bar` is the
//! test of trace 11 (`trace-11a-conditions-and-on-bar`). Not built yet.

use crate::exec::OrderRequest;
use crate::model::TsNanos;

use super::error::KernelError;
use super::kernel::Engine;

/// The arrays of one condition strategy on one instrument, aligned on strategy bars.
#[derive(Clone, Debug, PartialEq)]
pub struct SignalArrays {
    /// Open of each strategy bar, UTC nanoseconds, increasing.
    pub bar_opens: Vec<TsNanos>,
    /// `entry_when`, one value per bar.
    pub entry: Vec<bool>,
    /// `exit_when`.
    pub exit: Vec<bool>,
    /// `short_when`.
    pub short: Vec<bool>,
    /// `cover_when`.
    pub cover: Vec<bool>,
    /// Additional entries allowed while in a position.
    pub pyramiding: u32,
    /// The order `entry_when` places (side buy, its order arguments).
    pub entry_order: OrderRequest,
    /// The order `short_when` places (side sell).
    pub short_order: OrderRequest,
}

/// Run a condition strategy to the end of the grid.
///
/// Errors: `NotBuilt` until the signal driver exists.
pub fn run_signals(_engine: &mut Engine, _signals: &SignalArrays) -> Result<(), KernelError> {
    Err(KernelError::NotBuilt {
        feature: "condition and array strategies",
    })
}
