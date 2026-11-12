//! Condition and array strategies: decided entirely in Rust, with no Python call per
//! bar (14-strategy-api 4, `bt.from_arrays`, `bt.from_signals`).
//!
//! The Python layer evaluates the conditions once per run into boolean arrays, one
//! value per strategy bar of the instrument (the value of the condition at that bar's
//! close), and hands them over with the order templates. `run_signals` then drives the
//! engine through the same stepping API as an event strategy (`pause::run`), and at
//! every strategy-bar close of the instrument applies exactly the `on_bar` of
//! 14-strategy-api 4.1 (`decide`), reading the position in step 7 after the fills of
//! the slice (SPEC-29):
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
//! Conditions are level-triggered. The orders go through `Engine::submit` exactly as
//! the scripted `on_bar` places them (`buy(**entry_args)`, `exit()`), so both forms make
//! the same decisions at the same pauses; the equivalence is the test of trace 11
//! (`trace-11a-conditions-and-on-bar`) and of the property tests.

mod decide;
mod driver;

use crate::exec::{InstrumentId, OrderRequest, OrderType, Validity};
use crate::model::{Product, Side, TsNanos};
use crate::sizing::SkipReason;

use super::error::KernelError;
use super::kernel::Engine;
use super::pause;

pub use decide::{decide, BarSignals, Decision, Holding};

/// An entry the sizer skipped (it gave no quantity, so no order was created), as
/// `on_bar` records it for the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkippedEntry {
    /// The decision instant, UTC nanoseconds.
    pub decided_at: TsNanos,
    /// Why the sizer gave no quantity.
    pub reason: SkipReason,
}

/// One entry condition and the order it places while true.
#[derive(Clone, Debug, PartialEq)]
pub struct EntrySignal {
    /// The condition, one value per strategy bar.
    pub when: Vec<bool>,
    /// The order placed when the condition decides an entry or an add: a MARKET DAY
    /// order on the strategy's instrument, buy for `entry_when` and sell for
    /// `short_when`, with the order arguments of 14-strategy-api 4.1.
    pub order: OrderRequest,
}

/// The arrays of one condition strategy on one instrument, aligned on its strategy
/// bars.
#[derive(Clone, Debug, PartialEq)]
pub struct SignalArrays {
    /// The instrument traded.
    pub instrument: InstrumentId,
    /// The strategy's product: the position the conditions read, and the product of
    /// the exit order (as `self.position` and `exit()` in `on_bar`).
    pub product: Product,
    /// Open of each strategy bar, UTC nanoseconds, strictly increasing.
    pub bar_opens: Vec<TsNanos>,
    /// `entry_when`, if declared.
    pub entry: Option<EntrySignal>,
    /// `exit_when`, one value per bar, if declared.
    pub exit: Option<Vec<bool>>,
    /// `short_when`, if declared.
    pub short: Option<EntrySignal>,
    /// `cover_when`, one value per bar, if declared.
    pub cover: Option<Vec<bool>>,
    /// Additional entries allowed while in a position (the class attribute
    /// `pyramiding`, default 0).
    pub pyramiding: u32,
}

fn invalid(reason: String) -> KernelError {
    KernelError::InvalidConfig {
        field: "conditions",
        reason,
    }
}

impl SignalArrays {
    /// Check the arrays and the order templates before the run.
    ///
    /// Errors: `InvalidConfig` (field `conditions`) for an instrument the run does not
    /// have, bar opens that do not increase, an array whose length differs from the
    /// number of bars, or an order template that is not a MARKET DAY order opening on
    /// the right side of the strategy's instrument.
    pub fn validate(&self, instrument_count: usize) -> Result<(), KernelError> {
        if self.instrument.index() >= instrument_count {
            return Err(invalid(format!(
                "no instrument with index {} in this run",
                self.instrument.0
            )));
        }
        if self.bar_opens.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid(
                "the strategy-bar opens are not strictly increasing".to_string(),
            ));
        }
        let bars = self.bar_opens.len();
        let arrays = [
            ("entry_when", self.entry.as_ref().map(|e| e.when.len())),
            ("exit_when", self.exit.as_ref().map(Vec::len)),
            ("short_when", self.short.as_ref().map(|e| e.when.len())),
            ("cover_when", self.cover.as_ref().map(Vec::len)),
        ];
        for (name, length) in arrays {
            if let Some(length) = length.filter(|&length| length != bars) {
                return Err(invalid(format!(
                    "{name} has {length} values for {bars} strategy bars: give one value \
                     per strategy bar"
                )));
            }
        }
        self.check_template("entry_when", self.entry.as_ref(), Side::Buy)?;
        self.check_template("short_when", self.short.as_ref(), Side::Sell)
    }

    fn check_template(
        &self,
        name: &str,
        signal: Option<&EntrySignal>,
        side: Side,
    ) -> Result<(), KernelError> {
        let Some(order) = signal.map(|signal| &signal.order) else {
            return Ok(());
        };
        let problem = if order.instrument != self.instrument {
            Some("trades another instrument than the strategy")
        } else if order.side != side {
            Some("is on the wrong side (entry_when buys, short_when sells)")
        } else if order.order_type != OrderType::Market || order.validity != Validity::Day {
            Some("is not a MARKET DAY order (14-strategy-api 4.1)")
        } else if order.reduce_only || order.reverse {
            Some("only reduces or reverses; a condition entry opens or adds")
        } else {
            None
        };
        match problem {
            Some(problem) => Err(invalid(format!("the {name} order {problem}"))),
            None => Ok(()),
        }
    }

    /// The index of the strategy bar that opens at `ts_open`.
    ///
    /// Errors: `InvalidConfig` when the arrays have no value for that bar (they were not
    /// aligned on the run's strategy bars).
    fn bar_index(&self, ts_open: TsNanos) -> Result<usize, KernelError> {
        self.bar_opens.binary_search(&ts_open).map_err(|_| {
            invalid(format!(
                "the condition arrays have no value for the strategy bar opening at \
                 {ts_open} (UTC nanoseconds): align them on the run's strategy bars"
            ))
        })
    }

    /// The four conditions at a bar; an undeclared condition is false.
    fn at(&self, index: usize) -> BarSignals {
        let value = |array: Option<&Vec<bool>>| {
            array.and_then(|values| values.get(index)).copied() == Some(true)
        };
        BarSignals {
            entry: value(self.entry.as_ref().map(|signal| &signal.when)),
            exit: value(self.exit.as_ref()),
            short: value(self.short.as_ref().map(|signal| &signal.when)),
            cover: value(self.cover.as_ref()),
        }
    }
}

/// Run a condition strategy from the current position of the engine to the end of the
/// grid; then `Engine::finish` returns the result. Returns the entries a sizer
/// skipped.
///
/// Errors: `InvalidConfig` from `SignalArrays::validate` or for a strategy bar the
/// arrays do not cover; any error of `advance` or `submit` (an order the kernel cannot
/// run yet is `NotBuilt`). A market rule that refuses an order is not an error: it is a
/// rejected order, as in `on_bar`.
pub fn run_signals(
    engine: &mut Engine,
    signals: &SignalArrays,
) -> Result<Vec<SkippedEntry>, KernelError> {
    signals.validate(engine.instruments().len())?;
    let mut driver = driver::SignalDriver::new(signals);
    pause::run(engine, &mut driver)?;
    Ok(driver.skipped)
}
