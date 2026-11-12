//! What the market rules need to know about each instrument beyond `InstrumentSpec`:
//! the underlying (futures margin is per underlying; MCX futures and BFO options are
//! costed by it), the BSE scrip group (BSE equity is costed by it), the slippage class
//! (which row of the default slippage table applies) and the run's own slippage for
//! the instrument, when it overrides that row.
//!
//! The Python package resolves these from the reference tables and the run's
//! arguments; the kernel never looks anything up. A run that gives no facts (an empty
//! `MarketFacts`) gets the facts that follow from the exchange and kind alone.

use super::slippage::{SlippageClass, SlippageSpec};

/// The market facts of one instrument.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InstrumentMarket {
    /// Underlying symbol of a derivative (NIFTY, CRUDEOIL, ...), when known.
    pub underlying: Option<String>,
    /// BSE scrip group of a BSE equity (A, B, M, X, ...), when known.
    pub bse_group: Option<String>,
    /// Row of the default slippage table; `None` derives it from exchange and kind.
    pub slippage_class: Option<SlippageClass>,
    /// The run's slippage for this instrument, overriding the default table row.
    pub slippage: Option<SlippageSpec>,
}

/// The market facts of a run's instruments, in registration order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MarketFacts {
    /// One entry per instrument, or none at all.
    pub instruments: Vec<InstrumentMarket>,
}

impl MarketFacts {
    /// The facts of the instrument at `index`, or `None` when the run gave none.
    pub fn of(&self, index: usize) -> Option<&InstrumentMarket> {
        self.instruments.get(index)
    }

    /// Check that the facts, when given, cover exactly `instruments` instruments.
    ///
    /// Errors: a reason naming both counts.
    pub fn check_count(&self, instruments: usize) -> Result<(), String> {
        let given = self.instruments.len();
        if given == 0 || given == instruments {
            Ok(())
        } else {
            Err(format!(
                "market facts are given for {given} instruments but the run has {instruments}"
            ))
        }
    }
}
