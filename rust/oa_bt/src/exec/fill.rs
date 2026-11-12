//! Fills: an executed quantity at a price, where in the bar it was priced, and why.

use crate::model::Side;
use crate::rms::vocabulary::ExitReason;

use super::instrument::InstrumentId;
use super::order::OrderId;

/// Where in its bar a fill was priced (03-architecture 4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FillAt {
    /// The bar's open, plus slippage: queued orders (step 2), and levels already
    /// crossed at the open (step 3).
    Open = 0,
    /// A level inside the bar's range, plus slippage (step 3).
    Level = 1,
    /// The bar's close: an exit with no later bar in the session, flagged.
    Close = 2,
}

impl FillAt {
    /// The name used in results.
    pub fn code(self) -> &'static str {
        match self {
            FillAt::Open => "open",
            FillAt::Level => "level",
            FillAt::Close => "close",
        }
    }
}

/// Why a fill happened (decision SPEC-22).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FillReason {
    /// Opened a position from flat.
    Entry,
    /// Added a tranche to an open position.
    Add,
    /// One of the two fills of a futures roll (both carry this reason, SPEC-19).
    Roll,
    /// Reduced or closed a position.
    Exit(ExitReason),
}

impl FillReason {
    /// The reason string: "entry", "add", "roll", or the exit reason.
    pub fn code(self) -> &'static str {
        match self {
            FillReason::Entry => "entry",
            FillReason::Add => "add",
            FillReason::Roll => "roll",
            FillReason::Exit(reason) => reason.code(),
        }
    }
}

/// Flags that qualify a fill. A bit set; `FillFlags::NONE` is a plain fill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FillFlags(pub u8);

impl FillFlags {
    /// No flag.
    pub const NONE: FillFlags = FillFlags(0);
    /// The fill was priced at a last close because no later bar existed in the session
    /// (an exit queued with no bar left, a square-off with no bar at or after it).
    pub const FILLED_AT_LAST_CLOSE: FillFlags = FillFlags(1);

    /// True when every bit of `other` is set.
    pub fn contains(self, other: FillFlags) -> bool {
        self.0 & other.0 == other.0
    }

    /// The flags with `other` added.
    pub fn with(self, other: FillFlags) -> FillFlags {
        FillFlags(self.0 | other.0)
    }

    /// The names of the flags that are set, for results.
    pub fn names(self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.contains(FillFlags::FILLED_AT_LAST_CLOSE) {
            names.push("filled_at_last_close");
        }
        names
    }
}

/// One executed fill. The engine pairs it with the slice and step it happened in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fill {
    /// The order that filled, or `None` for an engine-held exit (stop, target,
    /// square-off, roll), which has no order (SPEC-21).
    pub order: Option<OrderId>,
    /// The instrument.
    pub instrument: InstrumentId,
    /// Buy or sell.
    pub side: Side,
    /// Executed quantity in units, positive.
    pub quantity_units: f64,
    /// Fill price in rupees, on the tick grid.
    pub price_rupees: f64,
    /// Where in the bar the price came from.
    pub at: FillAt,
    /// Why it happened.
    pub reason: FillReason,
    /// The target that filled (0 is the first), for a target exit.
    pub target_index: Option<u16>,
    /// Qualifiers.
    pub flags: FillFlags,
    /// Modelled charges of this fill in rupees (0 when the run is cost-free).
    pub costs_rupees: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasons_use_the_fixture_vocabulary() {
        assert_eq!(FillReason::Entry.code(), "entry");
        assert_eq!(FillReason::Exit(ExitReason::LegSl).code(), "leg_sl");
        assert_eq!(FillFlags::NONE.names(), Vec::<&str>::new());
        let flagged = FillFlags::NONE.with(FillFlags::FILLED_AT_LAST_CLOSE);
        assert_eq!(flagged.names(), vec!["filled_at_last_close"]);
    }
}
