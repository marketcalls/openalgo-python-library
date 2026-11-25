//! Fills: an executed quantity at a price, where in the bar it was priced, and why.

use crate::model::Side;
use crate::rms::vocabulary::ExitReason;

use super::gtt::GttId;
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
    /// The engine's own exit after a GTT's child order did not fill through one base bar
    /// (04-execution 4.4 item 3; the name is the trace vocabulary's, decision GTT-8).
    pub const GTT_CHILD_UNFILLED: FillFlags = FillFlags(2);
    /// A fill a live session's broker made with no order of the session: the broker's own
    /// square-off, or a position closed by hand (`engine::venue_external`).
    pub const EXTERNAL: FillFlags = FillFlags(4);
    /// The engine's exit of a CNC or NRML leg whose stop the price had already crossed
    /// when the stop was to be placed again: no stop was placed and the leg was exited at
    /// market (design 20-positional-live 3.4, `engine::venue_gtt_protect`).
    pub const STOP_CROSSED: FillFlags = FillFlags(8);
    /// An order of a daily or longer strategy's pre-open decision made after the session's
    /// open had passed (a late start): sent at once rather than at the open, so it fills
    /// later than a backtest of the same bars would (design 20-positional-live section 8,
    /// `engine::pre_open`).
    pub const LATE_DECISION: FillFlags = FillFlags(16);

    /// Every flag with its name, in bit order.
    pub const NAMED: [(FillFlags, &'static str); 5] = [
        (FillFlags::FILLED_AT_LAST_CLOSE, "filled_at_last_close"),
        (FillFlags::GTT_CHILD_UNFILLED, "gtt_child_unfilled"),
        (FillFlags::EXTERNAL, "external"),
        (FillFlags::STOP_CROSSED, "stop_crossed"),
        (FillFlags::LATE_DECISION, "late_decision"),
    ];

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
        FillFlags::NAMED
            .iter()
            .filter(|(flag, _)| self.contains(*flag))
            .map(|(_, name)| *name)
            .collect()
    }

    /// The flag `name` names (`NAMED`), if any.
    pub fn from_name(name: &str) -> Option<FillFlags> {
        FillFlags::NAMED
            .iter()
            .find(|(_, known)| *known == name)
            .map(|(flag, _)| *flag)
    }
}

/// How a fill's reference price (the open, a level, a limit) becomes its price.
///
/// Slippage is for orders that take the market (04-execution 4.3). A LIMIT that rests
/// and is filled by the market coming to it takes none; a limit that is marketable the
/// moment it is placed takes the market like a MARKET order but never past its limit
/// (decision GTT-5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FillPricing {
    /// The reference plus slippage: MARKET orders, SL-M, every engine exit and a leg's
    /// level (05-risk-management 3: "fills at the level plus slippage").
    Market,
    /// The reference plus slippage, never worse than `limit_rupees`: a GTT child or a
    /// triggered SL whose limit is marketable when it is placed (GTT-5).
    MarketableLimit { limit_rupees: f64 },
    /// Exactly the reference: a resting LIMIT filled at a better open or at its limit,
    /// a GTT child or a triggered SL filled when the bar trades back to its limit.
    Resting,
}

impl FillPricing {
    /// True when the fill takes slippage at all.
    pub fn takes_slippage(self) -> bool {
        self != FillPricing::Resting
    }

    /// The fill price for a `side` from its reference and the reference after slippage
    /// (rupees): `slipped` for a market fill, `slipped` held at the limit for a
    /// marketable limit, the reference itself for a resting limit.
    pub fn price(self, side: Side, reference_rupees: f64, slipped_rupees: f64) -> f64 {
        match self {
            FillPricing::Market => slipped_rupees,
            FillPricing::MarketableLimit { limit_rupees } => match side {
                Side::Buy => slipped_rupees.min(limit_rupees),
                Side::Sell => slipped_rupees.max(limit_rupees),
            },
            FillPricing::Resting => reference_rupees,
        }
    }
}

/// One executed fill. The engine pairs it with the slice and step it happened in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fill {
    /// The order that filled, or `None` for an engine-held exit (stop, target,
    /// square-off, roll), which has no order (SPEC-21).
    pub order: Option<OrderId>,
    /// The GTT whose child order this fill is, or whose unfilled child the engine's
    /// exit follows; `None` for every other fill.
    pub gtt: Option<GttId>,
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
    /// Modelled charges of this fill in rupees (0 when the run is cost-free). In a
    /// run's output, what the account paid for it, whole paise, contract-note rounding
    /// included; while the run advances, what the ledger posted when it filled.
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
        let unfilled = FillFlags::NONE.with(FillFlags::GTT_CHILD_UNFILLED);
        assert_eq!(unfilled.names(), vec!["gtt_child_unfilled"]);
    }

    #[test]
    fn a_crossed_stop_is_flagged_by_name_and_read_back_from_it() {
        let crossed = FillFlags::NONE.with(FillFlags::STOP_CROSSED);
        assert_eq!(crossed.names(), vec!["stop_crossed"]);
        assert_eq!(
            FillFlags::from_name("stop_crossed"),
            Some(FillFlags::STOP_CROSSED)
        );
        assert_eq!(FillFlags::from_name("crossed"), None);
    }

    #[test]
    fn a_market_fill_takes_its_slippage() {
        // A sell referenced at 1460.00 and slipped two ticks to 1459.90.
        assert_eq!(
            FillPricing::Market.price(Side::Sell, 1460.0, 1459.9),
            1459.9
        );
    }

    #[test]
    fn a_marketable_limit_takes_slippage_but_never_passes_its_limit() {
        // A sell child with limit 1445.40 fired at 1460.00: slipped to 1459.90 it is
        // inside the limit; slipped to 1440.00 it stops at the limit (GTT-5).
        let child = FillPricing::MarketableLimit {
            limit_rupees: 1445.4,
        };
        assert_eq!(child.price(Side::Sell, 1460.0, 1459.9), 1459.9);
        assert_eq!(child.price(Side::Sell, 1446.0, 1440.0), 1445.4);
        let buy = FillPricing::MarketableLimit {
            limit_rupees: 102.05,
        };
        assert_eq!(buy.price(Side::Buy, 101.0, 102.5), 102.05);
    }

    #[test]
    fn a_resting_limit_fills_at_its_reference_without_slippage() {
        // A buy limit at 99.50 filled at a 99.00 open stays at 99.00 though the run's
        // slippage would make it 99.10.
        assert!(!FillPricing::Resting.takes_slippage());
        assert_eq!(FillPricing::Resting.price(Side::Buy, 99.0, 99.1), 99.0);
    }
}
