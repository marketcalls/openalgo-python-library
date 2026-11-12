//! The strategy bars of a whole run, for indicators computed ahead of the run.
//!
//! Event strategies register indicators in `init`; they are computed once over the
//! whole series in Rust and revealed one strategy bar at a time, at its close
//! (03-architecture 5). This is the series they are computed on: exactly the bars the
//! pauses deliver, in the same order, so an indicator value and the bar it belongs to
//! cannot drift apart.

use crate::exec::InstrumentId;

use super::kernel::Engine;
use super::market_data::StrategyBar;

impl Engine {
    /// Every strategy bar of an instrument over the run, in time order: the bars
    /// `Pause::BarClose` delivers, one per strategy-bar close with a base bar present
    /// (SPEC-07). Empty for an instrument the run does not know.
    pub fn strategy_bars(&self, instrument: InstrumentId) -> Vec<StrategyBar> {
        (0..self.timeline.len())
            .filter_map(|index| self.timeline.slice(index))
            .filter_map(|slice| self.timeline.strategy_bar_closing(&slice))
            .filter_map(|(open, _)| self.market.strategy_bar(instrument, open))
            .collect()
    }
}
