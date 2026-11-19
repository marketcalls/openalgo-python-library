//! The price a backtest fills a MARKET order at, for a bar the caller names
//! (16-session-design 1.4, decided item 11 of 10.0): a live session's day summary sets
//! each MARKET fill the broker made against it.
//!
//! A backtest fills a MARKET order at the open of the first base bar that opens at or after
//! its decision (SPEC-35), worse by the slippage the run models for the instrument on the
//! bar's trade date, rounded to the tick away from the trader (`order_fill::fill_price`,
//! `market::slippage`; the run's `slippage=`, the instrument's own, or the dated default
//! table's row for its class). `Engine::market_fill_price` applies that rule to a bar given
//! by its open instant and open price. It records nothing (no fill, no use of a table row)
//! and changes nothing, so a session can ask it at any time, also for a bar it has not
//! handed over yet (the minute in progress when the session stops). Which bar a fill is set
//! against is the caller's: the one opening at the decision, as the backtest's rule says.

use crate::exec::{FillPricing, InstrumentId};
use crate::market::slippage::slipped_price;
use crate::model::{Side, TsNanos};

use super::error::KernelError;
use super::kernel::Engine;

impl Engine {
    /// Rupees: the price a backtest of this run fills a MARKET order of `side` on
    /// `instrument` at, at the open of the instrument's base bar that opens at
    /// `bar_open_ts` (UTC nanoseconds, the open of a slice of the grid) at `open_rupees`
    /// (module docs). A buy pays the open plus the slippage, a sale receives the open less
    /// it, each rounded to the instrument's tick away from the trader.
    ///
    /// Errors: `InvalidConfig` naming `symbol` for an instrument the run does not trade, and
    /// those of the slippage (`InvalidConfig` naming `slippage` for an NFO or BFO derivative
    /// with no slippage class, `Slippage` when no row of the default table covers the day or
    /// the price cannot be rounded); `InvalidData` for an instant that opens no slice of the
    /// grid and for an open that is not a price above 0.
    pub fn market_fill_price(
        &self,
        instrument: InstrumentId,
        side: Side,
        bar_open_ts: TsNanos,
        open_rupees: f64,
    ) -> Result<f64, KernelError> {
        let spec = self.spec(instrument)?;
        let invalid = |reason: String| KernelError::InvalidData {
            symbol: spec.symbol.clone(),
            reason,
        };
        if !(open_rupees.is_finite() && open_rupees > 0.0) {
            return Err(invalid(format!(
                "a bar's open of {open_rupees} rupees: give the open price of the bar, above 0"
            )));
        }
        let slice = self
            .timeline
            .slice_opens()
            .binary_search(&bar_open_ts)
            .ok()
            .and_then(|index| self.timeline.slice(index))
            .ok_or_else(|| {
                invalid(format!(
                    "no slice of the run opens at {}: give the open instant of a base bar",
                    crate::market::calendar::format_ist(bar_open_ts)
                ))
            })?;
        let tick_rupees = spec.tick_size_rupees;
        let (slippage, _row) = self.slippage_of(instrument, slice.day)?;
        let slipped = slipped_price(slippage, side, open_rupees, tick_rupees)?;
        Ok(FillPricing::Market.price(side, open_rupees, slipped))
    }
}
