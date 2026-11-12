//! The engine default size (03-architecture 7): what an order with no `qty`, `lots` or
//! `size` trades.
//!
//! - Futures and options: one lot.
//! - Cash equity: 1x the strategy's capital at the sizing price, net of the estimated
//!   costs of the order, split equally across the instruments of the strategy. The
//!   capital is the current equity (compounding), never more than the free cash, so
//!   no leverage is ever implied. In fractional mode the size is taken at the fill
//!   price; in venue mode (and in live) at the decision price, with
//!   `VENUE_CASH_BUFFER` kept back so a market order is not rejected on a move.

use super::{unit_value, SizingContext, SizingError};

/// The cash buffer of the engine default in venue mode and in live: 2 percent of the
/// budget, as a fraction. Source: 03-architecture section 7 ("the 2 percent cash buffer
/// applies in live and in venue-mode backtests").
pub const VENUE_CASH_BUFFER: f64 = 0.02;

/// Units of the engine default before the venue rule.
///
/// Errors: `InvalidSizer` for a non-positive sizing price or multiplier, or a buffer
/// outside [0, 1).
pub fn engine_default_units(ctx: &SizingContext) -> Result<f64, SizingError> {
    if ctx.derivative {
        return Ok(f64::from(ctx.lot_size_units.max(1)));
    }
    if !(0.0..1.0).contains(&ctx.cash_buffer) {
        return Err(SizingError::InvalidSizer {
            reason: format!(
                "the cash buffer is a fraction from 0 to below 1, got {}",
                ctx.cash_buffer
            ),
        });
    }
    let share = ctx.equity_rupees / f64::from(ctx.instruments.max(1));
    let capital = share.min(ctx.free_cash_rupees).max(0.0);
    let budget = capital * (1.0 - ctx.cash_buffer) - ctx.estimated_cost_rupees;
    Ok(budget.max(0.0) / unit_value(ctx)?)
}
