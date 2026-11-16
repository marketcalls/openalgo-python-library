//! Nothing is sent that a fill cannot record (`venue`): in an external session the checks
//! a backtest makes at the fill bar, and that the fill path would fail on, run when the
//! order is submitted, with the backtest's reject codes and errors.
//!
//! - The cost model must price the order (its segment and the rates of the day): an
//!   error, as in a backtest's fill.
//! - The margin rule of the part that opens or adds (a reversal's opening part): a
//!   leverage the margin table does not allow, a short option, or a kind and product with
//!   no rule are errors, as at a backtest's fill; a margin above the free cash, with what
//!   the order's closing part frees, rejects it `margin`. The margin is priced at the
//!   order's limit, else its trigger, else the last close; with no price at all the free
//!   cash is left to the venue (the rules do not depend on the price).
//!
//! The product rule (no delivery short) and the entry window are submission checks of
//! the backtest already.

use crate::exec::{OrderRequest, RejectCode};
use crate::model::Side;
use crate::portfolio::PositionKey;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::kernel::Engine;

impl Engine {
    /// The checks above for `request` of `units`, decided in `slice`; `Ok(Err(code))`
    /// rejects it. A backtest checks nothing here.
    ///
    /// Errors: those of `accounting::check_chargeable` and `accounting::margin_quote`.
    pub(super) fn venue_checks(
        &self,
        request: &OrderRequest,
        units: f64,
        slice: &SliceInfo,
    ) -> Result<Result<(), RejectCode>, KernelError> {
        if !self.is_external() || units <= 0.0 {
            return Ok(Ok(()));
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        let order_type = request.order_type;
        let price = order_type
            .limit_rupees()
            .or(order_type.trigger_rupees())
            .or(self.last_close(request.instrument));
        // The rules below do not depend on the price; one tick stands in for a price
        // the order does not know yet.
        let priced_at = price.unwrap_or(self.spec(request.instrument)?.tick_size_rupees);
        self.check_chargeable(key, request.side, units, priced_at, slice.day)?;
        let opening = match (request.reduce_only, reduces) {
            (true, _) => 0.0,
            (false, true) => (units - held.abs()).max(0.0),
            (false, false) => units,
        };
        if opening <= 0.0 {
            return Ok(Ok(()));
        }
        let (day, side, leverage) = (slice.day, request.side, request.leverage);
        let margin = self.margin_quote(day, key, side, opening, priced_at, leverage)?;
        if price.is_none() {
            return Ok(Ok(()));
        }
        let freed = match (reduces, self.portfolio.position(key)) {
            (true, Some(position)) => position.margin_rupees,
            _ => 0.0,
        };
        if margin.rupees > self.free_cash_rupees() + freed {
            return Ok(Err(RejectCode::Margin));
        }
        Ok(Ok(()))
    }
}
