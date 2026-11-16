//! Nothing is sent that a fill cannot record (`venue`): in an external session the checks
//! a backtest makes at the fill bar, and that the fill path would fail on, run when the
//! order is submitted or modified, with the backtest's reject codes and errors.
//!
//! - A reduce-only order must fit in the position with the others resting to close it
//!   (`venue_cover`): `exit_cover`.
//! - The cost model must price the order (its segment and the rates of the day): an
//!   error, as in a backtest's fill.
//! - The margin rule of the part that opens or adds (a reversal's opening part): a
//!   leverage the margin table does not allow, a short option, or a kind and product with
//!   no rule are errors, as at a backtest's fill; a margin above the free cash rejects it
//!   `margin`. The free cash counts what the order's closing part frees and what the
//!   session's other working orders that open or add would block if they filled now,
//!   since a backtest fills them first. Every margin is priced at the order's limit, else
//!   its trigger, else the last close; an order with no price at all is left to the venue
//!   (the rules do not depend on the price, the free cash does).
//!
//! The product rule (no delivery short) and the entry window are submission checks of
//! the backtest already.

use crate::exec::{Order, OrderId, OrderRequest, RejectCode};
use crate::model::Side;
use crate::portfolio::PositionKey;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::kernel::Engine;

impl Engine {
    /// The checks above for `request` of `units`, decided in `slice`; `replacing` is the
    /// order a modification replaces (its own units and margin are not counted against
    /// it). `Ok(Err(code))` rejects it. A backtest checks nothing here.
    ///
    /// Errors: those of `accounting::check_chargeable` and `accounting::margin_quote`.
    pub(super) fn venue_checks(
        &self,
        request: &OrderRequest,
        units: f64,
        slice: &SliceInfo,
        replacing: Option<OrderId>,
    ) -> Result<Result<(), RejectCode>, KernelError> {
        if !self.is_external() || units <= 0.0 {
            return Ok(Ok(()));
        }
        if let Err(code) = self.check_exit_cover(request, units, replacing) {
            return Ok(Err(code));
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        let price = self.decision_price(request);
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
        let blocked_by_working = self.working_margin_rupees(slice, replacing)?;
        if margin.rupees > self.free_cash_rupees() + freed - blocked_by_working {
            return Ok(Err(RejectCode::Margin));
        }
        Ok(Ok(()))
    }

    /// The price a margin is quoted at before the fill: the limit, else the trigger, else
    /// the instrument's last close.
    fn decision_price(&self, request: &OrderRequest) -> Option<f64> {
        let order_type = request.order_type;
        order_type
            .limit_rupees()
            .or(order_type.trigger_rupees())
            .or(self.last_close(request.instrument))
    }

    /// The margin, in rupees, the working orders that open or add would block if they
    /// filled now, but `replacing` (module docs).
    ///
    /// Errors: those of `accounting::margin_quote`.
    fn working_margin_rupees(
        &self,
        slice: &SliceInfo,
        replacing: Option<OrderId>,
    ) -> Result<f64, KernelError> {
        let working: Vec<&Order> = self
            .book
            .working()
            .filter(|order| !order.request.reduce_only && Some(order.id) != replacing)
            .collect();
        let mut rupees = 0.0;
        for order in working {
            let request = &order.request;
            let key = PositionKey {
                instrument: request.instrument,
                product: request.product,
            };
            let held = self.held_units(key);
            let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
            let left = self.units_to_fill(order);
            let opening = if reduces {
                (left - held.abs()).max(0.0)
            } else {
                left
            };
            let Some(price) = self.decision_price(request) else {
                continue;
            };
            if opening > 0.0 {
                let (side, leverage) = (request.side, request.leverage);
                let quote = self.margin_quote(slice.day, key, side, opening, price, leverage)?;
                rupees += quote.rupees;
            }
        }
        Ok(rupees)
    }
}
