//! `reverse=True` (14-strategy-api 2.4): a `buy` while short or a `sell` while long that
//! is larger than the position closes the position and opens the rest on the other
//! side. A reversal is two fills of the one order, the close and then the open, each
//! with its own costs and events (12-position-sizing 1); the order is `PartiallyFilled`
//! between them.
//!
//! The opening part passes the fill-bar checks of any entry (the entry window for MIS,
//! then margin) after the close has filled, as a separate entry order would. When it
//! fails one, the close stands and the rest of the order is rejected with that code.
//! Without `reverse=True` the order is rejected whole with `reverse_not_allowed`, at
//! submission (`submit`) and again at the fill bar (`fills::at_fill`).

use crate::exec::{FillAt, FillFlags, Order, OrderAction, RejectCode};
use crate::model::Product;
use crate::portfolio::PositionKey;
use crate::rms::{ExitReason, LegRules};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{OrderEventKind, Step};
use super::fills::FillRequest;
use super::kernel::Engine;

impl Engine {
    /// Fill a reversing order at the open: close `close_units`, then open `open_units`
    /// on the order's side with its leg rules.
    ///
    /// Errors: those of `execute_fill`, the session times and the margin model.
    pub(super) fn fill_reversal(
        &mut self,
        slice: &SliceInfo,
        order: &Order,
        (close_units, open_units): (f64, f64),
        open_rupees: f64,
    ) -> Result<(), KernelError> {
        let stamp = Self::stamp(slice, Step::OpenFills);
        let request = &order.request;
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let close = FillRequest {
            key,
            side: request.side,
            units: close_units,
            reference_rupees: open_rupees,
            at: FillAt::Open,
            order: Some(order.id),
            exit_reason: request.exit_reason.unwrap_or(ExitReason::Signal),
            flags: FillFlags::NONE,
            margin_rupees: 0.0,
            leg_rules: LegRules::default(),
            target_index: None,
        };
        self.execute_fill(stamp, close.clone())?;
        let margin = match self.reversal_margin(slice, order, open_units, open_rupees)? {
            Ok(margin) => margin,
            Err(code) => {
                let kind = OrderEventKind::Rejected(code);
                return self.end_order(stamp, order.id, OrderAction::Reject, kind);
            }
        };
        let open = FillRequest {
            units: open_units,
            margin_rupees: margin,
            leg_rules: request.leg_rules.clone(),
            ..close
        };
        self.execute_fill(stamp, open)?;
        Ok(())
    }

    /// The margin the opening part of a reversal blocks, or the fill-bar check that
    /// refuses it: the entry window for MIS, then margin.
    fn reversal_margin(
        &mut self,
        slice: &SliceInfo,
        order: &Order,
        open_units: f64,
        open_rupees: f64,
    ) -> Result<Result<f64, RejectCode>, KernelError> {
        let request = &order.request;
        if request.product == Product::Mis {
            if let Err(code) = self.session_times(slice.day)?.entry.check(slice.open) {
                return Ok(Err(code));
            }
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let margin = self.opening_margin(
            slice.day,
            key,
            request.side,
            open_units,
            open_rupees,
            request.leverage,
        )?;
        if margin > self.portfolio.free_cash_rupees() {
            return Ok(Err(RejectCode::Margin));
        }
        Ok(Ok(margin))
    }
}
