//! Where, when and how an order fills (`OrderFill`), and the price of any fill
//! (`fill_price`): the reference plus slippage, held at a marketable limit, or exactly
//! the reference for a resting limit (`exec::FillPricing`, decision GTT-5).

use crate::exec::{BarPoint, FillAt, FillPricing, GttId};
use crate::market::slippage::slipped_price;
use crate::model::Day;

use super::error::KernelError;
use super::event::Step;
use super::fills::FillRequest;
use super::kernel::Engine;

/// Where, when and how an order fills: at the open in step 2 (a queued order), or at a
/// point of the bar in step 3 (a resting order, a GTT's child).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct OrderFill {
    pub(super) step: Step,
    /// Where in the bar, and the price before slippage.
    pub(super) point: BarPoint,
    pub(super) pricing: FillPricing,
    pub(super) gtt: Option<GttId>,
}

impl OrderFill {
    /// A queued order at the open plus slippage (step 2).
    pub(super) fn at_open(open_rupees: f64) -> OrderFill {
        OrderFill {
            step: Step::OpenFills,
            point: BarPoint {
                at: FillAt::Open,
                price_rupees: open_rupees,
            },
            pricing: FillPricing::Market,
            gtt: None,
        }
    }

    /// An order filled at a point of the bar in step 3.
    pub(super) fn in_bar(point: BarPoint, pricing: FillPricing) -> OrderFill {
        OrderFill {
            step: Step::Levels,
            point,
            pricing,
            gtt: None,
        }
    }
}

impl Engine {
    /// The price of a fill: its reference plus the slippage of its instrument and day,
    /// as its pricing allows (a resting limit takes none, a marketable limit stops at
    /// its limit).
    ///
    /// Errors: `Slippage`, and those of `fill_slippage`.
    pub(super) fn fill_price(
        &mut self,
        request: &FillRequest,
        tick_rupees: f64,
        day: Day,
    ) -> Result<f64, KernelError> {
        let reference = request.reference_rupees;
        if !request.pricing.takes_slippage() {
            return Ok(reference);
        }
        let slippage = self.fill_slippage(request.key.instrument, day)?;
        let slipped = slipped_price(slippage, request.side, reference, tick_rupees)?;
        Ok(request.pricing.price(request.side, reference, slipped))
    }
}
