//! `target()`: trade to a target position (12-position-sizing 5, FR-SIZE-08), the
//! single-instrument form of rebalancing (11-portfolio-rebalancing 2.1).
//!
//! At a decision, `submit_target` takes the position the strategy will hold once its
//! queued MARKET orders on that instrument and product have filled (the position plus
//! each such order's signed units, an order sized at the fill estimated as the
//! decision would size it) and sends one MARKET DAY order for the difference to the
//! target (`sizing::target_units`):
//!
//! | From | To | The order |
//! |---|---|---|
//! | flat, or a smaller position on the target's side | a larger one on that side | opens or adds: the entry gates, the scale-in limits and margin apply |
//! | a larger position on the target's side | a smaller one, or flat | only reduces (`reduce_only`) |
//! | a position on one side | a position on the other | `reverse=True`: closes, then opens the rest (two fills, two sets of costs) |
//!
//! No order is sent when the position already is the target. A value or percent target
//! that rounds to no units while nothing is held or queued asks for less than the venue's
//! smallest quantity: it is skipped with `SkipReason::RoundedToZero`, as a sizer's order
//! is, so the result counts it. Resting orders (LIMIT, SL,
//! SL-M and GTT orders) are not counted and stay working: they may never fill, and the
//! next `target()` corrects the position after one does. The order carries no leg
//! rules: an add keeps the position's (SPEC-14), a reduction shrinks them to the
//! position, and the side a reversal opens has none. Sizer caps do not apply: the
//! target is the quantity asked for.

use crate::exec::{InstrumentId, OrderRequest, OrderType, QuantitySpec, Validity};
use crate::model::{Product, Side};
use crate::portfolio::PositionKey;
use crate::sizing::{
    round_down_to_venue, target_units, SizeOutcome, SizerSpec, SkipReason, TargetSize,
};

use super::clock::SliceInfo;
use super::config::QuantityMode;
use super::error::KernelError;
use super::kernel::Engine;
use super::submit::Submission;

/// Difference, relative to the larger of the target and the position (and at least one
/// unit), below which the position already is the target: a residue of fractional
/// arithmetic sends no order.
const TARGET_TOLERANCE: f64 = 1.0e-9;

/// The order a target leads to (`Engine::target_order`).
pub(super) enum TargetOrder {
    /// The order for the difference (boxed: a request is far larger than the others).
    Send(Box<OrderRequest>),
    /// The position already is the target: nothing to send.
    AtTarget,
    /// The target rounds to no units from flat (see the module documentation).
    Skipped(SkipReason),
}

/// True for a target given as a non-zero value or percent: one the venue rule rounds.
fn asks_for_value(size: TargetSize) -> bool {
    match size {
        TargetSize::Rupees(rupees) => rupees != 0.0,
        TargetSize::PercentOfEquity(percent) => percent != 0.0,
        TargetSize::Units(_) | TargetSize::Lots(_) => false,
    }
}

/// What `target()` asks for on one instrument and product.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetRequest {
    /// The instrument.
    pub instrument: InstrumentId,
    /// The product of the position (the strategy's).
    pub product: Product,
    /// The position to hold.
    pub size: TargetSize,
    /// Intraday leverage for an order that opens or adds (`None` is 1x).
    pub leverage: Option<f64>,
}

impl Engine {
    /// Trade to `target` at the current decision point (see the module documentation).
    /// Returns `None` when the position already is the target, `Submission::Skipped`
    /// for a value or percent target that rounds to no units from flat, else the outcome
    /// of the order's submission.
    ///
    /// Errors: `NotAtDecisionPoint` unless paused; `Sizing` for an amount that is not a
    /// finite number, or a value or percent target with no sizing price yet; `Exec`
    /// for an instrument the run does not have; those of `submit`.
    pub fn submit_target(
        &mut self,
        target: TargetRequest,
    ) -> Result<Option<Submission>, KernelError> {
        match self.target_order(target)? {
            TargetOrder::Send(request) => self.submit(*request).map(Some),
            TargetOrder::AtTarget => Ok(None),
            TargetOrder::Skipped(reason) => Ok(Some(Submission::Skipped(reason))),
        }
    }

    /// The order `submit_target` sends for `target`, or why it sends none. Errors:
    /// those of `submit_target` before it submits.
    pub(super) fn target_order(&self, target: TargetRequest) -> Result<TargetOrder, KernelError> {
        let Some((slice, _, _)) = self.decision_point() else {
            return Err(KernelError::NotAtDecisionPoint);
        };
        let key = PositionKey {
            instrument: target.instrument,
            product: target.product,
        };
        // A target is sized like any order on its instrument and product; the side and
        // quantity of this probe are not read.
        let probe = OrderRequest {
            product: target.product,
            ..OrderRequest::market(target.instrument, Side::Buy, QuantitySpec::Units(0.0))
        };
        let wanted = target_units(target.size, &self.sizing_inputs(&probe)?)?;
        let expected = self.held_units(key) + self.queued_market_units(key, &slice)?;
        let difference = wanted - expected;
        let scale = wanted.abs().max(expected.abs()).max(1.0);
        if difference.abs() <= TARGET_TOLERANCE * scale {
            if wanted == 0.0 && asks_for_value(target.size) {
                return Ok(TargetOrder::Skipped(SkipReason::RoundedToZero));
            }
            return Ok(TargetOrder::AtTarget);
        }
        let side = if difference > 0.0 {
            Side::Buy
        } else {
            Side::Sell
        };
        let crosses = expected != 0.0 && wanted != 0.0 && (wanted > 0.0) != (expected > 0.0);
        let reduces = expected != 0.0 && !crosses && wanted.abs() < expected.abs();
        Ok(TargetOrder::Send(Box::new(OrderRequest {
            product: target.product,
            reduce_only: reduces,
            reverse: crosses,
            leverage: if reduces { None } else { target.leverage },
            ..OrderRequest::market(
                target.instrument,
                side,
                QuantitySpec::Units(difference.abs()),
            )
        })))
    }

    /// Signed units the queued MARKET DAY orders on `key` add to the position when they
    /// fill, as the decision of `slice` sees them: a fraction of the position at the
    /// venue rule, an engine default sized at the fill as the decision would size it,
    /// and a reduction never beyond the position.
    fn queued_market_units(&self, key: PositionKey, slice: &SliceInfo) -> Result<f64, KernelError> {
        let held = self.held_units(key);
        let spec = self.spec(key.instrument)?;
        let fractional =
            self.config.quantity_mode == QuantityMode::Fractional && !spec.is_derivative();
        let lot = spec.lot_size_units;
        let mut units = 0.0;
        for order in self.book.working() {
            let request = &order.request;
            let queued = request.instrument == key.instrument
                && request.product == key.product
                && request.order_type == OrderType::Market
                && request.validity == Validity::Day;
            if !queued {
                continue;
            }
            let mut quantity = match request.quantity {
                QuantitySpec::PositionFraction(fraction) => {
                    round_down_to_venue(held.abs() * fraction, lot, fractional)
                }
                QuantitySpec::Sizer(SizerSpec::EngineDefault) if order.quantity_units == 0.0 => {
                    match self.size_at_decision(request, SizerSpec::EngineDefault, slice)? {
                        SizeOutcome::Units(units) => units,
                        SizeOutcome::Skipped(_) => 0.0,
                    }
                }
                _ => order.quantity_units,
            };
            if request.reduce_only {
                quantity = quantity.min(held.abs());
            }
            units += quantity * request.side.sign();
        }
        Ok(units)
    }
}
