//! Sizing orders in the engine (12-position-sizing 5, 03-architecture 7): the context
//! a sizer sees, the cost estimate in its budget, and the engine default.
//!
//! A sizer runs at the decision with the decision price (the last close). The engine
//! default of a cash-equity order that opens or adds in fractional mode is the one
//! exception: it is sized at the fill, at the fill bar's open plus slippage, so the
//! whole capital is used without a buffer. In venue mode it is sized at the decision
//! with the 2 percent cash buffer, as live trading does.
//!
//! Costs in a budget come from the cost model in two passes: the sizer runs without
//! costs, the order of that size is priced, and the sizer runs again with that
//! estimate (the entry for the engine default, the round trip for the risk sizers).
//! Costs only shrink a budget, so the second size never needs more than the estimate.

use crate::charges::{estimate_entry, estimate_round_trip};
use crate::exec::{InstrumentId, Order, OrderRequest, QuantitySpec};
use crate::market::slippage::slipped_price;
use crate::model::{Day, Product, Side};
use crate::rms::Distance;
use crate::sizing::{size, SizeOutcome, SizerSpec, SizingContext, SizingError, VENUE_CASH_BUFFER};

use super::accounting::charge_error;
use super::clock::SliceInfo;
use super::config::{CostConfig, QuantityMode};
use super::error::KernelError;
use super::kernel::Engine;

/// The distance from `price` to a stop, rupees of price, for the risk sizers.
fn stop_distance(
    stop: Option<Distance>,
    price: f64,
    tick: f64,
) -> Result<Option<f64>, KernelError> {
    Ok(match stop {
        None => None,
        Some(Distance::Points(points)) => Some(points),
        Some(Distance::Percent(percent)) => Some(price * percent / 100.0),
        Some(Distance::Ticks(ticks)) => Some(f64::from(ticks) * tick),
        Some(Distance::Atr {
            multiple,
            atr_rupees,
        }) => Some(multiple * atr_rupees),
        Some(Distance::Rupees(_)) => {
            return Err(KernelError::Sizing(SizingError::InvalidSizer {
                reason: "a stop in bt.rupees depends on the quantity, so it cannot size fixed \
                         risk; give the stop in points, percent, ticks or ATR"
                    .to_string(),
            }))
        }
    })
}

impl Engine {
    /// True for cash equity in fractional mode.
    fn fractional_equity(&self, instrument: InstrumentId) -> bool {
        let derivative = self
            .instruments
            .get(instrument.index())
            .map_or(true, |spec| spec.is_derivative());
        self.config.quantity_mode == QuantityMode::Fractional && !derivative
    }

    /// True when the request's engine default is sized at the fill: cash equity in
    /// fractional mode, an order that opens or adds, and not a delivery sale (which
    /// the product rule refuses at submission).
    pub(super) fn sizes_at_fill(&self, request: &OrderRequest) -> bool {
        let key_units = self
            .portfolio
            .position(crate::portfolio::PositionKey {
                instrument: request.instrument,
                product: request.product,
            })
            .map_or(0.0, |position| position.quantity_units);
        let reduces = key_units != 0.0 && (key_units > 0.0) != (request.side == Side::Buy);
        matches!(
            request.quantity,
            QuantitySpec::Sizer(SizerSpec::EngineDefault)
        ) && self.fractional_equity(request.instrument)
            && !request.reduce_only
            && !reduces
            && !(request.product == Product::Cnc && request.side == Side::Sell)
    }

    /// What a sizer sees for `request` at the decision: the last close as the sizing
    /// price, no cash buffer, no cost estimate. A strategy's sizing function receives
    /// these figures (`size=` a function).
    ///
    /// Errors: `Sizing` for a stop in `bt.rupees` (it depends on the quantity);
    /// `Exec` for an instrument the run does not have.
    pub fn sizing_inputs(&self, request: &OrderRequest) -> Result<SizingContext, KernelError> {
        let price = self.last_close(request.instrument).unwrap_or(0.0);
        self.sizing_context(request, price, 0.0)
    }

    /// What a sizer sees for `request` at `price` (rupees).
    pub(super) fn sizing_context(
        &self,
        request: &OrderRequest,
        price: f64,
        cash_buffer: f64,
    ) -> Result<SizingContext, KernelError> {
        let spec = self.spec(request.instrument)?;
        let account = self.account();
        let key = crate::portfolio::PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        Ok(SizingContext {
            equity_rupees: account.equity_rupees,
            initial_capital_rupees: self.config.capital_rupees,
            free_cash_rupees: account.free_cash_rupees,
            decision_price_rupees: price,
            stop_distance_rupees: stop_distance(
                request.leg_rules.stop,
                price,
                spec.tick_size_rupees,
            )?,
            lot_size_units: spec.lot_size_units,
            multiplier: spec.multiplier,
            estimated_cost_rupees: 0.0,
            fractional: self.fractional_equity(request.instrument),
            derivative: spec.is_derivative(),
            freeze_quantity_units: spec.freeze_quantity_units,
            instruments: u32::try_from(self.instruments.len()).unwrap_or(u32::MAX),
            cash_buffer,
            held_on_side_units: self.held_units(key) * request.side.sign(),
            reinvest: self.config.sizing.reinvest,
            caps: self.config.sizing.caps,
        })
    }

    /// Size `sizer` in two passes, the second with the cost model's estimate.
    fn sized(
        &self,
        sizer: SizerSpec,
        ctx: SizingContext,
        request: &OrderRequest,
        day: Day,
    ) -> Result<SizeOutcome, KernelError> {
        let first = size(sizer, &ctx)?;
        let (SizeOutcome::Units(units), CostConfig::Model(model)) = (first, &self.config.costs)
        else {
            return Ok(first);
        };
        let spec = self.spec(request.instrument)?;
        let facts = self.config.market.of(request.instrument.index());
        let price = ctx.decision_price_rupees;
        let estimate = match sizer {
            SizerSpec::EngineDefault if !ctx.derivative => estimate_entry(
                model,
                spec,
                facts,
                request.product,
                request.side,
                units,
                price,
                day,
            ),
            SizerSpec::FixedRisk { .. } | SizerSpec::AtrRisk { .. } => {
                estimate_round_trip(model, spec, facts, request.product, units, price, day)
            }
            _ => return Ok(first),
        }
        .map_err(charge_error)?;
        let costed = SizingContext {
            estimated_cost_rupees: estimate,
            ..ctx
        };
        Ok(size(sizer, &costed)?)
    }

    /// Size a request at the decision of `slice`: the decision price is the last
    /// close; the engine default keeps the venue cash buffer.
    pub(super) fn size_at_decision(
        &self,
        request: &OrderRequest,
        sizer: SizerSpec,
        slice: &SliceInfo,
    ) -> Result<SizeOutcome, KernelError> {
        let price = self.last_close(request.instrument).unwrap_or(0.0);
        let buffer = if sizer == SizerSpec::EngineDefault {
            VENUE_CASH_BUFFER
        } else {
            0.0
        };
        let ctx = self.sizing_context(request, price, buffer)?;
        self.sized(sizer, ctx, request, slice.day)
    }

    /// The units a queued order fills: `units` as resolved, or for an engine default
    /// left to the fill, the default at the fill bar's open plus slippage. `None` when
    /// nothing fits (no free cash, or the position changed side since the decision).
    pub(super) fn size_at_fill(
        &mut self,
        slice: &SliceInfo,
        order: &Order,
        units: f64,
        opening: bool,
        open: f64,
    ) -> Result<Option<f64>, KernelError> {
        let request = &order.request;
        let deferred = order.quantity_units == 0.0
            && matches!(
                request.quantity,
                QuantitySpec::Sizer(SizerSpec::EngineDefault)
            );
        if !deferred {
            return Ok(Some(units));
        }
        if !opening {
            return Ok(None);
        }
        let tick = self.spec(request.instrument)?.tick_size_rupees;
        let slippage = self.fill_slippage(request.instrument, slice.day)?;
        let price = slipped_price(slippage, request.side, open, tick)?;
        let ctx = self.sizing_context(request, price, 0.0)?;
        Ok(
            match self.sized(SizerSpec::EngineDefault, ctx, request, slice.day)? {
                SizeOutcome::Units(units) => Some(units),
                SizeOutcome::Skipped(_) => None,
            },
        )
    }
}
