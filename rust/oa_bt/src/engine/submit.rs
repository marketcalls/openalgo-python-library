//! Placing, cancelling and changing orders while the engine is paused.
//!
//! `submit` runs the submission checks against the decision instant and records the
//! order with `Submitted`, or `Rejected` alone (SPEC-21), stamped in step 7 of the
//! paused slice. The order then waits for the first base bar of its instrument that
//! opens at or after that instant (SPEC-35).

use crate::exec::checks::{freeze_rule, product_rule, quantity_rule};
use crate::exec::{
    ExecError, InstrumentId, Order, OrderAction, OrderId, OrderRequest, OrderStatus, OrderType,
    QuantitySpec, RejectCode, Validity,
};
use crate::model::{Product, Side, TsNanos};
use crate::portfolio::PositionKey;
use crate::sizing::{size, SizeOutcome, SizingContext, SkipReason};

use super::clock::SliceInfo;
use super::config::QuantityMode;
use super::error::KernelError;
use super::event::{Event, OrderEventKind, Step};
use super::kernel::{Engine, Phase};

/// Days searched for the next session of a daily strategy's order before it counts as
/// having no session to fill in: longer than any run of Indian market holidays.
const NEXT_SESSION_SEARCH_DAYS: i32 = 31;

/// The outcome of `Engine::submit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Submission {
    /// Passed the submission checks; queued.
    Submitted(OrderId),
    /// Refused by a submission check.
    Rejected(OrderId, RejectCode),
    /// A sizer gave no quantity; no order was created (12-position-sizing 5).
    Skipped(SkipReason),
}

/// The quantity of a request at submission.
enum Resolved {
    Units(f64),
    AtFill,
    Skipped(SkipReason),
}

fn not_built(feature: &'static str) -> KernelError {
    KernelError::NotBuilt { feature }
}

fn invalid(reason: &str) -> KernelError {
    KernelError::Exec(ExecError::InvalidRequest {
        reason: reason.to_string(),
    })
}

impl Engine {
    /// Place an order at the current decision point.
    ///
    /// Errors: `NotAtDecisionPoint` unless paused; `Exec(InvalidRequest)` for a
    /// quantity that is not positive or a fraction outside (0, 1]; `Rms(InvalidRule)`
    /// for contradictory leg rules; `NotBuilt` for order types, validities and
    /// arguments not built yet. A market rule that refuses the order is not an error:
    /// the result is `Submission::Rejected`.
    pub fn submit(&mut self, request: OrderRequest) -> Result<Submission, KernelError> {
        let Phase::Paused(slice) = self.phase else {
            return Err(KernelError::NotAtDecisionPoint);
        };
        self.check_buildable(&request)?;
        let units = match self.resolve_units(&request)? {
            Resolved::Units(units) => units,
            Resolved::AtFill => 0.0,
            Resolved::Skipped(reason) => return Ok(Submission::Skipped(reason)),
        };
        let checked = self.submission_checks(&request, units, &slice)?;
        let id = self.book.next_id();
        let (action, kind, good_until, reject) = match checked {
            Ok(good_until) => (
                OrderAction::PassChecks,
                OrderEventKind::Submitted,
                good_until,
                None,
            ),
            Err(code) => (
                OrderAction::FailChecks,
                OrderEventKind::Rejected(code),
                slice.close,
                Some(code),
            ),
        };
        let status = crate::exec::state::next_status(OrderStatus::Initialized, action)?;
        self.book.insert(Order {
            id,
            request,
            status,
            decided_at: slice.close,
            good_until,
            quantity_units: units,
            filled_units: 0.0,
            average_fill_rupees: 0.0,
            reject_code: reject,
        })?;
        let stamp = Self::stamp(&slice, Step::Callbacks);
        self.emit(Event::Order {
            stamp,
            order: id,
            kind,
        });
        Ok(match reject {
            Some(code) => Submission::Rejected(id, code),
            None => Submission::Submitted(id),
        })
    }

    /// Cancel a working order. An order that no longer works is left as it is and its
    /// state is returned.
    ///
    /// Errors: `NotAtDecisionPoint` unless paused; `UnknownOrder`.
    pub fn cancel(&mut self, id: OrderId) -> Result<OrderStatus, KernelError> {
        let Phase::Paused(slice) = self.phase else {
            return Err(KernelError::NotAtDecisionPoint);
        };
        let status = self
            .book
            .get(id)
            .ok_or(KernelError::UnknownOrder(id))?
            .status;
        if !status.is_working() {
            return Ok(status);
        }
        let stamp = Self::stamp(&slice, Step::Callbacks);
        self.end_order(stamp, id, OrderAction::Cancel, OrderEventKind::Canceled)?;
        Ok(OrderStatus::Canceled)
    }

    /// Change a working order's quantity or prices.
    ///
    /// Errors: `NotBuilt` (modification arrives with LIMIT and stop orders).
    pub fn modify(&mut self, _id: OrderId, _request: OrderRequest) -> Result<(), KernelError> {
        Err(not_built("order modification"))
    }

    /// Refuse, with `NotBuilt`, the request shapes the kernel cannot run yet.
    fn check_buildable(&self, request: &OrderRequest) -> Result<(), KernelError> {
        if request.order_type != OrderType::Market {
            return Err(not_built("LIMIT, SL and SL-M orders"));
        }
        if request.validity == Validity::Gtt {
            return Err(not_built("GTT orders"));
        }
        if request.reverse {
            return Err(not_built("reverse=True"));
        }
        if request.leverage.is_some_and(|leverage| leverage != 1.0) {
            return Err(not_built("intraday leverage"));
        }
        request.leg_rules.validate(self.config.risk.strict_config)?;
        if !request.leg_rules.is_empty() {
            return Err(not_built(
                "leg stops, targets, trailing stops and breakeven",
            ));
        }
        Ok(())
    }

    fn resolve_units(&self, request: &OrderRequest) -> Result<Resolved, KernelError> {
        let lot = self
            .instruments
            .get(request.instrument.index())
            .map_or(1, |spec| spec.lot_size_units);
        match request.quantity {
            QuantitySpec::Units(units) if units.is_finite() && units > 0.0 => {
                Ok(Resolved::Units(units))
            }
            QuantitySpec::Units(_) => Err(invalid("qty must be a positive number of units")),
            QuantitySpec::Lots(lots) if lots > 0 => {
                Ok(Resolved::Units(f64::from(lots) * f64::from(lot)))
            }
            QuantitySpec::Lots(_) => Err(invalid("lots must be at least 1")),
            QuantitySpec::PositionFraction(fraction) => {
                if !(fraction > 0.0 && fraction <= 1.0) || !request.reduce_only {
                    return Err(invalid(
                        "a fraction is above 0, at most 1, and only reduces",
                    ));
                }
                Ok(Resolved::AtFill)
            }
            QuantitySpec::Sizer(sizer) => match size(sizer, &self.sizing_context(request)?)? {
                SizeOutcome::Units(units) => Ok(Resolved::Units(units)),
                SizeOutcome::Skipped(reason) => Ok(Resolved::Skipped(reason)),
            },
        }
    }

    fn sizing_context(&self, request: &OrderRequest) -> Result<SizingContext, KernelError> {
        let spec = self.spec(request.instrument)?;
        let account = self.account();
        Ok(SizingContext {
            equity_rupees: account.equity_rupees,
            initial_capital_rupees: self.config.capital_rupees,
            free_cash_rupees: account.free_cash_rupees,
            decision_price_rupees: self.last_close(request.instrument).unwrap_or(0.0),
            stop_distance_rupees: None,
            lot_size_units: spec.lot_size_units,
            multiplier: spec.multiplier,
            estimated_cost_rupees: 0.0,
            fractional: self.fractional(request.instrument),
        })
    }

    fn fractional(&self, instrument: InstrumentId) -> bool {
        let derivative = self
            .instruments
            .get(instrument.index())
            .is_some_and(|s| s.is_derivative());
        self.config.quantity_mode == QuantityMode::Fractional && !derivative
    }

    /// The submission checks, in order; `Ok` carries the order's `good_until`.
    fn submission_checks(
        &mut self,
        request: &OrderRequest,
        units: f64,
        slice: &SliceInfo,
    ) -> Result<Result<TsNanos, RejectCode>, KernelError> {
        let Some(spec) = self.instruments.get(request.instrument.index()) else {
            return Ok(Err(RejectCode::InstrumentUnknown));
        };
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        if reduces && !request.reduce_only && units > held.abs() {
            return Err(not_built(
                "an order larger than the opposite position (the excess rule and reverse=True)",
            ));
        }
        let fractional = self.fractional(request.instrument);
        let rules = if units > 0.0 {
            quantity_rule(units, spec, fractional).and_then(|()| freeze_rule(units, spec))
        } else {
            Ok(())
        };
        let checked =
            rules.and_then(|()| product_rule(spec, request.product, request.side, units, held));
        if let Err(code) = checked {
            return Ok(Err(code));
        }
        let opening = !reduces && !request.reduce_only;
        if opening && request.product == Product::Mis {
            if let Err(code) = self.session_times(slice.day)?.entry.check(slice.close) {
                return Ok(Err(code));
            }
        }
        let validity = self.day_validity(slice);
        if opening && validity.is_ok() {
            if let Err(code) = self.entry_gates(key, held, slice) {
                return Ok(Err(code));
            }
        }
        Ok(validity)
    }

    /// When a DAY order decided at this slice expires, or `market_closed` when its fill
    /// would fall in a later session than the decision (lead ruling). Orders of daily
    /// and longer strategies are valid for the next session.
    fn day_validity(&self, slice: &SliceInfo) -> Result<TsNanos, RejectCode> {
        let exchange = self.timeline.exchange();
        if self.config.strategy_interval.is_intraday() {
            return match self.calendar.session(exchange, slice.day) {
                Some(session) if slice.close < session.close => Ok(session.close),
                _ => Err(RejectCode::MarketClosed),
            };
        }
        (1..=NEXT_SESSION_SEARCH_DAYS)
            .filter_map(|ahead| self.calendar.session(exchange, slice.day + ahead))
            .find(|session| session.open >= slice.close)
            .map(|session| session.close)
            .ok_or(RejectCode::MarketClosed)
    }
}
