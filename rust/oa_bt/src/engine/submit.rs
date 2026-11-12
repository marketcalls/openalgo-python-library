//! Placing, cancelling and changing orders while the engine is paused.
//!
//! `submit` runs the submission checks against the decision instant and records the
//! order with `Submitted`, or `Rejected` alone (SPEC-21), stamped in step 7 of the
//! paused slice. The order then waits for the first base bar of its instrument that
//! opens at or after that instant (SPEC-35). A GTT order (`validity="gtt"`) also places
//! its GTT (`gtt_orders`); `modify` is in `modify`.

use crate::exec::checks::{freeze_rule, gtt_rule, product_rule, quantity_rule, tick_rule};
use crate::exec::{
    ExecError, InstrumentId, Order, OrderAction, OrderId, OrderRequest, OrderStatus, QuantitySpec,
    RejectCode, Validity, GTT_GOOD_UNTIL,
};
use crate::model::{Product, Side, TsNanos};
use crate::portfolio::PositionKey;
use crate::sizing::{SizeOutcome, SkipReason};

use super::clock::SliceInfo;
use super::config::QuantityMode;
use super::error::KernelError;
use super::event::{Event, OrderEventKind};
use super::kernel::Engine;

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
pub(super) enum Resolved {
    Units(f64),
    AtFill,
    Skipped(SkipReason),
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
    /// quantity that is not positive, a fraction outside (0, 1], a limit or trigger
    /// that is not a usable price, or a GTT order without a price; `Rms(InvalidRule)`
    /// for contradictory leg rules, or an add whose leg rules differ from the
    /// position's (SPEC-14); `NotBuilt` for arguments not built yet. A market rule that
    /// refuses the order is not an error: the result is `Submission::Rejected`.
    pub fn submit(&mut self, request: OrderRequest) -> Result<Submission, KernelError> {
        let Some((slice, decided_at, step)) = self.decision_point() else {
            return Err(KernelError::NotAtDecisionPoint);
        };
        self.check_buildable(&request)?;
        let request = self.venue_request(request)?;
        let units = match self.resolve_units(&request, &slice)? {
            Resolved::Units(units) => units,
            Resolved::AtFill => 0.0,
            Resolved::Skipped(reason) => return Ok(Submission::Skipped(reason)),
        };
        let checked = self.submission_checks(&request, units, &slice, decided_at, None)?;
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
                decided_at,
                Some(code),
            ),
        };
        let status = crate::exec::state::next_status(OrderStatus::Initialized, action)?;
        self.book.insert(Order {
            id,
            request,
            status,
            decided_at,
            good_until,
            quantity_units: units,
            filled_units: 0.0,
            average_fill_rupees: 0.0,
            reject_code: reject,
        })?;
        let stamp = Self::stamp(&slice, step);
        self.emit(Event::Order {
            stamp,
            order: id,
            kind,
        });
        if reject.is_none() {
            self.place_order_gtt(stamp, id)?;
        }
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
        let Some((slice, _, step)) = self.decision_point() else {
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
        let stamp = Self::stamp(&slice, step);
        self.end_order(stamp, id, OrderAction::Cancel, OrderEventKind::Canceled)?;
        Ok(OrderStatus::Canceled)
    }

    /// Refuse a request that cannot be read: unusable prices, contradictory leg rules.
    pub(super) fn check_buildable(&self, request: &OrderRequest) -> Result<(), KernelError> {
        request.check_prices()?;
        request.leg_rules.validate(self.config.risk.strict_config)?;
        self.check_leg_rules(request)
    }

    pub(super) fn resolve_units(
        &self,
        request: &OrderRequest,
        slice: &SliceInfo,
    ) -> Result<Resolved, KernelError> {
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
            QuantitySpec::Sizer(_) if self.sizes_at_fill(request) => Ok(Resolved::AtFill),
            QuantitySpec::Sizer(sizer) => match self.size_at_decision(request, sizer, slice)? {
                SizeOutcome::Units(units) => Ok(Resolved::Units(units)),
                SizeOutcome::Skipped(reason) => Ok(Resolved::Skipped(reason)),
            },
        }
    }

    fn fractional(&self, instrument: InstrumentId) -> bool {
        let derivative = self
            .instruments
            .get(instrument.index())
            .is_some_and(|s| s.is_derivative());
        self.config.quantity_mode == QuantityMode::Fractional && !derivative
    }

    /// The submission checks, in order, at the decision instant `decided_at`; `Ok`
    /// carries the order's `good_until`. `replacing` is the order a modification
    /// replaces (the scale-in limits do not count it as waiting).
    pub(super) fn submission_checks(
        &mut self,
        request: &OrderRequest,
        units: f64,
        slice: &SliceInfo,
        decided_at: TsNanos,
        replacing: Option<OrderId>,
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
        // Beyond the opposite position: reversed with `reverse=True`, else rejected.
        let excess = reduces && !request.reduce_only && units > held.abs();
        let fractional = self.fractional(request.instrument);
        let rules = if units > 0.0 {
            quantity_rule(units, spec, fractional)
                .and_then(|()| tick_rule(request.order_type, spec))
                .and_then(|()| freeze_rule(units, spec))
        } else {
            tick_rule(request.order_type, spec)
        };
        let checked = rules
            .and_then(|()| product_rule(spec, request.product, request.side, units, held))
            .and_then(|()| gtt_rule(request.validity, request.product));
        if let Err(code) = checked {
            return Ok(Err(code));
        }
        let opening = (!reduces || excess) && !request.reduce_only;
        if opening && request.product == Product::Mis {
            if let Err(code) = self.session_times(slice.day)?.entry.check(decided_at) {
                return Ok(Err(code));
            }
        }
        // Pre-trade check order of 03-architecture 4.4: closed market (5), trading state
        // and entry gates (6, 7), the scale-in limits (7), then an excess over the
        // position without `reverse` (7).
        let validity = match request.validity {
            Validity::Day => self.day_validity(slice, decided_at),
            Validity::Gtt => Ok(GTT_GOOD_UNTIL),
        };
        if opening && validity.is_ok() {
            if let Err(code) = self.entry_gates(key, held, slice) {
                return Ok(Err(code));
            }
            if let Err(code) = self.scale_in_limits(request, units, slice, replacing)? {
                return Ok(Err(code));
            }
        }
        if validity.is_ok() && excess && !request.reverse {
            return Ok(Err(RejectCode::ReverseNotAllowed));
        }
        Ok(validity)
    }

    /// When a DAY order decided at this slice expires, or `market_closed` when its fill
    /// would fall in a later session than the decision (lead ruling). Orders of daily
    /// and longer strategies are valid for the first session that opens at or after the
    /// decision: the next session for a decision at a close, the same session for a
    /// timer at its open.
    fn day_validity(&self, slice: &SliceInfo, decided_at: TsNanos) -> Result<TsNanos, RejectCode> {
        let exchange = self.timeline.exchange();
        if self.config.strategy_interval.is_intraday() {
            return match self.calendar.session(exchange, slice.day) {
                Some(session) if decided_at < session.close => Ok(session.close),
                _ => Err(RejectCode::MarketClosed),
            };
        }
        (0..=NEXT_SESSION_SEARCH_DAYS)
            .filter_map(|ahead| self.calendar.session(exchange, slice.day + ahead))
            .find(|session| session.open >= decided_at)
            .map(|session| session.close)
            .ok_or(RejectCode::MarketClosed)
    }
}
