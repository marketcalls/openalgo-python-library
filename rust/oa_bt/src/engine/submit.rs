//! Placing, cancelling and changing orders while the engine is paused.
//!
//! `submit` runs the submission checks against the decision instant and records the
//! order with `Submitted`, or `Rejected` alone (SPEC-21), stamped in step 7 of the
//! paused slice. The order then waits for the first base bar of its instrument that
//! opens at or after that instant (SPEC-35). A GTT order (`validity="gtt"`) also places
//! its GTT (`gtt_orders`); `modify` is in `modify`. The daily caps come last, after the
//! other checks, and count the day's entries (`daily_caps`).

use crate::exec::checks::{freeze_rule, gtt_rule, product_rule, quantity_rule, tick_rule};
use crate::exec::{
    ExecError, InstrumentId, Order, OrderAction, OrderId, OrderRequest, OrderStatus, QuantitySpec,
    RejectCode, Validity, GTT_GOOD_UNTIL,
};
use crate::model::{Day, Product, Side, TsNanos};
use crate::portfolio::PositionKey;
use crate::sizing::{SizeOutcome, SkipReason};

use super::clock::SliceInfo;
use super::config::QuantityMode;
use super::daily_caps::Capped;
use super::error::KernelError;
use super::event::{Event, OrderEventKind};
use super::kernel::Engine;
use super::venue::{is_exit, SendAt};

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
    /// Refused unsized: the account's free cash alone left its sizer nothing
    /// (`sizing::size_submission`).
    Refused(RejectCode),
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
    /// position's (SPEC-14); `NotBuilt` for arguments not built yet (in a live session
    /// also leg rules, GTT orders and rebalancing, `venue`). A market rule that refuses
    /// the order is not an error: the result is `Submission::Rejected`. In a live session
    /// an order that passes is sent to the venue at once (`venue`). A reversal at a daily
    /// cap is cut to its close, whose submission is returned (`daily_caps`).
    pub fn submit(&mut self, request: OrderRequest) -> Result<Submission, KernelError> {
        let Some((slice, decided_at, _)) = self.decision_point() else {
            return Err(KernelError::NotAtDecisionPoint);
        };
        self.check_buildable(&request)?;
        self.check_venue_request(&request)?;
        let request = self.venue_request(request)?;
        let (units, refused) = match self.resolve_units(&request, &slice)? {
            Resolved::Units(units) => (units, None),
            Resolved::AtFill => (0.0, None),
            Resolved::Refused(code) => (0.0, Some(code)),
            Resolved::Skipped(reason) => return Ok(Submission::Skipped(reason)),
        };
        let units = self.venue_units(&request, units);
        // The checks come first, so a gate's code stands before the account's refusal.
        let checks = self.submission_checks(&request, units, &slice, decided_at, None)?;
        let checked = match (checks, refused) {
            (Err(code), _) | (Ok(_), Some(code)) => Err(code),
            (Ok(good_until), None) => self
                .live_checks(&request, units, &slice, None)?
                .map(|()| good_until),
        };
        let Ok(good_until) = checked else {
            return self.record_submission(request, units, checked);
        };
        match self.daily_caps(&request, units, &slice, false)? {
            Capped::Pass(entry) => {
                let submission = self.record_submission(request, units, Ok(good_until))?;
                if let (true, Submission::Submitted(id)) = (entry, submission) {
                    self.counted(&slice, id);
                }
                Ok(submission)
            }
            Capped::Refuse(code) => self.record_submission(request, units, Err(code)),
            Capped::CutReversal(code) => self.cut_reversal(request, units, code),
        }
    }

    /// Record `request` for `units` at the current decision point as its checks ended
    /// (`checked`: the order's `good_until`, or the code that refused it): `Submitted`, the
    /// order held for its delay, its GTT placed and, in a live session, sent; or
    /// `Rejected` alone.
    ///
    /// Errors: `NotAtDecisionPoint` unless paused; those of the order book, the GTT desk
    /// and the venue.
    pub(super) fn record_submission(
        &mut self,
        request: OrderRequest,
        units: f64,
        checked: Result<TsNanos, RejectCode>,
    ) -> Result<Submission, KernelError> {
        let Some((slice, decided_at, step)) = self.decision_point() else {
            return Err(KernelError::NotAtDecisionPoint);
        };
        let id = self.book.next_id();
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held_until = self.held_until(&request, self.held_units(key), decided_at);
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
        self.hold_order(id, held_until, decided_at);
        let stamp = Self::stamp(&slice, step);
        self.emit(Event::Order {
            stamp,
            order: id,
            kind,
        });
        if reject.is_none() {
            self.place_order_gtt(stamp, id)?;
            let at = SendAt {
                slice,
                step,
                decided_at,
            };
            self.send_order(at, id)?;
        }
        Ok(match reject {
            Some(code) => Submission::Rejected(id, code),
            None => Submission::Submitted(id),
        })
    }

    /// Cancel a working order. An order that no longer works is left as it is and its
    /// state is returned. Under a delay the cancellation takes effect when its delay
    /// ends, the order working until then (`held_changes`), and the state returned is the
    /// order's now. In a live session the cancellation is sent to the venue and
    /// the order works on until the venue confirms it (`venue`): its state then is the
    /// one returned. A live session takes it while it waits too, and sends it again for
    /// an order whose cancellation the venue refused (`venue_session`).
    ///
    /// Errors: `NotAtDecisionPoint` unless paused (in a live session, unless paused or
    /// waiting); `Venue(SessionOver)` after a live session; `UnknownOrder`.
    pub fn cancel(&mut self, id: OrderId) -> Result<OrderStatus, KernelError> {
        let at = self.control_point()?;
        let status = self
            .book
            .get(id)
            .ok_or(KernelError::UnknownOrder(id))?
            .status;
        if !status.is_working() || self.hold_cancel(id, at.decided_at)? {
            return Ok(status);
        }
        self.cancel_on_request(at.stamp(), id)
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
            QuantitySpec::Sizer(sizer) => match self.size_submission(request, sizer, slice)? {
                Ok(SizeOutcome::Units(units)) => Ok(Resolved::Units(units)),
                Ok(SizeOutcome::Skipped(reason)) => Ok(Resolved::Skipped(reason)),
                Err(code) => Ok(Resolved::Refused(code)),
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
        // A live session's exit is sized when it is sent, in orders of at most the freeze
        // quantity (`venue_owner`); entries keep the freeze rule, and so does a
        // modification, which replaces one order.
        let splits = self.is_external() && is_exit(request) && replacing.is_none();
        let rules = if units > 0.0 {
            quantity_rule(units, spec, fractional)
                .and_then(|()| tick_rule(request.order_type, spec))
                .and_then(|()| match splits {
                    true => Ok(()),
                    false => freeze_rule(units, spec),
                })
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
            Validity::Day => self.held_day_validity(slice, request, held, decided_at),
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
    pub(super) fn day_validity(
        &self,
        slice: &SliceInfo,
        decided_at: TsNanos,
    ) -> Result<TsNanos, RejectCode> {
        let exchange = self.timeline.exchange();
        if self.config.strategy_interval.is_intraday() {
            return match self.calendar.session(exchange, slice.day) {
                Some(session) if decided_at < session.close => Ok(session.close),
                _ => Err(RejectCode::MarketClosed),
            };
        }
        self.session_close_from(slice.day, decided_at)
            .ok_or(RejectCode::MarketClosed)
    }

    /// The close of the first session from `day` on that opens at or after `at`, within
    /// `NEXT_SESSION_SEARCH_DAYS`; `None` when there is none.
    pub(super) fn session_close_from(&self, day: Day, at: TsNanos) -> Option<TsNanos> {
        let exchange = self.timeline.exchange();
        (0..=NEXT_SESSION_SEARCH_DAYS)
            .filter_map(|ahead| self.calendar.session(exchange, day + ahead))
            .find(|session| session.open >= at)
            .map(|session| session.close)
    }
}
