//! The child of a GTT the venue fired, in a live session (`venue_gtts`; design
//! 20-positional-live section 2 and 3.2, 04-execution 4.4 item 3, LEAD-10; unit N1 item
//! N1-2).
//!
//! The kernel never fires a GTT in a session: the venue does, and the live layer says so
//! with `apply_gtt_triggered` once it has found the child order (design 3.2). The child is
//! then an order the kernel follows, whose fills, cancellation and rejection come back as
//! any order's do (`apply_fill`, `apply_cancel`, `apply_reject`); before the call the
//! child has no kernel order, so nothing of it can be recorded.
//!
//! - **A leg's stop GTT** fires as a backtest's does: `GttTriggered`, then the leg's
//!   `leg_sl_hit` at the trigger. Its child is a new engine order for the GTT's units, a
//!   LIMIT at the child's limit with the exit reason `leg_sl` and the role `leg_stop`, and
//!   it is the position's exit from then on (one exit owner, `venue_owner`): the exits
//!   that waited for the GTT's cancellation (a target, the strategy's exit) end
//!   `Canceled`, and no other exit is made while it works. Its first fill writes
//!   `GttChildFilled` before the position event, as a backtest's child does.
//! - **A strategy's GTT order** fires into that order itself, now working at the venue.
//! - **A child not filled through one base bar** after the trigger is cancelled
//!   (`withdraw_order`; a cancellation the venue refused goes again in a later slice, as
//!   long as the child works), and only once the venue confirms the cancellation does the
//!   engine send its own exit of what is left at market, flagged `gtt_child_unfilled`,
//!   reason `leg_sl` (`RiskExitReason::StopUnfilled`), as a backtest's after an unfilled
//!   child. A child the broker rejects exits the leg at market as a rejected stop does
//!   (`stop_rejected`). A strategy GTT order's child that ends unfilled ends the order.
//!   On a daily base the child is a DAY order the session close ends.
//! - **A child that filled all its units** while the leg holds more (units added before
//!   the GTT was cut up) leaves the rest to be protected again (`venue_gtt_protect`), as a
//!   backtest places a new GTT for the rest.

use crate::exec::{
    FillFlags, Gtt, GttAction, GttEventName, GttId, GttPurpose, GttStatus, OrderAction, OrderId,
    OrderRequest, OrderType, QuantitySpec,
};
use crate::market::calendar::BaseInterval;
use crate::model::TsNanos;
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;

use super::error::KernelError;
use super::event::{OrderEventKind, Step};
use super::kernel::Engine;
use super::venue::{ExitBy, PositionExit, SendAt};
use super::venue_gtts::GttChild;
use super::venue_legs::StopEnd;
use super::venue_roles::{EngineExit, OrderRole, RiskExitReason};

/// The exit that follows a stop GTT's child that ended unfilled: the leg's stop level was
/// reached (`leg_sl`).
const CHILD_EXIT: EngineExit = EngineExit::Risk(RiskExitReason::StopUnfilled);

fn key_of(gtt: &Gtt) -> PositionKey {
    PositionKey {
        instrument: gtt.instrument,
        product: gtt.product,
    }
}

impl Engine {
    /// Record that the venue fired GTT `gtt` at `at` (UTC nanoseconds) and placed its
    /// child (module docs). Returns the kernel order the session follows the child as:
    /// the engine's new exit for a leg's stop GTT, the strategy's order for its GTT order.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `Venue(AtBarClose)` while paused at a
    /// bar close and `Venue(SessionOver)` after the session; `Venue(UnknownGtt)`;
    /// `Venue(GttState)` for a GTT that does not rest (asked for and not confirmed, fired
    /// already, or ended); those of the order book. A refused call changes nothing.
    pub fn apply_gtt_triggered(&mut self, gtt: GttId, at: TsNanos) -> Result<OrderId, KernelError> {
        let slice = self.result_slice()?;
        let fired = self.gtt_answered(gtt, "triggered", &RESTING)?;
        self.venue.clock_ns = self.venue.clock_ns.max(at);
        let stamp = Self::stamp(&slice, Step::Levels);
        self.gtts.book.apply(gtt, GttAction::Fire)?;
        let follow = self.gtt_follow_mut(gtt);
        follow.pending = None;
        follow.cancel_wanted = false;
        self.emit_gtt(stamp, GttEventName::Triggered, gtt);
        let child = match fired.purpose {
            GttPurpose::Protect => {
                self.emit_stop_hit(stamp, &fired);
                self.stop_child(self.waiting_at(&slice), &fired)?
            }
            GttPurpose::Order(order) => order,
        };
        self.venue.gtts.children.push(GttChild {
            gtt,
            order: child,
            triggered_at: at,
        });
        self.after_result(&slice)?;
        Ok(child)
    }

    /// The child of the leg's stop GTT `gtt`: the engine's exit of its units, the
    /// position's exit from now on (module docs), decided as `at` says.
    ///
    /// Errors: those of the order book and of `fills::end_order`.
    fn stop_child(&mut self, at: SendAt, gtt: &Gtt) -> Result<OrderId, KernelError> {
        let key = key_of(gtt);
        let waiting: Vec<PositionExit> = self
            .venue
            .exits
            .iter()
            .filter(|exit| exit.key == key && exit.sent.is_empty())
            .cloned()
            .collect();
        for exit in waiting {
            self.drop_exit(at, &exit, OrderEventKind::Canceled)?;
        }
        let units = gtt.quantity_units;
        let request = OrderRequest {
            product: gtt.product,
            reduce_only: true,
            exit_reason: Some(ExitReason::LegSl),
            order_type: OrderType::Limit {
                limit_rupees: gtt.leg.child_limit_rupees,
            },
            ..OrderRequest::market(gtt.instrument, gtt.side, QuantitySpec::Units(units))
        };
        let good_until = self
            .day_validity(&at.slice, at.decided_at)
            .unwrap_or(at.slice.close);
        let id = self.insert_engine_order(at, request, units, good_until)?;
        self.record_role(id, OrderRole::LegStop);
        self.venue.exits.push(PositionExit {
            key,
            by: ExitBy::Engine(CHILD_EXIT),
            sent: vec![id],
            limit_rupees: None,
            flags: FillFlags::NONE,
        });
        Ok(id)
    }

    /// Cancel the children not filled through one base bar since their GTT fired, by the
    /// session's clock (module docs), decided and stamped as `at` says: once while the
    /// venue has not answered, and again in a later slice after it refused
    /// (`withdraw_order`), as long as the child works.
    ///
    /// Errors: those of `withdraw_order`.
    pub(super) fn cancel_unfilled_children(&mut self, at: SendAt) -> Result<(), KernelError> {
        let BaseInterval::Intraday(base_ns) = self.config.base else {
            return Ok(());
        };
        let now = self.venue.clock_ns;
        let book = &self.book;
        let due: Vec<OrderId> = self
            .venue
            .gtts
            .children
            .iter()
            .filter(|child| now.saturating_sub(child.triggered_at) >= base_ns)
            .map(|child| child.order)
            .filter(|id| book.get(*id).is_some_and(|o| o.status.is_working()))
            .collect();
        for id in due {
            self.withdraw_order(at.stamp(), id)?;
        }
        Ok(())
    }

    /// After the venue ended order `order` as `how`: when it is a fired GTT's child, the
    /// GTT is resolved and the leg is exited or protected again (module docs), decided as
    /// `at` says.
    ///
    /// Errors: those of the GTT book and of the exits.
    pub(super) fn after_gtt_child_end(
        &mut self,
        at: SendAt,
        order: OrderId,
        how: StopEnd,
    ) -> Result<(), KernelError> {
        let Some(child) = self.gtt_child_of(order) else {
            return Ok(());
        };
        let Some(gtt) = self.gtts.book.get(child.gtt).cloned() else {
            return Ok(());
        };
        if gtt.status == GttStatus::Triggered && how != StopEnd::Filled {
            self.gtts.book.apply(gtt.id, GttAction::CancelChild)?;
            self.emit_gtt(at.stamp(), GttEventName::ChildCanceled, gtt.id);
        }
        if gtt.purpose != GttPurpose::Protect {
            return Ok(());
        }
        // The child's own exit of the position ended with it.
        self.tidy_venue();
        let key = key_of(&gtt);
        if self.held_units(key) == 0.0 || self.exit_closes_leg(key)? {
            return Ok(());
        }
        match how {
            StopEnd::MarginRejected => Ok(()),
            StopEnd::Filled => self.protect_gtt_leg(at, key).map(|_| ()),
            StopEnd::Rejected => {
                let exit = EngineExit::Risk(RiskExitReason::StopRejected);
                self.request_flagged_exit(at, key, exit, FillFlags::NONE)
            }
            StopEnd::Canceled { .. } => {
                self.request_flagged_exit(at, key, CHILD_EXIT, FillFlags::GTT_CHILD_UNFILLED)
            }
        }
    }

    /// The fired GTT whose child order `order` is, when its child's fills resolve it
    /// (`venue_fills`): its id.
    pub(super) fn child_gtt_of(&self, order: OrderId) -> Option<GttId> {
        self.gtt_child_of(order).map(|child| child.gtt)
    }

    /// End the strategy order `order` a GTT rested as, with `action` and its event
    /// `kind`, stamped as `at` says: the GTT ended without firing.
    ///
    /// Errors: those of `fills::end_order`.
    pub(super) fn end_gtt_order(
        &mut self,
        at: SendAt,
        order: OrderId,
        (action, kind): (OrderAction, OrderEventKind),
    ) -> Result<(), KernelError> {
        let works = self.book.get(order).is_some_and(|o| o.status.is_working());
        if works {
            self.end_order(at.stamp(), order, action, kind)?;
        }
        Ok(())
    }
}

/// The states in which a GTT's trigger rests at the venue on known terms.
pub(super) const RESTING: [GttStatus; 3] = [
    GttStatus::Active,
    GttStatus::ModifyPending,
    GttStatus::CancelPending,
];
