//! A live session's controls (`venue`; 16-session-design 2.10, requirements R7, R9, R10,
//! R18 and R20): what the live layer asks of an external session besides recording its
//! venue's results.
//!
//! | Call | When | What |
//! |---|---|---|
//! | `request_exit(instrument, product, exit)` | paused at a decision point, or waiting | the engine's exit of the whole position for the session's reason (`venue_owner`): the orders on its closing side are cancelled, then it exits; it says what it did (`ExitRequest`, below) |
//! | `cancel(order)` (`submit`) | the same | `Cancel` is sent, and sent again for an order whose cancellation the venue refused, even in the slice it was refused in; not twice while one waits for the venue's answer |
//! | `lock_entries(reason)`, `unlock_entries(reason)`, `entry_locks()` | any time | entry locks are reasons in a set: while it is not empty, entries and adds are rejected at submission with `live_lock`, and exits are not (`venue_gates`); a reason is a name with no spaces around it |
//! | `cap_sizing_equity(rupees)` | any time | the engine default sizes from the lesser of the strategy's sizing equity and the cap, the account's available cash |
//! | `daily_loss_measure()` | any time | the strategy's daily loss measure, rupees on its risk basis: what the account's limit adds up |
//! | `stop_session()` | any time | the session ends where it stands (below) |
//!
//! "Waiting" is the slice in progress, where the venue's results are recorded too
//! (`venue_fills`): what is decided there is decided at its open (at its close in a
//! session close that waits for its orders) and stamped in its step 2. Every call but the
//! queries `entry_locks` and `daily_loss_measure` is refused in a backtest
//! (`VenueError::NotExternal`); `cancel` keeps a backtest's rule, at a decision point only.
//!
//! A session's exit is sent once the orders on the position's closing side have ended, so
//! `request_exit` may leave it waiting (`ExitRequest::WaitsForCancels`). If the market has
//! closed by the time they end, the exit is not sent and the position stays open, as for
//! an exit asked for when the market is closed already (`ExitRequest::MarketClosed`).
//! Asking again says where the position stands: `AlreadyExiting` while the exit works or
//! waits, `Flat` once it has closed the position, and a new exit once it has ended
//! without closing it (rejected by the venue, or dropped at the close).
//!
//! `stop_session` ends the session at once: nothing is sent (no cancellation, no exit),
//! the working orders keep their state and the open positions stay open, marked at their
//! last close, so `finish` reports them as they stand. A strategy exit that waits for its
//! cancels was never sent, so nothing of it is at the venue: it ends `Canceled`, and an
//! engine exit that waits is dropped. The slice in progress does not run:
//! what was recorded in it (fills while waiting) is in the fills and positions, while the
//! position history and the equity series end at the last slice that ran. Its contract
//! notes are settled, as at a run's end (`run_end`), so every fill carries its charges.
//! The events not yet delivered are returned, as `Pause::Finished` returns them, and
//! commands not taken yet stay for `take_commands`: the live layer reports them as not
//! sent. The run's end (`run_end`) stays the end of the session's range.

use std::collections::BTreeSet;

use crate::exec::{InstrumentId, OrderAction, OrderId, OrderStatus};
use crate::model::Product;
use crate::portfolio::PositionKey;
use crate::sizing::{SizerSpec, SizingContext};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, EventStamp, OrderEventKind};
use super::kernel::{Engine, Phase};
use super::venue::{ExitBy, SendAt};
use super::venue_error::VenueError;
use super::venue_roles::EngineExit;

/// What `Engine::request_exit` did (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitRequest {
    /// The exit's orders were sent.
    Sent,
    /// The exit waits for the cancellation of the orders on the position's closing side,
    /// sent now; it is sent once they have all ended, if the market is open then.
    WaitsForCancels,
    /// An engine exit holds the position already, sent or waiting: no second exit.
    AlreadyExiting,
    /// The position is flat: nothing to exit.
    Flat,
    /// The market is closed: no exit can be sent, and the position stays open.
    MarketClosed,
}

impl ExitRequest {
    /// Every outcome's code, in the order of the variants.
    pub const CODES: [&'static str; 5] = [
        "sent",
        "waits_for_cancels",
        "already_exiting",
        "flat",
        "market_closed",
    ];

    /// The code the bindings use.
    pub fn code(self) -> &'static str {
        match self {
            ExitRequest::Sent => "sent",
            ExitRequest::WaitsForCancels => "waits_for_cancels",
            ExitRequest::AlreadyExiting => "already_exiting",
            ExitRequest::Flat => "flat",
            ExitRequest::MarketClosed => "market_closed",
        }
    }
}

/// What the live layer has set on an external session.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct SessionControls {
    /// Why entries are locked, in sorted order; empty while entries are allowed.
    pub(super) entry_locks: BTreeSet<String>,
    /// The cap on the engine default's sizing equity, rupees, once set.
    pub(super) sizing_cap_rupees: Option<f64>,
}

/// A lock reason as given, refused when it names nothing or has spaces around it: two
/// reasons that differ only in those would be two locks, and one would stand after the
/// other was lifted.
///
/// Errors: `InvalidConfig` naming `entry_lock` for an empty or blank reason, and for one
/// with spaces around it.
fn lock_reason(reason: &str) -> Result<&str, KernelError> {
    if reason.trim().is_empty() || reason.trim() != reason {
        return Err(KernelError::InvalidConfig {
            field: "entry_lock",
            reason: format!(
                "{reason:?}: name why entries are locked, with no spaces around it, for \
                 example \"feed_dead\""
            ),
        });
    }
    Ok(reason)
}

impl Engine {
    /// Exit the position of `instrument` in `product` with the engine's `exit` (module
    /// docs): the square-off, or a risk exit for the session's reason. It goes through the
    /// position's exit owner (`venue_owner`): it takes over from the strategy's exits,
    /// cancels the working orders on the closing side and is sent once they have ended,
    /// for the position then open. Returns what it did: `Sent` or `WaitsForCancels`, and
    /// nothing at all for a position an engine exit holds already (`AlreadyExiting`), a
    /// flat one (`Flat`) and a closed market (`MarketClosed`).
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `Venue(SessionOver)` after the session;
    /// `InvalidConfig` naming `symbol` for an instrument the session does not trade; those
    /// of the exit owner.
    pub fn request_exit(
        &mut self,
        instrument: InstrumentId,
        product: Product,
        exit: EngineExit,
    ) -> Result<ExitRequest, KernelError> {
        self.request_exit_at(instrument, product, exit, None)
    }

    /// `request_exit` at market, or at `limit_rupees` (`venue_protect`).
    ///
    /// Errors: those of `request_exit`.
    pub(super) fn request_exit_at(
        &mut self,
        instrument: InstrumentId,
        product: Product,
        exit: EngineExit,
        limit_rupees: Option<f64>,
    ) -> Result<ExitRequest, KernelError> {
        self.require_session()?;
        self.spec(instrument)?;
        let at = self.control_point()?;
        let key = PositionKey {
            instrument,
            product,
        };
        if self.held_units(key) == 0.0 {
            return Ok(ExitRequest::Flat);
        }
        if self.has_engine_exit(key) {
            return Ok(ExitRequest::AlreadyExiting);
        }
        if self.day_validity(&at.slice, at.decided_at).is_err() {
            return Ok(ExitRequest::MarketClosed);
        }
        self.request_engine_exit(at, key, exit, limit_rupees)?;
        Ok(if self.has_waiting_exit(key) {
            ExitRequest::WaitsForCancels
        } else {
            ExitRequest::Sent
        })
    }

    /// Lock entries for `reason`: entries and adds are rejected at submission while any
    /// reason locks them (`venue_gates`). True when entries were not locked for it yet.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `InvalidConfig` naming `entry_lock`
    /// for a blank reason.
    pub fn lock_entries(&mut self, reason: &str) -> Result<bool, KernelError> {
        self.require_session()?;
        let reason = lock_reason(reason)?;
        Ok(self.venue.controls.entry_locks.insert(reason.to_string()))
    }

    /// Lift the entry lock of `reason`; entries resume once no reason locks them. True when
    /// entries were locked for it.
    ///
    /// Errors: those of `lock_entries`.
    pub fn unlock_entries(&mut self, reason: &str) -> Result<bool, KernelError> {
        self.require_session()?;
        let reason = lock_reason(reason)?;
        Ok(self.venue.controls.entry_locks.remove(reason))
    }

    /// The reasons entries are locked for, sorted; empty while entries are allowed (and
    /// always in a backtest).
    pub fn entry_locks(&self) -> Vec<&str> {
        let locks = &self.venue.controls.entry_locks;
        locks.iter().map(String::as_str).collect()
    }

    /// Cap the equity the engine default sizes from at `rupees`, the account's available
    /// cash (16-session-design 6.6): from now on it sizes from the lesser of the strategy's
    /// sizing equity and the cap. Other sizers and `target()` are not capped.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `InvalidConfig` naming
    /// `sizing_equity_cap` for rupees that are not a finite number of 0 or more.
    pub fn cap_sizing_equity(&mut self, rupees: f64) -> Result<(), KernelError> {
        self.require_session()?;
        if !(rupees.is_finite() && rupees >= 0.0) {
            return Err(KernelError::InvalidConfig {
                field: "sizing_equity_cap",
                reason: format!("{rupees}: pass the account's available cash in rupees, 0 or more"),
            });
        }
        self.venue.controls.sizing_cap_rupees = Some(rupees);
        Ok(())
    }

    /// The strategy's daily loss measure (05-risk-management 5), rupees rounded to the
    /// paisa, negative for a loss: the realized P&L of its runs ended in this session plus
    /// the open run's MTM at the last marks, less their charges on the net `risk_basis`.
    /// The daily loss limit fires at or below `-max_daily_loss`.
    pub fn daily_loss_measure(&self) -> f64 {
        self.risk.daily_loss.measure_rupees(
            self.config.risk.risk_basis,
            self.account_totals(),
            self.portfolio.unrealized_rupees(),
        )
    }

    /// End the session where it stands (module docs); `finish` can be called next. Returns
    /// the events not yet delivered. A session that has finished stays finished and returns
    /// none.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest.
    pub fn stop_session(&mut self) -> Result<Vec<Event>, KernelError> {
        self.require_session()?;
        if let Some(slice) = self.slice_in_progress() {
            self.drop_unsent_exits()?;
            self.settle_contract_notes(&slice);
        }
        self.phase = Phase::Finished;
        Ok(std::mem::take(&mut self.undelivered))
    }

    /// At the stop, drop every exit that waits for its cancels: none was sent (module
    /// docs). The strategy's end `Canceled`, stamped where a session control acts.
    ///
    /// Errors: those of `control_point` and of `fills::end_order`.
    fn drop_unsent_exits(&mut self) -> Result<(), KernelError> {
        let unsent: Vec<OrderId> = self
            .venue
            .exits
            .iter()
            .filter(|exit| exit.sent.is_empty())
            .filter_map(|exit| match exit.by {
                ExitBy::Strategy(id) => Some(id),
                ExitBy::Engine(_) => None,
            })
            .collect();
        self.venue.exits.retain(|exit| !exit.sent.is_empty());
        if unsent.is_empty() {
            return Ok(());
        }
        let stamp = self.control_point()?.stamp();
        for id in unsent {
            self.end_order(stamp, id, OrderAction::Cancel, OrderEventKind::Canceled)?;
        }
        Ok(())
    }

    /// Where a session control acts (module docs): the decision point the engine is paused
    /// at, or the slice in progress of an external session that waits.
    ///
    /// Errors: `NotAtDecisionPoint` in a backtest that is not paused; `Venue(SessionOver)`
    /// after an external session.
    pub(super) fn control_point(&self) -> Result<SendAt, KernelError> {
        if let Some((slice, decided_at, step)) = self.decision_point() {
            return Ok(SendAt {
                slice,
                step,
                decided_at,
            });
        }
        if !self.is_external() {
            return Err(KernelError::NotAtDecisionPoint);
        }
        let slice = self.result_slice()?;
        Ok(self.waiting_at(&slice))
    }

    /// Withdraw the working order `id` on the strategy's or the session's request
    /// (`Engine::cancel`): in an external session a cancellation the venue refused is sent
    /// again now, even in the slice it was refused in, and one waiting for the venue's
    /// answer is not sent twice (`withdraw_order`).
    ///
    /// Errors: those of `withdraw_order`.
    pub(super) fn cancel_on_request(
        &mut self,
        stamp: EventStamp,
        id: OrderId,
    ) -> Result<OrderStatus, KernelError> {
        if !self.venue.cancelling.contains(&id) {
            self.venue.cancels_sent.retain(|(known, _)| *known != id);
        }
        self.withdraw_order(stamp, id)
    }

    /// What the engine default sizes from once the session has a sizing cap: the
    /// strategy's equity, and its capital without `reinvest`, at most the cap, so its
    /// sizing equity is the lesser of the two. Any other sizer sees `ctx` as it is.
    pub(super) fn session_sizing(&self, sizer: SizerSpec, ctx: SizingContext) -> SizingContext {
        match self.venue.controls.sizing_cap_rupees {
            Some(cap) if sizer == SizerSpec::EngineDefault => SizingContext {
                equity_rupees: ctx.equity_rupees.min(cap),
                initial_capital_rupees: ctx.initial_capital_rupees.min(cap),
                ..ctx
            },
            _ => ctx,
        }
    }

    /// Refuse a session control in a backtest.
    ///
    /// Errors: `Venue(NotExternal)`.
    pub(super) fn require_session(&self) -> Result<(), KernelError> {
        if self.is_external() {
            Ok(())
        } else {
            Err(VenueError::NotExternal.into())
        }
    }

    /// The slice that has started and not completed, or the next one to run; `None` once
    /// the run has finished or every slice has run.
    fn slice_in_progress(&self) -> Option<SliceInfo> {
        match self.phase {
            Phase::Ready(index) => self.timeline.slice(index),
            Phase::Paused(slice)
            | Phase::Timers(slice)
            | Phase::Ahead(slice)
            | Phase::Closing(slice) => Some(slice),
            Phase::Finished => None,
        }
    }
}
