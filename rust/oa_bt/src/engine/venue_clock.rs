//! The end of a slice and the reach of the next open, in a backtest and in an external
//! session (`venue`).
//!
//! A backtest ends a slice with step 8 and its records, then runs the next slice. An
//! external session differs in two ways:
//!
//! - **The venue ends orders, not the clock.** At a session close the kernel expires no
//!   order: it sends `Cancel` for the DAY orders a backtest would expire (every working
//!   order at the session's last slice) and keeps them working; the rest of step 8 and
//!   the slice's records wait until the venue has ended them all (`Phase::Closing`, the
//!   engine returns `Pause::Waiting` meanwhile). The close's own cancellation is sent at
//!   most once in a slice (`withdraw_order`), so one the venue refuses at the close is not
//!   sent again by the engine: the close waits for the venue's own end of the order (a
//!   broker ends every DAY order at the close, or reports the fill that made it refuse),
//!   unless the session asks again with `cancel`, which sends it at once
//!   (`venue_session`). So the session cannot finish
//!   while an order works, and a fill the venue reports before the cancellation is
//!   recorded in the closing slice.
//! - **The next open is reached at the close of the slice before it.** A live session runs
//!   a slice once it has closed, which is the moment the next one opens. So what a
//!   backtest does at that open happens then: the timers due at it fire (`Pause::Timers`,
//!   decided at that open, `Phase::Ahead`), then the engine's exits due at it are made
//!   (`venue_exits`). Not across a session close: the first slice of a session fires its
//!   timers when it runs, as in a backtest.

use crate::exec::OrderId;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::Step;
use super::kernel::{Engine, Phase};
use super::pause::Pause;

impl Engine {
    /// End `slice` (step 8 and its records, `steps::close_slice`) and, in an external
    /// session, reach the next open. Returns the pause to stop at, if any; otherwise the
    /// engine stands before the next slice.
    ///
    /// Errors: those of the steps.
    pub(super) fn end_slice(&mut self, slice: &SliceInfo) -> Result<Option<Pause>, KernelError> {
        self.close_slice(slice)?;
        if self.session_close_waits(slice) {
            self.phase = Phase::Closing(*slice);
            return Ok(Some(Pause::Waiting {
                next_slice: slice.index + 1,
            }));
        }
        self.reach_next_open(slice)
    }

    /// A session close waiting for the venue to end its orders (module docs): its
    /// cancellations went out when the slice closed and are not sent twice in it, so the
    /// close completes once no order works.
    ///
    /// Errors: those of the steps.
    pub(super) fn wait_for_close(
        &mut self,
        slice: &SliceInfo,
    ) -> Result<Option<Pause>, KernelError> {
        if self.session_close_waits(slice) {
            return Ok(Some(Pause::Waiting {
                next_slice: slice.index + 1,
            }));
        }
        self.complete_close(slice)?;
        self.reach_next_open(slice)
    }

    /// Resume after the timers that fired ahead of `next`: the engine's exits due at its
    /// open, then the engine stands before it.
    ///
    /// Errors: those of `venue_exits`.
    pub(super) fn resume_ahead(&mut self, next: &SliceInfo) -> Result<(), KernelError> {
        self.venue_at_next_open(next)?;
        self.phase = Phase::Ready(next.index);
        Ok(())
    }

    /// After `slice` has ended: in an external session, the next open of the same
    /// session is reached (its timers, then the engine's exits); the engine then stands
    /// before the next slice.
    fn reach_next_open(&mut self, slice: &SliceInfo) -> Result<Option<Pause>, KernelError> {
        let next = match self.is_external() && !slice.closes_session {
            true => self.timeline.slice(slice.index + 1),
            false => None,
        };
        if let Some(next) = next {
            if let Some(pause) = self.timers_pause(&next) {
                self.phase = Phase::Ahead(next);
                return Ok(Some(pause));
            }
            self.venue_at_next_open(&next)?;
        }
        self.phase = Phase::Ready(slice.index + 1);
        Ok(None)
    }

    /// True when the timers due at the open of `slice` fired ahead of it, at the close of
    /// the slice before (module docs).
    pub(super) fn timers_fired_ahead(&self, slice: &SliceInfo) -> bool {
        self.is_external() && slice.index > 0 && !slice.opens_session
    }

    /// The orders a session close ends: those whose DAY validity ends at it, and every
    /// working order at the session's last slice.
    fn ending_at_close(&self, slice: &SliceInfo) -> Vec<OrderId> {
        let last = slice.index + 1 == self.timeline.len();
        self.book
            .working()
            .filter(|order| last || order.good_until <= slice.close)
            .map(|order| order.id)
            .collect()
    }

    /// Step 8: end the DAY orders whose validity ends with this session. A backtest
    /// expires them (`Expired`); an external session sends `Cancel` for them (module
    /// docs).
    ///
    /// Errors: those of the order book and of `withdraw_order`.
    pub(super) fn end_day_orders(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        if !self.is_external() {
            return self.expire_day_orders(slice);
        }
        let stamp = Self::stamp(slice, Step::SessionClose);
        for id in self.ending_at_close(slice) {
            self.withdraw_order(stamp, id)?;
        }
        Ok(())
    }

    /// True when an external session's close waits for the venue to end its orders.
    pub(super) fn session_close_waits(&self, slice: &SliceInfo) -> bool {
        self.is_external() && slice.closes_session && !self.ending_at_close(slice).is_empty()
    }
}
