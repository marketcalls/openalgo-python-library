//! The venue's answers about a live session's GTTs (`venue_gtts`; design
//! 20-positional-live 3.1, 3.3 to 3.5, unit N1 item N1-2), taken while the session waits,
//! at a timer pause, or while its close waits, as its fills are (`venue_fills`), and
//! recorded in the slice in progress. The fired GTT's answer is `venue_gtt_child`'s.
//!
//! | Call | The GTT | Then |
//! |---|---|---|
//! | `apply_gtt_placed` | `Requested` to `Active` (`GttPlaced`) | a cancellation wanted meanwhile is sent; the leg's newest terms follow (`venue_gtt_sync`) |
//! | `apply_gtt_refused` | `Requested` to `Rejected` (`GttRejected`) | a leg's stop: the leg is exited at market at once (`stop_rejected`) and entries are locked for the session (`gtt_refused`, decision 11.7); a strategy's GTT order ends `Rejected` |
//! | `apply_gtt_modified` | `ModifyPending` to `Active` with the terms sent (`GttModified`) | a partial exit waiting for the cut is sent |
//! | `apply_gtt_modify_refused` | `ModifyPending` to `Active`, the old terms standing | a level that still differs is sent again a minute after the last (`GTT_MODIFY_EVERY_NS`) |
//! | `apply_gtt_cancelled` | asked: `CancelPending` to `Canceled` (`GttCanceled`), and an exit that waited for it is sent; not asked: `VenueEnded` (`GttEnded`), the GTT ended outside the session | not asked: the leg is protected again (`venue_gtt_protect`); a strategy's GTT order ends `Canceled` |
//! | `apply_gtt_cancel_refused` | `CancelPending` to `Active` | it works on; the cancellation is sent again in a later slice |
//! | `apply_gtt_ended(how)` | `VenueEnded` (`GttEnded`): expired, or rejected by the broker after its placement | a leg's stop not being cancelled is protected again, but a leg whose stop GTT is rejected a second time is exited at market (`stop_rejected`); a strategy's GTT order ends `Expired` or `Rejected` |
//!
//! Each call refuses with a `VenueError`, changing nothing, for an unknown GTT
//! (`UnknownGtt`) and for a GTT whose state takes no such answer (`GttState`): none of
//! that request waits, as for a second answer to the same request.

use crate::exec::{FillFlags, RejectCode};
use crate::exec::{Gtt, GttAction, GttEventName, GttId, GttPurpose, GttStatus, OrderAction};
use crate::portfolio::PositionKey;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{EventStamp, OrderEventKind};
use super::kernel::Engine;
use super::venue::SendAt;
use super::venue_error::VenueError;
use super::venue_gtt_child::RESTING;
use super::venue_roles::{EngineExit, RiskExitReason};

/// The entry lock a stop GTT the venue refused sets for the session (decision 11.7).
pub const GTT_REFUSED_LOCK: &str = "gtt_refused";

/// The rejections after placement of a leg's stop GTTs at which the leg is exited at
/// market instead of protected again (design 20-positional-live 3.1, row `rejected`).
const REJECTIONS_BEFORE_EXIT: u8 = 2;

/// How a GTT ended at the venue without the session asking (`apply_gtt_ended`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GttEnd {
    /// The broker's expiry ended it.
    Expired,
    /// The broker refused it after its placement.
    Rejected,
}

impl GttEnd {
    /// Every way, in the order of the variants.
    pub const ALL: [GttEnd; 2] = [GttEnd::Expired, GttEnd::Rejected];

    /// The code the bindings use: "expired" or "rejected".
    pub fn code(self) -> &'static str {
        match self {
            GttEnd::Expired => "expired",
            GttEnd::Rejected => "rejected",
        }
    }

    /// The way `code` names (`code`), if any.
    pub fn from_code(code: &str) -> Option<GttEnd> {
        GttEnd::ALL.into_iter().find(|end| end.code() == code)
    }
}

fn key_of(gtt: &Gtt) -> PositionKey {
    PositionKey {
        instrument: gtt.instrument,
        product: gtt.product,
    }
}

impl Engine {
    /// The GTT `id` an answer `answer` names, when its state is one of `states`.
    ///
    /// Errors: `Venue(UnknownGtt)`, `Venue(GttState)`.
    pub(super) fn gtt_answered(
        &self,
        id: GttId,
        answer: &'static str,
        states: &[GttStatus],
    ) -> Result<Gtt, KernelError> {
        let gtt = self
            .gtts
            .book
            .get(id)
            .ok_or(VenueError::UnknownGtt { gtt: id })?;
        if !states.contains(&gtt.status) {
            let status = gtt.status;
            return Err(VenueError::GttState {
                gtt: id,
                status,
                answer,
            }
            .into());
        }
        Ok(gtt.clone())
    }

    /// Where an answer is recorded: the slice in progress and its stamp (`venue_fills`).
    fn gtt_answer_at(&self) -> Result<(SliceInfo, SendAt), KernelError> {
        let slice = self.result_slice()?;
        Ok((slice, self.waiting_at(&slice)))
    }

    /// Record that the venue confirmed the place of GTT `gtt` (module docs).
    ///
    /// Errors: those of `result_slice`; `Venue(UnknownGtt)`; `Venue(GttState)` unless its
    /// place waits for an answer.
    pub fn apply_gtt_placed(&mut self, gtt: GttId) -> Result<EventStamp, KernelError> {
        let (slice, at) = self.gtt_answer_at()?;
        self.gtt_answered(gtt, "placed", &[GttStatus::Requested])?;
        self.gtts.book.apply(gtt, GttAction::Confirm)?;
        self.emit_gtt(at.stamp(), GttEventName::Placed, gtt);
        self.send_wanted_cancel(at.stamp().slice, gtt)?;
        self.after_result(&slice)?;
        Ok(at.stamp())
    }

    /// Record that the venue refused the place of GTT `gtt` (module docs).
    ///
    /// Errors: those of `apply_gtt_placed`; those of the exit.
    pub fn apply_gtt_refused(&mut self, gtt: GttId) -> Result<EventStamp, KernelError> {
        let (slice, at) = self.gtt_answer_at()?;
        let refused = self.gtt_answered(gtt, "refused", &[GttStatus::Requested])?;
        self.gtts.book.apply(gtt, GttAction::Refuse)?;
        self.gtt_follow_mut(gtt).cancel_wanted = false;
        self.emit_gtt(at.stamp(), GttEventName::Rejected, gtt);
        match refused.purpose {
            GttPurpose::Protect => {
                self.venue
                    .controls
                    .entry_locks
                    .insert(GTT_REFUSED_LOCK.to_string());
                self.exit_unprotected_leg(at, key_of(&refused))?;
            }
            GttPurpose::Order(order) => {
                let kind = OrderEventKind::Rejected(RejectCode::Venue);
                self.end_gtt_order(at, order, (OrderAction::Reject, kind))?;
            }
        }
        self.after_result(&slice)?;
        Ok(at.stamp())
    }

    /// Record that the venue confirmed the modification of GTT `gtt`: it rests on the
    /// terms sent (module docs).
    ///
    /// Errors: those of `result_slice`; `Venue(UnknownGtt)`; `Venue(GttState)` unless a
    /// modification of it waits for an answer.
    pub fn apply_gtt_modified(&mut self, gtt: GttId) -> Result<EventStamp, KernelError> {
        let (slice, at) = self.gtt_answer_at()?;
        self.gtt_answered(gtt, "modified", &[GttStatus::ModifyPending])?;
        self.gtts.book.apply(gtt, GttAction::ConfirmModify)?;
        let sent = self.gtt_follow_mut(gtt).pending.take();
        let clock = self.venue.clock_ns;
        if let (Some(terms), Some(modified)) = (sent, self.gtts.book.get_mut(gtt)) {
            modified.leg = terms.leg(modified.leg.role);
            modified.quantity_units = terms.units;
            modified.last_modified_slice = Some(at.stamp().slice);
            modified.eligible_from = clock;
        }
        self.emit_gtt(at.stamp(), GttEventName::Modified, gtt);
        self.send_wanted_cancel(at.stamp().slice, gtt)?;
        self.after_result(&slice)?;
        Ok(at.stamp())
    }

    /// Record that the venue refused the modification of GTT `gtt`: its old terms stand
    /// (module docs).
    ///
    /// Errors: those of `apply_gtt_modified`.
    pub fn apply_gtt_modify_refused(&mut self, gtt: GttId) -> Result<(), KernelError> {
        let (slice, at) = self.gtt_answer_at()?;
        self.gtt_answered(gtt, "modify_refused", &[GttStatus::ModifyPending])?;
        self.gtts.book.apply(gtt, GttAction::RefuseModify)?;
        let follow = self.gtt_follow_mut(gtt);
        follow.pending = None;
        follow.refused = true;
        self.send_wanted_cancel(at.stamp().slice, gtt)?;
        self.after_result(&slice)
    }

    /// Record that the venue cancelled GTT `gtt`, on the session's request or not (module
    /// docs).
    ///
    /// Errors: those of `result_slice`; `Venue(UnknownGtt)`; `Venue(GttState)` for a GTT
    /// that does not rest; those of the exits and of `protect`.
    pub fn apply_gtt_cancelled(&mut self, gtt: GttId) -> Result<EventStamp, KernelError> {
        let (slice, at) = self.gtt_answer_at()?;
        let cancelled = self.gtt_answered(gtt, "cancelled", &RESTING)?;
        let asked = cancelled.status == GttStatus::CancelPending;
        let (action, event) = match asked {
            true => (GttAction::ConfirmCancel, GttEventName::Canceled),
            false => (GttAction::VenueEnd, GttEventName::Ended),
        };
        self.gtts.book.apply(gtt, action)?;
        self.emit_gtt(at.stamp(), event, gtt);
        match cancelled.purpose {
            GttPurpose::Order(order) => {
                let ended = (OrderAction::Cancel, OrderEventKind::Canceled);
                self.end_gtt_order(at, order, ended)?;
            }
            GttPurpose::Protect if !asked => {
                self.protect_gtt_leg(at, key_of(&cancelled))?;
            }
            GttPurpose::Protect => {}
        }
        self.after_result(&slice)?;
        Ok(at.stamp())
    }

    /// Record that the venue refused to cancel GTT `gtt`: it works on, and the
    /// cancellation is sent again in a later slice (module docs).
    ///
    /// Errors: those of `result_slice`; `Venue(UnknownGtt)`; `Venue(GttState)` unless a
    /// cancellation of it waits for an answer.
    pub fn apply_gtt_cancel_refused(&mut self, gtt: GttId) -> Result<(), KernelError> {
        let (slice, at) = self.gtt_answer_at()?;
        self.gtt_answered(gtt, "cancel_refused", &[GttStatus::CancelPending])?;
        self.gtts.book.apply(gtt, GttAction::RefuseCancel)?;
        let follow = self.gtt_follow_mut(gtt);
        follow.cancel_wanted = true;
        follow.cancel_refused_slice = Some(at.stamp().slice);
        self.after_result(&slice)
    }

    /// Record that GTT `gtt` ended at the venue as `how` says, without the session asking
    /// (module docs).
    ///
    /// Errors: those of `apply_gtt_cancelled`.
    pub fn apply_gtt_ended(&mut self, gtt: GttId, how: GttEnd) -> Result<EventStamp, KernelError> {
        let (slice, at) = self.gtt_answer_at()?;
        let ended = self.gtt_answered(gtt, "ended", &RESTING)?;
        let asked = ended.status == GttStatus::CancelPending;
        self.gtts.book.apply(gtt, GttAction::VenueEnd)?;
        self.emit_gtt(at.stamp(), GttEventName::Ended, gtt);
        match (ended.purpose, how) {
            (GttPurpose::Order(order), GttEnd::Expired) => {
                let expired = (OrderAction::Expire, OrderEventKind::Expired);
                self.end_gtt_order(at, order, expired)?;
            }
            (GttPurpose::Order(order), GttEnd::Rejected) => {
                let kind = OrderEventKind::Rejected(RejectCode::Venue);
                self.end_gtt_order(at, order, (OrderAction::Reject, kind))?;
            }
            (GttPurpose::Protect, _) if asked => {}
            (GttPurpose::Protect, GttEnd::Expired) => {
                self.protect_gtt_leg(at, key_of(&ended))?;
            }
            (GttPurpose::Protect, GttEnd::Rejected) => self.after_rejection(at, &ended)?,
        }
        self.after_result(&slice)?;
        Ok(at.stamp())
    }

    /// A leg's stop GTT rejected after its placement: protected again, or exited at market
    /// once it is the leg's second (module docs).
    ///
    /// Errors: those of the exit and of `protect`.
    fn after_rejection(&mut self, at: SendAt, gtt: &Gtt) -> Result<(), KernelError> {
        let key = key_of(gtt);
        let rejections = match self.leg_gtt_mut(key) {
            Some(leg) => {
                leg.rejections = leg.rejections.saturating_add(1);
                leg.rejections
            }
            None => 1,
        };
        if rejections >= REJECTIONS_BEFORE_EXIT {
            return self.exit_unprotected_leg(at, key);
        }
        self.protect_gtt_leg(at, key).map(|_| ())
    }

    /// Exit `key`'s leg at market for a rejected stop (`stop_rejected`), unless it is flat
    /// or an exit closes it already.
    ///
    /// Errors: those of the exit owner.
    fn exit_unprotected_leg(&mut self, at: SendAt, key: PositionKey) -> Result<(), KernelError> {
        if self.held_units(key) == 0.0 || self.exit_closes_leg(key)? {
            return Ok(());
        }
        let exit = EngineExit::Risk(RiskExitReason::StopRejected);
        self.request_flagged_exit(at, key, exit, FillFlags::NONE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gtt_end_is_named_by_its_code() {
        assert_eq!(GttEnd::from_code("expired"), Some(GttEnd::Expired));
        assert_eq!(GttEnd::from_code("rejected"), Some(GttEnd::Rejected));
        assert_eq!(GttEnd::from_code("cancelled"), None);
    }
}
