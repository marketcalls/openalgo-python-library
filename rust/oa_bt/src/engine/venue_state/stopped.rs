//! A live session of an earlier trading day that stopped without its close (a crash, a
//! power cut, a killed process), rebuilt so that its carried state can be taken (design
//! 20-positional-live 4.2 step 2, review M7; unit N5c item N5c-1).
//!
//! Such a session leaves only its last saved state (`live_state`) and what its venue
//! answered after that save, which the live layer journaled. The day-close rules are
//! `carried_state`'s and nothing else's, so the rebuild runs them on a session put back
//! where the save stood:
//!
//! 1. `Engine::restore_stopped_state` gives the save to a session built for the saved day
//!    (its instruments and capital) with none of that day's bars, before any of its slices
//!    runs. The session stands where the save stood: in the slice in progress at the save
//!    (in the close that waited for the venue, when the save was made there), and takes
//!    the venue's results there, as a restored session takes them (`venue_fills`). The
//!    save's parts are read and checked as `restore_live_state` reads and checks them, and
//!    each instrument's last close comes back from the save.
//! 2. The live layer gives the results the venue gave after the save through the calls a
//!    same-day restart gives them by (16-session-design 3.6): fills by `apply_fill` with
//!    their flags, order ends, GTT answers, a fired GTT's child.
//! 3. `stop_session` ends the session as the close would, and `carried_state` gives what
//!    it carries: the DAY orders end with no fill beyond the results given, a GTT request
//!    the venue had not answered rests on the terms asked, and a command never handed out
//!    is dropped (`carry`).
//!
//! **Never handed out.** The commands the save holds come back as commands the adapter
//! has not taken, and so do those the results make (a stop GTT placed for an entry that
//! filled after the save): no session sends them, so the carried state drops them. A
//! request the save shows sent (a GTT's place, modification or cancellation waiting for
//! the venue's answer) that the live layer's journal shows never reached the venue is
//! named in `unsent`, and goes back among the commands not taken.
//!
//! Errors refuse the state whole: nothing of it is restored.

use crate::exec::{GttId, GttPurpose, GttStatus, OrderId};
use crate::model::Day;

use super::super::error::KernelError;
use super::super::kernel::{Engine, Phase};
use super::super::venue::VenueCommand;
use super::super::venue_error::VenueError;
use super::super::venue_gtts::GttTerms;
use super::carry_expired::read_last_closes;
use super::error::LiveStateError;
use super::standing::{check_layout, saved_at, Stage};
use super::value::{Read, StateValue};
use super::Saved;

impl Engine {
    /// Give a session built for an earlier trading day the state its live session last
    /// saved (`live_state`), the session having stopped without its close, standing where
    /// that save stood (module docs). `today` is the trading day of the session that
    /// carries it on, and `unsent` the GTTs whose waiting request never reached the venue.
    /// Returns (saved id, new id) for every restored working order, in saved order; the
    /// GTTs keep their ids. Then the venue's later results, `stop_session` and
    /// `carried_state`.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `Venue(UnknownGtt)` for an id of
    /// `unsent` the state holds no GTT of, and `Venue(GttState)` for one with no request
    /// waiting for the venue; `LiveState`, the state refused whole: `NotRestorable` for a
    /// session that has run a slice, traded or was restored already, and for a state saved
    /// once the session had finished; `Version`, `OtherSession` (another day, other
    /// instruments, another capital, or a state of `today` or later, which a same-day
    /// restart resumes with `restore_live_state`) and `Malformed` for the state itself.
    pub fn restore_stopped_state(
        &mut self,
        state: &StateValue,
        today: Day,
        unsent: &[GttId],
    ) -> Result<Vec<(OrderId, OrderId)>, KernelError> {
        if !self.is_external() {
            return Err(VenueError::NotExternal.into());
        }
        self.check_restorable()?;
        self.check_unrun()?;
        let read = Read::root(state);
        let (slice, stage) = self.check_stopped(&read, today)?;
        let saved = self.read_parts(&read)?;
        check_unsent(&saved, unsent)?;
        let closes = read_last_closes(&read, self.instruments.len())?;
        let pairs = self.apply_saved(saved)?;
        let commands: Vec<VenueCommand> = unsent
            .iter()
            .filter_map(|gtt| self.unsent_command(*gtt))
            .collect();
        self.venue.commands.extend(commands);
        for (instrument, close) in closes {
            if let Some(slot @ None) = self.last_close.get_mut(instrument.index()) {
                *slot = Some(close);
            }
        }
        self.phase = self.stand_at(slice, stage)?;
        Ok(pairs)
    }

    /// Refuse a session that has run a slice: the save is put back before any.
    fn check_unrun(&self) -> Result<(), LiveStateError> {
        if self.standing() == (0, Stage::Ready) {
            return Ok(());
        }
        Err(LiveStateError::NotRestorable {
            reason: "a stopped session's state goes to a session built for its day before any \
                     of its slices runs"
                .to_string(),
        })
    }

    /// Check the state's layout, its session (day, instruments, capital), that its day is
    /// before `today`, and that it was saved before the session finished; the slice and
    /// stage it was saved at.
    fn check_stopped(&self, read: &Read<'_>, today: Day) -> Result<(usize, Stage), LiveStateError> {
        check_layout(read)?;
        self.check_same_session(read)?;
        let saved_day: Day = read.whole_of("day")?;
        if saved_day >= today {
            return Err(LiveStateError::OtherSession {
                reason: format!(
                    "day {saved_day} is not before day {today}, the day it is carried to: a \
                     session of the day it was saved resumes it with restore_live_state"
                ),
            });
        }
        let (slice, stage) = saved_at(read)?;
        if stage == Stage::Finished || slice > self.timeline.len() {
            return Err(LiveStateError::NotRestorable {
                reason: "the state was saved once the session had finished: its carried state \
                         is that session's own (carried_state)"
                    .to_string(),
            });
        }
        Ok((slice, stage))
    }

    /// Where the rebuilt session stands (module docs): in the close of slice `slice` when
    /// the save was made while it closed, else in slice `slice`, whose results it takes.
    ///
    /// Errors: `Malformed` for a slice the session's day does not have.
    fn stand_at(&self, slice: usize, stage: Stage) -> Result<Phase, LiveStateError> {
        if stage != Stage::Closing {
            return Ok(Phase::Ready(slice));
        }
        self.timeline
            .slice(slice)
            .map(Phase::Closing)
            .ok_or_else(|| LiveStateError::Malformed {
                path: "live_state.slice".to_string(),
                reason: format!("names slice {slice}, which the session's day does not have"),
            })
    }

    /// The command of the request GTT `gtt` waits for: its place, its modification (on
    /// the terms asked) or its cancellation; `None` for a GTT with none.
    fn unsent_command(&self, gtt: GttId) -> Option<VenueCommand> {
        let found = self.gtts.book.get(gtt)?;
        let terms = GttTerms::of(found);
        match found.status {
            GttStatus::Requested => Some(VenueCommand::PlaceGtt {
                gtt,
                order: match found.purpose {
                    GttPurpose::Order(order) => Some(order),
                    GttPurpose::Protect => None,
                },
                terms,
            }),
            GttStatus::ModifyPending => {
                let follows = &self.venue.gtts.follows;
                let follow = follows.iter().find(|follow| follow.gtt == gtt)?;
                Some(VenueCommand::ModifyGtt {
                    gtt,
                    terms: follow.pending?,
                })
            }
            GttStatus::CancelPending => Some(VenueCommand::CancelGtt { gtt, terms }),
            _ => None,
        }
    }
}

/// Refuse an id of `unsent` that names no GTT of `saved` with a request waiting for the
/// venue (`restore_stopped_state`), before anything of the state is applied.
///
/// Errors: `Venue(UnknownGtt)`, `Venue(GttState)`.
fn check_unsent(saved: &Saved, unsent: &[GttId]) -> Result<(), KernelError> {
    for gtt in unsent {
        let found = saved
            .gtts
            .get(*gtt)
            .ok_or(VenueError::UnknownGtt { gtt: *gtt })?;
        let waiting = [
            GttStatus::Requested,
            GttStatus::ModifyPending,
            GttStatus::CancelPending,
        ];
        if !waiting.contains(&found.status) {
            return Err(VenueError::GttState {
                gtt: *gtt,
                status: found.status,
                answer: "unsent",
            }
            .into());
        }
    }
    Ok(())
}
