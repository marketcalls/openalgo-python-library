//! Where a session stands and whether it can take a saved state now (`venue_state`): the
//! session must not have traded or been restored, the state must be of its day,
//! instruments and capital, saved before the session close began, and the session must
//! stand at or after the point it was saved at and before its own close.
//!
//! A state saved while the session closed (it waited for the venue to end the day's
//! orders) is refused for that reason: a session that restarts after its close finishes
//! without closing again, so it never stands where that state was saved, and nothing of the
//! day is left to decide. The day's open positions and orders are then the broker's books'.

use crate::model::Day;

use super::super::kernel::{Engine, Phase};
use super::error::LiveStateError;
use super::value::Read;
use super::LIVE_STATE_LAYOUT;

/// How far into a slice a session stands, in the order a slice runs them: the timers
/// due at its open fired ahead of it, before it, at its timer pause, its open phase run
/// (steps 1 and 2; a session passes it within one `advance`), paused at its close,
/// waiting at its session close, and after the last slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Stage {
    Ahead,
    Ready,
    Timers,
    Opened,
    Paused,
    Closing,
    Finished,
}

const STAGES: [Stage; 7] = [
    Stage::Ahead,
    Stage::Ready,
    Stage::Timers,
    Stage::Opened,
    Stage::Paused,
    Stage::Closing,
    Stage::Finished,
];

impl Stage {
    pub(super) fn code(self) -> &'static str {
        match self {
            Stage::Ahead => "ahead",
            Stage::Ready => "ready",
            Stage::Timers => "timers",
            Stage::Opened => "opened",
            Stage::Paused => "paused",
            Stage::Closing => "closing",
            Stage::Finished => "finished",
        }
    }
}

impl Engine {
    /// The slice the session stands at and how far into it (`Stage`).
    pub(super) fn standing(&self) -> (usize, Stage) {
        match self.phase {
            Phase::Ahead(slice) => (slice.index, Stage::Ahead),
            Phase::Ready(index) => (index, Stage::Ready),
            Phase::PreOpen(slice, _) => (slice.index, Stage::Ready),
            Phase::Timers(slice) => (slice.index, Stage::Timers),
            Phase::Opened(slice) => (slice.index, Stage::Opened),
            Phase::Paused(slice) => (slice.index, Stage::Paused),
            Phase::Closing(slice) => (slice.index, Stage::Closing),
            Phase::Finished => (self.timeline.len(), Stage::Finished),
        }
    }

    /// The IST trading day of the slice at `index` (the last slice's after the end).
    pub(super) fn day_at(&self, index: usize) -> Option<Day> {
        let last = self.timeline.len().checked_sub(1)?;
        self.timeline.slice(index.min(last)).map(|slice| slice.day)
    }

    /// Refuse a session that has traded or taken a state already (module docs).
    pub(super) fn check_restorable(&self) -> Result<(), LiveStateError> {
        let refuse = |reason: &str| {
            Err(LiveStateError::NotRestorable {
                reason: reason.to_string(),
            })
        };
        if self.venue.restored {
            return refuse("the session was restored already (a state is restored once)");
        }
        let traded = !self.book.orders().is_empty()
            || !self.portfolio.positions().is_empty()
            || !self.record.fills.is_empty()
            || !self.gtts.book.gtts().is_empty();
        if traded {
            return refuse(
                "the session has orders or positions of its own: restore into a session \
                 built for the restart, after its catch-up and before any order",
            );
        }
        Ok(())
    }

    /// Check that `read` is a state of this session saved before its close began, and that
    /// the session stands at or after that point and before its own close (module docs).
    pub(super) fn check_session(&self, read: &Read<'_>) -> Result<(), LiveStateError> {
        let layout: i64 = read.whole_of("layout")?;
        if layout != LIVE_STATE_LAYOUT {
            return Err(LiveStateError::Version { found: layout });
        }
        self.check_same_session(read)?;
        let saved_at = (
            read.whole_of::<usize>("slice")?,
            read.field("stage")?.code(&STAGES, Stage::code)?,
        );
        let not_restorable = |reason: String| Err(LiveStateError::NotRestorable { reason });
        if saved_at.1 >= Stage::Closing {
            return not_restorable(format!(
                "the state was saved while the session closed (stage {}): a session does not \
                 resume at or after its close; the day's open positions and orders are in the \
                 broker's books",
                saved_at.1.code()
            ));
        }
        if matches!(self.phase, Phase::Closing(_) | Phase::Finished) {
            return not_restorable(
                "the session's close has begun: restore before the session close".to_string(),
            );
        }
        let (slice, stage) = self.standing();
        if (slice, stage) < saved_at {
            return not_restorable(format!(
                "the session stands at slice {slice} ({}), before the saved slice {} ({}): \
                 catch it up first",
                stage.code(),
                saved_at.0,
                saved_at.1.code()
            ));
        }
        Ok(())
    }

    /// Check that `read` names this session's instruments, day and capital.
    fn check_same_session(&self, read: &Read<'_>) -> Result<(), LiveStateError> {
        let other = |reason: String| Err(LiveStateError::OtherSession { reason });
        let saved = read.items_of("instruments")?;
        let named: Vec<(String, String)> = saved
            .iter()
            .map(|item| {
                let symbol = item.field("symbol")?.text()?.to_string();
                Ok((symbol, item.field("exchange")?.text()?.to_string()))
            })
            .collect::<Result<_, LiveStateError>>()?;
        let ours: Vec<(String, String)> = self
            .instruments
            .iter()
            .map(|spec| (spec.symbol.clone(), spec.exchange.code().to_string()))
            .collect();
        if named != ours {
            return other(format!("instruments {named:?}, the session's {ours:?}"));
        }
        let (slice, _) = self.standing();
        let saved_day: Option<Day> = read.opt_whole_of("day")?;
        if saved_day != self.day_at(slice) {
            return other(format!(
                "day {saved_day:?}, the session's {:?}",
                self.day_at(slice)
            ));
        }
        let saved_capital = read.field("account")?.f64_of("capital_rupees")?;
        if saved_capital != self.config.capital_rupees {
            let ours = self.config.capital_rupees;
            return other(format!(
                "capital {saved_capital} rupees, the session's {ours}"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_opened_slice_stands_after_its_timers_and_before_its_pause() {
        let codes = STAGES.map(Stage::code);
        let expected = [
            "ahead", "ready", "timers", "opened", "paused", "closing", "finished",
        ];
        assert_eq!(codes, expected);
        assert!(STAGES.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
