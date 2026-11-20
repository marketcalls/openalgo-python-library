//! The fund's stepping (18-multi-strategy 2.3): the stage it stands at and one engine's
//! turn at a time, so that a pause handed to a strategy's driver resumes exactly where it
//! stopped.
//!
//! Every turn of an engine is the same: in a shared pool, link the engine to the account
//! from the ledger (but in phase S, whose step 8 opens nothing); check that the engine
//! stands where the fund left it; run its phase; then the ledger takes the engine's
//! figures and the record reads the orders the account refused or ended. A turn whose
//! phase pauses for a strategy's driver ends when the fund is advanced again, after the
//! driver's decisions.

use crate::engine::kernel::Phase;
use crate::engine::signals::decide_signals;
use crate::engine::{AccountLink, Engine, KernelError, Pause, SliceInfo};
use crate::model::{Day, TsNanos};

use super::rules::account_measure;
use super::timeline::{self, Cursor, Instant};
use super::{Decider, Fund, FundError, FundPause, Pool};

/// Where the fund stands between two engine turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stage {
    /// Before the next instant of the time line.
    Next,
    /// Phase O of the instant: the engine at `turn` of `Fund::turns` next; `paused` when it
    /// stands at a timer pause its driver has decided.
    Opens { turn: usize, paused: bool },
    /// Phase L of the round: the participant at `turn` next, then phase A.
    Levels { turn: usize },
    /// Phase C of the round: the participant at `turn` next; `paused` when it stands at its
    /// step 7 pause its driver has decided.
    Decisions { turn: usize, paused: bool },
    /// Phase S of the round: the participant at `turn` next, then the account's row.
    Close { turn: usize },
    /// The runs that ended at the round: the one at `turn` next.
    Ends { turn: usize },
    /// Every run has finished.
    Finished,
}

/// The round in progress: its close instant (UTC nanoseconds) and IST session day.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Round {
    pub(super) at: TsNanos,
    pub(super) day: Day,
}

/// What one step did: `None` to go on, or the pause a driver decides.
type Step = Option<FundPause>;

/// The fund's error for an error of strategy `strategy`'s engine.
fn of_engine(strategy: usize) -> impl Fn(KernelError) -> FundError {
    move |error| FundError::Engine { strategy, error }
}

/// `OutOfStep` unless `engine` stands at `phase`.
fn expect(strategy: usize, engine: &Engine, phase: Phase) -> Result<(), FundError> {
    match engine.phase() == phase {
        true => Ok(()),
        false => Err(FundError::OutOfStep { strategy }),
    }
}

impl Fund {
    /// Step until a driver must decide or every run has finished (`Fund::advance`).
    pub(super) fn run(&mut self, engines: &mut [&mut Engine]) -> Result<FundPause, FundError> {
        loop {
            let stop = match self.stage {
                Stage::Next => {
                    self.next_instant();
                    None
                }
                Stage::Opens { turn, paused } => self.open_turn(engines, turn, paused)?,
                Stage::Levels { turn } => self.levels_turn(engines, turn).map(|()| None)?,
                Stage::Decisions { turn, paused } => self.decision_turn(engines, turn, paused)?,
                Stage::Close { turn } => self.close_turn(engines, turn).map(|()| None)?,
                Stage::Ends { turn } => self.end_turn(engines, turn)?,
                Stage::Finished => return Err(FundError::Finished),
            };
            if let Some(pause) = stop {
                return Ok(pause);
            }
            if self.stage == Stage::Finished {
                return Ok(FundPause::Finished);
            }
        }
    }

    /// Take the next instant of the time line: a round, the open phases, or the end.
    fn next_instant(&mut self) {
        match timeline::next_instant(&self.lanes) {
            None => self.stage = Stage::Finished,
            Some(Instant::Close { at, day }) => {
                timeline::closing_at(&self.lanes, at, &mut self.turns);
                self.round = Some(Round { at, day });
                self.stage = Stage::Levels { turn: 0 };
            }
            Some(Instant::Open(at)) => {
                timeline::opening_at(&self.lanes, at, &mut self.turns);
                self.stage = Stage::Opens {
                    turn: 0,
                    paused: false,
                };
            }
        }
    }

    /// Phase O for the engine at `turn`: steps 1 and 2 of its slice; a timer pause goes to
    /// its driver, and step 2's rest runs at the next step (`paused`).
    fn open_turn(
        &mut self,
        engines: &mut [&mut Engine],
        turn: usize,
        paused: bool,
    ) -> Result<Step, FundError> {
        let Some(&strategy) = self.turns.get(turn) else {
            self.stage = Stage::Next;
            return Ok(None);
        };
        let Cursor::Before(slice) = self.lanes[strategy].cursor else {
            return Err(FundError::OutOfStep { strategy });
        };
        let engine = &mut *engines[strategy];
        self.link(strategy, engine)?;
        let failed = of_engine(strategy);
        if paused {
            expect(strategy, engine, Phase::Timers(slice))?;
            engine.resume_open_phase(&slice).map_err(&failed)?;
        } else {
            expect(strategy, engine, Phase::Ready(slice.index))?;
            if let Some(pause) = engine.open_phase(&slice).map_err(&failed)? {
                if !self.decide_in_rust(strategy, engine, &pause)? {
                    self.stage = Stage::Opens { turn, paused: true };
                    return Ok(Some(FundPause::Strategy { strategy, pause }));
                }
                engine.resume_open_phase(&slice).map_err(&failed)?;
            }
        }
        self.after_turn(strategy, engine);
        self.lanes[strategy].cursor = Cursor::Open(slice);
        self.stage = Stage::Opens {
            turn: turn + 1,
            paused: false,
        };
        Ok(None)
    }

    /// Phase L for the participant at `turn`: steps 3 to 6; after the last, phase A.
    fn levels_turn(&mut self, engines: &mut [&mut Engine], turn: usize) -> Result<(), FundError> {
        let Some(&strategy) = self.turns.get(turn) else {
            self.account_phase(engines)?;
            self.stage = Stage::Decisions {
                turn: 0,
                paused: false,
            };
            return Ok(());
        };
        let slice = self.open_slice(strategy)?;
        let engine = &mut *engines[strategy];
        self.link(strategy, engine)?;
        expect(strategy, engine, Phase::Opened(slice))?;
        engine.levels_phase(&slice).map_err(of_engine(strategy))?;
        self.after_turn(strategy, engine);
        self.stage = Stage::Levels { turn: turn + 1 };
        Ok(())
    }

    /// Phase A: the account's daily loss limit on the ledger; at a breach every engine
    /// still running takes it, in call order.
    fn account_phase(&mut self, engines: &mut [&mut Engine]) -> Result<(), FundError> {
        let Some(round) = self.round else {
            return Ok(());
        };
        let measure = account_measure(round.day, self.book.measures());
        let Some(breach) = self.rules.breach(round.at, round.day, measure) else {
            return Ok(());
        };
        self.record.breach(breach);
        for (strategy, engine) in engines.iter_mut().enumerate() {
            if self.lanes[strategy].cursor == Cursor::Ended {
                continue;
            }
            engine.account_breach(breach).map_err(of_engine(strategy))?;
            self.after_turn(strategy, engine);
        }
        Ok(())
    }

    /// Phase C for the participant at `turn`: its step 7 pause goes to its driver; the
    /// turn ends at the next step, after the driver's decisions (`paused`).
    fn decision_turn(
        &mut self,
        engines: &mut [&mut Engine],
        turn: usize,
        paused: bool,
    ) -> Result<Step, FundError> {
        let Some(&strategy) = self.turns.get(turn) else {
            self.stage = Stage::Close { turn: 0 };
            return Ok(None);
        };
        let slice = self.open_slice(strategy)?;
        let engine = &mut *engines[strategy];
        if paused {
            expect(strategy, engine, Phase::Paused(slice))?;
        } else {
            self.link(strategy, engine)?;
            expect(strategy, engine, Phase::Opened(slice))?;
            if let Some(pause) = engine.decision_phase(&slice) {
                if !self.decide_in_rust(strategy, engine, &pause)? {
                    self.stage = Stage::Decisions { turn, paused: true };
                    return Ok(Some(FundPause::Strategy { strategy, pause }));
                }
            }
        }
        self.after_turn(strategy, engine);
        self.stage = Stage::Decisions {
            turn: turn + 1,
            paused: false,
        };
        Ok(None)
    }

    /// Phase S for the participant at `turn`: step 8; after the last, the account's row
    /// and the runs that ended.
    fn close_turn(&mut self, engines: &mut [&mut Engine], turn: usize) -> Result<(), FundError> {
        let Some(&strategy) = self.turns.get(turn) else {
            self.end_round();
            return Ok(());
        };
        let slice = self.open_slice(strategy)?;
        let engine = &mut *engines[strategy];
        // Step 8 only closes (the session's MIS close, expiries) and opens nothing, so
        // the engine needs no new room: its link of phase C stands.
        let phase = engine.phase();
        if phase != Phase::Opened(slice) && phase != Phase::Paused(slice) {
            return Err(FundError::OutOfStep { strategy });
        }
        // A backtest's slice ends without a pause; only a live session's close waits.
        if engine
            .end_slice(&slice)
            .map_err(of_engine(strategy))?
            .is_some()
        {
            return Err(FundError::OutOfStep { strategy });
        }
        self.after_turn(strategy, engine);
        self.lanes[strategy].cursor = timeline::before(engine, slice.index + 1);
        self.stage = Stage::Close { turn: turn + 1 };
        Ok(())
    }

    /// The account's row of the round, then the participants whose grid ended at it.
    fn end_round(&mut self) {
        if let Some(round) = self.round.take() {
            let book = &self.book;
            let measure = account_measure(round.day, book.measures());
            let figures = [
                book.equity_rupees(),
                book.free_cash_rupees(),
                book.margin_rupees(),
                measure,
            ];
            self.record.row(round.at, round.day, figures);
        }
        timeline::keep_done(&self.lanes, &mut self.turns);
        self.stage = Stage::Ends { turn: 0 };
    }

    /// The run of the engine at `turn` ended at the round: its `Pause::Finished` goes to
    /// its driver, and it takes no more phases.
    fn end_turn(&mut self, engines: &mut [&mut Engine], turn: usize) -> Result<Step, FundError> {
        let Some(&strategy) = self.turns.get(turn) else {
            self.stage = Stage::Next;
            return Ok(None);
        };
        let engine = &mut *engines[strategy];
        expect(strategy, engine, Phase::Ready(engine.timeline().len()))?;
        let pause = engine.finish_grid().map_err(of_engine(strategy))?;
        self.after_turn(strategy, engine);
        self.lanes[strategy].cursor = Cursor::Ended;
        self.stage = Stage::Ends { turn: turn + 1 };
        match self.decide_in_rust(strategy, engine, &pause)? {
            true => Ok(None),
            false => Ok(Some(FundPause::Strategy { strategy, pause })),
        }
    }

    /// The slice whose open phase strategy `strategy`'s engine has run.
    fn open_slice(&self, strategy: usize) -> Result<SliceInfo, FundError> {
        match self.lanes[strategy].cursor {
            Cursor::Open(slice) => Ok(slice),
            _ => Err(FundError::OutOfStep { strategy }),
        }
    }

    /// In a shared pool, link the engine to the account for its turn (module docs).
    fn link(&self, strategy: usize, engine: &mut Engine) -> Result<(), FundError> {
        if self.pool == Pool::Separate {
            return Ok(());
        }
        let link = AccountLink {
            room_rupees: Some(self.book.room_rupees(strategy)),
        };
        engine.set_account_link(link).map_err(of_engine(strategy))
    }

    /// Decide `pause` with strategy `strategy`'s Rust driver; false when its caller
    /// decides.
    fn decide_in_rust(
        &mut self,
        strategy: usize,
        engine: &mut Engine,
        pause: &Pause,
    ) -> Result<bool, FundError> {
        match &mut self.deciders[strategy] {
            Decider::Caller => Ok(false),
            Decider::Signals { arrays, skipped } => {
                decide_signals(engine, arrays, pause, skipped).map_err(of_engine(strategy))?;
                Ok(true)
            }
        }
    }

    /// After a turn: the ledger takes the engine's figures, and the record reads the
    /// orders the account refused or ended since its last turn.
    fn after_turn(&mut self, strategy: usize, engine: &mut Engine) {
        self.book.take(strategy, engine.figures());
        let cancels = engine.take_account_cancels();
        let lane = &mut self.lanes[strategy];
        let events = engine.recorded_events();
        let new = events.get(lane.events_read..).unwrap_or(&[]);
        lane.events_read = events.len();
        self.record.refusals(strategy, new, cancels);
    }
}
