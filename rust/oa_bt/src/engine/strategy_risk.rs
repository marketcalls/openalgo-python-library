//! The strategy's risk bookkeeping between the steps of the slice sequence: its trading
//! state, the daily loss measure, the drawdown watch, the bars that the cooldown and
//! the time exit count, and the engine exits waiting for an open.
//!
//! The engine feeds it every fill (`after_fill`, from `fills::execute_fill`) and every
//! session start; the rules that read it run in `aggregate` (steps 1, 2, 6 and 8) and
//! `entry_rules` (submission gates, the time exit). Slices are counted by their index
//! on the run's grid.

use crate::exec::{FillReason, InstrumentId};
use crate::model::Day;
use crate::portfolio::{FillEffect, FillKind, PositionChange, PositionKey};
use crate::rms::StrategyRiskConfig;
use crate::rms::{AccountTotals, DailyLossMeasure, DrawdownWatch, ExitReason, TradingLock};

/// An engine exit waiting for its instrument's next base-bar open.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PendingExit {
    pub(super) key: PositionKey,
    pub(super) reason: ExitReason,
}

/// What a fill tells the risk bookkeeping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FillFacts {
    /// Index of the slice the fill happened in.
    pub(super) slice: usize,
    pub(super) key: PositionKey,
    pub(super) reason: FillReason,
    pub(super) effect: FillEffect,
    /// The account's totals after the fill.
    pub(super) totals: AccountTotals,
    /// Every position of the strategy is flat after the fill.
    pub(super) strategy_flat: bool,
}

/// The risk state of the run's one strategy.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct StrategyRisk {
    /// ACTIVE, REDUCING or HALTED, with the cause.
    pub(super) lock: TradingLock,
    /// The IST session date of the slice being run (`None` before the first).
    session_day: Option<Day>,
    /// The daily loss measure (SPEC-16).
    pub(super) daily_loss: DailyLossMeasure,
    /// The drawdown halt, when the run sets one.
    pub(super) drawdown: Option<DrawdownWatch>,
    /// Slice of the last exit that left each instrument flat, by registration index.
    last_exit_slice: Vec<Option<usize>>,
    /// Slice in which each open position was entered, in the order they opened.
    entered_at: Vec<(PositionKey, usize)>,
    /// Entry slice of a position closed by a roll, handed to the contract it rolls to.
    roll_carry: Option<usize>,
    /// Exits of the aggregate rules waiting for an open, in the order they were queued.
    pending_exits: Vec<PendingExit>,
}

impl StrategyRisk {
    /// The state at the start of a run: ACTIVE, nothing held, the drawdown peak at the
    /// capital.
    pub(super) fn new(
        config: &StrategyRiskConfig,
        capital_rupees: f64,
        instruments: usize,
    ) -> StrategyRisk {
        StrategyRisk {
            lock: TradingLock::default(),
            session_day: None,
            daily_loss: DailyLossMeasure::default(),
            drawdown: config
                .drawdown_halt_percent
                .map(|percent| DrawdownWatch::new(percent, capital_rupees)),
            last_exit_slice: vec![None; instruments],
            entered_at: Vec::new(),
            roll_carry: None,
            pending_exits: Vec::new(),
        }
    }

    /// Note the session of a slice. Returns true when it is a later session than the
    /// previous slice's (a session boundary was crossed); the first slice of the run
    /// starts the first session without crossing one.
    pub(super) fn enter_session(&mut self, day: Day) -> bool {
        let previous = self.session_day.replace(day);
        if previous == Some(day) {
            return false;
        }
        self.daily_loss.start_session();
        previous.is_some()
    }

    /// Account for one fill: the daily loss runs, the entry slice of a new position and
    /// the exit slice of a closed one. A roll's fills hand the entry slice from the
    /// near contract to the next and start no cooldown.
    pub(super) fn after_fill(&mut self, fill: FillFacts) {
        let roll = fill.reason == FillReason::Roll;
        self.daily_loss
            .after_fill(fill.totals, fill.strategy_flat, roll);
        match (fill.effect.kind, fill.effect.change) {
            (FillKind::Entry, _) => {
                let carried = if roll { self.roll_carry.take() } else { None };
                let entered = carried.unwrap_or(fill.slice);
                self.entered_at.retain(|(key, _)| *key != fill.key);
                self.entered_at.push((fill.key, entered));
            }
            (FillKind::Reduce, PositionChange::Closed) => {
                let entered = self.entered_at(fill.key);
                self.entered_at.retain(|(key, _)| *key != fill.key);
                if roll {
                    self.roll_carry = entered;
                } else if let Some(slot) = self.last_exit_slice.get_mut(fill.key.instrument.index())
                {
                    *slot = Some(fill.slice);
                }
            }
            _ => {}
        }
    }

    /// The slice of the last exit that left an instrument flat.
    pub(super) fn last_exit_slice(&self, instrument: InstrumentId) -> Option<usize> {
        self.last_exit_slice
            .get(instrument.index())
            .copied()
            .flatten()
    }

    /// The slice in which an open position was entered.
    pub(super) fn entered_at(&self, key: PositionKey) -> Option<usize> {
        self.entered_at
            .iter()
            .find(|(held, _)| *held == key)
            .map(|(_, slice)| *slice)
    }

    /// The open positions of an instrument with their entry slices, in opening order.
    pub(super) fn entries_on(&self, instrument: InstrumentId) -> Vec<(PositionKey, usize)> {
        self.entered_at
            .iter()
            .filter(|(key, _)| key.instrument == instrument)
            .copied()
            .collect()
    }

    /// Queue an exit for the next open, unless one is already waiting for the position.
    pub(super) fn queue_exit(&mut self, key: PositionKey, reason: ExitReason) {
        if self.pending_exits.iter().all(|pending| pending.key != key) {
            self.pending_exits.push(PendingExit { key, reason });
        }
    }

    /// True when an exit is waiting for the position.
    pub(super) fn exit_is_pending(&self, key: PositionKey) -> bool {
        self.pending_exits.iter().any(|pending| pending.key == key)
    }

    /// Take the waiting exits of an instrument (all of them with `None`).
    pub(super) fn take_exits(&mut self, instrument: Option<InstrumentId>) -> Vec<PendingExit> {
        let (taken, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending_exits)
            .into_iter()
            .partition(|pending| instrument.map_or(true, |i| pending.key.instrument == i));
        self.pending_exits = kept;
        taken
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Product;

    fn key(instrument: u32) -> PositionKey {
        PositionKey {
            instrument: InstrumentId(instrument),
            product: Product::Nrml,
        }
    }

    fn fill(slice: usize, instrument: u32, reason: FillReason, kind: FillKind) -> FillFacts {
        let (change, flat) = match kind {
            FillKind::Reduce => (PositionChange::Closed, true),
            _ => (PositionChange::Opened, false),
        };
        FillFacts {
            slice,
            key: key(instrument),
            reason,
            effect: FillEffect {
                kind,
                change,
                realized_rupees: 0.0,
                quantity_after: if flat { 0.0 } else { 75.0 },
            },
            totals: AccountTotals::default(),
            strategy_flat: flat,
        }
    }

    fn risk() -> StrategyRisk {
        StrategyRisk::new(&StrategyRiskConfig::default(), 1_000_000.0, 2)
    }

    #[test]
    fn a_closing_exit_starts_the_instruments_cooldown() {
        let mut risk = risk();
        risk.after_fill(fill(3, 1, FillReason::Entry, FillKind::Entry));
        let exit = FillReason::Exit(ExitReason::Signal);
        risk.after_fill(fill(7, 1, exit, FillKind::Reduce));
        assert_eq!(risk.last_exit_slice(InstrumentId(1)), Some(7));
        assert_eq!(risk.last_exit_slice(InstrumentId(0)), None);
        assert_eq!(risk.entered_at(key(1)), None);
    }

    #[test]
    fn a_roll_hands_the_entry_slice_to_the_next_contract_and_starts_no_cooldown() {
        let mut risk = risk();
        risk.after_fill(fill(1, 0, FillReason::Entry, FillKind::Entry));
        risk.after_fill(fill(3, 0, FillReason::Roll, FillKind::Reduce));
        risk.after_fill(fill(3, 1, FillReason::Roll, FillKind::Entry));
        assert_eq!(risk.entered_at(key(1)), Some(1));
        assert_eq!(risk.last_exit_slice(InstrumentId(0)), None);
        assert!(risk.daily_loss.run_is_live());
    }

    #[test]
    fn only_a_later_session_crosses_a_boundary() {
        let mut risk = risk();
        assert!(!risk.enter_session(20_000));
        assert!(!risk.enter_session(20_000));
        assert!(risk.enter_session(20_001));
    }

    #[test]
    fn an_exit_is_queued_once_per_position_and_taken_by_instrument() {
        let mut risk = risk();
        risk.queue_exit(key(0), ExitReason::DailyLossLimit);
        risk.queue_exit(key(0), ExitReason::DrawdownHalt);
        risk.queue_exit(key(1), ExitReason::DailyLossLimit);
        let first = risk.take_exits(Some(InstrumentId(0)));
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].reason, ExitReason::DailyLossLimit);
        assert!(risk.exit_is_pending(key(1)));
        assert_eq!(risk.take_exits(None).len(), 1);
        assert!(!risk.exit_is_pending(key(1)));
    }
}
