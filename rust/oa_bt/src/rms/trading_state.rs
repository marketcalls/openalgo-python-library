//! The trading state of a strategy: ACTIVE, REDUCING or HALTED, and what moves it.
//!
//! The three states drive the entry locks, the daily loss limit and the kill switch
//! (03-architecture 9). REDUCING refuses entries and adds but lets exits through; it
//! lasts until the next session boundary (03:00 IST by default) after a daily loss
//! breach (always, SPEC-18 and the lead ruling) or a combined-stop or lock-profit exit
//! with `after_risk_exit="lock_session"`. HALTED (drawdown halt, kill switch) latches
//! for the rest of a backtest, and until an operator reset in live.
//!
//! `TradingLock` holds one strategy's state with the cause that set it, and is the
//! trading-state gate of 03-architecture 4.4 (item 6) for orders that open or add.

use crate::exec::RejectCode;

/// Whether a strategy may open or add to positions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TradingState {
    /// Entries, adds and exits are allowed.
    #[default]
    Active = 0,
    /// Only orders that reduce a position are allowed, until the next session boundary.
    Reducing = 1,
    /// Only orders that reduce a position are allowed, until the end of the run.
    Halted = 2,
}

impl TradingState {
    /// The state name, as in the risk literature and the live module.
    pub fn code(self) -> &'static str {
        match self {
            TradingState::Active => "ACTIVE",
            TradingState::Reducing => "REDUCING",
            TradingState::Halted => "HALTED",
        }
    }

    /// True when new entries and adds are allowed.
    pub fn allows_entries(self) -> bool {
        self == TradingState::Active
    }
}

/// What moves the trading state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TradingStateCause {
    /// The daily loss limit was breached.
    DailyLossBreach = 0,
    /// A strategy-level exit (combined stop, lock profit) with entries locked for the
    /// session.
    StrategyExit = 1,
    /// The drawdown halt fired.
    DrawdownHalt = 2,
    /// The kill switch was used.
    KillSwitch = 3,
    /// The session boundary passed.
    SessionBoundary = 4,
}

impl TradingStateCause {
    /// The cause name used in events and results.
    pub fn code(self) -> &'static str {
        match self {
            TradingStateCause::DailyLossBreach => "daily_loss_breach",
            TradingStateCause::StrategyExit => "strategy_exit",
            TradingStateCause::DrawdownHalt => "drawdown_halt",
            TradingStateCause::KillSwitch => "kill_switch",
            TradingStateCause::SessionBoundary => "session_boundary",
        }
    }
}

use TradingState as T;
use TradingStateCause as C;

/// Every legal change as (from, cause, to). A cause with no row leaves the state as it
/// is: a second breach while REDUCING changes nothing.
pub const TRADING_STATE_TRANSITIONS: [(TradingState, TradingStateCause, TradingState); 7] = [
    (T::Active, C::DailyLossBreach, T::Reducing),
    (T::Active, C::StrategyExit, T::Reducing),
    (T::Active, C::DrawdownHalt, T::Halted),
    (T::Active, C::KillSwitch, T::Halted),
    (T::Reducing, C::DrawdownHalt, T::Halted),
    (T::Reducing, C::KillSwitch, T::Halted),
    (T::Reducing, C::SessionBoundary, T::Active),
];

/// The state after a cause; unchanged when the table has no row for it.
pub fn next_trading_state(from: TradingState, cause: TradingStateCause) -> TradingState {
    TRADING_STATE_TRANSITIONS
        .iter()
        .find(|(state, taken, _)| *state == from && *taken == cause)
        .map_or(from, |(_, _, to)| *to)
}

/// A change of trading state, for the `trading_state_changed` event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateChange {
    pub from: TradingState,
    pub to: TradingState,
    pub cause: TradingStateCause,
}

/// One strategy's trading state and the cause that left ACTIVE, which decides the
/// reject code of an entry it refuses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TradingLock {
    state: TradingState,
    /// What set the current state; `None` while ACTIVE.
    cause: Option<TradingStateCause>,
}

impl TradingLock {
    /// The current state.
    pub fn state(&self) -> TradingState {
        self.state
    }

    /// Apply a cause by the transition table; the change when the state moved.
    pub fn apply(&mut self, cause: TradingStateCause) -> Option<StateChange> {
        let from = self.state;
        let to = next_trading_state(from, cause);
        if to == from {
            return None;
        }
        self.state = to;
        self.cause = (to != T::Active).then_some(cause);
        Some(StateChange { from, to, cause })
    }

    /// The trading-state gate for an order that opens or adds.
    ///
    /// Errors: `DailyLossGate` while REDUCING after a daily loss breach (SPEC-18);
    /// `Halted` while HALTED, or REDUCING for any other cause.
    pub fn entry_gate(&self) -> Result<(), RejectCode> {
        match (self.state, self.cause) {
            (T::Active, _) => Ok(()),
            (T::Reducing, Some(C::DailyLossBreach)) => Err(RejectCode::DailyLossGate),
            _ => Err(RejectCode::Halted),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_daily_loss_breach_reduces_until_the_session_boundary() {
        let reducing = next_trading_state(T::Active, C::DailyLossBreach);
        assert_eq!(reducing, T::Reducing);
        assert!(!reducing.allows_entries());
        assert_eq!(next_trading_state(reducing, C::SessionBoundary), T::Active);
    }

    #[test]
    fn a_halt_latches() {
        let halted = next_trading_state(T::Active, C::DrawdownHalt);
        assert_eq!(next_trading_state(halted, C::SessionBoundary), T::Halted);
    }

    #[test]
    fn a_daily_loss_lock_rejects_entries_with_the_daily_loss_gate_until_the_boundary() {
        let mut lock = TradingLock::default();
        let change = lock.apply(C::DailyLossBreach);
        let reducing = StateChange {
            from: T::Active,
            to: T::Reducing,
            cause: C::DailyLossBreach,
        };
        assert_eq!(change, Some(reducing));
        assert_eq!(lock.entry_gate(), Err(RejectCode::DailyLossGate));
        assert_eq!(lock.apply(C::DailyLossBreach), None);
        assert_eq!(
            lock.apply(C::SessionBoundary).map(|c| c.to),
            Some(T::Active)
        );
        assert_eq!(lock.entry_gate(), Ok(()));
    }

    #[test]
    fn a_halt_after_a_daily_loss_rejects_entries_as_halted_and_survives_the_boundary() {
        let mut lock = TradingLock::default();
        lock.apply(C::DailyLossBreach);
        let change = lock.apply(C::DrawdownHalt);
        assert_eq!(
            change.map(|c| (c.from, c.to)),
            Some((T::Reducing, T::Halted))
        );
        assert_eq!(lock.apply(C::SessionBoundary), None);
        assert_eq!(lock.state(), T::Halted);
        assert_eq!(lock.entry_gate(), Err(RejectCode::Halted));
    }

    #[test]
    fn cause_codes_are_distinct() {
        let causes = [
            C::DailyLossBreach,
            C::StrategyExit,
            C::DrawdownHalt,
            C::KillSwitch,
            C::SessionBoundary,
        ];
        let mut codes: Vec<&str> = causes.iter().map(|cause| cause.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), causes.len());
    }
}
