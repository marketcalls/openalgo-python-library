//! The trading state of a strategy: ACTIVE, REDUCING or HALTED, and what moves it.
//!
//! The three states drive the entry locks, the daily loss limit and the kill switch
//! (03-architecture 9). REDUCING refuses entries and adds but lets exits through; it
//! lasts until the next session boundary (03:00 IST by default) after a daily loss
//! breach (always, SPEC-18 and the lead ruling) or a combined-stop or lock-profit exit
//! with `after_risk_exit="lock_session"`. HALTED (drawdown halt, kill switch) latches
//! for the rest of a backtest, and until an operator reset in live.

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
}
