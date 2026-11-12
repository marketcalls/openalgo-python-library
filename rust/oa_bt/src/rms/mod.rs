//! Risk management (RMS): rule configuration, the leg rules, the trading state and the
//! fixed vocabularies of exit reasons and risk events.
//!
//! The rules behave like OpenAlgo's live `/strategy` module by default
//! (05-risk-management 1); the deliberate differences are options in
//! `rules::StrategyRiskConfig`. In a backtest they are evaluated on base bars with the
//! split of 05-risk-management 6.2 (levels in step 3, ratchets in step 4, aggregate
//! rules in step 6); in tick mode on each price event in the live order (6.1).
//!
//! | Module | Responsibility |
//! |---|---|
//! | `rules` | `LegRules`, `Distance`, `TrailSpec`, `BreakevenSpec`, `StrategyRiskConfig`, scale-in limits |
//! | `leg` | `LegState`: the levels in force on one leg and the calls each step makes |
//! | `trading_state` | `TradingState` (ACTIVE, REDUCING, HALTED) and its transition table |
//! | `vocabulary` | `ExitReason`, `RiskEventName`, `RiskScope`, `RiskEvent` |
//! | `error` | `RmsError` |
//!
//! Not built yet: the leg rules themselves (`leg`), the aggregate rules of step 6
//! (combined stop and target, lock profit, daily loss limit, drawdown halt) and the
//! pre-entry gates. The engine refuses a run that asks for them with a typed
//! not-implemented error instead of ignoring them.

pub mod error;
pub mod leg;
pub mod rules;
pub mod trading_state;
pub mod vocabulary;

pub use error::RmsError;
pub use leg::{BarPrices, LegContext, LegState, LevelHit, LevelKind};
pub use rules::{
    AfterFirstTarget, AfterRiskExit, BreakevenSpec, Distance, LegRules, RiskBasis, ScaleLimits,
    StrategyRiskConfig, TargetSpec, TrailSpec,
};
pub use trading_state::{next_trading_state, TradingState, TradingStateCause};
pub use vocabulary::{ExitReason, RiskEvent, RiskEventName, RiskScope};
