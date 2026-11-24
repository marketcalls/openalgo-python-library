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
//! | `leg` | `LegState`: the levels in force on one leg (stop, targets, trail, breakeven) and the calls each step makes |
//! | `trading_state` | `TradingState` (ACTIVE, REDUCING, HALTED), its transition table, and `TradingLock` (the state with its cause, the trading-state gate) |
//! | `daily_loss` | `DailyLossMeasure`: the daily loss measure of SPEC-16, and the breach test |
//! | `drawdown` | `DrawdownWatch`: peak equity and the drawdown halt's floor |
//! | `entry_gates` | The cooldown, maximum-positions, trade-cap and order-value gates on new entries |
//! | `trade_count` | `TradeCount`: the day's entries a strategy has made, for the trade caps |
//! | `daily_profit` | When a day's measure has reached its daily profit target |
//! | `scale_in` | The scale-in limits on adds (`max_adds`, `max_position`, `min_spacing`, `max_risk`, averaging down) |
//! | `vocabulary` | `ExitReason`, `RiskEventName`, `RiskScope`, `RiskEvent` |
//! | `error` | `RmsError` |
//!
//! Built: the leg rules (stop, target and several targets, every trailing form,
//! breakeven, `after_first_target`), engine-held, with a CNC or NRML position's stop held
//! as a GTT order (`engine::gtt`, which keeps this module's leg state as the source of
//! the levels); the daily loss limit, the drawdown halt, the trading state
//! and the entry gates (daily loss gate, halted, cooldown, maximum positions) and the
//! scale-in limits on adds, the daily trade caps, the daily profit target and the order
//! value cap (18-multi-strategy 2.5a), which the engine evaluates in step 6 and at submission
//! (`engine::aggregate`, `engine::entry_rules`, `engine::scale_in`). Not built yet: the combined rules (combined stop and target,
//! lock profit, Phase 3). The engine refuses a run that asks for them with a typed
//! not-implemented error instead of ignoring them.

pub mod daily_loss;
pub mod daily_profit;
pub mod drawdown;
pub mod entry_gates;
pub mod error;
pub mod leg;
pub mod rules;
pub mod scale_in;
pub mod trade_count;
pub mod trading_state;
pub mod vocabulary;

pub use daily_loss::{is_breached, AccountTotals, DailyLossMeasure, DailyLossTotals};
pub use daily_profit::is_target_reached;
pub use drawdown::{DrawdownBreach, DrawdownWatch};
pub use entry_gates::{cooldown_gate, max_positions_gate, order_value_gate, trade_cap_gate};
pub use error::RmsError;
pub use leg::{
    BarPrices, LegContext, LegLevels, LegMemory, LegState, LevelHit, LevelKind, RatchetOutcome,
    TickOutcome, TrailPlan,
};
pub use rules::{
    AfterFirstTarget, AfterRiskExit, BreakevenSpec, Distance, LegRules, RiskBasis, ScaleLimits,
    StrategyRiskConfig, TargetSpec, TrailSpec,
};
pub use scale_in::{scale_in_gate, AddFacts, PositionCap, RiskCap, Spacing};
pub use trade_count::{AccountTrades, TradeCount};
pub use trading_state::{
    next_trading_state, StateChange, TradingLock, TradingState, TradingStateCause,
};
pub use vocabulary::{ExitReason, RiskEvent, RiskEventName, RiskScope};
