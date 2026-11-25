//! The fixed vocabularies of risk management: why a position was exited, the risk
//! events and their scopes (05-risk-management 7, decisions SPEC-18 and SPEC-22).
//!
//! Codes are the strings OpenAlgo's live strategy module writes, so results, journals
//! and the server's records use one vocabulary.

/// Why a fill reduced or closed a position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExitReason {
    /// A leg's stop (fixed, trailed or moved to breakeven).
    LegSl = 0,
    /// A leg's target; the fill records which target.
    LegTarget = 1,
    /// The leg's maximum holding time.
    TimeExit = 2,
    /// The combined stop of a run.
    OverallSl = 3,
    /// The combined target of a run.
    OverallTarget = 4,
    /// Lock profit fired.
    LockProfit = 5,
    /// The daily loss limit.
    DailyLossLimit = 6,
    /// The intraday square-off time.
    Scheduler = 7,
    /// A manual or kill-switch exit.
    Manual = 8,
    /// An intraday position closed at the session's last close (no bar at or after the
    /// square-off).
    Eod = 9,
    /// The drawdown halt.
    DrawdownHalt = 10,
    /// Liquidation after a margin call.
    MarginLiquidation = 11,
    /// Expiry settlement.
    Expiry = 12,
    /// The strategy's own exit (`exit()`, a reducing order, `exit_when`).
    Signal = 13,
    /// The account's daily loss limit of a portfolio run, across its strategies
    /// (`engine::account_breach`).
    AccountLossLimit = 14,
    /// The strategy's daily profit target (`daily_profit_target`, 18-multi-strategy 2.5a).
    DailyProfitTarget = 15,
    /// The account's daily profit target of a portfolio run or live session, across its
    /// strategies (`account_daily_profit_target`; `engine::account_breach`).
    AccountProfitTarget = 16,
    /// A derivatives position exited at the square-off of its positional exit day, before
    /// the exchange settles the contract by delivery (stock futures and options, every
    /// option, MCX contracts before their tender period; design 20-positional-live 9).
    ExpiryExit = 17,
}

impl ExitReason {
    /// Every reason, in code order.
    pub const ALL: [ExitReason; 18] = [
        ExitReason::LegSl,
        ExitReason::LegTarget,
        ExitReason::TimeExit,
        ExitReason::OverallSl,
        ExitReason::OverallTarget,
        ExitReason::LockProfit,
        ExitReason::DailyLossLimit,
        ExitReason::Scheduler,
        ExitReason::Manual,
        ExitReason::Eod,
        ExitReason::DrawdownHalt,
        ExitReason::MarginLiquidation,
        ExitReason::Expiry,
        ExitReason::Signal,
        ExitReason::AccountLossLimit,
        ExitReason::DailyProfitTarget,
        ExitReason::AccountProfitTarget,
        ExitReason::ExpiryExit,
    ];

    /// The reason string, as the live module writes it.
    pub fn code(self) -> &'static str {
        match self {
            ExitReason::LegSl => "leg_sl",
            ExitReason::LegTarget => "leg_target",
            ExitReason::TimeExit => "time_exit",
            ExitReason::OverallSl => "overall_sl",
            ExitReason::OverallTarget => "overall_target",
            ExitReason::LockProfit => "lock_profit",
            ExitReason::DailyLossLimit => "daily_loss_limit",
            ExitReason::Scheduler => "scheduler",
            ExitReason::Manual => "manual",
            ExitReason::Eod => "eod",
            ExitReason::DrawdownHalt => "drawdown_halt",
            ExitReason::MarginLiquidation => "margin_liquidation",
            ExitReason::Expiry => "expiry",
            ExitReason::Signal => "signal",
            ExitReason::AccountLossLimit => "account_loss_limit",
            ExitReason::DailyProfitTarget => "daily_profit_target",
            ExitReason::AccountProfitTarget => "account_profit_target",
            ExitReason::ExpiryExit => "expiry_exit",
        }
    }

    /// Parse a reason string. `None` for anything not in the vocabulary.
    pub fn from_code(code: &str) -> Option<ExitReason> {
        ExitReason::ALL
            .into_iter()
            .find(|reason| reason.code() == code)
    }
}

/// The names of risk events. A strategy receives them through `on_risk_event`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RiskEventName {
    /// A leg's stop was hit.
    LegSlHit = 0,
    /// A leg's target was hit.
    LegTargetHit = 1,
    /// A trail armed (written once per leg, not on every price event).
    LegTrailArmed = 2,
    /// A breakeven move armed.
    LegBreakevenArmed = 3,
    /// Lock profit armed.
    LockProfitArmed = 4,
    /// The lock-profit floor rose.
    LockProfitFloorAdvanced = 5,
    /// The combined stop was hit.
    OverallSlHit = 6,
    /// The combined target was hit.
    OverallTargetHit = 7,
    /// The daily loss limit was breached (SPEC-18).
    DailyLossLimitHit = 8,
    /// A latching halt (drawdown halt, kill switch).
    RiskHalt = 9,
    /// A GTT fired but its child order did not fill (a gap beyond the limit); the
    /// position is exited at the next base-bar open (04-execution 4.4).
    GttChildUnfilled = 10,
    /// The trading state changed (ACTIVE, REDUCING, HALTED).
    TradingStateChanged = 11,
    /// The account's daily loss limit of a portfolio run was breached, across its
    /// strategies (`engine::account_breach`).
    AccountLossHit = 12,
    /// The strategy's daily profit target was reached (18-multi-strategy 2.5a).
    DailyProfitHit = 13,
    /// The account's daily profit target of a portfolio run was reached, across its
    /// strategies (`engine::account_breach`).
    AccountProfitHit = 14,
}

impl RiskEventName {
    /// The event name.
    pub fn code(self) -> &'static str {
        match self {
            RiskEventName::LegSlHit => "leg_sl_hit",
            RiskEventName::LegTargetHit => "leg_target_hit",
            RiskEventName::LegTrailArmed => "leg_trail_armed",
            RiskEventName::LegBreakevenArmed => "leg_breakeven_armed",
            RiskEventName::LockProfitArmed => "lock_profit_armed",
            RiskEventName::LockProfitFloorAdvanced => "lock_profit_floor_advanced",
            RiskEventName::OverallSlHit => "overall_sl_hit",
            RiskEventName::OverallTargetHit => "overall_target_hit",
            RiskEventName::DailyLossLimitHit => "daily_loss_limit_hit",
            RiskEventName::RiskHalt => "risk_halt",
            RiskEventName::GttChildUnfilled => "gtt_child_unfilled",
            RiskEventName::TradingStateChanged => "trading_state_changed",
            RiskEventName::AccountLossHit => "account_loss_hit",
            RiskEventName::DailyProfitHit => "daily_profit_hit",
            RiskEventName::AccountProfitHit => "account_profit_hit",
        }
    }
}

/// What a risk rule applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RiskScope {
    /// One leg (one position).
    Leg = 0,
    /// A run of the strategy (combined rules).
    Strategy = 1,
    /// The session (daily loss limit, daily profit target).
    Session = 2,
    /// The account a portfolio run's strategies share (its daily loss limit and daily
    /// profit target).
    Account = 3,
}

impl RiskScope {
    /// The scope name.
    pub fn code(self) -> &'static str {
        match self {
            RiskScope::Leg => "leg",
            RiskScope::Strategy => "strategy",
            RiskScope::Session => "session",
            RiskScope::Account => "account",
        }
    }
}

/// One risk event: the rule, its scope, the threshold and what was observed. The
/// engine adds the time stamp and, for a leg, the instrument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiskEvent {
    /// Which event.
    pub name: RiskEventName,
    /// What the rule applies to.
    pub scope: RiskScope,
    /// The level in force when the rule fired, in rupees (a stop or target price, the
    /// stop a trail armed at), or the limit for a money rule.
    pub level_rupees: Option<f64>,
    /// The target that was hit (0 is the first), for `LegTargetHit`.
    pub target_index: Option<u16>,
    /// The value observed against the threshold (the run's MTM for the daily loss
    /// limit), in rupees.
    pub observed_rupees: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reason_codes_round_trip() {
        for reason in ExitReason::ALL {
            assert_eq!(ExitReason::from_code(reason.code()), Some(reason));
        }
        assert_eq!(ExitReason::from_code("roll"), None);
    }

    #[test]
    fn the_accounts_daily_loss_limit_has_its_exit_reason_event_and_scope() {
        assert_eq!(
            ExitReason::from_code("account_loss_limit"),
            Some(ExitReason::AccountLossLimit)
        );
        assert_eq!(RiskEventName::AccountLossHit.code(), "account_loss_hit");
        assert_eq!(RiskScope::Account.code(), "account");
    }

    #[test]
    fn the_daily_profit_targets_have_their_exit_reasons_and_events() {
        let reasons = [
            ExitReason::DailyProfitTarget,
            ExitReason::AccountProfitTarget,
        ];
        assert_eq!(
            reasons.map(ExitReason::code),
            ["daily_profit_target", "account_profit_target"]
        );
        let events = [
            RiskEventName::DailyProfitHit,
            RiskEventName::AccountProfitHit,
        ];
        assert_eq!(
            events.map(RiskEventName::code),
            ["daily_profit_hit", "account_profit_hit"]
        );
    }
}
