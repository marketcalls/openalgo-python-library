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
}

impl ExitReason {
    /// Every reason, in code order.
    pub const ALL: [ExitReason; 14] = [
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
    /// The session (daily loss limit).
    Session = 2,
}

impl RiskScope {
    /// The scope name.
    pub fn code(self) -> &'static str {
        match self {
            RiskScope::Leg => "leg",
            RiskScope::Strategy => "strategy",
            RiskScope::Session => "session",
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
}
