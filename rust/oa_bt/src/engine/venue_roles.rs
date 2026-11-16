//! What each order of an external session is for (`venue`): its role, carried by its
//! `VenueCommand::Place` and reported by `Engine::order_role`, so the live layer files the
//! order under its purpose, with the priority and the checks before sending that go with
//! it (16-session-design 2.7, requirement R1).
//!
//! | Role | Code | The order |
//! |---|---|---|
//! | `Entry` | `entry` | opens a position from flat, or reverses one (`reverse=True` beyond it: the rest opens the other side) |
//! | `Add` | `add` | adds to an open position on its side |
//! | `LegStop` | `leg_stop` | a leg's broker-held stop (decision D31); no order takes it until the broker-held stops are built |
//! | `Exit` | `exit` | the strategy's order that only reduces: `exit()`, `reduce()`, a reduce-only order resting to close, an order against the position within it, and the further orders of such an exit above the freeze quantity |
//! | `RiskExit(reason)` | `risk_exit` | the engine's exit of the whole position for a risk reason |
//! | `SquareOff` | `square_off` | the engine's exit at the square-off time |
//!
//! A strategy order's role is decided when it passes the submission checks, from the
//! position then held; the engine's orders take the role of the exit they are made for
//! (`EngineExit`).
//!
//! The risk reasons are the kernel's own rules (`daily_loss`, `drawdown`, `time_exit`) and
//! those the live layer names when it asks for an exit (`Engine::request_exit`:
//! `unprotected`, `stop_rejected`, `stop_unfilled`, `kill_switch`, `account_loss`,
//! `operator_stop`). The trade an engine exit closes records the backtest's exit reason:
//!
//! | Engine exit | Exit reason of the trade |
//! |---|---|
//! | `square_off` | `scheduler`, as the backtest's square-off |
//! | `daily_loss`, `drawdown`, `time_exit` | `daily_loss_limit`, `drawdown_halt`, `time_exit`, as the backtest's rules |
//! | `stop_unfilled` | `leg_sl`: the leg's stop level was reached |
//! | `unprotected`, `stop_rejected`, `kill_switch`, `account_loss`, `operator_stop` | `manual`: neither the strategy nor a kernel rule made it |

use crate::exec::{OrderId, OrderRequest};
use crate::model::Side;
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;

use super::kernel::Engine;

/// Why the engine of an external session exits a whole position for risk (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RiskExitReason {
    /// The daily loss limit (`max_daily_loss`).
    DailyLoss,
    /// The drawdown halt.
    Drawdown,
    /// The maximum holding time (`max_hold`).
    TimeExit,
    /// The leg's broker-held stop was not confirmed in time.
    Unprotected,
    /// The broker rejected the leg's stop.
    StopRejected,
    /// The leg's stop was reached and did not fill.
    StopUnfilled,
    /// The kill switch.
    KillSwitch,
    /// The account's daily loss limit, across the session's strategies.
    AccountLoss,
    /// The operator stopped the session (Ctrl+C).
    OperatorStop,
}

impl RiskExitReason {
    /// Every reason, in the order of the module docs.
    pub const ALL: [RiskExitReason; 9] = [
        RiskExitReason::DailyLoss,
        RiskExitReason::Drawdown,
        RiskExitReason::TimeExit,
        RiskExitReason::Unprotected,
        RiskExitReason::StopRejected,
        RiskExitReason::StopUnfilled,
        RiskExitReason::KillSwitch,
        RiskExitReason::AccountLoss,
        RiskExitReason::OperatorStop,
    ];

    /// The code the bindings use.
    pub fn code(self) -> &'static str {
        match self {
            RiskExitReason::DailyLoss => "daily_loss",
            RiskExitReason::Drawdown => "drawdown",
            RiskExitReason::TimeExit => "time_exit",
            RiskExitReason::Unprotected => "unprotected",
            RiskExitReason::StopRejected => "stop_rejected",
            RiskExitReason::StopUnfilled => "stop_unfilled",
            RiskExitReason::KillSwitch => "kill_switch",
            RiskExitReason::AccountLoss => "account_loss",
            RiskExitReason::OperatorStop => "operator_stop",
        }
    }
}

/// An exit the engine makes of a whole position in an external session: the square-off,
/// or a risk exit. It takes over from the strategy's exits of the position
/// (`venue_owner`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineExit {
    /// The square-off.
    SquareOff,
    /// A risk exit, for its reason.
    Risk(RiskExitReason),
}

impl EngineExit {
    /// The exit named by `code`: "square_off" or a risk reason's code (module docs);
    /// `None` for any other.
    pub fn from_code(code: &str) -> Option<EngineExit> {
        if code == OrderRole::SquareOff.code() {
            return Some(EngineExit::SquareOff);
        }
        RiskExitReason::ALL
            .into_iter()
            .find(|reason| reason.code() == code)
            .map(EngineExit::Risk)
    }

    /// The exit reason the trade it closes records (module docs).
    pub fn exit_reason(self) -> ExitReason {
        match self {
            EngineExit::SquareOff => ExitReason::Scheduler,
            EngineExit::Risk(reason) => match reason {
                RiskExitReason::DailyLoss => ExitReason::DailyLossLimit,
                RiskExitReason::Drawdown => ExitReason::DrawdownHalt,
                RiskExitReason::TimeExit => ExitReason::TimeExit,
                RiskExitReason::StopUnfilled => ExitReason::LegSl,
                RiskExitReason::Unprotected
                | RiskExitReason::StopRejected
                | RiskExitReason::KillSwitch
                | RiskExitReason::AccountLoss
                | RiskExitReason::OperatorStop => ExitReason::Manual,
            },
        }
    }
}

/// What an order of an external session is for (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderRole {
    /// Opens a position, or reverses one.
    Entry,
    /// Adds to an open position.
    Add,
    /// A leg's broker-held stop.
    LegStop,
    /// The strategy's order that only reduces.
    Exit,
    /// The engine's exit of the whole position for a risk reason.
    RiskExit(RiskExitReason),
    /// The engine's exit at the square-off time.
    SquareOff,
}

impl OrderRole {
    /// Every role's code, in the order of the module docs.
    pub const CODES: [&'static str; 6] = [
        "entry",
        "add",
        "leg_stop",
        "exit",
        "risk_exit",
        "square_off",
    ];

    /// The code the bindings use (module docs).
    pub fn code(self) -> &'static str {
        match self {
            OrderRole::Entry => "entry",
            OrderRole::Add => "add",
            OrderRole::LegStop => "leg_stop",
            OrderRole::Exit => "exit",
            OrderRole::RiskExit(_) => "risk_exit",
            OrderRole::SquareOff => "square_off",
        }
    }

    /// The reason of a risk exit; `None` for every other role.
    pub fn reason(self) -> Option<RiskExitReason> {
        match self {
            OrderRole::RiskExit(reason) => Some(reason),
            _ => None,
        }
    }

    /// True for an order that opens or adds: the entry gates judge it (`venue_gates`).
    pub fn opens(self) -> bool {
        matches!(self, OrderRole::Entry | OrderRole::Add)
    }
}

impl From<EngineExit> for OrderRole {
    fn from(exit: EngineExit) -> OrderRole {
        match exit {
            EngineExit::SquareOff => OrderRole::SquareOff,
            EngineExit::Risk(reason) => OrderRole::RiskExit(reason),
        }
    }
}

impl Engine {
    /// The role of an order an external session accepted (module docs); `None` for an
    /// order it never accepted (one rejected at submission, a backtest's) and an unknown
    /// id.
    pub fn order_role(&self, id: OrderId) -> Option<OrderRole> {
        let roles = &self.venue.roles;
        let slot = roles.binary_search_by_key(&id, |(known, _)| *known).ok()?;
        Some(roles[slot].1)
    }

    /// The role of the strategy's `request` for `units`, against the position held now
    /// (module docs).
    pub(super) fn submission_role(&self, request: &OrderRequest, units: f64) -> OrderRole {
        let held = self.held_units(PositionKey {
            instrument: request.instrument,
            product: request.product,
        });
        let against = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        match (request.reduce_only, against) {
            (true, _) => OrderRole::Exit,
            (false, true) if units <= held.abs() => OrderRole::Exit,
            (false, true) => OrderRole::Entry,
            (false, false) if held == 0.0 => OrderRole::Entry,
            (false, false) => OrderRole::Add,
        }
    }

    /// Record the role of order `id`; a role once recorded stands.
    pub(super) fn record_role(&mut self, id: OrderId, role: OrderRole) {
        let roles = &mut self.venue.roles;
        if let Err(slot) = roles.binary_search_by_key(&id, |(known, _)| *known) {
            roles.insert(slot, (id, role));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_exit_a_session_can_ask_for_is_named_by_its_code() {
        let table = [
            ("square_off", EngineExit::SquareOff),
            ("daily_loss", EngineExit::Risk(RiskExitReason::DailyLoss)),
            ("drawdown", EngineExit::Risk(RiskExitReason::Drawdown)),
            ("time_exit", EngineExit::Risk(RiskExitReason::TimeExit)),
            ("unprotected", EngineExit::Risk(RiskExitReason::Unprotected)),
            (
                "stop_rejected",
                EngineExit::Risk(RiskExitReason::StopRejected),
            ),
            (
                "stop_unfilled",
                EngineExit::Risk(RiskExitReason::StopUnfilled),
            ),
            ("kill_switch", EngineExit::Risk(RiskExitReason::KillSwitch)),
            (
                "account_loss",
                EngineExit::Risk(RiskExitReason::AccountLoss),
            ),
            (
                "operator_stop",
                EngineExit::Risk(RiskExitReason::OperatorStop),
            ),
        ];
        for (code, exit) in table {
            assert_eq!(EngineExit::from_code(code), Some(exit), "{code}");
        }
        assert_eq!(table.len(), RiskExitReason::ALL.len() + 1);
        assert_eq!(EngineExit::from_code("entry"), None);
        assert_eq!(EngineExit::from_code("daily_loss_limit"), None);
    }

    #[test]
    fn each_engine_exit_records_its_trades_exit_reason() {
        let table = [
            (EngineExit::SquareOff, ExitReason::Scheduler),
            (
                EngineExit::Risk(RiskExitReason::DailyLoss),
                ExitReason::DailyLossLimit,
            ),
            (
                EngineExit::Risk(RiskExitReason::Drawdown),
                ExitReason::DrawdownHalt,
            ),
            (
                EngineExit::Risk(RiskExitReason::TimeExit),
                ExitReason::TimeExit,
            ),
            (
                EngineExit::Risk(RiskExitReason::Unprotected),
                ExitReason::Manual,
            ),
            (
                EngineExit::Risk(RiskExitReason::StopRejected),
                ExitReason::Manual,
            ),
            (
                EngineExit::Risk(RiskExitReason::StopUnfilled),
                ExitReason::LegSl,
            ),
            (
                EngineExit::Risk(RiskExitReason::KillSwitch),
                ExitReason::Manual,
            ),
            (
                EngineExit::Risk(RiskExitReason::AccountLoss),
                ExitReason::Manual,
            ),
            (
                EngineExit::Risk(RiskExitReason::OperatorStop),
                ExitReason::Manual,
            ),
        ];
        for (exit, reason) in table {
            assert_eq!(exit.exit_reason(), reason, "{exit:?}");
        }
    }

    #[test]
    fn every_role_has_its_own_code_and_only_entries_and_adds_open() {
        let roles = [
            OrderRole::Entry,
            OrderRole::Add,
            OrderRole::LegStop,
            OrderRole::Exit,
            OrderRole::RiskExit(RiskExitReason::KillSwitch),
            OrderRole::SquareOff,
        ];
        let codes: Vec<&str> = roles.iter().map(|role| role.code()).collect();
        let expected = [
            "entry",
            "add",
            "leg_stop",
            "exit",
            "risk_exit",
            "square_off",
        ];
        assert_eq!(codes, expected);
        assert_eq!(OrderRole::CODES, expected);
        let opening: Vec<bool> = roles.iter().map(|role| role.opens()).collect();
        assert_eq!(opening, [true, true, false, false, false, false]);
    }
}
