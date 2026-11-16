//! What each order of an external session is for (`venue`): its role, carried by its
//! `VenueCommand::Place` and reported by `Engine::order_role`, so the live layer files the
//! order under its purpose, with the priority and the checks before sending that go with
//! it (16-session-design 2.7, requirement R1).
//!
//! | Role | Code | The order |
//! |---|---|---|
//! | `Entry` | `entry` | opens a position from flat, reverses one (`reverse=True` beyond it: the rest opens the other side), or goes against it beyond the room its closing side has (`venue_cover::closing_room`): the other orders working on that side could leave the position flat, and this one would then open the other side |
//! | `Add` | `add` | adds to an open position on its side |
//! | `LegStop` | `leg_stop` | a leg's broker-held stop (decision D31, `venue_legs`) |
//! | `Exit` | `exit` | the strategy's order that only reduces: `exit()`, `reduce()`, a reduce-only order resting to close, an order against the position within the room its closing side has, and the further orders of such an exit above the freeze quantity |
//! | `RiskExit(reason)` | `risk_exit` | the engine's exit of the whole position for a risk reason |
//! | `SquareOff` | `square_off` | the engine's exit at the square-off time |
//!
//! A MARKET order against a position whose leg's stop rests is judged in the room the stop
//! makes for it: the position's exit owner sends it once the stop is cut down to what stays
//! open, or cancelled for a sale of the whole position or a reversal (`venue_owner`), so
//! the order and the stop never both fill at the broker (16-session-design 4.5). A resting
//! order against it (LIMIT, SL, SL-M) is judged beside the stop, which keeps working.
//!
//! A strategy order's role is decided when it passes the submission checks, from the
//! position then held and the orders working to close it; the engine's orders take the
//! role of the exit they are made for (`EngineExit`). A role, once filed with the order's
//! `Place`, stands: a modification that would change it is refused (cancel the order and
//! place a new one), and the fill of an order whose role only closes is never recorded
//! beyond the position (`venue_fills`).
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

use crate::exec::{ExecError, Order, OrderId, OrderRequest, OrderType};
use crate::model::Side;
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;

use super::error::KernelError;
use super::kernel::Engine;
use super::venue::is_exit;

/// True when an order on `side` goes against a position of `held` signed units.
fn goes_against(held: f64, side: Side) -> bool {
    held != 0.0 && (held > 0.0) != (side == Side::Buy)
}

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
    /// and the orders working to close it (module docs). `except` is the order judged,
    /// once it is in the book (a modification's own order), so that its own units leave
    /// the room it is judged in.
    pub(super) fn submission_role(
        &self,
        request: &OrderRequest,
        units: f64,
        except: Option<OrderId>,
    ) -> OrderRole {
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held = self.held_units(key);
        let against = goes_against(held, request.side);
        let room = || self.closing_room(key, except) + self.room_the_stop_makes(request, except);
        match (request.reduce_only, against) {
            (true, _) => OrderRole::Exit,
            (false, true) if units <= room() => OrderRole::Exit,
            (false, true) => OrderRole::Entry,
            (false, false) if held == 0.0 => OrderRole::Entry,
            (false, false) => OrderRole::Add,
        }
    }

    /// The units of the leg's resting stop of `request`'s position when `request` is a
    /// MARKET order against the position that the exit owner sends (module docs); 0 for
    /// any other order, and for the stop itself (`except`).
    fn room_the_stop_makes(&self, request: &OrderRequest, except: Option<OrderId>) -> f64 {
        if request.order_type != OrderType::Market {
            return 0.0;
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        match self.resting_leg_stop(key) {
            Some(stop) if Some(stop) != except => {
                self.book.get(stop).map_or(0.0, |o| self.units_to_fill(o))
            }
            _ => 0.0,
        }
    }

    /// True for an order the exit owner of its position sends (`venue_owner`): an exit
    /// (`exit()`, `reduce()`), and a plain MARKET order against a position whose leg's
    /// stop rests (module docs). Any other order is sent at once.
    pub(super) fn owner_sends(&self, request: &OrderRequest) -> bool {
        if is_exit(request) {
            return true;
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        request.order_type == OrderType::Market
            && goes_against(self.held_units(key), request.side)
            && self.resting_leg_stop(key).is_some()
    }

    /// Refuse, in an external session, a replacement `request` of `units` for order `id`
    /// that would change the role filed with its `Place` (module docs): the live layer
    /// keeps the order under that purpose, with its checks before sending. A backtest
    /// passes.
    ///
    /// Errors: `Exec(InvalidRequest)` naming both roles and saying to cancel the order and
    /// place a new one.
    pub(super) fn check_replacement_role(
        &self,
        id: OrderId,
        request: &OrderRequest,
        units: f64,
    ) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        let placed = self.order_role(id);
        let replaced = self.submission_role(request, units, Some(id));
        if placed == Some(replaced) {
            return Ok(());
        }
        let was = placed.map_or("none", OrderRole::code);
        Err(KernelError::Exec(ExecError::InvalidRequest {
            reason: format!(
                "a modification keeps what the order is for: it was placed as {was} and the \
                 replacement would make it {}; cancel the order and place a new one instead",
                replaced.code()
            ),
        }))
    }

    /// True for an order sent only to close its position: a reduce-only order, or one whose
    /// role does not open (module docs). Its fills are never recorded beyond the position
    /// (`venue_fills`), and it is fitted to the position when that shrinks (`venue_cover`).
    pub(super) fn sent_to_close(&self, order: &Order) -> bool {
        let role = self.order_role(order.id);
        order.request.reduce_only || role.is_some_and(|role| !role.opens())
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
