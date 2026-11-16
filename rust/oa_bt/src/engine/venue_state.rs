//! A live session's state saved and restored across a restart (16-session-design R14,
//! section 3.6; decision 6 of section 10.0: a restart restores the kernel from this
//! export, not by replaying the day).
//!
//! `Engine::live_state` writes, as plain data (`StateValue`, the shapes of a JSON
//! document), everything the session's later decisions and books depend on:
//!
//! | Part | What |
//! |---|---|
//! | where | the trading day, the slice and stage the session stands at, its clock, its instruments |
//! | `account` | the cash: capital, realized P&L, charges, margin blocked, sale proceeds withheld |
//! | `positions` | every position with its tranches, adds, realized P&L, margin and mark |
//! | `legs` | each open position's leg state whole (levels, targets, trail, breakeven and what the leg keeps besides) |
//! | `orders` | every working order: its role, request, status, units, filled units, fill price, times and what its fills were charged so far |
//! | `desk` | the cancellations and modifications waiting for the venue's answer (a modification with its terms), the exits of the positions, the legs' broker-held stops, the targets fired, the reprices waiting, the entry locks and the sizing cap |
//! | `commands` | the commands the adapter has not taken yet |
//! | `risk` | the trading state and its cause, the daily loss totals, the drawdown peak, the slices the cooldown and time exit count from, the engine exits waiting for an open |
//! | `open_trades`, `session_buys` | the round trips still open, the day's CNC buys |
//!
//! `Engine::restore_live_state` gives a saved state to a session built for the restart,
//! once, after its catch-up (the day's bars added and released, nothing decided) and
//! before any order: the session must be external, of the same day and instruments and
//! capital, not closing, and stand at or after the point the state was saved at. The
//! saved working orders take new ids, from o1 in the order of their saved ids
//! (`orders::Ids`), returned as (saved id, new id) pairs so the live layer re-keys its
//! venue map. Nothing is sent and no event is written: the restored session is where the
//! saved one was, and from there it decides, sends and records as the saved one would
//! have. The entry locks set before the restore (the live layer's "starting") stay, with
//! the saved ones added. A state that cannot be restored is refused whole
//! (`LiveStateError`): nothing of it is restored.
//!
//! Not saved, so a restarted session's run begins at the restart (16-session-design 1.4:
//! each start's result covers that start): the fills, events, position history and
//! equity of the earlier start, and its charge records. Two charge rules therefore start
//! again at the restart: the contract notes of the day are settled per start (the fills
//! after the restart make a note of their own, rounded on its own), and a CNC sale after
//! the restart is not matched against a CNC buy of the same scrip before it for the
//! intraday rates (`charges::same_day`; the session's buys still count for the sale
//! credit). Timers, GTT orders and futures series are not part of a session's state
//! (missed timers are skipped; the other two are refused in a session).

mod book;
mod desk;
mod error;
mod legs;
mod orders;
mod requests;
mod risk;
mod value;

pub use error::LiveStateError;
pub use value::StateValue;

use crate::charges::OrderCharges;
use crate::exec::{InstrumentId, Order, OrderId};
use crate::model::Day;
use crate::portfolio::{Portfolio, PositionKey};
use crate::rms::LegState;

use super::error::KernelError;
use super::kernel::{Engine, Phase};
use super::output::OpenTrade;
use super::strategy_risk::StrategyRisk;
use super::venue::{ChargePart, VenueCommand};
use super::venue_error::VenueError;
use super::venue_roles::OrderRole;
use book::{
    read_account, read_buy, read_open_trade, read_position, repeats, write_account, write_buy,
    write_open_trade, write_position,
};
use desk::{read_desk, write_desk, SavedDesk};
use legs::{read_leg, write_leg};
use orders::{read_command, read_order, write_command, write_order, Ids};
use risk::{read_risk, write_risk};
use value::{list, map, Read};

/// The layout of the saved state this kernel writes and reads; a state of another layout
/// is refused (`LiveStateError::Version`).
pub const LIVE_STATE_LAYOUT: i64 = 1;

/// How far into a slice a session stands, in the order a slice runs them: the timers
/// due at its open fired ahead of it, before it, at its timer pause, paused at its close,
/// waiting at its session close, and after the last slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Ahead,
    Ready,
    Timers,
    Paused,
    Closing,
    Finished,
}

const STAGES: [Stage; 6] = [
    Stage::Ahead,
    Stage::Ready,
    Stage::Timers,
    Stage::Paused,
    Stage::Closing,
    Stage::Finished,
];

impl Stage {
    fn code(self) -> &'static str {
        match self {
            Stage::Ahead => "ahead",
            Stage::Ready => "ready",
            Stage::Timers => "timers",
            Stage::Paused => "paused",
            Stage::Closing => "closing",
            Stage::Finished => "finished",
        }
    }
}

/// A saved state read whole, before any of it is applied.
struct Saved {
    clock_ns: i64,
    portfolio: Portfolio,
    legs: Vec<(PositionKey, LegState)>,
    orders: Vec<(Order, Option<OrderRole>)>,
    charged: Vec<(ChargePart, OrderCharges)>,
    desk: SavedDesk,
    commands: Vec<VenueCommand>,
    risk: StrategyRisk,
    session_buys: Vec<(InstrumentId, f64)>,
    open_trades: Vec<OpenTrade>,
    ids: Ids,
}

impl Engine {
    /// The slice the session stands at and how far into it (`Stage`).
    fn standing(&self) -> (usize, Stage) {
        match self.phase {
            Phase::Ahead(slice) => (slice.index, Stage::Ahead),
            Phase::Ready(index) => (index, Stage::Ready),
            Phase::Timers(slice) => (slice.index, Stage::Timers),
            Phase::Paused(slice) => (slice.index, Stage::Paused),
            Phase::Closing(slice) => (slice.index, Stage::Closing),
            Phase::Finished => (self.timeline.len(), Stage::Finished),
        }
    }

    /// The IST trading day of the slice at `index` (the last slice's after the end).
    fn day_at(&self, index: usize) -> Option<Day> {
        let last = self.timeline.len().checked_sub(1)?;
        self.timeline.slice(index.min(last)).map(|slice| slice.day)
    }

    /// The session's state as plain data (module docs), to save before each batch of
    /// commands is sent and at each slice release, and to give to `restore_live_state`
    /// after a restart. A backtest writes its state the same way.
    pub fn live_state(&self) -> StateValue {
        let (slice, stage) = self.standing();
        let book = &self.book;
        let instruments = list(&self.instruments, |spec| {
            map([
                ("symbol", spec.symbol.as_str().into()),
                ("exchange", spec.exchange.code().into()),
            ])
        });
        let orders = list(book.working(), |order| {
            write_order(order, self.order_role(order.id), &self.venue.charged)
        });
        let commands = self.venue.commands.iter();
        map([
            ("layout", LIVE_STATE_LAYOUT.into()),
            ("day", self.day_at(slice).into()),
            ("slice", slice.into()),
            ("stage", stage.code().into()),
            ("clock_ns", self.venue.clock_ns.into()),
            ("instruments", instruments),
            ("account", write_account(&self.portfolio.account)),
            (
                "positions",
                list(self.portfolio.positions(), write_position),
            ),
            ("legs", list(&self.legs, |(key, leg)| write_leg(*key, leg))),
            ("orders", orders),
            ("desk", write_desk(book, &self.venue)),
            (
                "commands",
                StateValue::List(commands.filter_map(|c| write_command(book, c)).collect()),
            ),
            ("risk", write_risk(&self.risk)),
            (
                "open_trades",
                list(&self.record.trades.open, write_open_trade),
            ),
            ("session_buys", list(&self.session_buys.bought, write_buy)),
        ])
    }

    /// Give a session built for a restart the state `live_state` saved (module docs).
    /// Returns (saved id, new id) for every restored working order, in saved order.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `LiveState` when the state is refused,
    /// whole: `NotRestorable` for a session that has traded, was restored already, has
    /// begun its close or stands before the saved point; `Version`, `OtherSession` (another
    /// day, other instruments, another capital) and `Malformed` for the state itself.
    pub fn restore_live_state(
        &mut self,
        state: &StateValue,
    ) -> Result<Vec<(OrderId, OrderId)>, KernelError> {
        if !self.is_external() {
            return Err(VenueError::NotExternal.into());
        }
        self.check_restorable()?;
        let saved = self.read_saved(&Read::root(state))?;
        self.apply_saved(saved)
    }

    /// Refuse a session that cannot take a state now (module docs).
    fn check_restorable(&self) -> Result<(), LiveStateError> {
        let refuse = |reason: &str| {
            Err(LiveStateError::NotRestorable {
                reason: reason.to_string(),
            })
        };
        if self.venue.restored {
            return refuse("the session was restored already (a state is restored once)");
        }
        let traded = !self.book.orders().is_empty()
            || !self.portfolio.positions().is_empty()
            || !self.record.fills.is_empty();
        if traded {
            return refuse(
                "the session has orders or positions of its own: restore into a session \
                 built for the restart, after its catch-up and before any order",
            );
        }
        if matches!(self.phase, Phase::Closing(_) | Phase::Finished) {
            return refuse("the session's close has begun: restore before the session close");
        }
        Ok(())
    }

    /// Check that `read` is a state of this session saved at or before where it stands.
    fn check_session(&self, read: &Read<'_>) -> Result<(), LiveStateError> {
        let layout: i64 = read.whole_of("layout")?;
        if layout != LIVE_STATE_LAYOUT {
            return Err(LiveStateError::Version { found: layout });
        }
        let other = |reason: String| Err(LiveStateError::OtherSession { reason });
        let saved = read.items_of("instruments")?;
        let named: Vec<(String, String)> = saved
            .iter()
            .map(|item| {
                let symbol = item.field("symbol")?.text()?.to_string();
                Ok((symbol, item.field("exchange")?.text()?.to_string()))
            })
            .collect::<Result<_, LiveStateError>>()?;
        let ours: Vec<(String, String)> = self
            .instruments
            .iter()
            .map(|spec| (spec.symbol.clone(), spec.exchange.code().to_string()))
            .collect();
        if named != ours {
            return other(format!("instruments {named:?}, the session's {ours:?}"));
        }
        let (slice, stage) = self.standing();
        let saved_day: Option<Day> = read.opt_whole_of("day")?;
        if saved_day != self.day_at(slice) {
            return other(format!(
                "day {saved_day:?}, the session's {:?}",
                self.day_at(slice)
            ));
        }
        let saved_capital = read.field("account")?.f64_of("capital_rupees")?;
        if saved_capital != self.config.capital_rupees {
            let ours = self.config.capital_rupees;
            return other(format!(
                "capital {saved_capital} rupees, the session's {ours}"
            ));
        }
        let saved_at = (
            read.whole_of::<usize>("slice")?,
            read.field("stage")?.code(&STAGES, Stage::code)?,
        );
        if (slice, stage) < saved_at {
            return Err(LiveStateError::NotRestorable {
                reason: format!(
                    "the session stands at slice {slice} ({}), before the saved slice {} ({}): \
                     catch it up first",
                    stage.code(),
                    saved_at.0,
                    saved_at.1.code()
                ),
            });
        }
        Ok(())
    }

    /// Read the whole state (module docs), applying nothing.
    fn read_saved(&self, read: &Read<'_>) -> Result<Saved, LiveStateError> {
        self.check_session(read)?;
        let count = self.instruments.len();
        let saved_orders = read.items_of("orders")?;
        let first = self.book.next_id().0;
        let mut ids = Ids { pairs: Vec::new() };
        for (offset, order) in (0u64..).zip(&saved_orders) {
            let saved = OrderId(order.whole_of("order")?);
            if ids.pairs.last().is_some_and(|(last, _)| *last >= saved) {
                return Err(order.malformed("is not in id order after the order before it"));
            }
            ids.pairs.push((saved, OrderId(first + offset)));
        }
        let mut orders = Vec::new();
        let mut charged = Vec::new();
        for (order, (_, id)) in saved_orders.iter().zip(&ids.pairs) {
            let (order, role, parts) = read_order(order, *id, count)?;
            orders.push((order, role));
            charged.extend(parts);
        }
        let positions = read
            .items_of("positions")?
            .iter()
            .map(|position| read_position(position, count))
            .collect::<Result<Vec<_>, _>>()?;
        let legs = read
            .items_of("legs")?
            .iter()
            .map(|leg| read_leg(leg, count))
            .collect::<Result<Vec<_>, _>>()?;
        if repeats(positions.iter().map(|p| p.key)) || repeats(legs.iter().map(|(key, _)| *key)) {
            return Err(read.malformed("names one position twice"));
        }
        let drawdown_limit = self.config.risk.drawdown_halt_percent;
        Ok(Saved {
            clock_ns: read.whole_of("clock_ns")?,
            portfolio: Portfolio::resume(read_account(&read.field("account")?)?, positions),
            legs,
            orders,
            charged,
            desk: read_desk(&read.field("desk")?, &ids, count)?,
            commands: read
                .items_of("commands")?
                .iter()
                .map(|command| read_command(command, &ids, count))
                .collect::<Result<_, _>>()?,
            risk: read_risk(&read.field("risk")?, &self.risk, count, drawdown_limit)?,
            session_buys: read
                .items_of("session_buys")?
                .iter()
                .map(|buy| read_buy(buy, count))
                .collect::<Result<_, _>>()?,
            open_trades: read
                .items_of("open_trades")?
                .iter()
                .map(|trade| read_open_trade(trade, count))
                .collect::<Result<_, _>>()?,
            ids,
        })
    }

    /// Put a state read whole in place (module docs); returns the restored orders' ids.
    ///
    /// Errors: `Exec` from the order book, a kernel defect: the ids follow the empty
    /// book's next id (`read_saved`), so the book takes every order, before anything else
    /// is changed.
    fn apply_saved(&mut self, saved: Saved) -> Result<Vec<(OrderId, OrderId)>, KernelError> {
        for (order, role) in saved.orders {
            let id = self.book.insert(order)?;
            if let Some(role) = role {
                self.venue.roles.push((id, role));
            }
        }
        self.portfolio = saved.portfolio;
        self.legs = saved.legs;
        let desk = saved.desk;
        let venue = &mut self.venue;
        venue.charged = saved.charged;
        venue.cancelling = desk.cancelling;
        venue.cancels_sent = desk.cancels_sent;
        venue.modifying = desk.modifying;
        venue.exits = desk.exits;
        venue.leg_stops = desk.leg_stops;
        venue.target_exits = desk.target_exits;
        venue.fired_targets = desk.fired_targets;
        venue.reprices = desk.reprices;
        venue.controls.entry_locks.extend(desk.controls.entry_locks);
        if desk.controls.sizing_cap_rupees.is_some() {
            venue.controls.sizing_cap_rupees = desk.controls.sizing_cap_rupees;
        }
        venue.commands.extend(saved.commands);
        venue.clock_ns = venue.clock_ns.max(saved.clock_ns);
        venue.restored = true;
        self.risk = saved.risk;
        self.session_buys.bought = saved.session_buys;
        for trade in &saved.open_trades {
            self.charges.resume_trade(trade.key);
        }
        self.record.trades.open = saved.open_trades;
        Ok(saved.ids.pairs)
    }
}
