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
//! | `orders` | every working order: its role, whether it was sent, request, status, units, filled units, fill price, times and what its fills were charged so far |
//! | `gtts` | every GTT of the session: its id, state, purpose (a leg's stop, or the strategy order it rests as) and terms (`gtts`) |
//! | `desk` | the cancellations and modifications waiting for the venue's answer (a modification with its terms), the exits of the positions (an engine exit's fill flags with them), the legs' broker-held stops, the targets fired, the reprices waiting, what the session follows of its GTTs (a modification or cancellation waiting, the legs' stop GTTs, the children of fired GTTs), the entry locks and the sizing cap |
//! | `commands` | the commands the adapter has not taken yet |
//! | `risk` | the trading state and its cause, the daily loss totals, the drawdown peak, the slices the cooldown and time exit count from, the engine exits waiting for an open, the day's entries the trade caps count |
//! | `open_trades`, `session_buys` | the round trips still open (with their excursion and slippage so far, the working orders that hold their excursion back, their last slice's ticks and the flags of their fills so far), the day's CNC buys |
//! | `last_closes` | each instrument's last close so far (rupees), read only when the state rebuilds a session that stopped without its close (`stopped`) |
//!
//! `Engine::restore_live_state` gives a saved state to a session built for the restart,
//! once, after its catch-up (the day's bars added and released, nothing decided) and
//! before any order: the session must be external, of the same day and instruments and
//! capital, and stand at or after the point the state was saved at and before its own
//! close; a state saved while the session closed is refused (`standing`). A state that
//! cannot be read, or whose parts do not hold together (`checks`), is refused whole
//! (`LiveStateError`): nothing of it is restored. Nothing is sent and no event is written:
//! the restored session is where the saved one was, and from there it decides, sends and
//! records as the saved one would have, but for the events below.
//!
//! - **Ids.** The saved working orders take new ids, from o1 in the order of their saved
//!   ids (`orders::Ids`), returned as (saved id, new id) pairs. Later orders take the ids
//!   after them, which the earlier start may have given to orders that have ended since:
//!   the live layer rebuilds its venue map from the pairs alone, since a key it kept for
//!   an ended order of the earlier start names another order now.
//! - **Commands.** The commands the saved session had not handed out come back, to be
//!   taken first. Section 3.6 saves the state before each batch of commands, so each may
//!   or may not have reached the venue before the crash, which only the journal tells (the
//!   restart's order check, 3.6). The result the check gives an order ends it, and
//!   `take_commands` then gives none of its commands (`venue` module docs); a restored
//!   Place still to be taken is the live layer's to send or not, by its journal. A saved
//!   order's `sent` is false for a strategy exit that waits for its cancels: nothing of it
//!   can be at the venue, and the check has no intent to look for.
//! - **A pre-open decision's orders** that wait for the open go as the restored session
//!   reaches it; restored after the open (the session stands past its first slice, or
//!   `restore_live_state_at`'s present is at or after the open), they go at once and are
//!   late, as a start after the open sends its decision's (`pre_open_decision` tells a
//!   late start by its slice and its present alike): their fills are flagged
//!   `late_decision` (`pre_open`).
//! - **Controls.** The entry locks are not restored: the session keeps those set before
//!   the restore (the live layer's "starting"), and the saved ones stay readable in the
//!   state (`desk.entry_locks`) for the live layer to set again those that last the day
//!   (16-session-design 6.2: a reconciliation mismatch, failed exits, the account's daily
//!   loss; not the mode guard's, the dead feed's or Ctrl+C's, which a new start lifts). The
//!   sizing cap is restored only into a session that has none: a cap set at the restart
//!   comes from fresher funds (6.6).
//!
//! Not saved, so a restarted session's run begins at the restart (16-session-design 1.4:
//! each start's result covers that start): the fills, events, position history and
//! equity of the earlier start, and its charge records. Two charge rules therefore start
//! again at the restart: the contract notes of the day are settled per start (the fills
//! after the restart make a note of their own, rounded on its own), and a CNC sale after
//! the restart is not matched against a CNC buy of the same scrip before it for the
//! intraday rates (`charges::same_day`; the session's buys still count for the sale
//! credit). The events recorded since the session's last pause (the fills and results it
//! took while waiting), which its next pause would have delivered, are not saved either:
//! the restored session never delivers them, so the strategy's `on_fill`, `on_order` and
//! `on_position` do not run for them, and a strategy that keeps state in those callbacks
//! can differ from the uninterrupted run (the books do not: they hold every fill). Timers
//! and futures series are not part of a session's state (missed timers are skipped;
//! futures series are refused in a session). The GTTs keep their ids: the restored
//! session's book of GTTs is empty before the restore and takes them all back.
//!
//! The state the next trading day's session takes is `carried_state` (`carry`). A session
//! of an earlier day that stopped without its close gives it too, rebuilt from its last
//! saved state (`stopped`).

mod book;
mod carry;
mod carry_checks;
mod carry_expired;
mod carry_restore;
mod checks;
mod desk;
mod error;
mod gtts;
mod legs;
mod orders;
mod requests;
mod risk;
mod standing;
mod stopped;
mod value;

pub use carry::CarriedIds;
pub use error::LiveStateError;
pub use value::StateValue;

use crate::charges::OrderCharges;
use crate::exec::{FillFlags, GttBook, InstrumentId, OrderId};
use crate::model::TsNanos;
use crate::portfolio::{Portfolio, PositionKey};
use crate::rms::LegState;

use super::error::KernelError;
use super::kernel::Engine;
use super::output::OpenTrade;
use super::strategy_risk::StrategyRisk;
use super::venue::{ChargePart, VenueCommand};
use super::venue_error::VenueError;
use book::{
    read_account, read_buy, read_open_trade, read_position, repeats, write_account, write_buy,
    write_open_trade, write_position,
};
use checks::{check_parts, Parts};
use desk::{read_desk, write_desk, SavedDesk};
use gtts::{read_gtt_book, write_gtt};
use legs::{read_leg, write_leg};
use orders::{read_command, read_orders, write_command, write_order, Ids, SavedOrder};
use risk::{read_risk, write_risk};
use value::{list, map, Read};

/// The layout of the saved state this kernel writes and reads; a state of another layout
/// is refused (`LiveStateError::Version`). Layout 2 added to each open trade its excursion
/// and slippage so far and the orders that hold its excursion back (`excursion`); layout
/// 3 the orders that hold it back by the terms of a modification the venue has not
/// answered, which its refusal releases (`excursion_hold`); layout 4 the terms confirmed
/// modifications replaced, which judge the bar of the slice they were confirmed in;
/// layout 5 the day's entries the trade caps count (`risk`, 18-multi-strategy 2.5a);
/// layout 6 the GTTs and what the session follows of them, and an engine exit's fill flags
/// (design 20-positional-live 4.1), and then, before layout 6 was released, the flags of
/// each open trade's fills (`book`), read as absent (no flag) from a layout 6 state
/// written without them, and each instrument's last close (`stopped`), read as absent (no
/// close) from one written without it.
pub const LIVE_STATE_LAYOUT: i64 = 6;

/// A saved state read whole, before any of it is applied.
struct Saved {
    clock_ns: i64,
    portfolio: Portfolio,
    legs: Vec<(PositionKey, LegState)>,
    orders: Vec<SavedOrder>,
    gtts: GttBook,
    desk: SavedDesk,
    commands: Vec<VenueCommand>,
    risk: StrategyRisk,
    session_buys: Vec<(InstrumentId, f64)>,
    open_trades: Vec<OpenTrade>,
    ids: Ids,
}

impl Engine {
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
            let (role, sent) = (self.order_role(order.id), !self.is_unsent(order.id));
            write_order(order, role, sent, &self.venue.charged)
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
            (
                "gtts",
                list(self.gtts.book.gtts(), |gtt| write_gtt(book, gtt)),
            ),
            ("desk", write_desk(book, &self.venue)),
            (
                "commands",
                StateValue::List(
                    commands
                        .filter_map(|c| write_command(book, &self.gtts.book, c))
                        .collect(),
                ),
            ),
            (
                "risk",
                write_risk(&self.risk, |id| {
                    book.get(id).is_some_and(|order| order.status.is_working())
                }),
            ),
            (
                "open_trades",
                list(&self.record.trades.open, |trade| {
                    write_open_trade(book, trade)
                }),
            ),
            ("session_buys", list(&self.session_buys.bought, write_buy)),
            ("last_closes", self.write_last_closes()),
        ])
    }

    /// Give a session built for a restart the state `live_state` saved (module docs).
    /// Returns (saved id, new id) for every restored working order, in saved order.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `LiveState` when the state is refused,
    /// whole: `NotRestorable` for a session that has traded, was restored already, has
    /// begun its close or stands before the saved point, and for a state saved while the
    /// session closed; `Version`, `OtherSession` (another day, other instruments, another
    /// capital) and `Malformed` (a field that cannot be read, or parts that do not hold
    /// together) for the state itself.
    pub fn restore_live_state(
        &mut self,
        state: &StateValue,
    ) -> Result<Vec<(OrderId, OrderId)>, KernelError> {
        self.restore_live(state, None)
    }

    /// `restore_live_state` at `now_ns`, the present (UTC nanoseconds): a restore at or
    /// after the session's open, even before the catch-up has released a slice of it,
    /// sends a pre-open decision's waiting orders at once, flagged `late_decision`, as
    /// `pre_open_decision` decides a start at that present late (module docs).
    ///
    /// Errors: those of `restore_live_state`.
    pub fn restore_live_state_at(
        &mut self,
        state: &StateValue,
        now_ns: TsNanos,
    ) -> Result<Vec<(OrderId, OrderId)>, KernelError> {
        self.restore_live(state, Some(now_ns))
    }

    /// `restore_live_state`, at the present `now_ns` when it is known.
    fn restore_live(
        &mut self,
        state: &StateValue,
        now_ns: Option<TsNanos>,
    ) -> Result<Vec<(OrderId, OrderId)>, KernelError> {
        if !self.is_external() {
            return Err(VenueError::NotExternal.into());
        }
        self.check_restorable()?;
        let saved = self.read_saved(&Read::root(state))?;
        let pairs = self.apply_saved(saved)?;
        self.late_after_open(now_ns);
        Ok(pairs)
    }

    /// Read the whole state (module docs), applying nothing.
    fn read_saved(&self, read: &Read<'_>) -> Result<Saved, LiveStateError> {
        self.check_session(read)?;
        self.read_parts(read)
    }

    /// Read the parts of a state whose session was checked (`check_session`, or the
    /// checks of a stopped session's, `stopped`), applying nothing.
    fn read_parts(&self, read: &Read<'_>) -> Result<Saved, LiveStateError> {
        let count = self.instruments.len();
        let saved_orders = read.items_of("orders")?;
        let (orders, ids) = read_orders(&saved_orders, self.book.next_id(), count)?;
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
        let desk = read_desk(&read.field("desk")?, &ids, count)?;
        let parts = Parts {
            positions: &positions,
            legs: &legs,
            orders: &orders,
            exits: &desk.exits,
            leg_stops: &desk.leg_stops,
            at_open: &desk.orders_at_open,
        };
        check_parts(read, &parts)?;
        let drawdown_limit = self.config.risk.drawdown_halt_percent;
        Ok(Saved {
            clock_ns: read.whole_of("clock_ns")?,
            portfolio: Portfolio::resume(read_account(&read.field("account")?)?, positions),
            legs,
            orders,
            gtts: read_gtt_book(&read.field("gtts")?, &ids, count)?,
            desk,
            commands: read
                .items_of("commands")?
                .iter()
                .map(|command| read_command(command, &ids, count))
                .collect::<Result<_, _>>()?,
            risk: read_risk(
                &read.field("risk")?,
                &self.risk,
                (count, &ids),
                drawdown_limit,
            )?,
            session_buys: read
                .items_of("session_buys")?
                .iter()
                .map(|buy| read_buy(buy, count))
                .collect::<Result<_, _>>()?,
            open_trades: read
                .items_of("open_trades")?
                .iter()
                .map(|trade| read_open_trade(trade, count, &ids))
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
        let mut charged: Vec<(ChargePart, OrderCharges)> = Vec::new();
        for SavedOrder {
            order,
            role,
            charged: parts,
            ..
        } in saved.orders
        {
            let id = self.book.insert(order)?;
            if let Some(role) = role {
                self.venue.roles.push((id, role));
            }
            charged.extend(parts);
        }
        self.portfolio = saved.portfolio;
        self.legs = saved.legs;
        let desk = saved.desk;
        let venue = &mut self.venue;
        venue.charged = charged;
        venue.cancelling = desk.cancelling;
        venue.cancels_sent = desk.cancels_sent;
        venue.modifying = desk.modifying;
        venue.exits = desk.exits;
        venue.leg_stops = desk.leg_stops;
        venue.target_exits = desk.target_exits;
        venue.fired_targets = desk.fired_targets;
        venue.reprices = desk.reprices;
        venue.exit_flags = desk.exit_flags;
        venue.gtts = desk.gtts;
        venue.pre_open.at_open = desk.orders_at_open;
        self.gtts.book = saved.gtts;
        if venue.controls.sizing_cap_rupees.is_none() {
            venue.controls.sizing_cap_rupees = desk.sizing_cap_rupees;
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

    /// A restored pre-open decision's orders that wait for the open, in a session restored
    /// after its open: it stands past its first slice, or the present `now_ns` (UTC
    /// nanoseconds, when known) is at or after the open of the session it stands in. They
    /// go at once, late, as a start after the open sends its decision's (`pre_open`), so
    /// their fills are flagged `late_decision`.
    fn late_after_open(&mut self, now_ns: Option<TsNanos>) {
        let index = self.standing().0;
        let open = self
            .timeline
            .slice(index)
            .and_then(|slice| self.session_open_of(&slice).ok());
        let opened = now_ns.is_some_and(|now| open.is_some_and(|open| now >= open));
        if index == 0 && !opened {
            return;
        }
        let late = self.venue.pre_open.at_open.iter();
        let flagged = late.map(|id| (*id, FillFlags::LATE_DECISION));
        self.venue.exit_flags.extend(flagged.collect::<Vec<_>>());
    }
}
