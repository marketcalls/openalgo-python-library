//! A live session carried to the next trading day's session (design 20-positional-live
//! 4.1 and 4.2, unit N1 item N1-4): `Engine::carried_state` once the day's session has
//! closed, and `Engine::restore_carried_state` in the session of a later trading day.
//!
//! The carried state is `live_state` at the close without its DAY parts, as plain data
//! that survives a JSON round trip:
//!
//! | Part | What |
//! |---|---|
//! | `day`, `instruments` | the trading day it closed, and its instruments |
//! | `account` | the cash: capital, realized P&L, charges, margin blocked, sale proceeds withheld |
//! | `positions`, `legs` | the open positions with their tranches and entry times, and their legs whole (levels, trail and breakeven state) |
//! | `orders`, `gtts` | the strategy's GTT orders resting at the venue, and the GTTs at the venue: the legs' stop GTTs and those orders' GTTs, each with its kernel id and terms, and `to_cancel` (below) |
//! | `stop_gtt_rejections` | for each open CNC or NRML leg, how many of its stop GTTs the broker rejected after their placement (design 3.1, row `rejected`: the second exits the leg, on whichever day it comes) |
//! | `risk` | the trading state, the daily loss measure's totals (its run baseline), the drawdown peak, and the slices each open position has been held |
//! | `open_trades` | the round trips still open, with their excursion so far |
//! | `last_closes` | each instrument's last close of the day (rupees): what a position on a contract that expires that day is recorded exited at by the next day's session (below) |
//!
//! A stop GTT that holds no open leg's stop is carried with `to_cancel` true: one whose
//! position closed, or whose leg opened again, while the venue refused its cancellation up
//! to the close. It still rests at the broker, so the next day's session follows it and
//! asks its cancellation at once (`Engine::loose_stop_gtts`).
//!
//! Not carried, as a backtest's next session does not see them either: the orders that
//! expired at the close and everything the venue was asked of them, the exits, the
//! targets fired, the entry locks (`gtt_refused` included: a refusal kept across starts is
//! the live layer's, from its journal), the day's entries the trade caps count, the day's
//! CNC buys, the slices of the cooldown, and the session's clock.
//!
//! The next day's session takes it once, after its catch-up and before any order. Its
//! instruments must be the saved ones and its capital the saved capital (as
//! `restore_live_state` asks); the state must be of an earlier trading day; and every
//! carried position must be CNC or NRML (MIS is squared off each day): otherwise it is
//! refused as `CarriedMismatch`. The parts are checked as a same-day state's are
//! (`checks`), and so are the GTTs: a stop GTT not to cancel holds an open CNC or NRML
//! leg's stop on its closing side for at most its units, one per position, and a GTT
//! order names a carried GTT order. Anything else refuses the state whole as `Malformed`,
//! naming the part of `carried_state` that breaks.
//!
//! Restored: the positions, marked at the catch-up's last close where there is one; the
//! legs, with their stop GTTs' rejections; the GTT orders and GTTs under new ids, every GTT
//! resting (a request the venue had not answered when the session stopped rests on the
//! terms asked; the start's GTT book read settles it, design 4.2 step 5), and a stop GTT
//! to cancel asked to be cancelled at once; a carried leg without a stop GTT gets none by
//! itself, `protect` places it (`venue_gtt_protect`). The daily loss measure starts the
//! session as a backtest's does at a session boundary (a run still open counts from its
//! first fill, decision 11.9), and so does the trading state (a daily loss lock lifts, a
//! drawdown halt stays); the sale proceeds withheld are credited, the day after the sale.
//! A carried leg whose stop is a rearm_daily DAY stop (`rearm`) gets it placed again once
//! the session has reached its open.
//!
//! A carried position on a contract whose last trading day is before the session's day
//! (an index future or a cash-settled MCX contract left to the exchange at its last
//! close, `expiry`) is not traded by the session: its exit is recorded at the contract's
//! carried last close (the position's mark when the state has none; a state with neither
//! is refused as `Malformed`), reason `expiry`, flagged `filled_at_last_close`, with no
//! charges, once the books are in place and before the session's boundary (the exit
//! belongs to the carried day, as a backtest's settlement at expiry), and its stop GTTs are
//! not restored (the broker ends them with the contract). Its instrument is reported in
//! `CarriedIds::expired`. A position on a contract with a positional exit day carried past
//! its last trading day missed the engine's exit and was settled by the exchange: the
//! state is refused, `MissedExpiryExit` (`carry_expired`). Reading, checking and applying
//! it is `carry_restore`.

use crate::exec::{Gtt, GttId, GttPurpose, InstrumentId, OrderId};
use crate::portfolio::PositionKey;
use crate::rms::TradingStateCause;

use super::super::error::KernelError;
use super::super::excursion_hold::Hold;
use super::super::kernel::{Engine, Phase};
use super::super::output::OpenTrade;
use super::super::venue_error::VenueError;
use super::book::{write_account, write_open_trade, write_position};
use super::error::LiveStateError;
use super::gtts::write_gtt;
use super::legs::write_leg;
use super::orders::write_order;
use super::requests::key_fields;
use super::value::{list, map, Read, StateValue};
use super::LIVE_STATE_LAYOUT;

/// The kind a carried state names itself by, apart from a same-day `live_state`.
pub(super) const CARRIED: &str = "carried";

/// What a carried leg's stop GTT record names when no stop GTT of it was carried: no GTT
/// has this id (ids start at g1), so the leg has none and gets one only from `protect`.
pub(super) const NO_CARRIED_GTT: GttId = GttId(0);

/// The new ids of a carried state's GTTs and strategy GTT orders, as (saved id, new id)
/// in saved order (`Engine::restore_carried_state`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CarriedIds {
    /// The GTTs.
    pub gtts: Vec<(GttId, GttId)>,
    /// The strategy's GTT orders.
    pub orders: Vec<(OrderId, OrderId)>,
    /// The instruments whose contracts had expired before the session's day: their
    /// positions were recorded exited at their last close (module docs).
    pub expired: Vec<InstrumentId>,
}

impl Engine {
    /// The session's state carried to the next trading day (module docs), once the day's
    /// session has closed: after its close, or `stop_session`.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `LiveState(NotRestorable)` before the
    /// session has closed.
    pub fn carried_state(&self) -> Result<StateValue, KernelError> {
        if !self.is_external() {
            return Err(VenueError::NotExternal.into());
        }
        if self.phase != Phase::Finished {
            return Err(LiveStateError::NotRestorable {
                reason: "a carried state is taken once the day's session has closed (after \
                         its close, or stop_session)"
                    .to_string(),
            }
            .into());
        }
        let book = &self.book;
        let orders = book
            .working()
            .filter(|order| self.resting_gtt_of(order.id).is_some());
        let gtts = self
            .gtts
            .book
            .gtts()
            .iter()
            .filter(|gtt| gtt.status.at_venue());
        let open = self.portfolio.positions().iter().filter(|p| !p.is_flat());
        let gtt_legs = self.venue.gtts.legs.iter().filter(|leg| {
            let held = self.portfolio.position(leg.key);
            held.is_some_and(|position| !position.is_flat())
        });
        let trades = self.record.trades.open.iter().map(cleared);
        Ok(map([
            ("layout", LIVE_STATE_LAYOUT.into()),
            ("kind", CARRIED.into()),
            ("day", self.day_at(self.timeline.len()).into()),
            ("instruments", self.write_instruments()),
            ("account", write_account(&self.portfolio.account)),
            ("positions", list(open, write_position)),
            ("legs", list(&self.legs, |(key, leg)| write_leg(*key, leg))),
            (
                "orders",
                list(orders, |order| {
                    write_order(order, self.order_role(order.id), true, &[])
                }),
            ),
            ("gtts", list(gtts, |gtt| self.write_carried_gtt(gtt))),
            (
                "stop_gtt_rejections",
                list(gtt_legs, |leg| {
                    let [instrument, product] = key_fields(leg.key);
                    let rejections = u32::from(leg.rejections);
                    map([instrument, product, ("rejections", rejections.into())])
                }),
            ),
            ("risk", self.write_carried_risk()),
            (
                "open_trades",
                list(trades, |trade| write_open_trade(book, &trade)),
            ),
            ("last_closes", self.write_last_closes()),
        ]))
    }

    /// Each instrument's last close (module docs), by registration index; an instrument
    /// with none is left out.
    fn write_last_closes(&self) -> StateValue {
        let closes = self
            .last_close
            .iter()
            .enumerate()
            .filter_map(|(index, close)| close.map(|close| (index, close)));
        list(closes, |(index, close)| {
            let index = u32::try_from(index).unwrap_or(u32::MAX);
            map([("instrument", index.into()), ("close_rupees", close.into())])
        })
    }

    /// The session's instruments by symbol and exchange, in registration order: the next
    /// day's session must trade the same.
    fn write_instruments(&self) -> StateValue {
        list(&self.instruments, |spec| {
            map([
                ("symbol", spec.symbol.as_str().into()),
                ("exchange", spec.exchange.code().into()),
            ])
        })
    }

    /// A GTT at the venue as it is carried: `write_gtt`'s fields and `to_cancel` (module
    /// docs).
    fn write_carried_gtt(&self, gtt: &Gtt) -> StateValue {
        let mut written = write_gtt(&self.book, gtt);
        if let StateValue::Map(entries) = &mut written {
            entries.push(("to_cancel".to_string(), self.carried_to_cancel(gtt).into()));
        }
        written
    }

    /// True for a stop GTT that holds no open leg's stop (module docs): its position is
    /// flat, or its leg holds another.
    fn carried_to_cancel(&self, gtt: &Gtt) -> bool {
        let key = PositionKey {
            instrument: gtt.instrument,
            product: gtt.product,
        };
        let legs_own = self.leg_gtt_of(key).and_then(|leg| leg.gtt) == Some(gtt.id);
        gtt.purpose == GttPurpose::Protect && (self.held_units(key) == 0.0 || !legs_own)
    }

    fn write_carried_risk(&self) -> StateValue {
        let daily = self.risk.daily_loss.totals();
        let totals = |t: crate::rms::AccountTotals| {
            map([
                ("realized_rupees", t.realized_rupees.into()),
                ("costs_rupees", t.costs_rupees.into()),
            ])
        };
        let held = self.timeline.len();
        map([
            ("trading_state", self.risk.lock.state().code().into()),
            (
                "trading_state_cause",
                self.risk.lock.cause().map(TradingStateCause::code).into(),
            ),
            ("flat", totals(daily.flat)),
            ("live_run_from", daily.live_run_from.map(totals).into()),
            (
                "drawdown_peak_rupees",
                self.risk.drawdown.map(|watch| watch.peak_rupees()).into(),
            ),
            (
                "held_slices",
                list(&self.risk.entered_at, |(key, slice)| {
                    let [instrument, product] = key_fields(*key);
                    let before = self.risk.carried_slices_of(*key);
                    let slices = held.saturating_sub(*slice) + before;
                    map([instrument, product, ("slices", slices.into())])
                }),
            ),
        ])
    }

    /// Give the session of a later trading day the state `carried_state` saved (module
    /// docs). Returns the new ids of its GTTs and strategy GTT orders.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `LiveState` when the state is refused,
    /// whole: `NotRestorable` for a session that has traded or was restored already;
    /// `Version`, `CarriedMismatch` (the same or a later day, other instruments, another
    /// capital, an MIS position), `MissedExpiryExit` (a position on a contract with a
    /// positional exit day past its last trading day) and `Malformed` (a field that cannot
    /// be read, parts that do not hold together, an expired position with no price to
    /// record its exit at) for the state itself.
    pub fn restore_carried_state(&mut self, state: &StateValue) -> Result<CarriedIds, KernelError> {
        if !self.is_external() {
            return Err(VenueError::NotExternal.into());
        }
        self.check_restorable()?;
        let carried = self.read_carried(&Read::carried_root(state))?;
        self.apply_carried(carried)
    }
}

/// `trade` as it is carried: what held its excursion back ended with the day's orders.
fn cleared(trade: &OpenTrade) -> OpenTrade {
    let mut carried = trade.clone();
    carried.excursion.hold = Hold::default();
    carried.excursion.inside_fill_slice = None;
    carried
}
