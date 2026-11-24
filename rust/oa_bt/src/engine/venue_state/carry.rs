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
//! | `orders`, `gtts` | the strategy's GTT orders resting at the venue, and the GTTs at the venue: the legs' stop GTTs and those orders' GTTs, each with its kernel id and terms |
//! | `risk` | the trading state, the daily loss measure's totals (its run baseline), the drawdown peak, and the slices each open position has been held |
//! | `open_trades` | the round trips still open, with their excursion so far |
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
//! carried position must be CNC or NRML (MIS is squared off each day). The parts are
//! checked as a same-day state's are (`checks`), and so are the GTTs: a stop GTT holds an
//! open CNC or NRML leg's stop on its closing side for at most its units, and a GTT order
//! names a carried GTT order. Anything else refuses the state whole (`LiveStateError`).
//!
//! Restored: the positions, marked at the catch-up's last close where there is one; the
//! legs; the GTT orders and GTTs under new ids, every GTT resting (a request the venue had
//! not answered when the session stopped rests on the terms asked; the start's GTT book
//! read settles it, design 4.2 step 5); a carried leg without a stop GTT gets none by
//! itself, `protect` places it (`venue_gtt_protect`). The daily loss measure starts the
//! session as a backtest's does at a session boundary (a run still open counts from its
//! first fill, decision 11.9), and so does the trading state (a daily loss lock lifts, a
//! drawdown halt stays); the sale proceeds withheld are credited, the day after the sale.
//! Reading, checking and applying it is `carry_restore`.

use crate::exec::{GttId, OrderId};
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
        let trades = self.record.trades.open.iter().map(cleared);
        let instruments = list(&self.instruments, |spec| {
            map([
                ("symbol", spec.symbol.as_str().into()),
                ("exchange", spec.exchange.code().into()),
            ])
        });
        Ok(map([
            ("layout", LIVE_STATE_LAYOUT.into()),
            ("kind", CARRIED.into()),
            ("day", self.day_at(self.timeline.len()).into()),
            ("instruments", instruments),
            ("account", write_account(&self.portfolio.account)),
            ("positions", list(open, write_position)),
            ("legs", list(&self.legs, |(key, leg)| write_leg(*key, leg))),
            (
                "orders",
                list(orders, |order| {
                    write_order(order, self.order_role(order.id), true, &[])
                }),
            ),
            ("gtts", list(gtts, |gtt| write_gtt(book, gtt))),
            ("risk", self.write_carried_risk()),
            (
                "open_trades",
                list(trades, |trade| write_open_trade(book, &trade)),
            ),
        ]))
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
    /// `Version`, `OtherSession` (the same or a later day, other instruments, another
    /// capital, an MIS position) and `Malformed` (a field that cannot be read, or parts
    /// that do not hold together) for the state itself.
    pub fn restore_carried_state(&mut self, state: &StateValue) -> Result<CarriedIds, KernelError> {
        if !self.is_external() {
            return Err(VenueError::NotExternal.into());
        }
        self.check_restorable()?;
        let carried = self.read_carried(&Read::root(state))?;
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
