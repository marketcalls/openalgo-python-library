//! The pre-open decision of a daily or longer strategy in a live session (design
//! 20-positional-live section 8, decision 11.1, review M8; unit N4 item N4-3).
//!
//! A backtest of a daily strategy decides at each session's close and fills a MARKET
//! order at the next session's first base-bar open. A live session decides the same bar
//! at the next start, before the open, on the broker's candle of the session that closed:
//!
//! - **In the session** a daily or longer strategy makes no decision of its own: no
//!   strategy bar closes for it while the session runs (`steps::callbacks_pause`); its
//!   session's DAY orders are valid to the session's close, as an intraday strategy's
//!   (`submit::day_validity`).
//! - **`Engine::pre_open_decision(bars, last_decided, now)`**, once, after the catch-up and
//!   any restore, before the first slice the session runs live: of the bars the live layer
//!   gives (the broker's candles of the strategy's interval, one or more per instrument),
//!   those whose interval has completed by the session's open are fed (an instrument with
//!   no last close yet takes the latest one's close, which sizes and values the orders as a
//!   backtest's decision at that close does), and the latest of them is decided, unless its
//!   close is at or before `last_decided` (the journal's last decided bar): a bar is never
//!   decided twice, and bars of missed days feed the indicators only. The decision is a
//!   `Pause::BarClose` decided at the bar's close, stamped in step 7 of the session's next
//!   slice; the engine stands at it (`Phase::PreOpen`) until the next `advance`, and the
//!   strategy's orders are submitted as at any decision. The answer names the bar decided
//!   (`PreOpen::Decided`) or that none was (`PreOpen::NotDecided`).
//! - **Its orders wait for the open.** An order decided before the session's open (by
//!   `now` and the session's clock) is not sent at once: it is sent as the session reaches
//!   its open (a tick at or after it, or the first slice's open phase), in the order it was
//!   decided, so a MARKET order goes at the open, as the backtest of the same bars fills it
//!   at that open; LIMIT and stop orders too. A strategy's GTT order rests as its GTT at
//!   once, as brokers take GTTs outside market hours. Until then the order is not at the
//!   venue: a cancellation ends it at once, and the venue's results are refused.
//! - **A late start.** A decision made once the session's open has passed is sent at once,
//!   flagged `late_decision`: the order (`Engine::order_flags`) and its fills
//!   (`FillFlags::LATE_DECISION`).
//!
//! A bar of the strategy's interval opens at the open of its interval's first session and
//! closes at the close of its last on the calendar (a daily bar at its session's open and
//! close; a weekly, monthly, quarterly or yearly bar at the open of the first session of its
//! period and the close of the last), stamped so by the live layer; it is completed when
//! that close is at or before the session's open. A weekly strategy is therefore decided
//! only at the start that follows the week's last session. A session is built for a
//! weekly or longer strategy with daily strategy bars of its own (none is decided in the
//! session); the period is kept for the pre-open decision (`Engine::new_session`).

use crate::data::resample::CalendarUnit;
use crate::exec::{FillFlags, OrderId};
use crate::market::calendar::format_ist;
use crate::model::{ist_day, Day, TsNanos};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::Step;
use super::kernel::{Engine, Phase};
use super::market_data::StrategyBar;
use super::pause::Pause;
use super::venue::SendAt;
use super::venue_error::VenueError;

/// What `Engine::pre_open_decision` did.
#[derive(Clone, Debug, PartialEq)]
pub enum PreOpen {
    /// The latest completed bar was decided: the engine stands at `pause` (a
    /// `Pause::BarClose`) for the strategy's `on_bar`; `bar_close` is that bar's close
    /// (UTC nanoseconds), what the live layer records as its last decided bar; `late` when
    /// the session's open had passed, so its orders go at once, flagged `late_decision`.
    Decided {
        pause: Pause,
        bar_close: TsNanos,
        late: bool,
    },
    /// No bar was decided: none of those given had completed by the session's open, or
    /// the latest had been decided already (at or before `last_decided`).
    NotDecided,
}

/// A live session's pre-open decision and its orders, and whether the session has
/// reached its open (module docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct PreOpenDesk {
    /// The period of a weekly or longer strategy, whose session runs on daily strategy
    /// bars; `None` for a daily or intraday strategy.
    pub(super) period: Option<CalendarUnit>,
    /// The session took its pre-open decision call (it takes one).
    pub(super) asked: bool,
    /// The decision was made after the session's open had passed.
    pub(super) late: bool,
    /// Orders of the decision waiting for the session's open, in the order they were
    /// decided.
    pub(super) at_open: Vec<OrderId>,
    /// The latest trading day whose open the session has reached, the expiry exits due
    /// there made and the rearm_daily stops asked for again (`reach_session_open`); `None`
    /// before the first. Keyed by the day, so a session of several days reaches each
    /// day's open as it reaches the first.
    pub(super) open_reached_on: Option<Day>,
}

fn invalid(reason: String) -> KernelError {
    KernelError::InvalidConfig {
        field: "pre_open_decision",
        reason,
    }
}

impl Engine {
    /// Decide before the session's open on the latest completed bar among `bars` (module
    /// docs), unless its close is at or before `last_decided_ns`; `now_ns` is the present
    /// (UTC nanoseconds), which tells a late start.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `InvalidConfig` naming
    /// `pre_open_decision` for an intraday strategy, a second call, a call once the
    /// session has started a slice live or finished, and a bar that is not a bar of the
    /// strategy's interval (its close not its interval's close on the calendar, a price
    /// that is not positive) or names an unknown instrument.
    pub fn pre_open_decision(
        &mut self,
        bars: &[StrategyBar],
        last_decided_ns: Option<TsNanos>,
        now_ns: TsNanos,
    ) -> Result<PreOpen, KernelError> {
        self.require_session()?;
        let Phase::Ready(index) = self.phase else {
            return Err(invalid(
                "a pre-open decision is taken between slices, before the session runs live"
                    .to_string(),
            ));
        };
        if self.config.strategy_interval.is_intraday() || self.venue.pre_open.asked {
            return Err(invalid(
                "a daily or longer strategy's session takes one pre-open decision".to_string(),
            ));
        }
        let slice = self.timeline.slice(index).ok_or(VenueError::SessionOver)?;
        let open = self.session_open_of(&slice)?;
        for bar in bars {
            self.check_interval_bar(bar)?;
        }
        self.venue.pre_open.asked = true;
        let completed: Vec<StrategyBar> = bars
            .iter()
            .filter(|bar| bar.ts_close <= open)
            .copied()
            .collect();
        self.feed_last_closes(&completed);
        let Some(latest) = completed.iter().map(|bar| bar.ts_close).max() else {
            return Ok(PreOpen::NotDecided);
        };
        if last_decided_ns.is_some_and(|last| latest <= last) {
            return Ok(PreOpen::NotDecided);
        }
        self.venue.clock_ns = self.venue.clock_ns.max(now_ns);
        let late = index > 0 || self.venue.clock_ns >= open;
        self.venue.pre_open.late = late;
        self.phase = Phase::PreOpen(slice, latest);
        let pause = Pause::BarClose {
            stamp: Self::stamp(&slice, Step::Callbacks),
            decided_at: latest,
            strategy_bar_closed: true,
            bars: completed
                .into_iter()
                .filter(|bar| bar.ts_close == latest)
                .collect(),
            events: std::mem::take(&mut self.undelivered),
        };
        Ok(PreOpen::Decided {
            pause,
            bar_close: latest,
            late,
        })
    }

    /// The open of `slice`'s session on the calendar, UTC nanoseconds.
    ///
    /// Errors: `InvalidConfig` when the exchange has no session that day.
    fn session_open_of(&self, slice: &SliceInfo) -> Result<TsNanos, KernelError> {
        let exchange = self.timeline.exchange();
        self.calendar
            .session(exchange, slice.day)
            .map(|session| session.open)
            .ok_or_else(|| {
                invalid(format!(
                    "{} has no session on day {}",
                    exchange.code(),
                    slice.day
                ))
            })
    }

    /// Refuse a bar that is not a bar of the strategy's interval (module docs).
    fn check_interval_bar(&self, bar: &StrategyBar) -> Result<(), KernelError> {
        let spec = self.spec(bar.instrument)?;
        let prices = [bar.open, bar.high, bar.low, bar.close];
        if !prices.iter().all(|price| price.is_finite() && *price > 0.0) {
            return Err(invalid(format!(
                "{}: the bar closing at {} has a price that is not positive",
                spec.symbol,
                format_ist(bar.ts_close)
            )));
        }
        let exchange = self.timeline.exchange();
        let day = ist_day(bar.ts_open);
        let bounds = match self.venue.pre_open.period {
            None => self
                .calendar
                .session(exchange, day)
                .map(|session| (session.open, session.close)),
            Some(unit) => {
                let (first, last) = unit.range(day);
                self.calendar.period_bounds(exchange, first, last)
            }
        };
        if bounds != Some((bar.ts_open, bar.ts_close)) {
            let what = match self.venue.pre_open.period {
                None => "day",
                Some(_) => "period",
            };
            return Err(invalid(format!(
                "{}: the bar opening at {} and closing at {} is not its {what}'s bar on the \
                 calendar (the open of its first session to the close of its last): give the \
                 bars of the strategy's interval",
                spec.symbol,
                format_ist(bar.ts_open),
                format_ist(bar.ts_close)
            )));
        }
        Ok(())
    }

    /// Each instrument with no last close yet takes the close of its latest completed bar
    /// (module docs).
    fn feed_last_closes(&mut self, completed: &[StrategyBar]) {
        for bar in completed {
            let Some(slot) = self.last_close.get_mut(bar.instrument.index()) else {
                continue;
            };
            let latest = completed
                .iter()
                .filter(|other| other.instrument == bar.instrument)
                .all(|other| other.ts_close <= bar.ts_close);
            if slot.is_none() && latest {
                *slot = Some(bar.close);
            }
        }
    }

    /// In a live session, whether order `id`, which has passed its checks at a pre-open
    /// decision, waits for the session's open instead of being sent now (module docs); a
    /// late decision's order is sent now and kept as late. False at any other decision.
    pub(super) fn waits_for_open(&mut self, id: OrderId, rests_as_gtt: bool) -> bool {
        if !matches!(self.phase, Phase::PreOpen(..)) {
            return false;
        }
        if self.venue.pre_open.late {
            // Its fills are flagged as an engine exit's are (`venue_exit_orders`).
            let flags = FillFlags::LATE_DECISION;
            self.venue.exit_flags.push((id, flags));
            return false;
        }
        if rests_as_gtt {
            return false;
        }
        self.venue.pre_open.at_open.push(id);
        true
    }

    /// True for an order of a pre-open decision still waiting for the session's open.
    pub(super) fn waiting_for_open(&self, id: OrderId) -> bool {
        self.venue.pre_open.at_open.contains(&id)
    }

    /// The orders of a pre-open decision that wait for the open and still work, taken to
    /// be sent now (module docs); none before the session's open.
    pub(super) fn take_orders_at_open(&mut self) -> Vec<OrderId> {
        let book = &self.book;
        let waiting = std::mem::take(&mut self.venue.pre_open.at_open);
        waiting
            .into_iter()
            .filter(|id| book.get(*id).is_some_and(|order| order.status.is_working()))
            .collect()
    }

    /// What waited for the session's open goes once the session has reached it, decided
    /// as `at` says, whatever reached it first (a tick at or after it, a venue's result, or
    /// the first slice's open phase): the expiry exit of a position carried past its
    /// contract's exit day (`venue_exits::expiry_exits_at_session_open`) and a new day's
    /// stop for each rearm_daily leg (`rearm`), once per trading day, then the orders of a
    /// pre-open decision. Called first as the exits move on (`progress_exits`), so a
    /// rearm_daily leg being exited gets no day's stop when its leg stops are synced after.
    ///
    /// Errors: those of the expiry exits and of `send_order`.
    pub(super) fn reach_session_open(&mut self, at: SendAt) -> Result<(), KernelError> {
        if self.before_session_open(&at) {
            return Ok(());
        }
        let day = at.slice.day;
        if at.slice.opens_session && self.venue.pre_open.open_reached_on != Some(day) {
            self.venue.pre_open.open_reached_on = Some(day);
            self.expiry_exits_at_session_open(at)?;
            self.rearm_leg_stops_at_open();
        }
        self.send_orders_at_open(at)
    }

    /// Send the orders of a pre-open decision that wait for the session's open, once the
    /// session has reached it, decided as `at` says (module docs).
    ///
    /// Errors: those of `send_order`.
    fn send_orders_at_open(&mut self, at: SendAt) -> Result<(), KernelError> {
        if self.venue.pre_open.at_open.is_empty() || self.before_session_open(&at) {
            return Ok(());
        }
        for id in self.take_orders_at_open() {
            self.send_order(at, id)?;
        }
        Ok(())
    }

    /// At the open of a session's first slice, in a live session: the session's clock
    /// reaches that open, so what waited for it goes as the exits move on
    /// (`progress_exits`): the expiry exit of a position carried past its contract's exit
    /// day and a pre-open decision's orders (`reach_session_open`), then a rearm_daily
    /// leg's stop (`rearm`). A tick at or after the open has moved them on already when one
    /// came first; when none did, the open of the slice's bar is the session's first price,
    /// and its open positions are marked there first, so a rearm_daily stop the open is
    /// already beyond exits the leg rather than being placed (`rearm_waits_or_crosses`). A
    /// backtest does nothing here.
    ///
    /// Errors: those of `progress_exits`.
    pub(super) fn venue_at_session_open(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        if !self.is_external() || !slice.opens_session {
            return Ok(());
        }
        if self.venue.clock_ns < slice.open {
            self.mark_at_first_open(slice);
        }
        self.venue.clock_ns = self.venue.clock_ns.max(slice.open);
        let at = SendAt {
            slice: *slice,
            step: Step::OpenFills,
            decided_at: slice.open,
        };
        self.progress_exits(at)
    }

    /// Mark every open position whose instrument has a bar in the session's first slice
    /// at that bar's open: no tick reached the open before the slice ran (module docs of
    /// `venue_at_session_open`).
    fn mark_at_first_open(&mut self, slice: &SliceInfo) {
        for position in self.portfolio.positions_mut() {
            if position.is_flat() {
                continue;
            }
            if let Some(bar) = self.market.bar(position.key.instrument, slice.index) {
                position.mark_rupees = Some(bar.open);
            }
        }
    }

    /// The flags of working order `id`'s fills (`FillFlags`): `late_decision` for an order
    /// of a late pre-open decision (module docs), `gtt_child_unfilled` or `stop_crossed`
    /// for the engine's exit orders that carry them; `NONE` otherwise and once the order
    /// has ended (its fills keep their flags).
    pub fn order_flags(&self, id: OrderId) -> FillFlags {
        self.exit_flags_of(id)
    }
}
