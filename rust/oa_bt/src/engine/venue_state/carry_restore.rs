//! Reading, checking and applying a carried state (`carry`; design 20-positional-live
//! 4.2, unit N1 item N1-4): the checks of its day, instruments, capital and products,
//! its parts read whole and checked (`checks`, `carry_checks`) before any of them is
//! applied, then the positions, legs, GTT orders and GTTs put back under new ids, and the
//! risk bookkeeping started as a backtest's next session starts it.

use crate::exec::{GttPurpose, GttStatus, OrderId};
use crate::model::Day;
use crate::portfolio::{Portfolio, PositionKey};
use crate::rms::{DailyLossMeasure, DailyLossTotals, DrawdownWatch, TradingStateCause};

use super::super::error::KernelError;
use super::super::gtt::held_as_gtt;
use super::super::kernel::Engine;
use super::super::output::OpenTrade;
use super::super::venue_gtts::{GttFollow, GttLegStop};
use super::book::{read_account, read_open_trade, read_position};
use super::carry::{CarriedIds, CARRIED, NO_CARRIED_GTT};
use super::carry_checks::{check_carried_gtts, CarriedGtt};
use super::checks::{check_parts, Parts};
use super::error::LiveStateError;
use super::gtts::read_gtt;
use super::legs::read_leg;
use super::orders::{read_order, Ids};
use super::requests::read_key;
use super::value::Read;
use super::LIVE_STATE_LAYOUT;

/// A carried state read whole, before any of it is applied.
pub(super) struct Carried {
    portfolio: Portfolio,
    legs: Vec<(PositionKey, crate::rms::LegState)>,
    orders: Vec<super::orders::SavedOrder>,
    gtts: Vec<CarriedGtt>,
    lock: crate::rms::TradingLock,
    daily: DailyLossTotals,
    drawdown_peak: Option<f64>,
    held_slices: Vec<(PositionKey, usize)>,
    open_trades: Vec<OpenTrade>,
    ids: Ids,
}

impl Engine {
    pub(super) fn read_carried(&self, read: &Read<'_>) -> Result<Carried, LiveStateError> {
        self.check_carried(read)?;
        let count = self.instruments.len();
        let first = self.book.next_id().0;
        let saved_orders = read.items_of("orders")?;
        let mut ids = Ids { pairs: Vec::new() };
        for (offset, order) in (0u64..).zip(&saved_orders) {
            ids.pairs
                .push((OrderId(order.whole_of("order")?), OrderId(first + offset)));
        }
        let orders = saved_orders
            .iter()
            .zip(&ids.pairs)
            .map(|(order, (_, id))| read_order(order, *id, count))
            .collect::<Result<Vec<_>, _>>()?;
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
        let parts = Parts {
            positions: &positions,
            legs: &legs,
            orders: &[],
            exits: &[],
            leg_stops: &[],
        };
        check_parts(read, &parts)?;
        let gtts = read
            .items_of("gtts")?
            .iter()
            .map(|gtt| Ok((read_gtt(gtt, &ids, count)?, gtt.clone())))
            .collect::<Result<Vec<_>, LiveStateError>>()?;
        let gtts = check_carried_gtts(&gtts, &positions, &legs, &orders)?;
        let risk = read.field("risk")?;
        let held_slices = risk
            .items_of("held_slices")?
            .iter()
            .map(|held| Ok((read_key(held, count)?, held.whole_of("slices")?)))
            .collect::<Result<_, LiveStateError>>()?;
        Ok(Carried {
            portfolio: Portfolio::resume(read_account(&read.field("account")?)?, positions),
            legs,
            orders,
            gtts,
            lock: super::risk::read_lock(&risk)?,
            daily: DailyLossTotals {
                flat: super::risk::read_totals(&risk.field("flat")?)?,
                live_run_from: super::risk::read_opt_totals(&risk, "live_run_from")?,
                counted_from: None,
            },
            drawdown_peak: risk.opt_f64_of("drawdown_peak_rupees")?,
            held_slices,
            open_trades: read
                .items_of("open_trades")?
                .iter()
                .map(|trade| read_open_trade(trade, count, &ids))
                .collect::<Result<_, _>>()?,
            ids,
        })
    }

    /// Check the state's layout, kind, day, instruments, capital and products (module
    /// docs).
    fn check_carried(&self, read: &Read<'_>) -> Result<(), LiveStateError> {
        let layout: i64 = read.whole_of("layout")?;
        if layout != LIVE_STATE_LAYOUT {
            return Err(LiveStateError::Version { found: layout });
        }
        if !read.field("kind")?.is_text(CARRIED) {
            return Err(read.malformed("is not a carried state: pass carried_state()"));
        }
        let other = |reason: String| Err(LiveStateError::OtherSession { reason });
        let saved_day: Day = read.whole_of("day")?;
        let (slice, _) = self.standing();
        match self.day_at(slice) {
            Some(today) if saved_day < today => {}
            today => {
                return other(format!(
                    "a carried state of day {saved_day} goes to a later trading day's \
                     session, not to day {today:?}"
                ))
            }
        }
        self.check_carried_instruments(read)?;
        let saved_capital = read.field("account")?.f64_of("capital_rupees")?;
        if saved_capital != self.config.capital_rupees {
            let ours = self.config.capital_rupees;
            return other(format!(
                "capital {saved_capital} rupees, the session's {ours}"
            ));
        }
        for position in read.items_of("positions")? {
            let key = read_key(&position, self.instruments.len())?;
            if !held_as_gtt(key.product) {
                return other(format!(
                    "a carried {} position: only CNC and NRML positions carry overnight",
                    key.product.code()
                ));
            }
        }
        Ok(())
    }

    fn check_carried_instruments(&self, read: &Read<'_>) -> Result<(), LiveStateError> {
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
            return Err(LiveStateError::OtherSession {
                reason: format!(
                    "instruments {named:?}, the session's {ours:?}: the session trades the \
                     carried instruments"
                ),
            });
        }
        Ok(())
    }

    /// Put a carried state read whole in place (module docs).
    ///
    /// Errors: `Exec` from the books, a kernel defect: the ids follow the empty books'
    /// next ids.
    pub(super) fn apply_carried(&mut self, carried: Carried) -> Result<CarriedIds, KernelError> {
        for saved in carried.orders {
            let id = self.book.insert(saved.order)?;
            if let Some(role) = saved.role {
                self.record_role(id, role);
            }
        }
        self.portfolio = carried.portfolio;
        for position in self.portfolio.positions_mut() {
            let close = self.last_close.get(position.key.instrument.index());
            if let Some(close) = close.copied().flatten() {
                position.mark_rupees = Some(close);
            }
            position.stale = false;
        }
        self.portfolio.account.credit_sale_proceeds();
        self.legs = carried.legs;
        let mut ids = CarriedIds {
            gtts: Vec::new(),
            orders: carried.ids.pairs,
        };
        for CarriedGtt { saved_id, gtt } in carried.gtts {
            let id = self.gtts.book.next_id();
            self.gtts.book.insert(crate::exec::Gtt {
                id,
                status: GttStatus::Active,
                eligible_from: self.venue.clock_ns,
                last_modified_slice: None,
                ..gtt
            })?;
            self.venue.gtts.follows.push(GttFollow::new(id));
            ids.gtts.push((saved_id, id));
        }
        self.carry_leg_gtts();
        self.resume_carried_risk(
            carried.lock,
            carried.daily,
            carried.drawdown_peak,
            carried.held_slices,
        );
        for trade in &carried.open_trades {
            self.charges.resume_trade(trade.key);
        }
        self.record.trades.open = carried.open_trades;
        self.venue.restored = true;
        Ok(ids)
    }

    /// Every carried CNC or NRML leg's stop GTT record: its carried stop GTT, or
    /// `NO_CARRIED_GTT`.
    fn carry_leg_gtts(&mut self) {
        let keys: Vec<PositionKey> = self
            .legs
            .iter()
            .map(|(key, _)| *key)
            .filter(|key| held_as_gtt(key.product))
            .collect();
        for key in keys {
            let stop = self.gtts.book.gtts().iter().find(|gtt| {
                gtt.purpose == GttPurpose::Protect
                    && gtt.instrument == key.instrument
                    && gtt.product == key.product
                    && gtt.status.at_venue()
            });
            let gtt = stop.map_or(NO_CARRIED_GTT, |gtt| gtt.id);
            self.venue.gtts.legs.push(GttLegStop {
                key,
                gtt: Some(gtt),
                rejections: 0,
            });
        }
    }

    /// The carried risk bookkeeping, then the session boundary a backtest crosses at the
    /// next session's open (module docs).
    fn resume_carried_risk(
        &mut self,
        lock: crate::rms::TradingLock,
        daily: DailyLossTotals,
        drawdown_peak: Option<f64>,
        held_slices: Vec<(PositionKey, usize)>,
    ) {
        let risk = &mut self.risk;
        risk.lock = lock;
        risk.daily_loss = DailyLossMeasure::resume(daily);
        risk.daily_loss.start_session();
        if let (Some(limit), Some(peak)) = (self.config.risk.drawdown_halt_percent, drawdown_peak) {
            risk.drawdown = Some(DrawdownWatch::new(limit, peak));
        }
        risk.entered_at = held_slices.iter().map(|(key, _)| (*key, 0)).collect();
        risk.carried_slices = held_slices;
        self.carried_session_boundary();
    }

    /// The trading state at the carried session's boundary (a daily loss lock lifts),
    /// written as a backtest's step 1 writes it.
    fn carried_session_boundary(&mut self) {
        let Ok(at) = self.control_point() else {
            return;
        };
        self.change_trading_state(at.stamp(), TradingStateCause::SessionBoundary);
    }
}
