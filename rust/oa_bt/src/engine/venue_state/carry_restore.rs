//! Reading, checking and applying a carried state (`carry`; design 20-positional-live
//! 4.2, unit N1 item N1-4): the checks of its day, instruments, capital and products,
//! its parts read whole and checked (`checks`, `carry_checks`) before any of them is
//! applied, then the positions, legs, GTT orders and GTTs put back under new ids (a stop
//! GTT to cancel asked to be cancelled at once), and the risk bookkeeping started as a
//! backtest's next session starts it.

use crate::exec::{GttId, GttPurpose, GttStatus, InstrumentId};
use crate::model::Day;
use crate::portfolio::{Portfolio, PositionKey};
use crate::rms::{
    DailyLossMeasure, DailyLossTotals, DrawdownWatch, LegState, TradingLock, TradingStateCause,
};

use super::super::error::KernelError;
use super::super::gtt::held_as_gtt;
use super::super::kernel::Engine;
use super::super::output::OpenTrade;
use super::super::venue_gtts::{GttFollow, GttLegStop};
use super::book::{read_account, read_open_trade, read_position};
use super::carry::{CarriedIds, CARRIED, NO_CARRIED_GTT};
use super::carry_checks::{check_carried_gtts, CarriedGtt, ReadGtt};
use super::carry_expired::{expired_instruments, read_last_closes, ExpiredExit};
use super::checks::{check_parts, Parts};
use super::error::LiveStateError;
use super::gtts::read_gtt;
use super::legs::read_leg;
use super::orders::{read_orders, Ids, SavedOrder};
use super::requests::read_key;
use super::value::Read;
use super::LIVE_STATE_LAYOUT;

/// A carried state read whole, before any of it is applied.
pub(super) struct Carried {
    portfolio: Portfolio,
    legs: Vec<(PositionKey, LegState)>,
    orders: Vec<SavedOrder>,
    gtts: Vec<CarriedGtt>,
    /// Each open leg's stop GTTs the broker rejected after their placement (`carry`).
    rejections: Vec<(PositionKey, u8)>,
    risk: CarriedRisk,
    open_trades: Vec<OpenTrade>,
    /// The positions on contracts that expired before the session's day, with the price
    /// each exit is recorded at (`carry_expired`).
    expired: Vec<ExpiredExit>,
    ids: Ids,
}

/// The risk bookkeeping of a carried state (`carry`, part `risk`).
struct CarriedRisk {
    lock: TradingLock,
    daily: DailyLossTotals,
    drawdown_peak: Option<f64>,
    held_slices: Vec<(PositionKey, usize)>,
}

/// The error of a carried state that does not fit the session (`CarriedMismatch`).
fn mismatch<T>(reason: String) -> Result<T, LiveStateError> {
    Err(LiveStateError::CarriedMismatch { reason })
}

impl Engine {
    /// Read and check a carried state whole (module docs).
    ///
    /// Errors: `Version`, `CarriedMismatch`, `MissedExpiryExit` and `Malformed`
    /// (`restore_carried_state`).
    pub(super) fn read_carried(&self, read: &Read<'_>) -> Result<Carried, LiveStateError> {
        self.check_carried(read)?;
        let count = self.instruments.len();
        let (orders, ids) = self.read_carried_orders(read)?;
        let position_reads = read.items_of("positions")?;
        let positions = position_reads
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
            at_open: &[],
        };
        check_parts(read, &parts)?;
        let gtts = read
            .items_of("gtts")?
            .iter()
            .map(|gtt| {
                Ok((
                    read_gtt(gtt, &ids, count)?,
                    gtt.bool_of("to_cancel")?,
                    gtt.clone(),
                ))
            })
            .collect::<Result<Vec<ReadGtt<'_>>, LiveStateError>>()?;
        let gtts = check_carried_gtts(&gtts, &positions, &legs, &orders)?;
        let rejections = read
            .items_of("stop_gtt_rejections")?
            .iter()
            .map(|leg| Ok((read_key(leg, count)?, leg.whole_of("rejections")?)))
            .collect::<Result<_, LiveStateError>>()?;
        let last_closes = read_last_closes(read, count)?;
        let expired = self.read_expired_exits(&positions, &position_reads, &last_closes)?;
        Ok(Carried {
            portfolio: Portfolio::resume(read_account(&read.field("account")?)?, positions),
            legs,
            orders,
            gtts,
            rejections,
            risk: read_carried_risk(&read.field("risk")?, count)?,
            open_trades: read
                .items_of("open_trades")?
                .iter()
                .map(|trade| read_open_trade(trade, count, &ids))
                .collect::<Result<_, _>>()?,
            expired,
            ids,
        })
    }

    /// The carried strategy GTT orders, numbered from the session's next order id, and
    /// the (saved id, new id) of each.
    fn read_carried_orders(
        &self,
        read: &Read<'_>,
    ) -> Result<(Vec<SavedOrder>, Ids), LiveStateError> {
        let saved_orders = read.items_of("orders")?;
        read_orders(&saved_orders, self.book.next_id(), self.instruments.len())
    }

    /// Check the state's layout, kind, day, instruments, capital and products (module
    /// docs).
    fn check_carried(&self, read: &Read<'_>) -> Result<(), LiveStateError> {
        let layout: i64 = read.whole_of("layout")?;
        if layout != LIVE_STATE_LAYOUT {
            return Err(LiveStateError::Version { found: layout });
        }
        let kind = read.field("kind")?;
        if !kind.is_text(CARRIED) {
            return Err(kind.malformed(format!("is not {CARRIED:?}: this is no carried state")));
        }
        let saved_day: Day = read.whole_of("day")?;
        let (slice, _) = self.standing();
        match self.day_at(slice) {
            Some(today) if saved_day < today => {}
            today => {
                return mismatch(format!(
                    "a carried state of day {saved_day} goes to a later trading day's \
                     session, not to day {today:?}"
                ))
            }
        }
        self.check_carried_instruments(read)?;
        let saved_capital = read.field("account")?.f64_of("capital_rupees")?;
        if saved_capital != self.config.capital_rupees {
            let ours = self.config.capital_rupees;
            return mismatch(format!(
                "capital {saved_capital} rupees, the session's {ours}"
            ));
        }
        for position in read.items_of("positions")? {
            let key = read_key(&position, self.instruments.len())?;
            if !held_as_gtt(key.product) {
                return mismatch(format!(
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
            return mismatch(format!(
                "instruments {named:?}, the session's {ours:?}: the session trades the \
                 carried instruments"
            ));
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
        let expired = expired_instruments(&carried.expired);
        let mut ids = CarriedIds {
            gtts: Vec::new(),
            orders: carried.ids.pairs,
            expired: expired.clone(),
        };
        let to_cancel = self.restore_carried_gtts(carried.gtts, &expired, &mut ids)?;
        self.carry_leg_gtts(&to_cancel, &carried.rejections);
        let slice = u32::try_from(self.standing().0).unwrap_or(u32::MAX);
        for id in to_cancel {
            self.ask_cancel_gtt(slice, id)?;
        }
        self.resume_carried_risk(carried.risk);
        for trade in &carried.open_trades {
            self.charges.resume_trade(trade.key);
        }
        self.record.trades.open = carried.open_trades;
        // The exchange settled them at the carried day's close: their exits end their runs
        // before the session's boundary, as a backtest's settlement at expiry does.
        self.record_expired_exits(&carried.expired)?;
        self.start_carried_session();
        self.venue.restored = true;
        Ok(ids)
    }

    /// Put the carried GTTs at the venue under new ids, each resting and followed, but a
    /// stop GTT of an `expired` instrument (`carry`); their (saved id, new id) go in
    /// `ids`. Returns the new ids of the stop GTTs to cancel at once.
    ///
    /// Errors: `Exec` from the GTT book, a kernel defect.
    fn restore_carried_gtts(
        &mut self,
        gtts: Vec<CarriedGtt>,
        expired: &[InstrumentId],
        ids: &mut CarriedIds,
    ) -> Result<Vec<GttId>, KernelError> {
        let mut to_cancel = Vec::new();
        for CarriedGtt {
            saved_id,
            gtt,
            to_cancel: cancel,
        } in gtts
        {
            if gtt.purpose == GttPurpose::Protect && expired.contains(&gtt.instrument) {
                continue;
            }
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
            if cancel {
                to_cancel.push(id);
            }
        }
        Ok(to_cancel)
    }

    /// Every carried CNC or NRML leg's stop record. A stop held as a GTT: its carried stop
    /// GTT (none of `to_cancel`), or `NO_CARRIED_GTT`, and its stop GTTs' `rejections`. A
    /// rearm_daily stop: a stop not placed yet, which goes out once the session has
    /// reached its open (`rearm`); a stop GTT carried for such a leg is held by no leg and
    /// is cancelled (`loose_stop_gtts`).
    fn carry_leg_gtts(&mut self, to_cancel: &[GttId], rejections: &[(PositionKey, u8)]) {
        let carried: Vec<PositionKey> = self.legs.iter().map(|(key, _)| *key).collect();
        let rearmed: Vec<PositionKey> = carried
            .iter()
            .copied()
            .filter(|key| self.rearms_daily(*key))
            .collect();
        for key in rearmed {
            self.renew_leg_stop(key, false);
        }
        let keys: Vec<PositionKey> = carried
            .into_iter()
            .filter(|key| self.stop_is_gtt(key.product))
            .collect();
        for key in keys {
            let stop = self.gtts.book.gtts().iter().find(|gtt| {
                gtt.purpose == GttPurpose::Protect
                    && gtt.instrument == key.instrument
                    && gtt.product == key.product
                    && gtt.status.at_venue()
                    && !to_cancel.contains(&gtt.id)
            });
            let gtt = stop.map_or(NO_CARRIED_GTT, |gtt| gtt.id);
            let rejected = rejections.iter().find(|(known, _)| *known == key);
            self.venue.gtts.legs.push(GttLegStop {
                key,
                gtt: Some(gtt),
                rejections: rejected.map_or(0, |(_, count)| *count),
            });
        }
    }

    /// The carried risk bookkeeping as it stood at the carried day's close (module docs);
    /// the session's boundary is `start_carried_session`.
    fn resume_carried_risk(&mut self, carried: CarriedRisk) {
        let risk = &mut self.risk;
        risk.lock = carried.lock;
        risk.daily_loss = DailyLossMeasure::resume(carried.daily);
        let limit = self.config.risk.drawdown_halt_percent;
        if let (Some(limit), Some(peak)) = (limit, carried.drawdown_peak) {
            risk.drawdown = Some(DrawdownWatch::new(limit, peak));
        }
        risk.entered_at = carried
            .held_slices
            .iter()
            .map(|(key, _)| (*key, 0))
            .collect();
        risk.carried_slices = carried.held_slices;
    }

    /// The session boundary a backtest crosses at the next session's open: the daily loss
    /// measure starts the session (a run still open counts from its first fill), then the
    /// trading state.
    fn start_carried_session(&mut self) {
        self.risk.daily_loss.start_session();
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

/// The carried state's `risk` part (`carry`).
///
/// Errors: `Malformed` for a field missing or unusable.
fn read_carried_risk(risk: &Read<'_>, instruments: usize) -> Result<CarriedRisk, LiveStateError> {
    let held_slices = risk
        .items_of("held_slices")?
        .iter()
        .map(|held| Ok((read_key(held, instruments)?, held.whole_of("slices")?)))
        .collect::<Result<_, LiveStateError>>()?;
    Ok(CarriedRisk {
        lock: super::risk::read_lock(risk)?,
        daily: DailyLossTotals {
            flat: super::risk::read_totals(&risk.field("flat")?)?,
            live_run_from: super::risk::read_opt_totals(risk, "live_run_from")?,
            counted_from: None,
        },
        drawdown_peak: risk.opt_f64_of("drawdown_peak_rupees")?,
        held_slices,
    })
}
