//! The eight steps of one slice (03-architecture 4.2), each across every instrument in
//! registration order before the next step starts.
//!
//! Steps 1 to 6 run in `open_slice`, step 7 decides whether to pause
//! (`callbacks_pause`), and step 8 plus the end-of-slice records run in `close_slice`.

use crate::exec::{FillAt, FillFlags, InstrumentId, OrderAction};
use crate::model::Product;
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, OrderEventKind, Step};
use super::fills::FillRequest;
use super::kernel::Engine;
use super::output::{LevelRow, StaleMark};
use super::pause::Pause;

impl Engine {
    /// Steps 1 to 6 of a slice.
    pub(super) fn open_slice(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        // Step 1: session open and reference events. The session's intraday times are
        // resolved on first use (`session_times`); no reference event (lot-size change,
        // roll) can be configured yet.
        self.risk_session_open(slice);
        self.step_open_fills(slice)?;
        self.step_levels(slice)?;
        self.step_ratchets(slice)?;
        self.step_mark(slice);
        self.step_aggregate_rules(slice)
    }

    /// Step 2: queued orders fill at the open; then the square-off timer.
    fn step_open_fills(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        for index in 0..self.instruments.len() {
            let instrument = InstrumentId(index as u32);
            let Some(bar) = self.market.bar(instrument, slice.index) else {
                continue;
            };
            self.risk_exits_at_open(slice, instrument, bar.open)?;
            for id in self.book.queued_for(instrument) {
                self.fill_queued(slice, id, bar.open)?;
            }
            self.time_exit_at_open(slice, instrument, bar.open)?;
            self.square_off_at_open(slice, instrument, bar.open)?;
        }
        Ok(())
    }

    /// Step 3: leg levels against the bar (`legs::test_leg_levels`).
    fn step_levels(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        self.test_leg_levels(slice)
    }

    /// Step 4: trail and breakeven ratchets, in force from the next slice
    /// (`legs::ratchet_legs`).
    fn step_ratchets(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        self.ratchet_legs(slice)
    }

    /// Step 5: mark every open position at its close, or at its last close and flagged
    /// stale when it has no bar (SPEC-07).
    fn step_mark(&mut self, slice: &SliceInfo) {
        for (index, last) in self.last_close.iter_mut().enumerate() {
            if let Some(bar) = self.market.bar(InstrumentId(index as u32), slice.index) {
                *last = Some(bar.close);
            }
        }
        for position in self.portfolio.positions_mut() {
            if position.is_flat() {
                position.stale = false;
                continue;
            }
            let instrument = position.key.instrument;
            let bar = self.market.bar(instrument, slice.index);
            position.stale = bar.is_none();
            position.mark_rupees = bar.map(|b| b.close).or(position.mark_rupees);
            if let (true, Some(mark)) = (position.stale, position.mark_rupees) {
                self.record.stale_marks.push(StaleMark {
                    slice: Self::stamp(slice, Step::Mark).slice,
                    slice_ts: slice.open,
                    instrument,
                    mark_rupees: mark,
                });
            }
        }
    }

    /// Step 7: pause when a strategy bar closes, or when events wait for callbacks.
    pub(super) fn callbacks_pause(&mut self, slice: &SliceInfo) -> Option<Pause> {
        let closing = self.timeline.strategy_bar_closing(slice);
        let waiting = self.config.deliver_events && !self.undelivered.is_empty();
        if closing.is_none() && !waiting {
            return None;
        }
        let bars = closing.map_or_else(Vec::new, |(open, _)| {
            (0..self.instruments.len())
                .filter_map(|i| self.market.strategy_bar(InstrumentId(i as u32), open))
                .collect()
        });
        Some(Pause::BarClose {
            stamp: Self::stamp(slice, Step::Callbacks),
            decided_at: slice.close,
            strategy_bar_closed: closing.is_some(),
            bars,
            events: std::mem::take(&mut self.undelivered),
        })
    }

    /// Step 8 when the session ends at this slice, then the end-of-slice records.
    pub(super) fn close_slice(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        if slice.closes_session {
            self.expire_day_orders(slice)?;
            self.risk_exits_at_session_close(slice)?;
            self.close_intraday_positions(slice)?;
        }
        self.record_levels(slice);
        self.record_equity(slice);
        Ok(())
    }

    /// Expire every DAY order whose validity ends with this session.
    fn expire_day_orders(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let ending: Vec<_> = self
            .book
            .working()
            .filter(|order| order.good_until <= slice.close)
            .map(|order| order.id)
            .collect();
        for id in ending {
            self.book.apply(id, OrderAction::Expire)?;
            let stamp = Self::stamp(slice, Step::SessionClose);
            let kind = OrderEventKind::Expired;
            self.emit(Event::Order {
                stamp,
                order: id,
                kind,
            });
        }
        Ok(())
    }

    /// The square-off check: an intraday position still open at the session close is
    /// closed at its last close, flagged, reason `eod` (SPEC-05).
    fn close_intraday_positions(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let open: Vec<(PositionKey, f64)> = self
            .portfolio
            .positions()
            .iter()
            .filter(|p| p.key.product == Product::Mis && !p.is_flat())
            .map(|p| (p.key, p.quantity_units))
            .collect();
        for (key, units) in open {
            let Some(last) = self.last_close(key.instrument) else {
                continue;
            };
            let request = FillRequest::engine_exit(key, units, last, ExitReason::Eod)
                .at(FillAt::Close)
                .flagged(FillFlags::FILLED_AT_LAST_CLOSE);
            self.execute_fill(Self::stamp(slice, Step::SessionClose), request)?;
        }
        Ok(())
    }

    /// A position-history row for every position that changed in this slice.
    fn record_levels(&mut self, slice: &SliceInfo) {
        for key in std::mem::take(&mut self.record.changed) {
            let Some(position) = self.portfolio.position(key) else {
                continue;
            };
            let leg = self
                .legs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, leg)| leg);
            let row = LevelRow {
                slice: Self::stamp(slice, Step::SessionClose).slice,
                slice_ts: slice.open,
                instrument: key.instrument,
                product: key.product,
                quantity_units: position.quantity_units,
                average_price_rupees: (!position.is_flat())
                    .then_some(position.average_price_rupees),
                stop: leg.and_then(|leg| leg.stop),
                targets: leg.map(|leg| leg.targets.clone()).unwrap_or_default(),
                remainder_units: leg.map_or(0.0, |leg| leg.remainder_units),
                trail: leg.and_then(|leg| leg.trail),
                breakeven: leg.and_then(|leg| leg.breakeven),
                realized_rupees: position.realized_rupees,
            };
            self.record.levels.push(row);
        }
    }

    /// One row of the equity series, at the slice's close.
    fn record_equity(&mut self, slice: &SliceInfo) {
        let account = self.portfolio.account;
        let unrealized = self.portfolio.unrealized_rupees();
        let exposure = self
            .portfolio
            .positions()
            .iter()
            .filter_map(|p| {
                p.mark_rupees
                    .map(|mark| p.quantity_units.abs() * mark * p.multiplier)
            })
            .sum();
        let equity = &mut self.record.equity;
        equity.ts.push(slice.close);
        equity.day.push(slice.day);
        equity.equity_rupees.push(account.equity_rupees(unrealized));
        equity.realized_rupees.push(account.realized_rupees);
        equity.unrealized_rupees.push(unrealized);
        equity.costs_rupees.push(account.costs_rupees);
        equity.margin_rupees.push(account.margin_blocked_rupees);
        equity.exposure_rupees.push(exposure);
    }
}
