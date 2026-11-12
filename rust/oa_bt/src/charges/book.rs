//! `ChargeBook`: the charge state of one run and the record of every charged fill.

use crate::exec::{InstrumentId, InstrumentSpec};
use crate::market::costs::{
    ContractNoteAggregator, CostBreakdown, CostModel, DpCharge, NoteGroup, RateStatus, Rounding,
    Segment,
};
use crate::market::facts::InstrumentMarket;
use crate::market::table_rows::RowMeta;
use crate::model::{Day, Exchange, InstrumentKind, Product, Side};
use crate::portfolio::{round_to_paise, FillEffect, FillKind, PositionChange, PositionKey};

use super::notes::{settle, SettledNote};
use super::same_day::{difference, split_cost, Repriced, SameDayBook, Segments};
use super::usage::{MarketTable, TableUse};
use super::{fill_segment, ChargeError};

/// What one fill was charged, rupees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillCharge {
    /// Position of the fill in the run's fills table.
    pub fill: u32,
    /// IST trade date.
    pub day: Day,
    /// Cost segment.
    pub segment: Segment,
    /// The six components as the cost model returns them (unrounded accruals in
    /// contract-note rounding).
    pub cost: CostBreakdown,
    /// The model's reported total of `cost`.
    pub cost_rupees: f64,
    /// The DP charge, on the first delivery sell of a scrip in a day.
    pub dp_charge: Option<DpCharge>,
    /// Which schedule version priced the fill and whether its rates are verified.
    pub status: RateStatus,
    /// Number of the closed trade the fill belongs to (as in the trades table), `None`
    /// while its trade is still open.
    pub trade: Option<u32>,
}

impl FillCharge {
    /// What the fill costs in all, rupees: the cost plus the DP charge.
    pub fn total_rupees(&self) -> f64 {
        self.cost_rupees + self.dp_charge.map_or(0.0, |dp| dp.total())
    }
}

/// One fill to charge.
#[derive(Clone, Copy, Debug)]
pub struct FillToCharge<'a> {
    /// Position of the fill in the run's fills table.
    pub fill: u32,
    /// The instrument.
    pub spec: &'a InstrumentSpec,
    /// Its market facts, when the run gave them.
    pub facts: Option<&'a InstrumentMarket>,
    /// Instrument and product.
    pub key: PositionKey,
    /// Buy or sell.
    pub side: Side,
    /// Units, positive.
    pub units: f64,
    /// Fill price after slippage, rupees.
    pub price_rupees: f64,
    /// IST trade date of the fill.
    pub day: Day,
    /// What the fill did to the position.
    pub effect: FillEffect,
}

/// Everything charged in a run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChargesOutput {
    /// Rounding mode of the run's cost model; `None` when the run is cost-free.
    pub rounding: Option<Rounding>,
    /// One record per charged fill, in fill order.
    pub fills: Vec<FillCharge>,
    /// Contract notes, in order of settlement (by day, then note group).
    pub notes: Vec<SettledNote>,
    /// Rows of the dated market tables the run used, in order of first use.
    pub tables: Vec<TableUse>,
}

/// The charge state of a run.
#[derive(Clone, Debug, Default)]
pub struct ChargeBook {
    notes: ContractNoteAggregator,
    /// Rupees posted to the ledger per contract note not yet settled, whole paise.
    posted: Vec<((Day, NoteGroup), f64)>,
    /// The day's CNC cash-equity fills and DP charges (`same_day`).
    same_day: SameDayBook,
    /// Open round trips: the position and the records of their fills.
    open_trades: Vec<(PositionKey, Vec<usize>)>,
    closed_trades: u32,
    output: ChargesOutput,
}

fn is_cash_equity(spec: &InstrumentSpec) -> bool {
    spec.kind == InstrumentKind::Equity && matches!(spec.exchange, Exchange::Nse | Exchange::Bse)
}

impl ChargeBook {
    /// A book for a run charged by a model with `rounding` (`None`: cost-free).
    pub fn new(rounding: Option<Rounding>) -> ChargeBook {
        ChargeBook {
            output: ChargesOutput {
                rounding,
                ..ChargesOutput::default()
            },
            ..ChargeBook::default()
        }
    }

    /// Charge one fill under `model`; returns the rupees to post to the ledger (the
    /// fill's cost plus any DP charge, unrounded; the ledger rounds to the paisa). A CNC
    /// fill of cash equity is charged by the same-day rule (`same_day`): its rupees also
    /// carry the new charges of the earlier fills of the day it turned into intraday.
    ///
    /// Errors: `Cost` when the fill cannot be priced.
    pub fn charge(
        &mut self,
        model: &CostModel,
        fill: &FillToCharge<'_>,
    ) -> Result<f64, ChargeError> {
        if fill.key.product == Product::Cnc && is_cash_equity(fill.spec) {
            return self.charge_delivery(model, fill);
        }
        let segment = fill_segment(fill.spec, fill.facts, fill.key.product)?;
        let cost = model.fill_cost(segment, fill.side, fill.price_rupees, fill.units, fill.day)?;
        self.book_fill(model, fill, segment, cost, None)
    }

    /// Record a priced fill: its note accrual, the record, its round trip. Returns what
    /// it costs in all.
    fn book_fill(
        &mut self,
        model: &CostModel,
        fill: &FillToCharge<'_>,
        segment: Segment,
        cost: CostBreakdown,
        dp_charge: Option<DpCharge>,
    ) -> Result<f64, ChargeError> {
        let status = model.status(segment, fill.day)?;
        let cost_rupees = model.reported_total(&cost);
        if model.rounding() == Rounding::ContractNote {
            self.notes.add(fill.day, segment, &cost);
            self.add_posted(
                (fill.day, segment.note_group()),
                round_to_paise(cost_rupees),
            );
        }
        let record = FillCharge {
            fill: fill.fill,
            day: fill.day,
            segment,
            cost,
            cost_rupees,
            dp_charge,
            status,
            trade: None,
        };
        let total = record.total_rupees();
        self.output.fills.push(record);
        let index = self.output.fills.len() - 1;
        self.track_trade(fill.key, fill.effect, Some(index));
        Ok(total)
    }

    /// A CNC fill of cash equity: the units matched against the day's other side at the
    /// intraday rates, the rest at the delivery rates, the DP charge on a sale from
    /// holdings, and the earlier fills it matched charged again (`same_day`).
    fn charge_delivery(
        &mut self,
        model: &CostModel,
        fill: &FillToCharge<'_>,
    ) -> Result<f64, ChargeError> {
        let segments = Segments {
            delivery: fill_segment(fill.spec, fill.facts, Product::Cnc)?,
            intraday: fill_segment(fill.spec, fill.facts, Product::Mis)?,
        };
        let (instrument, day) = (fill.key.instrument, fill.day);
        let record = self.output.fills.len();
        let matched = self.same_day.match_fill(
            record,
            instrument,
            day,
            fill.side,
            fill.price_rupees,
            fill.units,
        );
        let delivery_units = fill.units - matched.units;
        let split = (matched.units, delivery_units);
        let cost = split_cost(model, segments, fill.side, fill.price_rupees, split, day)?;
        let from_holdings = fill.side == Side::Sell && fill.effect.kind == FillKind::Reduce;
        let dp_charge = if from_holdings
            && delivery_units > 0.0
            && self.same_day.dp_record(instrument, day).is_none()
        {
            self.same_day.note_dp(instrument, day, record);
            Some(model.dp_charge(day)?)
        } else {
            None
        };
        let mut repriced = 0.0;
        for earlier in &matched.repriced {
            repriced += self.charge_again(model, segments, day, earlier)?;
        }
        repriced += self.refund_dp(instrument, day);
        let segment = if matched.units > 0.0 {
            segments.intraday
        } else {
            segments.delivery
        };
        Ok(self.book_fill(model, fill, segment, cost, dp_charge)? + repriced)
    }

    /// Charge an earlier fill of the day again with its new split; returns the change,
    /// rupees, which the note and the ledger take with the fill that caused it.
    fn charge_again(
        &mut self,
        model: &CostModel,
        segments: Segments,
        day: Day,
        earlier: &Repriced,
    ) -> Result<f64, ChargeError> {
        let Some(old) = self.output.fills.get(earlier.record).copied() else {
            return Ok(0.0);
        };
        let split = (earlier.intraday_units, earlier.delivery_units);
        let cost = split_cost(
            model,
            segments,
            earlier.side,
            earlier.price_rupees,
            split,
            day,
        )?;
        let cost_rupees = model.reported_total(&cost);
        let change = cost_rupees - old.cost_rupees;
        if model.rounding() == Rounding::ContractNote {
            self.notes
                .amend(day, segments.intraday, &difference(&cost, &old.cost));
            self.add_posted(
                (day, segments.intraday.note_group()),
                round_to_paise(change),
            );
        }
        if let Some(record) = self.output.fills.get_mut(earlier.record) {
            record.cost = cost;
            record.cost_rupees = cost_rupees;
            if earlier.delivery_units <= 0.0 {
                record.segment = segments.intraday;
            }
        }
        Ok(change)
    }

    /// Refund the scrip's DP charge of `day` once no unit of it is sold from holdings
    /// that day any more; returns the change, rupees (0 or minus the charge).
    fn refund_dp(&mut self, instrument: InstrumentId, day: Day) -> f64 {
        if self.same_day.delivery_sold(instrument, day) > 0.0 {
            return 0.0;
        }
        let Some(record) = self.same_day.dp_record(instrument, day) else {
            return 0.0;
        };
        self.same_day.forget_dp(instrument, day);
        let refunded = self
            .output
            .fills
            .get_mut(record)
            .and_then(|charge| charge.dp_charge.take());
        refunded.map_or(0.0, |dp| -dp.total())
    }

    fn add_posted(&mut self, note: (Day, NoteGroup), rupees: f64) {
        match self.posted.iter_mut().find(|(key, _)| *key == note) {
            Some((_, posted)) => *posted = round_to_paise(*posted + rupees),
            None => self.posted.push((note, rupees)),
        }
    }

    /// A fill that is not charged (a futures position settled at expiry): no record,
    /// but its round trip ends, so later trades keep the trades table's numbers.
    pub fn note_uncharged(&mut self, key: PositionKey, effect: FillEffect) {
        self.track_trade(key, effect, None);
    }

    /// Number the round trips the way the trades table does: in order of closing.
    /// `record` is the charge record of the fill, `None` for an uncharged one.
    fn track_trade(&mut self, key: PositionKey, effect: FillEffect, record: Option<usize>) {
        if effect.kind == FillKind::Entry {
            self.open_trades.retain(|(k, _)| *k != key);
            self.open_trades.push((key, Vec::new()));
        }
        let Some(slot) = self.open_trades.iter().position(|(k, _)| *k == key) else {
            return;
        };
        self.open_trades[slot].1.extend(record);
        if effect.change == PositionChange::Closed {
            let (_, records) = self.open_trades.remove(slot);
            self.closed_trades += 1;
            for index in records {
                if let Some(charge) = self.output.fills.get_mut(index) {
                    charge.trade = Some(self.closed_trades);
                }
            }
        }
    }

    /// Settle the contract notes of the IST trade date `day` (step 8 of its session's
    /// last slice). Returns the adjustments to post to the ledger, rupees, one per note.
    pub fn settle_day(&mut self, day: Day) -> Vec<f64> {
        let mut adjustments = Vec::new();
        let due: Vec<(Day, NoteGroup)> = self
            .posted
            .iter()
            .map(|(key, _)| *key)
            .filter(|(note_day, _)| *note_day == day)
            .collect();
        for (note_day, group) in due {
            let Some(slot) = self
                .posted
                .iter()
                .position(|(key, _)| *key == (note_day, group))
            else {
                continue;
            };
            let (_, posted) = self.posted.remove(slot);
            if let Some(line) = self.notes.note(note_day, group) {
                let settled = settle(line, posted);
                adjustments.push(settled.adjustment_rupees);
                self.output.notes.push(settled);
            }
        }
        adjustments
    }

    /// Note that the run used a row of a dated market table on `day`.
    pub fn record_use(&mut self, table: MarketTable, row: &'static RowMeta, day: Day) {
        let seen = self
            .output
            .tables
            .iter()
            .any(|used| used.table == table && used.row.id == row.id);
        if !seen {
            self.output.tables.push(TableUse {
                table,
                row,
                first_day: day,
            });
        }
    }

    /// Everything charged in the run. Notes of a day whose session close the run did
    /// not reach are recorded unsettled (no adjustment was posted).
    pub fn into_output(mut self) -> ChargesOutput {
        for ((day, group), posted) in std::mem::take(&mut self.posted) {
            if let Some(line) = self.notes.note(day, group) {
                self.output.notes.push(SettledNote::unsettled(line, posted));
            }
        }
        self.output
    }
}
