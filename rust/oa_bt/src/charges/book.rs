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
    /// The last day each instrument paid the DP charge.
    dp_days: Vec<(InstrumentId, Day)>,
    /// Instruments bought on delivery (CNC), and on which day last.
    cnc_buys: Vec<(InstrumentId, Day)>,
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
    /// fill's cost plus any DP charge, unrounded; the ledger rounds to the paisa).
    ///
    /// Errors: `Cost` when the fill cannot be priced; `NotBuilt` for a delivery sell of
    /// a scrip bought on delivery the same day (charged as intraday by brokers).
    pub fn charge(
        &mut self,
        model: &CostModel,
        fill: &FillToCharge<'_>,
    ) -> Result<f64, ChargeError> {
        let segment = fill_segment(fill.spec, fill.facts, fill.key.product)?;
        let cost = model.fill_cost(segment, fill.side, fill.price_rupees, fill.units, fill.day)?;
        let status = model.status(segment, fill.day)?;
        let dp_charge = self.delivery_charges(model, fill)?;
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
        self.track_trade(fill.key, fill.effect);
        Ok(total)
    }

    /// The DP charge of a delivery sell from holdings, and the same-day check.
    fn delivery_charges(
        &mut self,
        model: &CostModel,
        fill: &FillToCharge<'_>,
    ) -> Result<Option<DpCharge>, ChargeError> {
        let instrument = fill.key.instrument;
        if fill.key.product != Product::Cnc || !is_cash_equity(fill.spec) {
            return Ok(None);
        }
        if fill.side == Side::Buy {
            self.cnc_buys.retain(|(i, _)| *i != instrument);
            self.cnc_buys.push((instrument, fill.day));
            return Ok(None);
        }
        if self.cnc_buys.contains(&(instrument, fill.day)) {
            return Err(ChargeError::NotBuilt {
                feature: "charges for a CNC buy and sell of one scrip on the same day (brokers \
                          charge it as intraday); trade it as MIS or run with costs=None",
            });
        }
        if fill.effect.kind != FillKind::Reduce || self.dp_days.contains(&(instrument, fill.day)) {
            return Ok(None);
        }
        self.dp_days.retain(|(i, _)| *i != instrument);
        self.dp_days.push((instrument, fill.day));
        Ok(Some(model.dp_charge(fill.day)?))
    }

    fn add_posted(&mut self, note: (Day, NoteGroup), rupees: f64) {
        match self.posted.iter_mut().find(|(key, _)| *key == note) {
            Some((_, posted)) => *posted = round_to_paise(*posted + rupees),
            None => self.posted.push((note, rupees)),
        }
    }

    /// Number the round trips the way the trades table does: in order of closing.
    fn track_trade(&mut self, key: PositionKey, effect: FillEffect) {
        let record = self.output.fills.len().saturating_sub(1);
        if effect.kind == FillKind::Entry {
            self.open_trades.retain(|(k, _)| *k != key);
            self.open_trades.push((key, Vec::new()));
        }
        let Some(slot) = self.open_trades.iter().position(|(k, _)| *k == key) else {
            return;
        };
        self.open_trades[slot].1.push(record);
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
