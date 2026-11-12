//! What a run produces: the result tables, as plain records and columns that the
//! bindings turn into NumPy arrays and the Python layer into DataFrames.
//!
//! | Table | One row per | Notes |
//! |---|---|---|
//! | `fills` | fill | slice, step, side, units, price, where in the bar, reason, order, flags, costs |
//! | `levels` | slice in which a position or its levels changed | the position history: quantity, average price, stop, targets, remainder, trail, breakeven (as of the end of the slice; in force from the next) |
//! | `equity` | slice | the equity series the metrics are computed from |
//! | `trades` | closed round trip (flat to flat) | positions still open at the end are not trades; they are in `final_positions` |
//! | `orders` | order | every order with its final state |
//! | `events` | event | the stream of `event` |
//! | `stale_marks` | open position marked at a last close for want of a bar | SPEC-07 |
//! | `charges` | charged fill, contract note, market-table row used | `charges` |
//!
//! Equity is `capital + realized - costs + unrealized`, recorded at the end of every
//! slice, after its session-close step.

use crate::charges::ChargesOutput;
use crate::exec::{Fill, InstrumentId, InstrumentSpec, Order};
use crate::model::{Day, Product, Side, TsNanos};
use crate::portfolio::{FillEffect, FillKind, PositionChange, PositionKey};
use crate::rms::leg::{BreakevenState, LevelInForce, TargetLevel, TrailState};
use crate::rms::ExitReason;

use super::event::{Event, EventStamp};
use super::run_end::RunEnd;

/// A fill and when it happened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillRecord {
    pub stamp: EventStamp,
    pub product: Product,
    pub fill: Fill,
}

/// One row of the position history: the state at the end of a slice in which the
/// position or its levels changed.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelRow {
    /// Index and open instant (UTC nanoseconds) of the slice.
    pub slice: u32,
    pub slice_ts: TsNanos,
    pub instrument: InstrumentId,
    pub product: Product,
    /// Signed quantity in units (0 when flat).
    pub quantity_units: f64,
    /// Average price in rupees, `None` when flat.
    pub average_price_rupees: Option<f64>,
    pub stop: Option<LevelInForce>,
    pub targets: Vec<TargetLevel>,
    /// Units assigned to no target.
    pub remainder_units: f64,
    pub trail: Option<TrailState>,
    pub breakeven: Option<BreakevenState>,
    /// P&L realized on this instrument and product so far, rupees, gross.
    pub realized_rupees: f64,
}

/// An open position marked at a last close for want of a bar (SPEC-07).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaleMark {
    pub slice: u32,
    pub slice_ts: TsNanos,
    pub instrument: InstrumentId,
    pub mark_rupees: f64,
}

/// The equity series: one row per slice, in rupees.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EquityColumns {
    /// Close of the slice's base bar, UTC nanoseconds: the instant the marks are valid.
    pub ts: Vec<TsNanos>,
    /// IST session date of the slice (days since 1970-01-01).
    pub day: Vec<Day>,
    /// Capital plus realized minus costs plus unrealized.
    pub equity_rupees: Vec<f64>,
    /// Gross P&L realized since the start.
    pub realized_rupees: Vec<f64>,
    /// Open P&L at the marks.
    pub unrealized_rupees: Vec<f64>,
    /// Charges paid since the start.
    pub costs_rupees: Vec<f64>,
    /// Margin blocked by open positions.
    pub margin_rupees: Vec<f64>,
    /// Gross notional of open positions at the marks (sum of |units| x mark x multiplier).
    pub exposure_rupees: Vec<f64>,
}

/// One closed round trip: from the fill that opened a position to the fill that
/// closed it, adds and partial exits included.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TradeRecord {
    /// 1 for the first trade of the run, in order of closing.
    pub number: u32,
    pub instrument: InstrumentId,
    pub product: Product,
    /// `Buy` for a long trade, `Sell` for a short one.
    pub direction: Side,
    /// Slice index and open instant of the opening fill.
    pub entry_slice: u32,
    pub entry_ts: TsNanos,
    /// Slice index and open instant of the closing fill.
    pub exit_slice: u32,
    pub exit_ts: TsNanos,
    /// Units bought (long) or sold (short) to build the position, adds included.
    pub entry_units: f64,
    /// Largest absolute quantity held.
    pub max_units: f64,
    /// Quantity-weighted average price of the opening fills, rupees.
    pub entry_price_rupees: f64,
    /// Quantity-weighted average price of the closing fills, rupees.
    pub exit_price_rupees: f64,
    /// Realized P&L at average cost, before charges.
    pub gross_pnl_rupees: f64,
    /// Charges of every fill of the trade.
    pub costs_rupees: f64,
    /// Gross P&L less charges.
    pub net_pnl_rupees: f64,
    /// The reason of the closing fill.
    pub exit_reason: ExitReason,
    /// Number of fills.
    pub fills: u32,
}

/// A position at the end of the run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinalPosition {
    pub instrument: InstrumentId,
    pub product: Product,
    /// Signed units.
    pub quantity_units: f64,
    /// `None` when flat.
    pub average_price_rupees: Option<f64>,
    /// The last close seen, rupees.
    pub mark_rupees: Option<f64>,
    pub unrealized_rupees: f64,
    /// True when the mark is a last close carried past missing bars.
    pub stale: bool,
}

/// Gross P&L of the run, rupees.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GrossPnl {
    pub realized_rupees: f64,
    pub unrealized_rupees: f64,
    pub total_rupees: f64,
}

/// The result of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct RunOutput {
    pub capital_rupees: f64,
    pub instruments: Vec<InstrumentSpec>,
    pub fills: Vec<FillRecord>,
    pub levels: Vec<LevelRow>,
    pub equity: EquityColumns,
    pub trades: Vec<TradeRecord>,
    pub orders: Vec<Order>,
    pub events: Vec<Event>,
    pub stale_marks: Vec<StaleMark>,
    /// Charges per fill, contract notes and the dated market-table rows used.
    pub charges: ChargesOutput,
    pub final_positions: Vec<FinalPosition>,
    pub gross_pnl: GrossPnl,
    /// Where the run ended and why, when before the end of its range (`run_end`).
    pub run_end: RunEnd,
}

/// A round trip still open.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OpenTrade {
    key: PositionKey,
    direction: Side,
    entry_slice: u32,
    entry_ts: TsNanos,
    entry_units: f64,
    entry_value: f64,
    exit_units: f64,
    exit_value: f64,
    max_units: f64,
    gross: f64,
    costs: f64,
    fills: u32,
}

/// Builds the trades table from fills.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TradeBook {
    open: Vec<OpenTrade>,
    closed: Vec<TradeRecord>,
}

impl TradeBook {
    /// Account for one fill and its effect on the position.
    pub fn record(&mut self, record: &FillRecord, effect: &FillEffect) {
        let key = PositionKey {
            instrument: record.fill.instrument,
            product: record.product,
        };
        let fill = &record.fill;
        if effect.kind == FillKind::Entry {
            self.open.retain(|trade| trade.key != key);
            self.open.push(OpenTrade {
                key,
                direction: fill.side,
                entry_slice: record.stamp.slice,
                entry_ts: record.stamp.slice_ts,
                entry_units: 0.0,
                entry_value: 0.0,
                exit_units: 0.0,
                exit_value: 0.0,
                max_units: 0.0,
                gross: 0.0,
                costs: 0.0,
                fills: 0,
            });
        }
        let Some(index) = self.open.iter().position(|trade| trade.key == key) else {
            return;
        };
        let trade = &mut self.open[index];
        trade.fills += 1;
        trade.costs += fill.costs_rupees;
        if effect.kind == FillKind::Reduce {
            trade.exit_units += fill.quantity_units;
            trade.exit_value += fill.quantity_units * fill.price_rupees;
            trade.gross += effect.realized_rupees;
        } else {
            trade.entry_units += fill.quantity_units;
            trade.entry_value += fill.quantity_units * fill.price_rupees;
        }
        trade.max_units = trade.max_units.max(effect.quantity_after.abs());
        if effect.change == PositionChange::Closed {
            let done = self.open.remove(index);
            let reason = match fill.reason {
                crate::exec::FillReason::Exit(reason) => reason,
                _ => ExitReason::Signal,
            };
            self.close(done, record.stamp, reason);
        }
    }

    fn close(&mut self, trade: OpenTrade, stamp: EventStamp, reason: ExitReason) {
        let average = |value: f64, units: f64| if units > 0.0 { value / units } else { 0.0 };
        self.closed.push(TradeRecord {
            number: self.closed.len() as u32 + 1,
            instrument: trade.key.instrument,
            product: trade.key.product,
            direction: trade.direction,
            entry_slice: trade.entry_slice,
            entry_ts: trade.entry_ts,
            exit_slice: stamp.slice,
            exit_ts: stamp.slice_ts,
            entry_units: trade.entry_units,
            max_units: trade.max_units,
            entry_price_rupees: average(trade.entry_value, trade.entry_units),
            exit_price_rupees: average(trade.exit_value, trade.exit_units),
            gross_pnl_rupees: trade.gross,
            costs_rupees: trade.costs,
            net_pnl_rupees: trade.gross - trade.costs,
            exit_reason: reason,
            fills: trade.fills,
        });
    }

    /// The closed trades, in order of closing.
    pub fn into_closed(self) -> Vec<TradeRecord> {
        self.closed
    }
}

/// The tables being built while the run advances.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Recorder {
    pub(super) fills: Vec<FillRecord>,
    pub(super) levels: Vec<LevelRow>,
    pub(super) stale_marks: Vec<StaleMark>,
    pub(super) equity: EquityColumns,
    pub(super) trades: TradeBook,
    pub(super) events: Vec<Event>,
    /// Positions whose state changed in the current slice, in order of first change.
    pub(super) changed: Vec<PositionKey>,
}

impl Recorder {
    /// Note that a position or its levels changed in the current slice.
    pub(super) fn touch(&mut self, key: PositionKey) {
        if !self.changed.contains(&key) {
            self.changed.push(key);
        }
    }
}
