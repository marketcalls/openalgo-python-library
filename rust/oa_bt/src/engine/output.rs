//! What a run produces: the result tables, as plain records and columns that the
//! bindings turn into NumPy arrays and the Python layer into DataFrames.
//!
//! | Table | One row per | Notes |
//! |---|---|---|
//! | `fills` | fill | slice, step, side, units, price, where in the bar, reason, order, flags, costs, slippage |
//! | `levels` | slice in which a position or its levels changed | the position history: quantity, average price, stop, targets, remainder, trail, breakeven (as of the end of the slice; in force from the next) |
//! | `equity` | slice | the equity series the metrics are computed from |
//! | `trades` | closed round trip (flat to flat) | positions still open at the end are not trades; they are in `final_positions`; each has its MAE, MFE and slippage (`excursion`) |
//! | `orders` | order | every order with its final state |
//! | `events` | event | the stream of `event` |
//! | `stale_marks` | open position marked at a last close for want of a bar | SPEC-07 |
//! | `charges` | charged fill, contract note, market-table row used | `charges` |
//!
//! Equity is `capital + realized - costs + unrealized`, recorded at the end of every
//! slice, after its session-close step.

use crate::charges::ChargesOutput;
use crate::exec::{Fill, FillReason, InstrumentId, InstrumentSpec, Order};
use crate::model::{Day, Product, Side, TsNanos};
use crate::portfolio::{round_to_paise, FillEffect, FillKind, PositionChange, PositionKey};
use crate::rms::leg::{BreakevenState, LevelInForce, TargetLevel, TrailState};

use super::event::{Event, EventStamp};
use super::excursion::{add_slippage, paise, Excursion, FillView};
use super::futures::FuturesOutput;
use super::run_end::RunEnd;

/// A fill and when it happened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillRecord {
    pub stamp: EventStamp,
    pub product: Product,
    pub fill: Fill,
    /// What slippage cost the fill, rupees to the paisa, positive when it was worse
    /// than its price before slippage; `None` for a fill the venue priced (a live
    /// session's), which has none modelled (`excursion`). Already in the fill price.
    pub slippage_rupees: Option<f64>,
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
    /// Realized P&L at average cost, before charges: each closing fill's rounded to
    /// the paisa, as the ledger posts it.
    pub gross_pnl_rupees: f64,
    /// Charges of every fill of the trade, whole paise: what the account paid for
    /// them, contract-note rounding included, set when the run finishes
    /// (`settle_trade_charges`).
    pub costs_rupees: f64,
    /// Gross P&L less charges, whole paise.
    pub net_pnl_rupees: f64,
    /// The reason of the closing fill: an exit reason, or `roll` for a futures
    /// contract closed by a roll (the position goes on in the next contract's trade).
    pub exit_reason: FillReason,
    /// Number of fills.
    pub fills: u32,
    /// What the opening fill would have lost at the stop its entry set, rupees; `None`
    /// when the position opened with no stop (`initial_risk`). The R-multiple of the
    /// trade is its P&L over this.
    pub initial_risk_rupees: Option<f64>,
    /// Maximum adverse excursion: the lowest running P&L the trade showed, before
    /// charges, rupees to the paisa, at most 0 (`excursion` defines the running P&L and
    /// the prices that count).
    pub mae_rupees: f64,
    /// Maximum favourable excursion: the highest running P&L, at least 0.
    pub mfe_rupees: f64,
    /// What slippage cost the trade's fills, rupees: the sum of theirs, `None` when a
    /// fill has none modelled. Already in the gross P&L.
    pub slippage_rupees: Option<f64>,
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
    /// Futures switches and the settlement postings (`futures`, `settlement`).
    pub futures: FuturesOutput,
    pub final_positions: Vec<FinalPosition>,
    pub gross_pnl: GrossPnl,
    /// Where the run ended and why, when before the end of its range (`run_end`).
    pub run_end: RunEnd,
}

/// A round trip still open (a live session's saved state carries it, `venue_state`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct OpenTrade {
    pub(super) key: PositionKey,
    pub(super) direction: Side,
    pub(super) entry_slice: u32,
    pub(super) entry_ts: TsNanos,
    pub(super) entry_units: f64,
    pub(super) entry_value: f64,
    pub(super) exit_units: f64,
    pub(super) exit_value: f64,
    pub(super) max_units: f64,
    pub(super) gross: f64,
    pub(super) fills: u32,
    pub(super) initial_risk: Option<f64>,
    pub(super) excursion: Excursion,
    pub(super) slippage: Option<f64>,
}

/// Builds the trades table from fills.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TradeBook {
    pub(super) open: Vec<OpenTrade>,
    closed: Vec<TradeRecord>,
}

impl TradeBook {
    /// Account for one fill, its effect on the position and what it showed (`view`).
    pub(super) fn record(&mut self, record: &FillRecord, effect: &FillEffect, view: &FillView) {
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
                fills: 0,
                initial_risk: None,
                excursion: Excursion::default(),
                slippage: Some(0.0),
            });
        }
        let Some(index) = self.open.iter().position(|trade| trade.key == key) else {
            return;
        };
        let trade = &mut self.open[index];
        let realized_before = trade.gross;
        trade.fills += 1;
        if effect.kind == FillKind::Reduce {
            trade.exit_units += fill.quantity_units;
            trade.exit_value += fill.quantity_units * fill.price_rupees;
            // As the ledger posts it, so the trades add up to the account.
            trade.gross += round_to_paise(effect.realized_rupees);
        } else {
            trade.entry_units += fill.quantity_units;
            trade.entry_value += fill.quantity_units * fill.price_rupees;
        }
        trade.max_units = trade.max_units.max(effect.quantity_after.abs());
        trade.slippage = add_slippage(trade.slippage, record.slippage_rupees);
        let opened = effect.kind == FillKind::Entry;
        let realized = (realized_before, trade.gross);
        let when = (record.stamp.slice, fill.at);
        let price = fill.price_rupees;
        trade
            .excursion
            .see_fill(when, price, view, opened, realized);
        if effect.change == PositionChange::Closed {
            let done = self.open.remove(index);
            self.close(done, record.stamp, fill.reason);
        }
    }

    fn close(&mut self, trade: OpenTrade, stamp: EventStamp, reason: FillReason) {
        let average = |value: f64, units: f64| if units > 0.0 { value / units } else { 0.0 };
        let gross = round_to_paise(trade.gross);
        let (mae, mfe) = trade.excursion.rounded();
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
            gross_pnl_rupees: gross,
            // Charged when the run finishes, once its contract notes are settled
            // (`settle_trade_charges`); a cost-free run keeps these.
            costs_rupees: 0.0,
            net_pnl_rupees: gross,
            exit_reason: reason,
            fills: trade.fills,
            initial_risk_rupees: trade.initial_risk,
            mae_rupees: mae,
            mfe_rupees: mfe,
            slippage_rupees: trade.slippage.map(paise),
        });
    }

    /// Note the initial risk of the open trade on `key`, which a fill has just opened.
    pub fn set_initial_risk(&mut self, key: PositionKey, rupees: Option<f64>) {
        if let Some(trade) = self.open.iter_mut().find(|trade| trade.key == key) {
            trade.initial_risk = rupees;
        }
    }

    /// The closed trades, in order of closing.
    pub fn into_closed(self) -> Vec<TradeRecord> {
        self.closed
    }
}

/// Give the fills and the closed trades of a finished run their final charges, now
/// that its contract notes are settled: a charged fill costs what the account paid
/// for it (`FillCharge::paid_rupees`: its share of its note plus its DP charge), and a
/// trade the sum over its fills, its net P&L its gross P&L less that. The fills then
/// add up to the account's charges, and so do the trades when no position is open at
/// the end. A cost-free run has no charge records and is left as it is.
///
/// The charge book numbers round trips as this table does (in order of closing, from
/// the same fills), so every record of a closed trade names a trade of the table and
/// every trade has at least its opening fill charged; the debug checks hold the two
/// numberings to that.
pub(super) fn settle_trade_charges(
    fills: &mut [FillRecord],
    trades: &mut [TradeRecord],
    charges: &ChargesOutput,
) {
    if charges.rounding.is_none() {
        return;
    }
    let mut trade_costs: Vec<Option<f64>> = vec![None; trades.len()];
    for record in &charges.fills {
        let paid = record.paid_rupees();
        if let Some(fill) = fills.get_mut(record.fill as usize) {
            fill.fill.costs_rupees = paid;
        }
        let Some(number) = record.trade else {
            continue;
        };
        let slot = number
            .checked_sub(1)
            .and_then(|at| trade_costs.get_mut(at as usize));
        debug_assert!(
            slot.is_some(),
            "charge record of trade {number}, not in the table"
        );
        if let Some(costs) = slot {
            *costs = Some(costs.unwrap_or(0.0) + paid);
        }
    }
    for (trade, costs) in trades.iter_mut().zip(trade_costs) {
        debug_assert!(
            costs.is_some(),
            "trade {} has no charged fill",
            trade.number
        );
        trade.costs_rupees = round_to_paise(costs.unwrap_or(0.0));
        trade.net_pnl_rupees = round_to_paise(trade.gross_pnl_rupees - trade.costs_rupees);
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
