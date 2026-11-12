//! The engine's market accounting, called from the one fill path and the session
//! close: the slippage that prices a fill, the margin an opening fill blocks, the
//! charges of every fill, and the contract notes of a day (04-execution 4.3, 5, 6).
//!
//! The rules themselves live below the engine (`market::slippage`, `market::margin`,
//! `charges`); this module connects them to the run's configuration, instruments and
//! ledger, and records which dated table rows the run used.

use crate::charges::{ChargeBook, ChargeError, FillToCharge, MarketTable};
use crate::exec::InstrumentId;
use crate::market::margin::MarginRequest;
use crate::market::slippage::{default_slippage, SlippageClass, SlippageSpec};
use crate::model::{ist_day, Day, Side};
use crate::portfolio::{FillEffect, PositionKey};

use super::clock::SliceInfo;
use super::config::CostConfig;
use super::error::KernelError;
use super::event::EventStamp;
use super::kernel::Engine;

/// A charging error as a kernel error: a cost the model cannot price is a problem of
/// the run's `costs` setting.
pub(super) fn charge_error(error: ChargeError) -> KernelError {
    match error {
        ChargeError::NotBuilt { feature, instead } => KernelError::NotBuilt { feature, instead },
        ChargeError::Cost(inner) => KernelError::InvalidConfig {
            field: "costs",
            reason: inner.to_string(),
        },
    }
}

/// The charge book of a run with this cost setting.
pub(super) fn charge_book(costs: &CostConfig) -> ChargeBook {
    match costs {
        CostConfig::Off => ChargeBook::new(None),
        CostConfig::Model(model) => ChargeBook::new(Some(model.rounding())),
    }
}

impl Engine {
    /// The IST trade date of a stamp: its slice's session date.
    pub(super) fn trade_day(&self, stamp: EventStamp) -> Day {
        self.timeline
            .slice(stamp.slice as usize)
            .map_or_else(|| ist_day(stamp.slice_ts), |slice| slice.day)
    }

    /// The slippage that prices a fill of `instrument` on `day`: the run's own, the
    /// instrument's own, or the row of the default table for its class.
    ///
    /// Errors: `InvalidConfig` for an NFO or BFO derivative with no slippage class;
    /// `Slippage` when no row covers the day.
    pub(super) fn fill_slippage(
        &mut self,
        instrument: InstrumentId,
        day: Day,
    ) -> Result<SlippageSpec, KernelError> {
        if self.config.slippage != SlippageSpec::SegmentDefault {
            return Ok(self.config.slippage);
        }
        let facts = self.config.market.of(instrument.index());
        if let Some(own) = facts.and_then(|f| f.slippage) {
            return Ok(own);
        }
        let spec = self.spec(instrument)?;
        let class = facts
            .and_then(|f| f.slippage_class)
            .or_else(|| SlippageClass::of(spec.exchange, spec.kind))
            .ok_or_else(|| KernelError::InvalidConfig {
                field: "slippage",
                reason: format!(
                    "{}: the default slippage of an NFO or BFO derivative depends on whether its \
                     underlying is an index; give the instrument's underlying or pass slippage=",
                    spec.symbol
                ),
            })?;
        let row = default_slippage(class, day)?;
        self.charges
            .record_use(MarketTable::Slippage, &row.meta, day);
        Ok(row.spec)
    }

    /// Margin an opening fill blocks, rupees: the run's own futures margin for the
    /// instrument (`margin=`), else the dated margin table; `leverage` is what the order
    /// asked for (`None` is 1x).
    ///
    /// Errors: `Margin` (no row, a leverage the row does not allow, short options).
    pub(super) fn opening_margin(
        &mut self,
        day: Day,
        key: PositionKey,
        side: Side,
        units: f64,
        price_rupees: f64,
        leverage: Option<f64>,
    ) -> Result<f64, KernelError> {
        let spec = self.spec(key.instrument)?;
        let facts = self.config.market.of(key.instrument.index());
        let request = MarginRequest {
            exchange: spec.exchange,
            kind: spec.kind,
            product: key.product,
            side,
            quantity_units: units,
            price_rupees,
            multiplier: spec.multiplier,
            lot_size_units: f64::from(spec.lot_size_units),
            leverage: leverage.unwrap_or(1.0),
            day,
            futures_margin: facts.and_then(|f| f.futures_margin),
        };
        let quote = self.config.margin.initial_margin(&request)?;
        if let Some(row) = quote.row {
            self.charges.record_use(MarketTable::Margin, &row.meta, day);
        }
        Ok(quote.rupees)
    }

    /// Charge a fill and post its cost to the ledger; returns the fill's cost in rupees
    /// (0 when the run is cost-free). A CNC sale of a scrip bought the same day (or a
    /// buy back of one sold) also carries the new charges of the earlier fill it turned
    /// into intraday (`charges::same_day`). Call it before the fill is recorded.
    ///
    /// Errors: `InvalidConfig` (field `costs`) when the cost model cannot price the
    /// fill.
    pub(super) fn charge_fill(
        &mut self,
        key: PositionKey,
        side: Side,
        units: f64,
        price_rupees: f64,
        day: Day,
        effect: FillEffect,
    ) -> Result<f64, KernelError> {
        let CostConfig::Model(model) = &self.config.costs else {
            return Ok(0.0);
        };
        let spec =
            self.instruments
                .get(key.instrument.index())
                .ok_or(KernelError::InvalidConfig {
                    field: "symbol",
                    reason: format!("no instrument with index {}", key.instrument.0),
                })?;
        let fill = FillToCharge {
            fill: u32::try_from(self.record.fills.len()).unwrap_or(u32::MAX),
            spec,
            facts: self.config.market.of(key.instrument.index()),
            key,
            side,
            units,
            price_rupees,
            day,
            effect,
        };
        let rupees = self.charges.charge(model, &fill).map_err(charge_error)?;
        self.portfolio.account.post_costs(rupees);
        Ok(rupees)
    }

    /// A fill that is not a trade (a futures position settled at expiry): it costs
    /// nothing, but it ends its round trip in the charge book too, so the charges keep
    /// the trade numbers of the trades table. Returns its cost, 0.
    pub(super) fn uncharged_fill(&mut self, key: PositionKey, effect: FillEffect) -> f64 {
        if matches!(self.config.costs, CostConfig::Model(_)) {
            self.charges.note_uncharged(key, effect);
        }
        0.0
    }

    /// Step 8: settle the day's contract notes, posting each note's rounding
    /// adjustment to the ledger.
    pub(super) fn settle_contract_notes(&mut self, slice: &SliceInfo) {
        for adjustment in self.charges.settle_day(slice.day) {
            self.portfolio.account.post_costs(adjustment);
        }
    }
}
