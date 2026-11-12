//! Testing the levels in force against a base bar (step 3) or a tick.
//!
//! On a bar (03-architecture 4.2 step 3, SPEC-28, SPEC-33):
//! 1. The open comes first. A stop already crossed at the open exits the whole position
//!    at the open. Otherwise, when a target is crossed at the open, the stop is not
//!    tested against the rest of the bar.
//! 2. Otherwise a stop the range reaches exits the whole position at the level, and no
//!    target fills: the stop wins inside one bar.
//! 3. When the stop does not fill, every working target the bar reaches fills, nearest
//!    first, at its level, or at the open when crossed there.
//!
//! On a tick (tick mode, and the server's 35 risk vectors): the ratchet first, then the
//! stop, then the targets, all on the one price.

use crate::exec::fill::FillAt;
use crate::model::Side;

use super::super::error::RmsError;
use super::levels::{positive, stop_reached, target_reached, units};
use super::types::{BarPrices, LevelHit, LevelKind, TargetLevel, TickOutcome};
use super::LegState;

impl LegState {
    /// Test the levels in force against a base bar (step 3). The hits are in the order
    /// the engine fills them; a stop hit is the only hit. Bar prices are validated
    /// when the data is loaded, so nothing here can fail.
    pub fn test_bar(&self, bar: &BarPrices) -> Vec<LevelHit> {
        let side = self.context.side;
        if let Some(hit) = self.stop_hit(bar.open, FillAt::Open) {
            return vec![hit];
        }
        let targets = self.working_targets_nearest_first();
        let crossed_at_open = targets
            .iter()
            .any(|target| target_reached(side, bar.open, target.level_rupees));
        if !crossed_at_open {
            if let Some(hit) = self.stop_hit(bar.adverse(side), FillAt::Level) {
                return vec![hit];
            }
        }
        let best = bar.favourable(side);
        targets
            .into_iter()
            .filter(|target| target_reached(side, best, target.level_rupees))
            .map(|target| {
                if target_reached(side, bar.open, target.level_rupees) {
                    target_hit(target, bar.open, FillAt::Open)
                } else {
                    target_hit(target, target.level_rupees, FillAt::Level)
                }
            })
            .collect()
    }

    /// One price event in tick mode: update the extreme, move the trail and the
    /// breakeven, then test the stop and then the targets on that price (the live
    /// module's order). A level moved by this tick is tested on this tick.
    ///
    /// Errors: `InvalidValue` for a price that is not positive and finite; the leg is
    /// then left exactly as it was. `InvalidRule` for a level that is not a price.
    pub fn on_tick(&mut self, last_price_rupees: f64) -> Result<TickOutcome, RmsError> {
        let price = positive("last price", last_price_rupees)?;
        let ratchet = self.ratchet_at(price)?;
        let hits = match self.stop_hit(price, FillAt::Level) {
            Some(hit) => vec![LevelHit {
                price_rupees: price,
                ..hit
            }],
            None => self
                .working_targets_nearest_first()
                .into_iter()
                .filter(|target| target_reached(self.context.side, price, target.level_rupees))
                .map(|target| target_hit(target, price, FillAt::Level))
                .collect(),
        };
        Ok(TickOutcome { ratchet, hits })
    }

    /// The stop hit when `price` reaches the stop, priced at `price` for the open and
    /// at the level otherwise.
    fn stop_hit(&self, price: f64, at: FillAt) -> Option<LevelHit> {
        let stop = self.stop?;
        if !stop_reached(self.context.side, price, stop.level_rupees) {
            return None;
        }
        let price_rupees = match at {
            FillAt::Open => price,
            _ => stop.level_rupees,
        };
        Some(LevelHit {
            kind: LevelKind::Stop,
            level_rupees: stop.level_rupees,
            price_rupees,
            at,
            quantity_units: self.held_units,
        })
    }

    /// The targets that can still fill, nearest to the price first (lowest level first
    /// for a long), ties by index.
    fn working_targets_nearest_first(&self) -> Vec<TargetLevel> {
        let mut working: Vec<TargetLevel> = self
            .targets
            .iter()
            .copied()
            .filter(TargetLevel::is_working)
            .collect();
        let side = self.context.side;
        working.sort_by_key(|target| {
            let level = units(target.level_rupees);
            let distance = match side {
                Side::Buy => level,
                Side::Sell => -level,
            };
            (distance, target.index)
        });
        working
    }
}

fn target_hit(target: TargetLevel, price_rupees: f64, at: FillAt) -> LevelHit {
    LevelHit {
        kind: LevelKind::Target(target.index),
        level_rupees: target.level_rupees,
        price_rupees,
        at,
        quantity_units: target.quantity_units,
    }
}
