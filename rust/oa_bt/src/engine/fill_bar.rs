//! The checks of a queued or resting order at its fill bar (`fills::fill_order`): what
//! the position makes of it (reduce, reverse, nothing left to reduce), the entry window of
//! an MIS entry, and no delivery short.

use crate::exec::{Order, QuantitySpec, RejectCode};
use crate::model::{Product, Side};
use crate::portfolio::PositionKey;
use crate::sizing::round_down_to_venue;

use super::clock::SliceInfo;
use super::config::QuantityMode;
use super::error::KernelError;
use super::kernel::Engine;

/// What a queued order does at its fill bar.
pub(super) enum AtFill {
    /// Fill this many units; `true` when it opens or adds.
    Fill(f64, bool),
    /// Refused by a check at the fill bar.
    Reject(RejectCode),
    /// Nothing left to reduce: cancelled.
    Cancel,
    /// `reverse=True` beyond the position: close these units, then open these.
    Reverse(f64, f64),
}

impl Engine {
    /// Resolve a queued order against the position and the checks of the fill bar.
    pub(super) fn at_fill(
        &mut self,
        slice: &SliceInfo,
        order: &Order,
        key: PositionKey,
    ) -> Result<AtFill, KernelError> {
        let request = &order.request;
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        if request.reduce_only {
            if !reduces {
                return Ok(AtFill::Cancel);
            }
            let spec = self.spec(key.instrument)?;
            let fractional =
                self.config.quantity_mode == QuantityMode::Fractional && !spec.is_derivative();
            let wanted = match request.quantity {
                QuantitySpec::PositionFraction(fraction) => {
                    round_down_to_venue(held.abs() * fraction, spec.lot_size_units, fractional)
                }
                _ => order.quantity_units,
            };
            let units = wanted.min(held.abs());
            return Ok(if units > 0.0 {
                AtFill::Fill(units, false)
            } else {
                AtFill::Cancel
            });
        }
        if reduces {
            if order.quantity_units > held.abs() {
                return Ok(if request.reverse {
                    AtFill::Reverse(held.abs(), order.quantity_units - held.abs())
                } else {
                    AtFill::Reject(RejectCode::ReverseNotAllowed)
                });
            }
            return Ok(AtFill::Fill(order.quantity_units, false));
        }
        if request.product == Product::Mis {
            if let Err(code) = self.session_times(slice.day)?.entry.check(slice.open) {
                return Ok(AtFill::Reject(code));
            }
        }
        // Delivery cannot be shorted (04-execution 2): a resting sale whose long was
        // closed after it was placed must not open a short.
        if request.product == Product::Cnc && request.side == Side::Sell {
            return Ok(AtFill::Reject(RejectCode::ProductRule));
        }
        Ok(AtFill::Fill(order.quantity_units, true))
    }
}
