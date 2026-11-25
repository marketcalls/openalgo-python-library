//! A carried position on a contract that expired before the session's day (`carry`;
//! design 20-positional-live section 9, unit N4 item N4-2).
//!
//! A contract the exchange settles in cash and the engine leaves to it (an index future, a
//! cash-settled MCX contract: no positional exit day) was settled overnight, so the session
//! never trades it; its exit is recorded at the contract's last close of the carried day,
//! reason `expiry`, flagged `filled_at_last_close`, as a backtest's settlement with no
//! settlement price records one (`engine::settlement`). The exit belongs to the carried
//! day: it is recorded before the session's boundary, so it ends its run there and does
//! not count toward the new day's loss limit, as a backtest's settlement at expiry ends it
//! on its last day.
//!
//! A contract with a positional exit day (a stock future, a stock option, every option, an
//! MCX delivery contract) is exited by the engine before the exchange settles it
//! (`engine::expiry`). One carried past its last trading day missed that exit, and the
//! exchange has settled it by delivery (or, for an index option, at its settlement price),
//! which the kernel cannot record: the restore is refused, `MissedExpiryExit`, so the live
//! layer can alert rather than record a flat position the account does not hold.

use crate::exec::{FillAt, FillFlags, FillPricing, InstrumentId};
use crate::market::calendar::format_day;
use crate::portfolio::{Position, PositionKey};
use crate::rms::ExitReason;

use super::super::error::KernelError;
use super::super::fills::FillRequest;
use super::super::futures::FillOrigin;
use super::super::kernel::Engine;
use super::error::LiveStateError;
use super::requests::read_instrument;
use super::value::Read;

/// A carried position on a contract that expired before the session's day, and the price
/// in rupees its exit is recorded at (module docs).
pub(super) struct ExpiredExit {
    key: PositionKey,
    price_rupees: f64,
}

/// The instruments of `exits`, in registration order, each once.
pub(super) fn expired_instruments(exits: &[ExpiredExit]) -> Vec<InstrumentId> {
    let mut instruments: Vec<InstrumentId> = exits.iter().map(|exit| exit.key.instrument).collect();
    instruments.sort();
    instruments.dedup();
    instruments
}

/// The carried state's `last_closes` (`carry`): (instrument, close in rupees), empty for a
/// state saved before the part was added.
///
/// Errors: `Malformed` for an entry that cannot be read.
pub(super) fn read_last_closes(
    read: &Read<'_>,
    instruments: usize,
) -> Result<Vec<(InstrumentId, f64)>, LiveStateError> {
    let Some(closes) = read.added("last_closes")? else {
        return Ok(Vec::new());
    };
    closes
        .items()?
        .iter()
        .map(|entry| {
            let instrument = read_instrument(&entry.field("instrument")?, instruments)?;
            let close = entry.f64_of("close_rupees")?;
            if !(close.is_finite() && close > 0.0) {
                return Err(entry.malformed("has a close that is not a positive price"));
            }
            Ok((instrument, close))
        })
        .collect()
}

impl Engine {
    /// The carried open `positions` on contracts whose last trading day is before the
    /// session's day, each with the price its exit is recorded at: its contract's carried
    /// last close in `last_closes`, else the position's mark. `reads` are the state's
    /// `positions` entries in the same order, named in errors. Checked before any of the
    /// state is applied (module docs).
    ///
    /// Errors: `MissedExpiryExit` for a contract with a positional exit day; `Malformed`,
    /// naming the position, for one with neither a last close nor a mark to record it at.
    pub(super) fn read_expired_exits(
        &self,
        positions: &[Position],
        reads: &[Read<'_>],
        last_closes: &[(InstrumentId, f64)],
    ) -> Result<Vec<ExpiredExit>, LiveStateError> {
        let (slice, _) = self.standing();
        let Some(today) = self.day_at(slice) else {
            return Ok(Vec::new());
        };
        let mut expired = Vec::new();
        for (position, read) in positions.iter().zip(reads) {
            let spec = self.instruments.get(position.key.instrument.index());
            let ended = spec.and_then(|spec| Some((spec, spec.last_trading_day?)));
            let Some((spec, last)) = ended.filter(|(_, last)| *last < today) else {
                continue;
            };
            if position.is_flat() {
                continue;
            }
            if spec.positional_exit_day.is_some() {
                return Err(LiveStateError::MissedExpiryExit {
                    symbol: spec.symbol.clone(),
                    last_trading_day: last,
                });
            }
            let carried = last_closes
                .iter()
                .find(|(instrument, _)| *instrument == position.key.instrument)
                .map(|(_, close)| *close);
            let Some(price_rupees) = carried.or(position.mark_rupees) else {
                return Err(read.malformed(format!(
                    "holds {} past its last trading day {} with neither its last close in \
                     last_closes nor a mark, so its expiry has no price to be recorded at",
                    spec.symbol,
                    format_day(last)
                )));
            };
            expired.push(ExpiredExit {
                key: position.key,
                price_rupees,
            });
        }
        Ok(expired)
    }

    /// Record the exit of each of `expired` at its price, reason `expiry`, flagged
    /// `filled_at_last_close`, not charged (module docs).
    ///
    /// Errors: those of `execute_fill`.
    pub(super) fn record_expired_exits(
        &mut self,
        expired: &[ExpiredExit],
    ) -> Result<(), KernelError> {
        if expired.is_empty() {
            return Ok(());
        }
        let stamp = self.control_point()?.stamp();
        for exit in expired {
            let held = self.held_units(exit.key);
            let request =
                FillRequest::engine_exit(exit.key, held, exit.price_rupees, ExitReason::Expiry)
                    .at(FillAt::Close)
                    .flagged(FillFlags::FILLED_AT_LAST_CLOSE);
            let request = FillRequest {
                pricing: FillPricing::Resting,
                origin: FillOrigin::ExpirySettlement,
                ..request
            };
            self.execute_fill(stamp, request)?;
        }
        Ok(())
    }
}
