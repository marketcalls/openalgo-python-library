//! A carried position on a contract that expired before the session's day (`carry`;
//! design 20-positional-live section 9, unit N4 item N4-2): the exchange settled it
//! overnight, so the session never trades it; its exit is recorded at the contract's last
//! close of the carried day, reason `expiry`, flagged `filled_at_last_close`, as a
//! backtest's settlement with no settlement price records one (`engine::settlement`).

use crate::exec::{FillAt, FillFlags, FillPricing, InstrumentId};
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;

use super::super::error::KernelError;
use super::super::fills::FillRequest;
use super::super::futures::FillOrigin;
use super::super::kernel::Engine;
use super::error::LiveStateError;
use super::requests::read_instrument;
use super::value::Read;

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
    /// The instruments of the carried open positions whose last trading day is before the
    /// session's day, in registration order.
    pub(super) fn expired_instruments(&self) -> Vec<InstrumentId> {
        let (slice, _) = self.standing();
        let Some(today) = self.day_at(slice) else {
            return Vec::new();
        };
        let mut expired: Vec<InstrumentId> = self
            .portfolio
            .positions()
            .iter()
            .filter(|position| !position.is_flat())
            .map(|position| position.key.instrument)
            .filter(|instrument| {
                let spec = self.instruments.get(instrument.index());
                spec.and_then(|spec| spec.last_trading_day)
                    .is_some_and(|last| last < today)
            })
            .collect();
        expired.sort();
        expired.dedup();
        expired
    }

    /// Record the exit of every open position on the `expired` instruments at its
    /// contract's last close in `last_closes` (the position's mark when it has none),
    /// reason `expiry`, flagged `filled_at_last_close`, not charged (module docs).
    ///
    /// Errors: those of `execute_fill`.
    pub(super) fn record_expired_exits(
        &mut self,
        expired: &[InstrumentId],
        last_closes: &[(InstrumentId, f64)],
    ) -> Result<(), KernelError> {
        if expired.is_empty() {
            return Ok(());
        }
        let stamp = self.control_point()?.stamp();
        let open: Vec<(PositionKey, f64, Option<f64>)> = self
            .portfolio
            .positions()
            .iter()
            .filter(|position| !position.is_flat() && expired.contains(&position.key.instrument))
            .map(|position| (position.key, position.quantity_units, position.mark_rupees))
            .collect();
        for (key, held, mark) in open {
            let carried = last_closes
                .iter()
                .find(|(instrument, _)| *instrument == key.instrument)
                .map(|(_, close)| *close);
            let Some(price) = carried.or(mark) else {
                continue;
            };
            let exit = FillRequest::engine_exit(key, held, price, ExitReason::Expiry)
                .at(FillAt::Close)
                .flagged(FillFlags::FILLED_AT_LAST_CLOSE);
            let request = FillRequest {
                pricing: FillPricing::Resting,
                origin: FillOrigin::ExpirySettlement,
                ..exit
            };
            self.execute_fill(stamp, request)?;
        }
        Ok(())
    }
}
