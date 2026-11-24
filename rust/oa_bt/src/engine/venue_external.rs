//! A fill in a live session with no order of the session's (16-session-design R15,
//! section 5.3): the broker closed or reduced a position the session holds by itself,
//! with its own square-off or a position closed by hand, and the reconciliation adopts
//! it once the books explain it.
//!
//! It takes the one fill path (`fills::execute_fill`) at the broker's price exactly, as a
//! venue's fill of an order does (`venue_fills`): the position and its leg, the ledger,
//! the cost model's charges, the trade book (the trade it closes records `reason`) and
//! the position event, with no order and flagged `external` (`FillFlags::EXTERNAL`). It
//! is recorded in the slice in progress, at the open when it is stamped at the slice's
//! open instant and at a level otherwise, in step 3. Then, as after any venue result, the
//! orders working to close the position are fitted to what is left (`venue_cover`: a leg's
//! stop is cut down, or cancelled once the position is flat; a CNC or NRML leg's stop GTT
//! likewise, `venue_gtt_sync`) and the waiting exits move on.
//!
//! Such a fill only closes or reduces: one on the position's own side, of a flat position,
//! or of more units than the position has (through zero) is refused whole and records
//! nothing (`VenueError::BeyondPosition`): the session never takes on quantity it did not
//! open (08-live-trading section 7).

use crate::exec::{FillAt, FillFlags, FillPricing, InstrumentId};
use crate::model::{Product, Side, TsNanos};
use crate::portfolio::PositionKey;
use crate::rms::{ExitReason, LegRules};

use super::error::KernelError;
use super::event::{EventStamp, Step};
use super::fills::FillRequest;
use super::futures::FillOrigin;
use super::kernel::Engine;
use super::venue_error::VenueError;
use super::venue_fills::check_fill;

impl Engine {
    /// Record the broker's fill of `units` on `side` of the position of `instrument` in
    /// `product`, made with no order of the session at `price_rupees` at `at` (UTC
    /// nanoseconds), closing or reducing the position for `reason` (module docs). Returns
    /// the stamp it was recorded with.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `Venue(AtBarClose)` while paused at a bar
    /// close and `Venue(SessionOver)` after the session; `Venue(InvalidFill)` for units or a
    /// price that is not finite or not above 0; `InvalidConfig` for an unknown instrument;
    /// `Venue(BeyondPosition)` for a fill that would not only close or reduce the position;
    /// and those of the cost model, before anything is recorded.
    // The call takes the facts of the broker's fill as 16-session-design R15 names them.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_external_fill(
        &mut self,
        instrument: InstrumentId,
        product: Product,
        side: Side,
        units: f64,
        price_rupees: f64,
        at: TsNanos,
        reason: ExitReason,
    ) -> Result<EventStamp, KernelError> {
        let slice = self.result_slice()?;
        check_fill(units, price_rupees)?;
        self.spec(instrument)?;
        let key = PositionKey {
            instrument,
            product,
        };
        let held = self.held_units(key);
        let closes = held != 0.0 && (held > 0.0) != (side == Side::Buy);
        if !closes || units > held.abs() {
            let refused = VenueError::BeyondPosition {
                key,
                side,
                units,
                held_units: held,
            };
            return Err(refused.into());
        }
        self.check_chargeable(key, side, units, price_rupees, slice.day)?;
        self.venue.clock_ns = self.venue.clock_ns.max(at);
        let stamp = Self::stamp(&slice, Step::Levels);
        let fill = FillRequest {
            key,
            side,
            units,
            reference_rupees: price_rupees,
            at: if at == slice.open {
                FillAt::Open
            } else {
                FillAt::Level
            },
            order: None,
            exit_reason: reason,
            flags: FillFlags::EXTERNAL,
            margin_rupees: 0.0,
            leg_rules: LegRules::default(),
            target_index: None,
            pricing: FillPricing::Resting,
            gtt: None,
            origin: FillOrigin::Trade,
        };
        self.execute_fill(stamp, fill)?;
        self.after_result(&slice)?;
        Ok(stamp)
    }
}
