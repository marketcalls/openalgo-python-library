//! A CNC or NRML leg's levels against a bar: its targets held by the engine exactly as
//! an MIS leg's are, its stop by its SINGLE GTT (design 20-positional-live P6,
//! 04-execution 4.4 item 2).
//!
//! The leg's own test (`rms::LegState::test_bar_against`) decides the order, with the
//! stop at the trigger of the stop GTT in force: a level crossed at the open first; else
//! the stop when the range reaches it, and then no target; else every target the range
//! reaches, nearest first (03-architecture 4.2 step 3). A stop hit fires the GTT, whose
//! child fills by the GTT rules (`gtt_fire::test_gtt`); a target hit is the engine's exit
//! at the target or at the open, as an MIS leg's (`legs::exit_at_level`), which cancels
//! the GTT first when it closes the position and leaves it to be cut to the units left
//! in step 4 otherwise (`gtt::cancel_gtts_before_exit`).
//!
//! The GTT's trigger, not the leg's stop, is the stop tested: the GTT follows the leg's
//! stop from the next base bar (a trail, a breakeven or an add moves it in step 4) and is
//! on the tick grid even with `round_levels` off, and it is what fires. A leg with a stop
//! whose GTT is not in force in the bar (cancelled before a partial exit at the open, its
//! replacement in force from the next base bar) has no level tested in that bar: its
//! targets tested alone would exit at a target in a bar that also reaches the stop, where
//! the stop comes first. A leg without a stop has no GTT, and its targets are tested in
//! every bar, as an MIS leg's.

use crate::exec::{GttId, GttLegRole};
use crate::portfolio::PositionKey;
use crate::rms::{BarPrices, LevelKind};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::Step;
use super::kernel::Engine;

impl Engine {
    /// The stop GTT in force on `key` in `slice`, with its trigger in rupees: active, and
    /// tested from this slice's open.
    fn stop_gtt_in_force(&self, key: PositionKey, slice: &SliceInfo) -> Option<(GttId, f64)> {
        self.gtts
            .book
            .protecting(key.instrument, key.product)
            .into_iter()
            .filter_map(|id| self.gtts.book.get(id))
            .filter(|gtt| gtt.eligible_from <= slice.open)
            .find_map(|gtt| {
                gtt.trigger(GttLegRole::Stop)
                    .map(|trigger| (gtt.id, trigger))
            })
    }

    /// The CNC or NRML leg `key` against a bar (module docs): its levels tested on
    /// `levels_bar`, a fired stop GTT's child on `child_bar`, every event stamped with
    /// `step`. In step 3 both bars are the slice's; at a futures roll (step 2) the levels
    /// are tested at the near contract's open alone and the child on its whole bar.
    ///
    /// Errors: those of `execute_fill` and of the leg rules.
    pub(super) fn test_positional_leg(
        &mut self,
        slice: &SliceInfo,
        key: PositionKey,
        (levels_bar, child_bar): (&BarPrices, &BarPrices),
        step: Step,
    ) -> Result<(), KernelError> {
        let stop_gtt = self.stop_gtt_in_force(key, slice);
        let hits = match self.legs.iter().find(|(k, _)| *k == key) {
            // A stop not held in this bar: no level is tested (module docs).
            Some((_, leg)) if leg.stop.is_some() && stop_gtt.is_none() => return Ok(()),
            Some((_, leg)) => leg.test_bar_against(levels_bar, stop_gtt.map(|(_, at)| at)),
            None => return Ok(()),
        };
        for hit in hits {
            if self.held_units(key) == 0.0 {
                break;
            }
            match (hit.kind, stop_gtt) {
                (LevelKind::Stop, Some((id, _))) => self.test_gtt(slice, id, child_bar, step)?,
                // A stop hit comes only from the GTT's trigger given above.
                (LevelKind::Stop, None) => {}
                (LevelKind::Target(_), _) => {
                    self.exit_at_level(Self::stamp(slice, step), key, hit)?;
                }
            }
        }
        Ok(())
    }
}
