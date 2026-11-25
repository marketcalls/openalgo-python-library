//! A CNC or NRML leg protected again in a live session (`venue_gtts`; design
//! 20-positional-live 3.4, review M11; unit N1 item N1-3): `Engine::protect` for a leg
//! whose stop is a GTT, and the kernel's own protection of a leg whose stop GTT ended
//! without a fill (`venue_gtt_answers`).
//!
//! A leg with a stop level, no stop GTT at the venue and no exit holding the position
//! gets a new SINGLE GTT at the leg's current stop for the units its exits leave
//! (`Protect::Placed`), also while the market is closed: brokers take GTTs outside market
//! hours, so a session started before the open protects a carried leg at once. When the
//! session's last price of the instrument (the position's mark: the latest tick, or the
//! catch-up's last close) is already at or beyond the stop, no GTT is placed (the broker
//! would fire it at once, or refuse it): the leg is exited at market, its order's role
//! `risk_exit` with the reason `stop_crossed`, its fills flagged `stop_crossed`, the
//! trade's exit reason `leg_sl` (`Protect::StopCrossed`). Before the session's open
//! that exit waits for the open (`venue_owner`); at the session close no exit can go, and
//! the outcome is `MarketClosed`. The other outcomes keep their meaning: `StopWorks` while
//! a stop GTT of the leg is at the venue (asked for, resting, or with a request waiting),
//! `Exiting` while an exit waits for or holds the position (a fired GTT's child among
//! them), `NoStopLevel`, `Flat`, and `WaitsForCancels` when the GTT goes out once the
//! strategy's resting orders sent only to close the position, cancelled for its room,
//! have ended (`venue_stop_terms`).

use crate::exec::FillFlags;
use crate::portfolio::PositionKey;

use super::error::KernelError;
use super::kernel::Engine;
use super::venue::SendAt;
use super::venue_protect::Protect;
use super::venue_roles::{EngineExit, RiskExitReason};

/// The exit of a leg whose stop the price has crossed with no stop at the broker: the
/// leg's stop level was reached (`leg_sl`).
const CROSSED_EXIT: EngineExit = EngineExit::Risk(RiskExitReason::StopCrossed);

impl Engine {
    /// Protect `key`'s leg, whose stop is a GTT, again (module docs), decided as `at`
    /// says.
    ///
    /// Errors: those of the order book, of the GTT book and of the stop's terms.
    pub(super) fn protect_gtt_leg(
        &mut self,
        at: SendAt,
        key: PositionKey,
    ) -> Result<Protect, KernelError> {
        if self.held_units(key) == 0.0 {
            return Ok(Protect::Flat);
        }
        if self.contract_has_ended(key.instrument) {
            return Ok(Protect::MarketClosed);
        }
        let Some(trigger) = self.stop_trigger(key)? else {
            return Ok(Protect::NoStopLevel);
        };
        let cover = self.stop_cover_units(key)?;
        if self.has_waiting_exit(key) || self.has_engine_exit(key) || cover <= 0.0 {
            return Ok(Protect::Exiting);
        }
        if self.stop_gtt_at_venue(key).is_some() {
            return Ok(Protect::StopWorks);
        }
        if self.stop_crossed(key, trigger) {
            if self.day_validity(&at.slice, at.decided_at).is_err() {
                return Ok(Protect::MarketClosed);
            }
            self.request_flagged_exit(at, key, CROSSED_EXIT, FillFlags::STOP_CROSSED)?;
            return Ok(Protect::StopCrossed);
        }
        if self.leg_gtt_of(key).is_none() {
            self.open_leg_gtt(key);
        }
        if let Some(leg) = self.leg_gtt_mut(key) {
            leg.gtt = None;
        }
        self.sync_stop_gtt(at, key)?;
        Ok(match self.stop_gtt_at_venue(key) {
            Some(_) => Protect::Placed,
            None => Protect::WaitsForCancels,
        })
    }

    /// True when the session's last price of `key`'s instrument is at or beyond the stop
    /// `trigger_rupees`: at or below it for a long, at or above it for a short.
    pub(super) fn stop_crossed(&self, key: PositionKey, trigger_rupees: f64) -> bool {
        let held = self.held_units(key);
        let mark = self
            .portfolio
            .position(key)
            .and_then(|position| position.mark_rupees)
            .or_else(|| self.last_close(key.instrument));
        mark.is_some_and(|price| {
            if held > 0.0 {
                price <= trigger_rupees
            } else {
                price >= trigger_rupees
            }
        })
    }

    /// True while the session has not reached the open of `at`'s slice, the first of its
    /// session: a `stop_crossed` exit waits for it (module docs).
    pub(super) fn before_session_open(&self, at: &SendAt) -> bool {
        at.slice.opens_session && self.venue.clock_ns < at.slice.open
    }
}
