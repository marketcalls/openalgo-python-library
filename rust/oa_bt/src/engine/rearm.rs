//! `positional_stops="rearm_daily"` (design 20-positional-live section 6; decisions P2,
//! P5 and 11.4): a CNC or NRML leg's stop held as a DAY stop order, placed again at each
//! session's open, in a backtest and in a live session alike.
//!
//! With the default, `"gtt"`, a CNC or NRML leg's stop is one SINGLE GTT that rests
//! overnight (`gtt`, `venue_gtt_sync`). With `"rearm_daily"` no GTT is ever placed for a
//! leg (a strategy's own `validity="gtt"` orders are unchanged), and its stop is the stop
//! an MIS leg has, by the same code path and fill rules:
//!
//! | When | Backtest | Live session |
//! |---|---|---|
//! | During a session | the leg's stop tested on the bars as an MIS leg's (`legs`: SL-M for cash equity and futures, the stop-limit with its limit 5 percent beyond the trigger for NFO and BFO options); targets, trail and breakeven as for a GTT stop | the leg's broker-held stop order (`venue_legs`, role `leg_stop`), its trigger following a trail or a breakeven |
//! | The close | the stop is a DAY order: it ends with the session, and nothing protects the position until the next open | the close's cancellation of the DAY orders ends it (`venue_clock`), an expected end: the leg is neither protected again nor exited |
//! | The next open | at the instrument's first base bar of the session the stop is in force again, unless the open is at or beyond it: then the leg exits at that open plus the run's slippage, flagged `stop_crossed`, reason `leg_sl` (`rearm_stops_at_open`) | the stop is placed again for the units open at the leg's stop once the session has reached its open, unless the session's last price is then at or beyond it: then the leg is exited at market, flagged `stop_crossed` (`rearm_waits_or_crosses`) |
//!
//! The broker ends a DAY order at its session's close by itself, and may report that
//! before the session runs its closing slice: a rearm_daily leg's stop ended by the venue
//! without a cancellation the kernel sent, once the session stands at its closing slice, is
//! that expiry (`rearm_stop_expired`), not a stop lost, so it is neither placed again (a
//! stop sent after the close would be refused, or held by a broker for the next day) nor is
//! the leg exited.
//!
//! A live session asks for each rearm_daily leg's new day stop as it reaches each
//! trading day's open, the first day's (a leg carried from an earlier day, `carry`) and
//! every later day's of a session of several days (`pre_open::reach_session_open`).
//!
//! No day's stop is placed for a leg whose contract's expiry exit is due (`expiry`: its
//! positional exit has passed): the expiry exit goes first and alone, as the session
//! reaches the open it is due at, and the stop waits for it rather than being placed and
//! cancelled at once. At a session's first open the expiry exit is made before the stops
//! are synced; a session started late (its catch-up past the open, or past the exit day's
//! square-off) makes it at the next open it reaches, and meanwhile neither a tick nor
//! `protect` places the stop (`protect` then returns `Protect::Exiting`).
//!
//! A backtest of it assumes the engine was started before the open on every trading day
//! the position was held; the result says so (`bt.backtest`). A live session started
//! before the open waits for the open to place the stop: `protect` on such a leg then
//! returns `Protect::WaitsForOpen` (`venue_protect`).

use crate::exec::{FillFlags, InstrumentId};
use crate::model::{Day, Product};
use crate::portfolio::PositionKey;
use crate::rms::{ExitReason, PositionalStops};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::Step;
use super::fills::FillRequest;
use super::gtt::held_as_gtt;
use super::kernel::Engine;
use super::venue::SendAt;
use super::venue_legs::StopEnd;
use super::venue_roles::{EngineExit, RiskExitReason};

/// The live exit of a rearm_daily leg whose stop the price had crossed when the stop was
/// to be placed again: the leg's stop level was reached (`leg_sl`), as `venue_gtt_protect`
/// exits a CNC or NRML leg whose stop GTT it cannot place again.
const CROSSED_EXIT: EngineExit = EngineExit::Risk(RiskExitReason::StopCrossed);

/// What a backtest keeps for `rearm_daily`: the session day of each instrument's latest
/// base bar, so the first bar of a session is known in step 2.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct RearmDesk {
    /// Per instrument, in registration order: the IST date of its latest bar seen in
    /// step 2.
    seen_day: Vec<Option<Day>>,
}

/// True when a long (`held` above 0) or a short is at or beyond its stop `stop_rupees` at
/// `price_rupees`: at or below it for a long, at or above it for a short.
pub(super) fn beyond_stop(held: f64, stop_rupees: f64, price_rupees: f64) -> bool {
    if held > 0.0 {
        price_rupees <= stop_rupees
    } else {
        price_rupees >= stop_rupees
    }
}

impl Engine {
    /// True when the stop of a leg in `product` is held as a GTT in this run: CNC and NRML
    /// with the default `positional_stops="gtt"`; never with `"rearm_daily"`, nor for MIS.
    pub(super) fn stop_is_gtt(&self, product: Product) -> bool {
        held_as_gtt(product) && self.config.risk.positional_stops == PositionalStops::Gtt
    }

    /// True when `key`'s leg stop is a DAY stop placed again at each session's open
    /// (module docs): a CNC or NRML leg with `positional_stops="rearm_daily"`.
    pub(super) fn rearms_daily(&self, key: PositionKey) -> bool {
        held_as_gtt(key.product) && self.config.risk.positional_stops == PositionalStops::RearmDaily
    }

    /// Step 2 of a backtest, before the orders queued for this open fill: at
    /// `instrument`'s first base bar of a session, every rearm_daily leg of it whose stop
    /// the bar opens at or beyond exits at `open_rupees` plus the run's slippage, flagged
    /// `stop_crossed`, reason `leg_sl` (module docs). A run without `rearm_daily` does
    /// nothing here.
    ///
    /// Errors: those of `execute_fill`.
    pub(super) fn rearm_stops_at_open(
        &mut self,
        slice: &SliceInfo,
        instrument: InstrumentId,
        open_rupees: f64,
    ) -> Result<(), KernelError> {
        if self.config.risk.positional_stops != PositionalStops::RearmDaily {
            return Ok(());
        }
        let seen = &mut self.rearm.seen_day;
        if seen.len() <= instrument.index() {
            seen.resize(instrument.index() + 1, None);
        }
        let first_bar = seen[instrument.index()] != Some(slice.day);
        seen[instrument.index()] = Some(slice.day);
        if !first_bar {
            return Ok(());
        }
        let crossed: Vec<PositionKey> = self
            .legs
            .iter()
            .filter(|(key, _)| key.instrument == instrument && self.rearms_daily(*key))
            .filter_map(|(key, leg)| {
                let stop = leg.stop?.level_rupees;
                let held = self.held_units(*key);
                (held != 0.0 && beyond_stop(held, stop, open_rupees)).then_some(*key)
            })
            .collect();
        let stamp = Self::stamp(slice, Step::OpenFills);
        for key in crossed {
            let held = self.held_units(key);
            let exit = FillRequest::engine_exit(key, held, open_rupees, ExitReason::LegSl)
                .flagged(FillFlags::STOP_CROSSED);
            self.execute_fill(stamp, exit)?;
        }
        Ok(())
    }

    /// True when `key`'s rearm_daily stop ended at the venue, cancelled with no request of
    /// the kernel, while the session stands at its closing slice (`at`, module docs): the
    /// DAY order's expiry.
    pub(super) fn rearm_stop_expired(&self, at: &SendAt, key: PositionKey, how: StopEnd) -> bool {
        let unasked = how == StopEnd::Canceled { requested: false };
        unasked && at.slice.closes_session && self.rearms_daily(key)
    }

    /// In a live session, before a rearm_daily leg's stop is placed for `key` at
    /// `trigger_rupees`, decided as `at` says (module docs): true while the session has not
    /// reached its open (the stop waits for it), and true after the leg was exited at
    /// market, flagged `stop_crossed`, because the session's last price is at or beyond the
    /// stop; false when the stop is to be placed now. Always false for any other leg.
    ///
    /// Errors: those of the exit owner.
    pub(super) fn rearm_waits_or_crosses(
        &mut self,
        at: SendAt,
        key: PositionKey,
        trigger_rupees: f64,
    ) -> Result<bool, KernelError> {
        if !self.rearms_daily(key) {
            return Ok(false);
        }
        let market_closed = self.day_validity(&at.slice, at.decided_at).is_err();
        if market_closed || self.before_session_open(&at) || self.rearm_leg_expiring(&at, key)? {
            return Ok(true);
        }
        if !self.stop_crossed(key, trigger_rupees) {
            return Ok(false);
        }
        self.request_flagged_exit(at, key, CROSSED_EXIT, FillFlags::STOP_CROSSED)?;
        Ok(true)
    }

    /// True when `key` is a rearm_daily leg whose contract's expiry exit is due in `at`'s
    /// slice: no day's stop is placed for it (module docs).
    ///
    /// Errors: those of `expiry_exit_due`.
    pub(super) fn rearm_leg_expiring(
        &mut self,
        at: &SendAt,
        key: PositionKey,
    ) -> Result<bool, KernelError> {
        Ok(self.rearms_daily(key) && self.expiry_exit_due(&at.slice, key.instrument)?)
    }

    /// As a live session reaches a trading day's open: each open rearm_daily leg whose day
    /// stop no longer works (the close before ended it) asks for a new one, which
    /// `sync_leg_stops` places, or exits the leg instead (`rearm_waits_or_crosses`).
    pub(super) fn rearm_leg_stops_at_open(&mut self) {
        let renewed: Vec<PositionKey> = self
            .legs
            .iter()
            .map(|(key, _)| *key)
            .filter(|key| self.rearms_daily(*key) && self.held_units(*key) != 0.0)
            .filter(|key| self.working_leg_stop(*key).is_none())
            .collect();
        for key in renewed {
            self.renew_leg_stop(key, false);
        }
    }
}
