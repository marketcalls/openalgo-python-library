//! A live session's legs across the boundary (`oa_bt::engine::venue_legs`, `venue_ticks`,
//! `venue_protect`): the broker-held stops, tick mode and the session's own calls for a
//! leg (16-session-design 4, requirements R2 to R7).
//!
//! - Leg rules (`stop`, `target`, `targets`, `trail`, `breakeven`, `after_first_target`)
//!   are taken on MIS orders of a session; on CNC and NRML they raise
//!   `NotImplementedError` (those legs are held as GTTs).
//! - A leg's stop is the broker's: once the leg has a stop level (the fill that opens it,
//!   or a trail or breakeven that sets one) a "place" command with the role "leg_stop"
//!   goes out for the units filled, a reduce-only "SL-M" at the leg's level for cash
//!   equity and futures, an "SL" with its limit 5 percent beyond the trigger for options.
//!   Its units are modified with each further fill or add, its trigger as the trail or
//!   the breakeven moves it, at most once every 5 s and only for a move of at least 0.25
//!   percent of the trigger the venue last confirmed; one "modify" of it waits for its
//!   answer at a time. A partial exit ("reduce", a partial target) modifies the stop down
//!   first and is placed once `apply_modify` confirms that; a full exit cancels it first.
//!   Its fill closes the leg (a fill beyond the position raises `VenueRefusalError` of
//!   kind "would_reverse" after recording the units that closed it).
//! - A stop that ends without a fill while its leg is open: `apply_reject` exits the leg
//!   at market (a "risk_exit" with the reason "stop_rejected"); `apply_cancel` of a stop
//!   whose cancellation the session never sent (cancelled by the broker, or not placed)
//!   re-places it once, and a re-placed stop that ends so exits the leg ("stop_rejected"
//!   when rejected, "unprotected" otherwise). A stop that filled all its units while the
//!   leg holds more exits the rest ("stop_unfilled").
//! - `push_tick(instrument, price, at)`: the session's last price, rupees, at UTC
//!   nanoseconds, taken while the session waits. It ratchets the trail and the breakeven,
//!   exits a target the price reaches (a "place" with the role "exit"), marks the position
//!   and tests the daily loss limit and the drawdown halt on that mark. A stop reached is
//!   left to the broker. A price that is not above 0 raises `NativeEngineInputError`; a
//!   call at a bar close, after the session or in a backtest raises `VenueRefusalError`.
//! - `protect(instrument, product)`: the leg's stop placed again at its level, for what
//!   the position's exits leave; returns one of `PROTECT_OUTCOMES`: "placed", and nothing
//!   placed for "stop_works", "exiting" (an exit of the position waits or works),
//!   "no_stop_level", "flat" and "market_closed".
//! - `reprice_exit(order_id, limit_rupees)`: a leg's stop (or the engine's exit) modified
//!   to a "LIMIT" at the price for the order's total units, partly filled or not; True when
//!   the "modify" went now, False when it goes with the answer to the one that waits. The
//!   repriced stop no longer follows its leg. An order that is neither raises
//!   `NativeEngineInputError`, an order that has ended `VenueRefusalError`.
//! - `request_exit(..., limit_rupees=price)`: the session's exit as "LIMIT" orders at the
//!   price, the price rule's marketable fallback.
//!
//! Every price is rupees on the instrument's tick.

use oa_bt::engine::Protect;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Add `PROTECT_OUTCOMES` to the module, from the kernel's own list.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "PROTECT_OUTCOMES",
        PyTuple::new_bound(m.py(), Protect::CODES),
    )
}
