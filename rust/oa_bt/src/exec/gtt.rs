//! GTT (good till triggered) orders: the one kind of order that rests across sessions.
//!
//! Every ordinary order is a DAY order. Anything that must survive the session on a
//! CNC or NRML position (its stop, a resting entry) is a GTT with one trigger, a SINGLE
//! GTT (04-execution 4.4). A position's targets are held by the engine on the bars of
//! every session, as an MIS leg's are: no OCO GTT is used, since OpenAlgo's OCO behaves
//! differently from one broker to the next (design 20-positional-live P6). MIS cannot
//! hold a GTT.
//!
//! A GTT has a trigger and a child order. When a base bar reaches the trigger, or opens
//! beyond it, the GTT fires (`GttAction::Fire`) and places its child: a LIMIT beyond the
//! trigger by the segment buffer (`child_limit`). The child fills, or does not fill
//! through its bar and is cancelled (`FillChild`, `CancelChild`); either way the GTT is
//! spent. A modification is a full replacement of the trigger and quantity under the
//! same id (`Modify`).
//!
//! A live session's GTT is held by the broker, so it has the states of a request too
//! (design 20-positional-live 3.1, unit N1): `Requested` until the venue confirms the
//! place (`Active`) or refuses it (`Rejected`), `ModifyPending` and `CancelPending` while
//! a modification or cancellation waits for the venue's answer (the old terms rest until
//! then), and `VenueEnded` for a GTT that ended at the venue without the session asking
//! (expired, rejected after placement, cancelled outside the session). A backtest never
//! takes those states. `GTT_TRANSITIONS` lists every legal move (`gtt_state`).
//!
//! The four decisions on GTTs (all reading A) are the defaults of `GttPolicy`: GTT-1 the
//! child's limit buffer, GTT-2 re-creation after 360 days, GTT-3 at most one
//! modification per base bar and only for a move of at least one tick, GTT-4 the engine
//! is assumed running at every session open. GTT-6 (the child's rounding) is
//! `child_limit`.
//!
//! The book of a run's GTTs is `gtt_book`, their events `gtt_event`; the slice sequence
//! that fires and resolves them is in `engine::gtt_fire`.

use std::fmt;

use crate::model::{Day, InstrumentKind, Product, Side, TsNanos};

use super::error::ExecError;
pub use super::gtt_state::{next_gtt_status, GttAction, GttStatus, GTT_TRANSITIONS};
use super::instrument::InstrumentId;
use super::matching::TriggerDirection;
use super::order::OrderId;
use super::price::limit_beyond;
use super::segment::OPTIONS_STOP_LIMIT_BUFFER_PERCENT;

/// Identifier of a GTT, unique within a run, in creation order. Shown as `g1`, `g2`, ...
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GttId(pub u64);

impl fmt::Display for GttId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}", self.0)
    }
}

/// What a GTT's trigger is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GttLegRole {
    /// The stop of a carried position, placed by the engine.
    Stop,
    /// A strategy's own GTT (`validity="gtt"`): an entry, an add or a resting exit.
    Order,
}

impl GttLegRole {
    /// The name used in events ("stop", "order").
    pub fn code(self) -> &'static str {
        match self {
            GttLegRole::Stop => "stop",
            GttLegRole::Order => "order",
        }
    }
}

/// The trigger of a GTT and the limit price of the child order it places.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GttLeg {
    /// What the leg is for.
    pub role: GttLegRole,
    /// Trigger price in rupees, on the tick grid.
    pub trigger_rupees: f64,
    /// From which side the price reaches the trigger.
    pub direction: TriggerDirection,
    /// Limit price of the child order in rupees, on the tick grid.
    pub child_limit_rupees: f64,
}

/// Why a GTT exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GttPurpose {
    /// Placed by the engine to hold a CNC or NRML position's stop, for all its open units.
    Protect,
    /// Placed by the strategy with `validity="gtt"`; the order in the book it belongs to.
    Order(OrderId),
}

/// The decided GTT rules (GTT-1 to GTT-3), as values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GttPolicy {
    /// Child limit beyond the trigger for cash equity and futures, percent (GTT-1).
    pub buffer_cash_and_futures_percent: f64,
    /// Child limit beyond the trigger for options, percent, the same as the options
    /// stop-limit (GTT-1).
    pub buffer_options_percent: f64,
    /// A GTT older than this many calendar days is re-created at the next session open,
    /// in backtest and live alike (GTT-2).
    pub max_age_days: Day,
    /// A trailing or breakeven move modifies the GTT only when the level moves by at
    /// least this many ticks, at most once per base bar (GTT-3).
    pub min_modify_ticks: u32,
}

/// The child's buffer beyond the trigger for cash equity and futures, percent (GTT-1).
pub const CASH_AND_FUTURES_CHILD_BUFFER_PERCENT: f64 = 1.0;

/// Age in calendar days beyond which a GTT is re-created (GTT-2): under the broker's
/// usual one-year expiry.
pub const GTT_MAX_AGE_DAYS: Day = 360;

impl Default for GttPolicy {
    /// The values decided as GTT-1 to GTT-3.
    fn default() -> GttPolicy {
        GttPolicy {
            buffer_cash_and_futures_percent: CASH_AND_FUTURES_CHILD_BUFFER_PERCENT,
            buffer_options_percent: OPTIONS_STOP_LIMIT_BUFFER_PERCENT,
            max_age_days: GTT_MAX_AGE_DAYS,
            min_modify_ticks: 1,
        }
    }
}

impl GttPolicy {
    /// The child's buffer beyond the trigger for an instrument kind, percent (GTT-1).
    pub fn child_buffer_percent(&self, kind: InstrumentKind) -> f64 {
        match kind {
            InstrumentKind::Option => self.buffer_options_percent,
            _ => self.buffer_cash_and_futures_percent,
        }
    }

    /// True when a GTT created on `created_on` is older than the policy allows on
    /// `today` and must be re-created (GTT-2). Days are IST session dates.
    pub fn is_due_for_renewal(&self, created_on: Day, today: Day) -> bool {
        today - created_on > self.max_age_days
    }
}

/// The limit price of a GTT child order: beyond the trigger by `buffer_percent` in the
/// direction of the child's side (above for a buy, below for a sell), computed from the
/// trigger as rounded to the tick and rounded to the tick away from the trader, so the
/// buffer is never smaller than asked, and never below one tick (`limit_beyond`;
/// decision GTT-6).
///
/// Units: rupees and percent. Errors: `InvalidPrice` for an unusable trigger or a
/// negative buffer, `InvalidTick` for an unusable tick.
pub fn child_limit(
    trigger_rupees: f64,
    child_side: Side,
    buffer_percent: f64,
    tick_rupees: f64,
) -> Result<f64, ExecError> {
    limit_beyond(trigger_rupees, child_side, buffer_percent, tick_rupees)
}

/// A GTT in the book.
#[derive(Clone, Debug, PartialEq)]
pub struct Gtt {
    /// Its identifier.
    pub id: GttId,
    /// The instrument it trades.
    pub instrument: InstrumentId,
    /// The product of the position it protects or trades.
    pub product: Product,
    /// Side of the child order.
    pub side: Side,
    /// Quantity of the child order in units.
    pub quantity_units: f64,
    /// Its trigger and the child's limit.
    pub leg: GttLeg,
    /// Why it exists.
    pub purpose: GttPurpose,
    /// Where it stands.
    pub status: GttStatus,
    /// IST date it was created, for the age rule (GTT-2).
    pub created_on: Day,
    /// It is tested on base bars that open at or after this instant (UTC nanoseconds).
    pub eligible_from: TsNanos,
    /// Grid slice of its last modification, for the once-per-base-bar rule (GTT-3).
    pub last_modified_slice: Option<u32>,
}

impl Gtt {
    /// Its trigger, rupees, when it is for `role`.
    pub fn trigger(&self, role: GttLegRole) -> Option<f64> {
        (self.leg.role == role).then_some(self.leg.trigger_rupees)
    }

    /// True while its trigger rests (`GttStatus::is_active`): a backtest tests it on every
    /// base bar.
    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_child_limit_lies_beyond_the_trigger_by_the_buffer() {
        // A long's protective stop sells: 1 percent below 1460.00 is 1445.40.
        assert_eq!(child_limit(1460.0, Side::Sell, 1.0, 0.05), Ok(1445.4));
        // 1 percent above 101.03 is 102.0403, away from the buyer: 102.05.
        assert_eq!(child_limit(101.03, Side::Buy, 1.0, 0.05), Ok(102.05));
        assert!(child_limit(100.0, Side::Buy, -1.0, 0.05).is_err());
    }

    #[test]
    fn the_child_limit_rounds_away_from_the_trader() {
        // Trace 06b: 1473.90 x 0.99 = 1459.161; down for a sell is 1459.15 (GTT-6).
        assert_eq!(child_limit(1473.9, Side::Sell, 1.0, 0.05), Ok(1459.15));
    }

    #[test]
    fn a_sell_child_at_one_tick_keeps_a_limit_of_one_tick() {
        // 5 percent below 0.05 rounds down to 0.00, which no order can carry.
        assert_eq!(child_limit(0.05, Side::Sell, 5.0, 0.05), Ok(0.05));
        assert_eq!(child_limit(0.01, Side::Sell, 5.0, 0.01), Ok(0.01));
        // Two ticks keep their rounding: 0.095 down to the tick is 0.05.
        assert_eq!(child_limit(0.1, Side::Sell, 5.0, 0.05), Ok(0.05));
    }

    #[test]
    fn the_buffer_is_one_percent_for_cash_and_futures_and_five_for_options() {
        let policy = GttPolicy::default();
        assert_eq!(policy.child_buffer_percent(InstrumentKind::Equity), 1.0);
        assert_eq!(policy.child_buffer_percent(InstrumentKind::Future), 1.0);
        assert_eq!(policy.child_buffer_percent(InstrumentKind::Option), 5.0);
    }

    #[test]
    fn a_gtt_is_renewed_once_it_is_older_than_360_days() {
        let policy = GttPolicy::default();
        assert!(!policy.is_due_for_renewal(100, 460));
        assert!(policy.is_due_for_renewal(100, 461));
        assert_eq!(policy.min_modify_ticks, 1);
    }

    #[test]
    fn gtt_ids_print_as_in_results() {
        assert_eq!(GttId(2).to_string(), "g2");
    }
}
