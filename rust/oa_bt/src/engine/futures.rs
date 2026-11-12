//! Futures series (04-execution 7, 14-strategy-api 2.1, decisions SPEC-19, SPEC-20,
//! SPEC-30 and the lead rulings LEAD-04, LEAD-05, LEAD-16).
//!
//! A futures strategy names an underlying and a contract rank, never a contract. The
//! Python package resolves the contracts of that rank over the run from the reference
//! tables, registers them as instruments and hands the kernel one `FuturesSeries` per
//! underlying: the contracts in the order they are traded, and for each the IST session
//! on which the series hands over to the next contract (from `roll_session_day` for an
//! auto-roll, from `session_after` for `roll=None`).
//!
//! The kernel turns each hand-over into a `Switch` before the run, from the bars:
//!
//! - Auto-roll (`RollRule::Roll`): the first slice from the hand-over session up to the
//!   near contract's last trading day in which both contracts have a bar (the roll
//!   session's first base-bar open when nothing is missing). The spread is the next
//!   contract's raw open minus the near contract's raw open there, before slippage.
//!   Open positions roll in step 2 of that slice (`roll`).
//! - `roll=None` (`RollRule::SettleAtExpiry`): the first slice of the hand-over session
//!   (the session after the near contract's last trading day). Positions are settled at
//!   expiry in step 8 of the last trading day (`settlement`). The spread that adjusts the
//!   series is taken from the two contracts' closes at the near contract's last slice
//!   with both bars on its last trading day (the opens of a later session would include
//!   the overnight move of the next contract).
//!
//! The switches are the one source of the series' contract at any instant: from a
//! switch's slice on, entries go to the next contract and the spliced series of the
//! Python package (`openalgo/backtest/_spliced.py`) shows its bars, back-adjusted by the
//! spreads known at each decision point. Registration and the planning of the switches
//! are in `futures_plan`, the roll in `roll`, settlement in `settlement`.

use crate::exec::InstrumentId;
use crate::market::calendar::{format_day, Calendar};
use crate::model::{Day, Exchange, TsNanos};
use crate::portfolio::PositionKey;

use super::error::KernelError;

/// Longest run of closed days walked over when counting sessions; a longer run means
/// the calendar has no sessions there (an unsupported range, not a holiday).
const MAX_CLOSED_RUN_DAYS: u32 = 15;

/// What happens to an open position when its contract hands over to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RollRule {
    /// Roll at the hand-over session (`roll=N`, N sessions before expiry).
    Roll,
    /// Hold the position to its contract's expiry and settle it there (`roll=None`).
    SettleAtExpiry,
}

/// One contract of a series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeriesContract {
    /// The registered instrument (a future with a last trading day).
    pub instrument: InstrumentId,
    /// IST date of the session from which the next contract of the series takes over;
    /// `None` for the last contract of the run.
    pub hand_over: Option<Day>,
}

/// The contracts one futures strategy trades for one underlying, in trading order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuturesSeries {
    pub contracts: Vec<SeriesContract>,
    pub rule: RollRule,
}

/// A move of a series from one contract to the next, as the run's bars fix it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Switch {
    /// Index of the series, in registration order.
    pub series: u32,
    /// The contract handed over.
    pub from: InstrumentId,
    /// The contract that takes over.
    pub to: InstrumentId,
    /// Index of the slice from which `to` is the series' contract.
    pub slice: usize,
    /// Open of that slice, UTC nanoseconds.
    pub slice_ts: TsNanos,
    /// IST session date of that slice.
    pub day: Day,
    /// Rupees added to the series' history before the switch (back-adjustment): next
    /// minus near, opens at a roll, closes at a settlement hand-over (module docs).
    pub spread_rupees: f64,
    /// True when open positions roll at this slice (`RollRule::Roll`).
    pub rolls_positions: bool,
}

/// Why a fill happened beyond its order: what sets its reason and its charges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum FillOrigin {
    /// A trade at the market: an order, a level, an engine exit.
    #[default]
    Trade,
    /// One of the two fills of a futures roll: reason `roll`, charged as any trade.
    Roll,
    /// A futures position settled at its contract's expiry: not a trade, so no
    /// charges and no slippage.
    ExpirySettlement,
}

/// Which posting a settlement record is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SettlementKind {
    /// The daily mark to market of a futures position (step 8 of every session).
    Daily = 0,
    /// The final settlement of a contract on its last trading day.
    Expiry = 1,
}

impl SettlementKind {
    /// The name used in results.
    pub fn code(self) -> &'static str {
        match self {
            SettlementKind::Daily => "daily",
            SettlementKind::Expiry => "expiry",
        }
    }
}

/// One mark-to-market cash flow of a futures position at a session close.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SettlementRecord {
    /// IST session date.
    pub day: Day,
    /// Open of the session's last slice, UTC nanoseconds (the slice of step 8).
    pub slice_ts: TsNanos,
    pub instrument: InstrumentId,
    pub product: crate::model::Product,
    /// Signed units held after the session close (0 once closed or settled).
    pub quantity_units: f64,
    /// The settlement price, rupees.
    pub price_rupees: f64,
    /// True when the data gave no settlement price and the last close was used.
    pub last_close_used: bool,
    /// The cash flow, rupees: the position's P&L at this price (realized plus open)
    /// less what earlier postings settled.
    pub amount_rupees: f64,
    pub kind: SettlementKind,
}

/// What the futures part of a run produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FuturesOutput {
    /// Every switch of every series, in slice order.
    pub switches: Vec<Switch>,
    /// Every settlement posting, in order.
    pub settlements: Vec<SettlementRecord>,
}

/// The futures state of a run.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct FuturesDesk {
    pub(super) series: Vec<FuturesSeries>,
    pub(super) switches: Vec<Switch>,
    /// Settlement prices by instrument, (day, rupees) in day order.
    pub(super) settlement_prices: Vec<(InstrumentId, Vec<(Day, f64)>)>,
    /// The P&L each futures position has had settled so far, rupees.
    pub(super) settled: Vec<(PositionKey, f64)>,
    pub(super) records: Vec<SettlementRecord>,
}

impl FuturesDesk {
    /// The data's settlement price of an instrument on a day, if it gave one.
    pub(super) fn settlement_price(&self, instrument: InstrumentId, day: Day) -> Option<f64> {
        let (_, prices) = self
            .settlement_prices
            .iter()
            .find(|(i, _)| *i == instrument)?;
        let row = prices.binary_search_by_key(&day, |(d, _)| *d).ok()?;
        prices.get(row).map(|(_, price)| *price)
    }

    /// The output of the run.
    pub(super) fn into_output(self) -> FuturesOutput {
        FuturesOutput {
            switches: self.switches,
            settlements: self.records,
        }
    }
}

pub(super) fn invalid(reason: String) -> KernelError {
    KernelError::InvalidConfig {
        field: "roll",
        reason,
    }
}

/// The IST date of the session `sessions_before` sessions before `last_trading_day` on
/// `exchange`: the roll session of an auto-roll (`roll=1`, the default, is the session
/// before the last trading day; `roll=0` is the last trading day itself).
///
/// Errors: `InvalidConfig` (field `roll`) when the last trading day is not a session
/// of the calendar, or when the calendar has no session that far back.
pub fn roll_session_day(
    calendar: &Calendar,
    exchange: Exchange,
    last_trading_day: Day,
    sessions_before: u32,
) -> Result<Day, KernelError> {
    if !calendar.is_open(exchange, last_trading_day) {
        return Err(invalid(format!(
            "the last trading day {} is not a session of {} in the calendar",
            format_day(last_trading_day),
            exchange.code()
        )));
    }
    let mut day = last_trading_day;
    for _ in 0..sessions_before {
        day = step_to_session(calendar, exchange, day, -1)?;
    }
    Ok(day)
}

/// The IST date of the first session of `exchange` after `day`: where a series held to
/// expiry (`roll=None`) moves to its next contract.
///
/// Errors: `InvalidConfig` (field `roll`) when the calendar has no session soon after.
pub fn session_after(
    calendar: &Calendar,
    exchange: Exchange,
    day: Day,
) -> Result<Day, KernelError> {
    step_to_session(calendar, exchange, day, 1)
}

fn step_to_session(
    calendar: &Calendar,
    exchange: Exchange,
    from: Day,
    direction: Day,
) -> Result<Day, KernelError> {
    let mut day = from;
    for _ in 0..=MAX_CLOSED_RUN_DAYS {
        day += direction;
        if calendar.is_open(exchange, day) {
            return Ok(day);
        }
    }
    Err(invalid(format!(
        "the calendar of {} has no session within {MAX_CLOSED_RUN_DAYS} days of {}",
        exchange.code(),
        format_day(from)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::calendar::testing::{calendar, d};
    use crate::market::calendar::Coverage;

    /// The embedded calendar with Christmas 2024 closed, as NSE's holiday list has it.
    fn with_christmas_2024() -> Calendar {
        let mut calendar = calendar();
        let closed = calendar.set_closed(Exchange::Nfo, d(2024, 12, 25), Coverage::Sourced);
        assert_eq!(closed, Ok(()));
        calendar
    }

    #[test]
    fn the_default_roll_session_is_the_session_before_the_last_trading_day() {
        // NIFTY's September 2024 contract stops trading on Thursday 2024-09-26.
        let day = roll_session_day(&calendar(), Exchange::Nfo, d(2024, 9, 26), 1);
        assert_eq!(day, Ok(d(2024, 9, 25)));
    }

    #[test]
    fn the_roll_session_counts_sessions_not_calendar_days() {
        // 2024-12-25 is a holiday, so one session before Thursday 2024-12-26 is the
        // 24th, and two are the 23rd (Monday).
        let calendar = with_christmas_2024();
        let roll = |n| roll_session_day(&calendar, Exchange::Nfo, d(2024, 12, 26), n);
        assert_eq!(roll(1), Ok(d(2024, 12, 24)));
        assert_eq!(roll(2), Ok(d(2024, 12, 23)));
    }

    #[test]
    fn roll_zero_rolls_on_the_last_trading_day() {
        let day = roll_session_day(&calendar(), Exchange::Nfo, d(2024, 12, 26), 0);
        assert_eq!(day, Ok(d(2024, 12, 26)));
    }

    #[test]
    fn a_last_trading_day_that_is_not_a_session_is_refused() {
        let day = roll_session_day(&with_christmas_2024(), Exchange::Nfo, d(2024, 12, 25), 1);
        assert!(matches!(
            day,
            Err(KernelError::InvalidConfig { field: "roll", .. })
        ));
    }

    #[test]
    fn a_series_held_to_expiry_moves_on_at_the_next_session() {
        // After Friday 2024-09-27 comes Monday 2024-09-30.
        let day = session_after(&calendar(), Exchange::Nfo, d(2024, 9, 27));
        assert_eq!(day, Ok(d(2024, 9, 30)));
    }
}
