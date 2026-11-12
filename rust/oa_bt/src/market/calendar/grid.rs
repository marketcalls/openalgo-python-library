//! The union-calendar grid: the base-interval time slices the kernel iterates.
//!
//! Every exchange contributes one slice per base bar of each of its sessions, on the
//! bar lattice (`lattice`): bars of interval N open at the regular open plus `k * N`,
//! the first bar of a session may open before the session open, and the last is cut
//! at the session close. With the daily base an exchange contributes one slice per
//! session, from its session open to its session close.
//!
//! The grid is the sorted union of the bar opens across the requested exchanges. Each
//! slice carries bit masks (bit `i` stands for `exchanges[i]`) saying which exchanges
//! have a bar opening there, which sessions open with that bar and which end with it,
//! and for every exchange the instant at which its bar closes.
//!
//! What a consumer must not assume. Slices are ordered by bar open only. Bars of
//! different exchanges overlap whenever their lattices differ: with the daily base the
//! MCX bar opens at 09:00 and closes at 23:30, while the NSE bar of the same date
//! opens at 09:15 and closes at 15:30, so the NSE slice comes after the MCX slice
//! although NSE closes first. A value decided at one bar's close is therefore not
//! known at every later slice. Order decisions and fills by instant (`bar_close`
//! against `ts`), never by slice position.

use crate::model::{Day, Exchange, TsNanos, NANOS_PER_DAY, NANOS_PER_SEC};

use super::error::CalendarError;
use super::lattice::DayLattice;
use super::query::check_range;
use super::store::Calendar;

/// The base interval of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseInterval {
    /// An intraday interval in nanoseconds (1 minute is 60_000_000_000).
    Intraday(i64),
    /// One bar per session.
    Daily,
}

impl BaseInterval {
    /// Errors: `InvalidInterval` unless an intraday interval is positive, a whole
    /// number of seconds and shorter than a day.
    fn validate(self) -> Result<(), CalendarError> {
        match self {
            BaseInterval::Intraday(nanos)
                if nanos <= 0 || nanos >= NANOS_PER_DAY || nanos % NANOS_PER_SEC != 0 =>
            {
                Err(CalendarError::InvalidInterval { nanos })
            }
            _ => Ok(()),
        }
    }

    /// The (open, close) of every base bar of a date's session, in order.
    fn bars(self, lattice: &DayLattice) -> Vec<(TsNanos, TsNanos)> {
        match (self, lattice.session) {
            (_, None) => Vec::new(),
            (BaseInterval::Daily, Some(session)) => vec![(session.open, session.close)],
            (BaseInterval::Intraday(nanos), Some(_)) => lattice.session_bars(nanos).collect(),
        }
    }

    /// Number of base bars in a date's session.
    fn bar_count(self, lattice: &DayLattice) -> u64 {
        match self {
            BaseInterval::Daily => u64::from(lattice.session.is_some()),
            BaseInterval::Intraday(nanos) => lattice.session_bar_count(nanos),
        }
    }
}

/// Upper bound on the number of slices one call may produce.
pub const MAX_GRID_POINTS: u64 = 50_000_000;

/// The `bar_close` value of an exchange that has no bar at a slice.
pub const NO_BAR: TsNanos = TsNanos::MIN;

/// Sorted union of base-bar opens across exchanges. All columns have one length.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnionGrid {
    /// The exchanges the masks and `bar_close` refer to, in the order given.
    pub exchanges: Vec<Exchange>,
    /// Slice instants (base-bar opens) in UTC nanoseconds, strictly increasing.
    pub ts: Vec<TsNanos>,
    /// IST session date of each slice.
    pub day: Vec<Day>,
    /// Exchanges that have a base bar opening at the slice (bit `i` is `exchanges[i]`).
    pub active: Vec<u16>,
    /// Exchanges whose session opens with this bar (their first slice of the day).
    pub session_open: Vec<u16>,
    /// Exchanges whose session ends with this bar (their last slice of the day).
    pub session_close: Vec<u16>,
    /// `bar_close[i][slice]` is the instant, in UTC nanoseconds, at which the bar of
    /// `exchanges[i]` that opens at the slice closes, or `NO_BAR`.
    pub bar_close: Vec<Vec<TsNanos>>,
}

impl UnionGrid {
    /// Record one bar of the exchange at `index`, adding a slice when the instant is new.
    fn add_bar(&mut self, day: Day, index: usize, bar: &Bar) {
        if self.ts.last() != Some(&bar.open) {
            self.ts.push(bar.open);
            self.day.push(day);
            self.active.push(0);
            self.session_open.push(0);
            self.session_close.push(0);
            for closes in &mut self.bar_close {
                closes.push(NO_BAR);
            }
        }
        let bit = 1u16 << index;
        let flag = |on: bool| if on { bit } else { 0 };
        set_last(&mut self.active, |mask| *mask |= bit);
        set_last(&mut self.session_open, |mask| {
            *mask |= flag(bar.opens_session)
        });
        set_last(&mut self.session_close, |mask| {
            *mask |= flag(bar.closes_session)
        });
        if let Some(closes) = self.bar_close.get_mut(index) {
            set_last(closes, |close| *close = bar.close);
        }
    }
}

fn set_last<T>(column: &mut [T], update: impl FnOnce(&mut T)) {
    if let Some(last) = column.last_mut() {
        update(last);
    }
}

/// One base bar of one exchange, before merging across exchanges.
struct Bar {
    open: TsNanos,
    close: TsNanos,
    opens_session: bool,
    closes_session: bool,
}

/// The exchanges in argument order with duplicates dropped.
fn distinct(exchanges: &[Exchange]) -> Result<Vec<Exchange>, CalendarError> {
    let mut unique: Vec<Exchange> = Vec::with_capacity(exchanges.len());
    for &exchange in exchanges {
        if !unique.contains(&exchange) {
            unique.push(exchange);
        }
    }
    if unique.is_empty() {
        Err(CalendarError::NoExchanges)
    } else {
        Ok(unique)
    }
}

/// Refuse a grid above `MAX_GRID_POINTS` before anything is allocated.
fn check_size(
    calendar: &Calendar,
    exchanges: &[Exchange],
    from: Day,
    to: Day,
    base: BaseInterval,
) -> Result<(), CalendarError> {
    let mut points: u64 = 0;
    for day in from..=to {
        for &exchange in exchanges {
            if let Some(lattice) = calendar.day_lattice(exchange, day) {
                points += base.bar_count(&lattice);
            }
        }
        if points > MAX_GRID_POINTS {
            return Err(CalendarError::GridTooLarge {
                points,
                limit: MAX_GRID_POINTS,
            });
        }
    }
    Ok(())
}

/// The base bars of every exchange on one date, as (exchange index, bar), by open.
fn day_bars(
    calendar: &Calendar,
    exchanges: &[Exchange],
    day: Day,
    base: BaseInterval,
) -> Vec<(usize, Bar)> {
    let mut bars = Vec::new();
    for (index, &exchange) in exchanges.iter().enumerate() {
        let Some(lattice) = calendar.day_lattice(exchange, day) else {
            continue;
        };
        let of_exchange = base.bars(&lattice);
        let last = of_exchange.len().saturating_sub(1);
        for (position, (open, close)) in of_exchange.into_iter().enumerate() {
            let opens_session = position == 0;
            let closes_session = position == last;
            bars.push((
                index,
                Bar {
                    open,
                    close,
                    opens_session,
                    closes_session,
                },
            ));
        }
    }
    bars.sort_by_key(|(index, bar)| (bar.open, *index));
    bars
}

/// Build the union grid for `exchanges` from `from` to `to` (IST dates, inclusive).
///
/// Duplicate exchanges are dropped, keeping the first position, so mask bit `i` and
/// `bar_close[i]` stand for the i-th distinct exchange.
///
/// Errors: `InvalidInterval` for an unusable base, `UnsupportedDay`, `InvalidRange`
/// when `from` is after `to`, `NoExchanges` for an empty list and `GridTooLarge` above
/// `MAX_GRID_POINTS`.
pub fn union_grid(
    calendar: &Calendar,
    exchanges: &[Exchange],
    from: Day,
    to: Day,
    base: BaseInterval,
) -> Result<UnionGrid, CalendarError> {
    base.validate()?;
    check_range(from, to)?;
    let exchanges = distinct(exchanges)?;
    check_size(calendar, &exchanges, from, to, base)?;

    let bar_close = vec![Vec::new(); exchanges.len()];
    let mut grid = UnionGrid {
        exchanges,
        bar_close,
        ..UnionGrid::default()
    };
    for day in from..=to {
        for (index, bar) in day_bars(calendar, &grid.exchanges, day, base) {
            grid.add_bar(day, index, &bar);
        }
    }
    Ok(grid)
}
