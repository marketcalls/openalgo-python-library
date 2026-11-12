//! The union grid: slices, masks, bar closes and the limits it enforces.

use crate::market::calendar::testing::{calendar, d, MINUTE};
use crate::market::calendar::{union_grid, BaseInterval, CalendarError, NO_BAR};
use crate::model::Exchange;

const HOUR: BaseInterval = BaseInterval::Intraday(60 * MINUTE);

#[test]
fn an_hourly_grid_merges_two_lattices() {
    let day = d(2025, 6, 2);
    let grid = union_grid(&calendar(), &[Exchange::Cds, Exchange::Nse], day, day, HOUR).unwrap();
    // CDS 09:00 to 17:00 gives 8 bars on the hour; NSE 09:15 to 15:30 gives 7 at a
    // quarter past. None coincide, so the grid has 15 slices.
    assert_eq!(grid.ts.len(), 15);
    assert!(grid.ts.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(grid.ts[0], 1_748_835_000_000_000_000); // 2025-06-02 09:00 IST
    assert_eq!(grid.ts[1], 1_748_835_900_000_000_000); // 2025-06-02 09:15 IST
    assert_eq!((grid.active[0], grid.active[1]), (0b01, 0b10));
    assert_eq!((grid.session_open[0], grid.session_open[1]), (0b01, 0b10));
}

#[test]
fn each_exchange_has_its_own_bar_close_at_each_slice() {
    let day = d(2025, 6, 2);
    let grid = union_grid(&calendar(), &[Exchange::Cds, Exchange::Nse], day, day, HOUR).unwrap();
    // Slice 1 is the NSE 09:15 bar; CDS has no bar opening there.
    assert_eq!(grid.bar_close[0][1], NO_BAR);
    assert_eq!(grid.bar_close[1][1], 1_748_839_500_000_000_000); // 2025-06-02 10:15 IST

    // The last NSE bar (15:15) is cut at the 15:30 close and ends the NSE session.
    let slice = (0..grid.ts.len())
        .rev()
        .find(|&slice| grid.active[slice] & 0b10 != 0)
        .unwrap();
    assert_eq!(grid.ts[slice], 1_748_857_500_000_000_000); // 2025-06-02 15:15 IST
    assert_eq!(grid.bar_close[1][slice], 1_748_858_400_000_000_000); // 2025-06-02 15:30 IST
    assert_eq!(grid.session_close[slice] & 0b10, 0b10);
}

#[test]
fn a_shared_bar_open_is_one_slice() {
    let day = d(2025, 6, 2);
    let grid = union_grid(&calendar(), &[Exchange::Nse, Exchange::Bse], day, day, HOUR).unwrap();
    assert_eq!(grid.ts.len(), 7);
    assert!(grid.active.iter().all(|&mask| mask == 0b11));
}

#[test]
fn a_repeated_exchange_keeps_its_first_position() {
    let day = d(2025, 6, 2);
    let exchanges = [Exchange::Nse, Exchange::Mcx, Exchange::Nse];
    let grid = union_grid(&calendar(), &exchanges, day, day, BaseInterval::Daily).unwrap();
    assert_eq!(grid.exchanges, vec![Exchange::Nse, Exchange::Mcx]);
    assert_eq!(grid.bar_close.len(), 2);
}

#[test]
fn days_without_a_session_add_no_slices() {
    let grid = union_grid(
        &calendar(),
        &[Exchange::Nse],
        d(2025, 6, 7),
        d(2025, 6, 8),
        HOUR,
    )
    .unwrap();
    assert!(grid.ts.is_empty());
}

#[test]
fn unusable_requests_are_errors() {
    let cal = calendar();
    let day = d(2025, 6, 2);
    let bad_interval = [0, -MINUTE, MINUTE + 1, 1_440 * MINUTE];
    for nanos in bad_interval {
        let result = union_grid(
            &cal,
            &[Exchange::Nse],
            day,
            day,
            BaseInterval::Intraday(nanos),
        );
        assert_eq!(result, Err(CalendarError::InvalidInterval { nanos }));
    }
    assert_eq!(
        union_grid(&cal, &[], day, day, HOUR),
        Err(CalendarError::NoExchanges)
    );
    let reversed = union_grid(&cal, &[Exchange::Nse], day, day - 1, HOUR);
    assert!(matches!(reversed, Err(CalendarError::InvalidRange { .. })));
}

#[test]
fn a_grid_above_the_limit_is_refused_before_it_is_built() {
    let exchanges = [Exchange::Mcx, Exchange::Nse, Exchange::Cds];
    let second = BaseInterval::Intraday(1_000_000_000);
    let result = union_grid(
        &calendar(),
        &exchanges,
        d(2020, 1, 1),
        d(2025, 12, 31),
        second,
    );
    assert!(
        matches!(result, Err(CalendarError::GridTooLarge { .. })),
        "{result:?}"
    );
}
