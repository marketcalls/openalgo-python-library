//! The bar lattice: where a bar sits, where its bucket starts and the bars of a session.

use crate::market::calendar::testing::{calendar, d, ist, MINUTE};
use crate::market::calendar::{hm, Calendar, Coverage, Placement, RuleClose};
use crate::model::Exchange;

/// A calendar with a special NSE session from 14:20 to 15:30 on Sunday 2026-11-08.
fn with_afternoon_session() -> Calendar {
    let mut cal = calendar();
    let day = d(2026, 11, 8);
    cal.set_session(
        Exchange::Nse,
        day,
        hm(14, 20),
        RuleClose::At(hm(15, 30)),
        false,
        Coverage::Sourced,
    )
    .unwrap();
    cal
}

#[test]
fn bars_of_a_regular_day_are_placed_against_the_session() {
    let day = d(2025, 6, 2);
    let lattice = calendar().day_lattice(Exchange::Nse, day).unwrap();
    let cases = [
        ((9, 0), 15, Placement::PreOpen),     // ends 09:15, at the open
        ((9, 14), 1, Placement::PreOpen),     // ends 09:15
        ((9, 10), 15, Placement::OffLattice), // overlaps the session off the lattice
        ((9, 15), 15, Placement::InSession),
        ((9, 20), 15, Placement::OffLattice),
        ((15, 15), 15, Placement::InSession),
        ((15, 29), 1, Placement::InSession),
        ((15, 30), 1, Placement::PostClose),
        ((15, 45), 15, Placement::PostClose),
    ];
    for ((hour, minute), interval_min, expected) in cases {
        let placement = lattice.place(ist(day, hour, minute), interval_min * MINUTE);
        assert_eq!(
            placement, expected,
            "{hour:02}:{minute:02} at {interval_min} minutes"
        );
    }
}

#[test]
fn every_bar_of_a_closed_day_is_on_a_closed_day() {
    let day = d(2025, 6, 7); // a Saturday
    let lattice = calendar().day_lattice(Exchange::Nse, day).unwrap();
    assert_eq!(lattice.place(ist(day, 9, 15), MINUTE), Placement::ClosedDay);
}

#[test]
fn a_bucket_straddling_a_special_session_open_belongs_to_the_session() {
    let day = d(2026, 11, 8);
    let lattice = with_afternoon_session()
        .day_lattice(Exchange::Nse, day)
        .unwrap();
    // The 15-minute lattice runs from 09:15, so 14:15 is a lattice point.
    assert_eq!(
        lattice.place(ist(day, 14, 15), 15 * MINUTE),
        Placement::InSession
    );
    assert_eq!(lattice.place(ist(day, 14, 15), MINUTE), Placement::PreOpen);
    assert_eq!(
        lattice.place(ist(day, 14, 20), 15 * MINUTE),
        Placement::OffLattice
    );
}

#[test]
fn the_bucket_of_an_instant_starts_on_the_lattice() {
    let day = d(2026, 11, 8);
    let lattice = with_afternoon_session()
        .day_lattice(Exchange::Nse, day)
        .unwrap();
    let start = lattice.bucket_start(ist(day, 14, 20), 15 * MINUTE);
    assert_eq!(start, 1_794_127_500_000_000_000); // 2026-11-08 14:15 IST
}

#[test]
fn the_bars_of_a_special_session_start_before_its_open() {
    let day = d(2026, 11, 8);
    let lattice = with_afternoon_session()
        .day_lattice(Exchange::Nse, day)
        .unwrap();
    let bars: Vec<_> = lattice.session_bars(15 * MINUTE).collect();
    assert_eq!(bars.len(), 5);
    assert_eq!(lattice.session_bar_count(15 * MINUTE), 5);
    let first_open = 1_794_127_500_000_000_000; // 2026-11-08 14:15 IST
    let first_close = 1_794_128_400_000_000_000; // 2026-11-08 14:30 IST
    assert_eq!(bars[0], (first_open, first_close));
}

#[test]
fn the_last_bar_of_a_session_is_cut_at_the_close() {
    let day = d(2025, 6, 2);
    let lattice = calendar().day_lattice(Exchange::Nse, day).unwrap();
    let bars: Vec<_> = lattice.session_bars(60 * MINUTE).collect();
    // 09:15, 10:15, ..., 15:15: seven bars, the last one 15 minutes long.
    assert_eq!(bars.len(), 7);
    assert_eq!(lattice.session_bar_count(60 * MINUTE), 7);
    let last_open = 1_748_857_500_000_000_000; // 2025-06-02 15:15 IST
    let last_close = 1_748_858_400_000_000_000; // 2025-06-02 15:30 IST
    assert_eq!(bars[6], (last_open, last_close));
}

#[test]
fn a_closed_day_has_no_bars() {
    let lattice = calendar()
        .day_lattice(Exchange::Nse, d(2025, 6, 7))
        .unwrap();
    assert_eq!(lattice.session_bars(MINUTE).count(), 0);
    assert_eq!(lattice.session_bar_count(MINUTE), 0);
}
