//! Sessions out of the calendar: regular hours, weekends, dated rules, coverage and
//! the supported date range.

use crate::market::calendar::testing::{calendar, d};
use crate::market::calendar::{hm, CalendarError, Coverage, RuleClose, SessionKind};
use crate::model::{Exchange, MAX_DAY, MIN_DAY};

#[test]
fn a_regular_nse_session_is_pinned() {
    let session = calendar().session(Exchange::Nse, d(2025, 6, 2)).unwrap();
    assert_eq!(session.open, 1_748_835_900_000_000_000); // 2025-06-02 09:15 IST
    assert_eq!(session.close, 1_748_858_400_000_000_000); // 2025-06-02 15:30 IST
    assert_eq!(session.kind, SessionKind::Regular);
    assert_eq!(session.coverage, Coverage::Template);
    // The NSE row before 2026-04-28 is assumed, not read from an NSE document.
    assert!(session.hours_unverified);
}

#[test]
fn derivatives_sessions_close_later_from_each_cited_date() {
    let cases = [
        (d(2026, 7, 31), 1_785_492_000_000_000_000), // 2026-07-31 15:30 IST
        (d(2026, 8, 3), 1_785_751_800_000_000_000),  // 2026-08-03 15:40 IST
        (d(2026, 12, 7), 1_796_638_500_000_000_000), // 2026-12-07 15:45 IST
    ];
    for (day, close) in cases {
        assert_eq!(
            calendar().session(Exchange::Nfo, day).unwrap().close,
            close,
            "day {day}"
        );
    }
}

#[test]
fn mcx_sessions_close_by_united_states_daylight_saving() {
    let cases = [
        (d(2025, 6, 2), 1_748_887_200_000_000_000), // 2025-06-02 23:30 IST
        (d(2025, 12, 1), 1_764_613_500_000_000_000), // 2025-12-01 23:55 IST
    ];
    for (day, close) in cases {
        assert_eq!(
            calendar().session(Exchange::Mcx, day).unwrap().close,
            close,
            "day {day}"
        );
    }
}

#[test]
fn saturday_and_sunday_are_closed() {
    for day in [d(2025, 6, 7), d(2025, 6, 8)] {
        assert!(
            calendar().session(Exchange::Nse, day).is_none(),
            "day {day}"
        );
    }
}

#[test]
fn index_exchanges_share_the_calendar_they_follow() {
    let mut cal = calendar();
    cal.set_closed(Exchange::NseIndex, d(2025, 6, 2), Coverage::Sourced)
        .unwrap();
    assert!(cal.session(Exchange::Nse, d(2025, 6, 2)).is_none());
    assert_eq!(
        cal.session(Exchange::BseIndex, d(2025, 6, 3)),
        cal.session(Exchange::Bse, d(2025, 6, 3))
    );
}

#[test]
fn a_closure_closes_a_weekday_with_its_own_coverage() {
    let mut cal = calendar();
    cal.set_closed(Exchange::Nse, d(2025, 6, 2), Coverage::Sourced)
        .unwrap();
    assert!(cal.session(Exchange::Nse, d(2025, 6, 2)).is_none());
    assert_eq!(
        cal.coverage(Exchange::Nse, d(2025, 6, 2)),
        Coverage::Sourced
    );
}

#[test]
fn a_rule_is_never_stored_as_a_template() {
    let mut cal = calendar();
    cal.set_closed(Exchange::Nse, d(2025, 6, 2), Coverage::Template)
        .unwrap();
    assert_eq!(
        cal.coverage(Exchange::Nse, d(2025, 6, 2)),
        Coverage::Unverified
    );
}

#[test]
fn a_weekend_session_is_special() {
    let mut cal = calendar();
    let day = d(2026, 11, 8); // a Sunday
    cal.set_session(
        Exchange::Nse,
        day,
        hm(14, 20),
        RuleClose::At(hm(15, 30)),
        false,
        Coverage::Sourced,
    )
    .unwrap();
    let session = cal.session(Exchange::Nse, day).unwrap();
    assert_eq!(session.open, 1_794_127_800_000_000_000); // 2026-11-08 14:20 IST
    assert_eq!(session.close, 1_794_132_000_000_000_000); // 2026-11-08 15:30 IST
    assert_eq!(session.kind, SessionKind::Special);
}

#[test]
fn a_rule_with_the_regular_times_on_a_weekday_is_regular() {
    let mut cal = calendar();
    let day = d(2025, 6, 2);
    cal.set_session(
        Exchange::Nse,
        day,
        hm(9, 15),
        RuleClose::At(hm(15, 30)),
        false,
        Coverage::Sourced,
    )
    .unwrap();
    assert_eq!(
        cal.session(Exchange::Nse, day).unwrap().kind,
        SessionKind::Regular
    );
}

#[test]
fn a_rule_can_flag_its_times_as_unverified() {
    let mut cal = calendar();
    let day = d(2026, 1, 16);
    cal.set_session(
        Exchange::Nfo,
        day,
        hm(9, 15),
        RuleClose::At(hm(12, 30)),
        true,
        Coverage::Sourced,
    )
    .unwrap();
    assert!(cal.session(Exchange::Nfo, day).unwrap().hours_unverified);
}

#[test]
fn a_session_must_open_before_it_closes_on_its_date() {
    let cases = [
        (hm(15, 30), RuleClose::At(hm(9, 15))),
        (hm(9, 15), RuleClose::At(hm(24, 0) + 60)),
    ];
    for (open_sec, close) in cases {
        let result = calendar().set_session(
            Exchange::Nse,
            d(2025, 6, 2),
            open_sec,
            close,
            false,
            Coverage::Sourced,
        );
        assert!(
            matches!(result, Err(CalendarError::InvalidSession { .. })),
            "{result:?}"
        );
    }
}

#[test]
fn new_regular_hours_apply_from_their_date() {
    let mut cal = calendar();
    cal.set_regular_hours(Exchange::Cds, Some(d(2025, 6, 3)), hm(9, 0), hm(15, 0))
        .unwrap();
    let before = cal.session(Exchange::Cds, d(2025, 6, 2)).unwrap();
    let after = cal.session(Exchange::Cds, d(2025, 6, 3)).unwrap();
    assert_eq!(before.close, 1_748_863_800_000_000_000); // 2025-06-02 17:00 IST
    assert_eq!(after.close, 1_748_943_000_000_000_000); // 2025-06-03 15:00 IST
    assert!(
        !after.hours_unverified,
        "hours set by the caller count as verified"
    );
}

#[test]
fn new_regular_hours_cannot_close_before_an_evening_rule_opens() {
    let mut cal = calendar();
    let day = d(2025, 6, 2);
    cal.set_session(
        Exchange::Mcx,
        day,
        hm(17, 0),
        RuleClose::Regular,
        false,
        Coverage::Sourced,
    )
    .unwrap();
    let result = cal.set_regular_hours(Exchange::Mcx, None, hm(9, 0), hm(16, 0));
    assert!(
        matches!(result, Err(CalendarError::InvalidSession { .. })),
        "{result:?}"
    );
}

#[test]
fn an_evening_rule_closes_at_the_regular_close_of_its_date() {
    let mut cal = calendar();
    for day in [d(2025, 6, 2), d(2025, 12, 1)] {
        cal.set_session(
            Exchange::Mcx,
            day,
            hm(17, 0),
            RuleClose::Regular,
            false,
            Coverage::Sourced,
        )
        .unwrap();
    }
    let summer = cal.session(Exchange::Mcx, d(2025, 6, 2)).unwrap();
    let winter = cal.session(Exchange::Mcx, d(2025, 12, 1)).unwrap();
    assert_eq!(summer.close, 1_748_887_200_000_000_000); // 2025-06-02 23:30 IST
    assert_eq!(winter.close, 1_764_613_500_000_000_000); // 2025-12-01 23:55 IST
    assert!(
        summer.hours_unverified,
        "the MCX regular close is unverified"
    );
}

#[test]
fn dates_outside_the_supported_range_never_panic() {
    let mut cal = calendar();
    for day in [MIN_DAY - 1, MAX_DAY + 1, i32::MIN, i32::MAX] {
        assert!(cal.session(Exchange::Nse, day).is_none());
        assert!(cal.regular_hours(Exchange::Nse, day).is_none());
        let closed = cal.set_closed(Exchange::Nse, day, Coverage::Sourced);
        assert_eq!(closed, Err(CalendarError::UnsupportedDay(day)));
        let range = cal.sessions(Exchange::Nse, day, day);
        assert_eq!(range, Err(CalendarError::UnsupportedDay(day)));
        let hours = cal.set_regular_hours(Exchange::Nse, Some(day), hm(9, 15), hm(15, 30));
        assert_eq!(hours, Err(CalendarError::UnsupportedDay(day)));
    }
}

#[test]
fn the_ends_of_the_supported_range_have_sessions_when_weekdays() {
    let cal = calendar();
    for day in [MIN_DAY, MAX_DAY] {
        let weekday_or_closed = cal
            .session(Exchange::Nse, day)
            .map_or(true, |s| s.open < s.close);
        assert!(weekday_or_closed, "day {day}");
    }
}

#[test]
fn a_reversed_range_is_an_error() {
    let result = calendar().sessions(Exchange::Nse, d(2025, 6, 3), d(2025, 6, 2));
    assert!(
        matches!(result, Err(CalendarError::InvalidRange { .. })),
        "{result:?}"
    );
}

#[test]
fn sessions_of_a_week_are_its_five_weekdays() {
    let sessions = calendar()
        .sessions(Exchange::Nse, d(2025, 6, 2), d(2025, 6, 8))
        .unwrap();
    let days: Vec<_> = sessions.iter().map(|session| session.day).collect();
    assert_eq!(days, (d(2025, 6, 2)..=d(2025, 6, 6)).collect::<Vec<_>>());
}

#[test]
fn coverage_is_the_rule_level_else_the_year_level() {
    let mut cal = calendar();
    cal.set_year_coverage(Exchange::Nse, 2025, Coverage::Sourced);
    cal.set_closed(Exchange::Nse, d(2025, 6, 2), Coverage::Unverified)
        .unwrap();
    assert_eq!(
        cal.coverage(Exchange::Nse, d(2025, 6, 2)),
        Coverage::Unverified
    );
    assert_eq!(
        cal.coverage(Exchange::Nse, d(2025, 6, 3)),
        Coverage::Sourced
    );
    assert_eq!(
        cal.coverage(Exchange::Nse, d(2024, 6, 3)),
        Coverage::Template
    );
    cal.set_year_coverage(Exchange::Nse, 2025, Coverage::Template);
    assert_eq!(
        cal.coverage(Exchange::Nse, d(2025, 6, 3)),
        Coverage::Template
    );
}

#[test]
fn rules_come_back_in_date_order() {
    let mut cal = calendar();
    cal.set_closed(Exchange::Nse, d(2025, 6, 4), Coverage::Sourced)
        .unwrap();
    cal.set_closed(Exchange::Nse, d(2025, 6, 2), Coverage::Sourced)
        .unwrap();
    cal.set_closed(Exchange::Bse, d(2025, 6, 3), Coverage::Sourced)
        .unwrap();
    let days: Vec<_> = cal
        .rules_in(Exchange::Nse, d(2025, 6, 1), d(2025, 6, 30))
        .iter()
        .map(|r| r.0)
        .collect();
    assert_eq!(days, vec![d(2025, 6, 2), d(2025, 6, 4)]);
    assert!(cal
        .rules_in(Exchange::Nse, d(2025, 6, 30), d(2025, 6, 1))
        .is_empty());
}

#[test]
fn period_bounds_span_the_first_and_last_session() {
    let mut cal = calendar();
    cal.set_closed(Exchange::Nse, d(2025, 6, 2), Coverage::Sourced)
        .unwrap();
    let bounds = cal.period_bounds(Exchange::Nse, d(2025, 6, 2), d(2025, 6, 8));
    let first_open = 1_748_922_300_000_000_000; // 2025-06-03 09:15 IST
    let last_close = 1_749_204_000_000_000_000; // 2025-06-06 15:30 IST
    assert_eq!(bounds, Some((first_open, last_close)));
    assert_eq!(
        cal.period_bounds(Exchange::Nse, d(2025, 6, 7), d(2025, 6, 8)),
        None
    );
}
