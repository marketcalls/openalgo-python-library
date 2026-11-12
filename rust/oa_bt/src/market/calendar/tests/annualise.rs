//! Sessions per year: the count, the scaling, the minimum window and the coverage.

use crate::market::calendar::testing::{calendar, d};
use crate::market::calendar::{hm, CalendarError, Coverage, RuleClose, MIN_ANNUALISATION_DAYS};
use crate::model::Exchange;

#[test]
fn four_weeks_of_weekdays_annualise_to_their_share_of_a_year() {
    let result = calendar()
        .sessions_per_year(&[Exchange::Nse], d(2025, 6, 2), d(2025, 6, 29))
        .unwrap();
    // 20 weekdays in 28 calendar days: 20 * 365.25 / 28.
    assert_eq!((result.sessions, result.calendar_days), (20, 28));
    assert!(
        (result.value - 260.892_857_142_857_1).abs() < 1e-9,
        "{}",
        result.value
    );
}

#[test]
fn the_union_counts_a_date_once_when_any_exchange_trades() {
    let mut cal = calendar();
    let saturday = d(2025, 6, 7);
    cal.set_session(
        Exchange::Mcx,
        saturday,
        hm(9, 0),
        RuleClose::At(hm(17, 0)),
        false,
        Coverage::Sourced,
    )
    .unwrap();
    let exchanges = [Exchange::Nse, Exchange::Mcx];
    let result = cal
        .sessions_per_year(&exchanges, d(2025, 6, 2), d(2025, 6, 29))
        .unwrap();
    assert_eq!(result.sessions, 21);
}

#[test]
fn a_window_shorter_than_four_weeks_is_refused() {
    let result = calendar().sessions_per_year(&[Exchange::Nse], d(2025, 6, 2), d(2025, 6, 28));
    let minimum = MIN_ANNUALISATION_DAYS;
    assert_eq!(
        result,
        Err(CalendarError::WindowTooShort { days: 27, minimum })
    );
}

#[test]
fn no_exchange_is_an_error() {
    let result = calendar().sessions_per_year(&[], d(2025, 1, 1), d(2025, 12, 31));
    assert_eq!(result, Err(CalendarError::NoExchanges));
}

#[test]
fn the_coverage_is_the_weakest_of_the_exchange_years() {
    let mut cal = calendar();
    cal.set_year_coverage(Exchange::Nse, 2025, Coverage::Sourced);
    cal.set_year_coverage(Exchange::Nse, 2026, Coverage::Sourced);
    cal.set_year_coverage(Exchange::Mcx, 2025, Coverage::Unverified);
    let both_years = (d(2025, 1, 1), d(2026, 12, 31));
    let nse = cal
        .sessions_per_year(&[Exchange::Nse], both_years.0, both_years.1)
        .unwrap();
    assert_eq!(
        (nse.coverage, nse.template_only.len()),
        (Coverage::Sourced, 0)
    );
    let union = cal.sessions_per_year(&[Exchange::Nse, Exchange::Mcx], both_years.0, both_years.1);
    let union = union.unwrap();
    assert_eq!(union.coverage, Coverage::Template);
    assert_eq!(union.template_only, vec![(Exchange::Mcx, 2026)]);
}
