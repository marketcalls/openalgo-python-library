//! The embedded tables: they parse, they hold the cited rows, and a malformed table is
//! a typed error that names the line.

use crate::market::calendar::tables::{
    builtin, parse_intraday_defaults, parse_regular_hours, us_daylight_saving,
};
use crate::market::calendar::testing::{calendar, d};
use crate::market::calendar::{hm, RegularHours};
use crate::model::Exchange;

const HOURS_HEADER: &str = "exchange,valid_from,valid_to,open,close,close_us_dst,source,status\n";
const INTRADAY_HEADER: &str = "exchange,valid_from,valid_to,entry_until,square_off,\
                               entry_before_close_min,square_off_before_close_min,source,status\n";
const OTHER_EXCHANGES: [&str; 6] = ["BSE", "NFO", "BFO", "CDS", "BCD", "MCX"];

/// A regular-hours table with the given NSE rows and a valid row for every other
/// exchange that has its own calendar.
fn hours_table(nse_rows: &str) -> String {
    let others: String = OTHER_EXCHANGES
        .iter()
        .map(|code| format!("{code},,,09:15,15:30,,test,verified\n"))
        .collect();
    format!("{HOURS_HEADER}{nse_rows}{others}")
}

fn intraday_table(nse_rows: &str) -> String {
    let others: String = OTHER_EXCHANGES
        .iter()
        .map(|code| format!("{code},,,15:00,15:10,30,20,test,verified\n"))
        .collect();
    format!("{INTRADAY_HEADER}{nse_rows}{others}")
}

#[test]
fn the_embedded_tables_parse() {
    assert!(builtin().is_ok(), "{:?}", builtin().err());
}

#[test]
fn derivatives_hours_follow_the_cited_dates() {
    // (date, close, verified): NSE/FAOP/72352 and 73928, then 74467, then 77356.
    let cases = [
        (d(2026, 1, 15), hm(15, 30), false),
        (d(2026, 1, 16), hm(15, 30), true),
        (d(2026, 8, 2), hm(15, 30), true),
        (d(2026, 8, 3), hm(15, 40), true),
        (d(2026, 12, 6), hm(15, 40), true),
        (d(2026, 12, 7), hm(15, 45), true),
    ];
    for (day, close_sec, verified) in cases {
        let expected = RegularHours {
            open_sec: hm(9, 15),
            close_sec,
            unverified: !verified,
        };
        assert_eq!(
            calendar().regular_hours(Exchange::Nfo, day),
            Some(expected),
            "day {day}"
        );
    }
}

#[test]
fn bse_derivatives_hours_rest_on_the_regulator_only() {
    let hours = calendar()
        .regular_hours(Exchange::Bfo, d(2026, 12, 7))
        .unwrap();
    assert_eq!((hours.close_sec, hours.unverified), (hm(15, 45), true));
}

#[test]
fn the_mcx_close_follows_united_states_daylight_saving() {
    let summer = calendar()
        .regular_hours(Exchange::Mcx, d(2025, 6, 2))
        .unwrap();
    let winter = calendar()
        .regular_hours(Exchange::Mcx, d(2025, 12, 1))
        .unwrap();
    assert_eq!(
        (summer.close_sec, winter.close_sec),
        (hm(23, 30), hm(23, 55))
    );
    assert!(summer.unverified && winter.unverified);
}

#[test]
fn united_states_daylight_saving_dates() {
    // Second Sunday of March to the first Sunday of November.
    let cases = [
        (d(2025, 3, 8), false),
        (d(2025, 3, 9), true),
        (d(2025, 11, 1), true),
        (d(2025, 11, 2), false),
        (d(2026, 3, 7), false),
        (d(2026, 3, 8), true),
        (d(2026, 10, 31), true),
        (d(2026, 11, 1), false),
    ];
    for (day, expected) in cases {
        assert_eq!(us_daylight_saving(day), expected, "day {day}");
    }
}

#[test]
fn a_malformed_regular_hours_table_names_the_problem() {
    let cases = [
        ("NSE,,,9:15,15:30,,s,verified\n", "is not a time"),
        (
            "NSE,,,09:15,15:30,,s,maybe\n",
            "must be verified or unverified",
        ),
        (
            "NSE,,,15:30,09:15,,s,verified\n",
            "the open must be before the close",
        ),
        (
            "NSE,,,09:15,15:30,09:00,s,verified\n",
            "the open must be before the close",
        ),
        ("NSE,,2026-02-30,09:15,15:30,,s,verified\n", "is not a date"),
        ("NSE,,,09:15,15:30,,,verified\n", "the source is empty"),
        (
            "XYZ,,,09:15,15:30,,s,verified\n",
            "is not an OpenAlgo exchange code",
        ),
        (
            "NSE,2020-01-01,,09:15,15:30,,s,verified\n",
            "must be contiguous",
        ),
        (
            "NSE,,2025-12-31,09:15,15:30,,s,verified\n",
            "has no open-ended last row",
        ),
        (
            "NSE,2026-01-01,2025-12-31,09:15,15:30,,s,verified\n",
            "valid_from is after valid_to",
        ),
    ];
    for (rows, problem) in cases {
        let error = parse_regular_hours("test.csv", &hours_table(rows)).unwrap_err();
        assert!(
            error.problem.contains(problem),
            "{rows:?}: {}",
            error.problem
        );
    }
}

#[test]
fn a_gap_between_rows_is_an_error() {
    let rows = "NSE,,2025-12-31,09:15,15:30,,s,verified\nNSE,2026-01-02,,09:15,15:30,,s,verified\n";
    let error = parse_regular_hours("test.csv", &hours_table(rows)).unwrap_err();
    assert_eq!(error.line, 3);
    assert!(
        error.problem.contains("must be contiguous"),
        "{}",
        error.problem
    );
}

#[test]
fn a_row_after_an_open_ended_row_is_an_error() {
    let rows = "NSE,,,09:15,15:30,,s,verified\nNSE,2027-01-01,,09:15,15:30,,s,verified\n";
    let error = parse_regular_hours("test.csv", &hours_table(rows)).unwrap_err();
    assert!(
        error.problem.contains("already has an open-ended row"),
        "{}",
        error.problem
    );
}

#[test]
fn an_index_exchange_cannot_have_its_own_rows() {
    let rows = "NSE,,,09:15,15:30,,s,verified\nNSE_INDEX,,,09:15,15:30,,s,verified\n";
    let error = parse_regular_hours("test.csv", &hours_table(rows)).unwrap_err();
    assert!(
        error.problem.contains("follows the NSE calendar"),
        "{}",
        error.problem
    );
}

#[test]
fn a_malformed_intraday_table_names_the_problem() {
    let cases = [
        (
            "NSE,,,15:00,,30,20,s,verified\n",
            "give both entry_until and square_off",
        ),
        (
            "NSE,,,15:10,15:00,30,20,s,verified\n",
            "entries must stop at or before the square-off",
        ),
        (
            "NSE,,,15:00,15:10,20,30,s,verified\n",
            "entries must stop at or before the square-off",
        ),
        (
            "NSE,,,15:00,15:10,thirty,20,s,verified\n",
            "is not a whole number of minutes",
        ),
    ];
    for (rows, problem) in cases {
        let error = parse_intraday_defaults("test.csv", &intraday_table(rows)).unwrap_err();
        assert!(
            error.problem.contains(problem),
            "{rows:?}: {}",
            error.problem
        );
    }
}

#[test]
fn a_table_error_message_says_where_to_fix_it() {
    let error = parse_regular_hours("test.csv", &hours_table("NSE,,,9:15,15:30,,s,verified\n"))
        .unwrap_err();
    let message = error.to_string();
    assert!(message.starts_with("test.csv line 2: "), "{message}");
    assert!(
        message.contains("Fix the row in the table file"),
        "{message}"
    );
}
