//! The schedule data file: the embedded file parses, and a malformed one is a typed
//! error naming the line.

use crate::model::days_from_civil;

use super::super::builtin::parse_schedules;
use super::super::{builtin_schedules, CostError, Segment, TaxBase, TaxUnit};

const HEADER: &str = "schedule_id,valid_from,valid_to,segment,verified_through,stt_buy_rate,\
                      stt_sell_rate,exchange_txn_rate,ipft_rate,sebi_rate,stamp_buy_rate,\
                      stt_exercise_rate,gst_rate,gst_on_sebi_fee,calculator_tax_base,\
                      calculator_tax_unit,source,notes";
const VERSION: &str =
    "T1,2025-01-01,,*,2025-06-30,,,,,,,,0.18,yes,,,\"a source, with a comma\",note";
const ROW: &str = "T1,,,NSE_EQ_INTRADAY,2025-06-30,0,0.00025,0.0000297,0.000001,0.000001,\
                   0.00003,0,,,sides,rupee,,";

fn file(lines: &[&str]) -> String {
    let mut text = String::from(HEADER);
    for line in lines {
        text.push('\n');
        text.push_str(line);
    }
    text
}

/// The line number and reason of the error a malformed file gives.
fn failure(lines: &[&str]) -> (usize, String) {
    match parse_schedules(&file(lines)) {
        Err(CostError::InvalidScheduleData { line, reason }) => (line, reason),
        other => panic!("expected InvalidScheduleData, got {other:?}"),
    }
}

#[test]
fn the_embedded_data_file_parses() {
    assert_eq!(builtin_schedules().map(<[_]>::len), Ok(13));
}

#[test]
fn a_well_formed_file_becomes_schedule_versions() {
    let versions = parse_schedules(&file(&[VERSION, ROW])).unwrap();
    let version = &versions[0];
    assert_eq!(version.id, "T1");
    assert_eq!(version.valid_from, days_from_civil(2025, 1, 1));
    assert_eq!(version.valid_to, None);
    assert_eq!(version.gst_rate, 0.18);
    assert_eq!(
        version.gst_verified_through,
        Some(days_from_civil(2025, 6, 30))
    );
    assert!(version.gst_on_sebi_fee);
    assert_eq!(version.source, "a source, with a comma");
    let row = version.row(Segment::NseEqIntraday).unwrap();
    assert_eq!(row.rates.stt_sell_rate, 0.00025);
    assert_eq!(row.rates.calculator_tax_base, TaxBase::Sides);
    assert_eq!(row.rates.calculator_tax_unit, TaxUnit::Rupee);
    assert_eq!(row.verified_through, Some(days_from_civil(2025, 6, 30)));
}

#[test]
fn a_version_says_whether_gst_falls_on_the_sebi_fee() {
    let exempt = VERSION.replace("0.18,yes", "0.18,no");
    let versions = parse_schedules(&file(&[&exempt, ROW])).unwrap();
    assert!(!versions[0].gst_on_sebi_fee);
}

#[test]
fn windows_line_endings_and_blank_lines_are_accepted() {
    let text = file(&[VERSION, "", ROW]).replace('\n', "\r\n");
    assert_eq!(parse_schedules(&text).map(|versions| versions.len()), Ok(1));
}

#[test]
fn a_file_without_the_header_is_rejected_at_line_1() {
    for text in ["", "schedule_id,valid_from", VERSION] {
        let error = parse_schedules(text).unwrap_err();
        assert!(
            matches!(error, CostError::InvalidScheduleData { line: 1, .. }),
            "{error:?}"
        );
    }
}

#[test]
fn a_file_with_no_versions_is_rejected() {
    assert_eq!(parse_schedules(HEADER), Err(CostError::EmptyScheduleBook));
}

/// Assert that each case fails at the given line with a reason holding the text.
fn assert_failures(cases: &[(&str, Vec<&str>, usize, &str)]) {
    for (problem, lines, line, text) in cases {
        let (at, reason) = failure(lines);
        assert_eq!(at, *line, "{problem}: {reason}");
        assert!(reason.contains(text), "{problem}: {reason}");
    }
}

#[test]
fn a_malformed_value_is_reported_with_its_line_number() {
    let bad_rate = ROW.replace("0.00025", "0.025%");
    let bad_date = VERSION.replace("2025-01-01", "01-01-2025");
    let impossible_date = VERSION.replace("2025-01-01", "2025-02-30");
    let bad_base = ROW.replace("sides", "both");
    let bad_unit = ROW.replace("rupee", "rupees");
    let bad_segment = ROW.replace("NSE_EQ_INTRADAY", "NSE_EQ");
    let bad_gst_base = VERSION.replace("0.18,yes", "0.18,maybe");
    let no_gst_base = VERSION.replace("0.18,yes", "0.18,");
    assert_failures(&[
        (
            "an unknown GST base",
            vec![&bad_gst_base],
            2,
            "gst_on_sebi_fee \"maybe\" must be yes or no",
        ),
        (
            "a version without its GST base",
            vec![&no_gst_base],
            2,
            "gst_on_sebi_fee \"\" must be yes or no",
        ),
        (
            "a rate that is not a number",
            vec![VERSION, &bad_rate],
            3,
            "stt_sell_rate",
        ),
        ("a date that is not ISO", vec![&bad_date], 2, "not a date"),
        (
            "a date that does not exist",
            vec![&impossible_date],
            2,
            "not a date",
        ),
        (
            "an unknown tax base",
            vec![VERSION, &bad_base],
            3,
            "sides or turnover",
        ),
        (
            "an unknown tax unit",
            vec![VERSION, &bad_unit],
            3,
            "rupee or paise",
        ),
        (
            "an unknown segment",
            vec![VERSION, &bad_segment],
            3,
            "unknown cost segment",
        ),
    ]);
}

#[test]
fn a_misplaced_row_or_field_is_reported_with_its_line_number() {
    let wrong_columns = ROW.replacen(",sides", "", 1);
    let dated_row = ROW.replacen("T1,,", "T1,2025-01-01,", 1);
    let rate_on_version = VERSION.replacen(",,,,,,,,0.18", ",0.001,,,,,,,0.18", 1);
    let open_quote = VERSION.replace("note", "\"note");
    let gst_base_on_row = ROW.replace("0,,,sides", "0,,yes,sides");
    assert_failures(&[
        (
            "a GST base on a segment row",
            vec![VERSION, &gst_base_on_row],
            3,
            "must be empty",
        ),
        (
            "too few columns",
            vec![VERSION, &wrong_columns],
            3,
            "expected 18 columns",
        ),
        (
            "a segment row before its version row",
            vec![ROW],
            2,
            "must come before",
        ),
        (
            "a segment twice",
            vec![VERSION, ROW, ROW],
            4,
            "appears twice",
        ),
        (
            "a version twice",
            vec![VERSION, ROW, VERSION],
            4,
            "two * rows",
        ),
        (
            "a date on a segment row",
            vec![VERSION, &dated_row],
            3,
            "must be empty",
        ),
        (
            "a rate on a version row",
            vec![&rate_on_version],
            2,
            "must be empty",
        ),
        ("an unclosed quote", vec![&open_quote], 2, "not closed"),
    ]);
}

#[test]
fn a_rate_in_percent_units_is_rejected_with_the_unit_spelled_out() {
    // 0.025 was meant as 0.025 percent; as a fraction it is a 2.5 percent tax.
    let slip = ROW.replace("0.00025", "0.025");
    let error = parse_schedules(&file(&[VERSION, &slip])).unwrap_err();
    assert!(matches!(error, CostError::InvalidSchedule(_)), "{error:?}");
    assert!(
        error
            .to_string()
            .contains("rates are fractions: 0.001 is 0.1 percent"),
        "{error}"
    );
}

#[test]
fn text_that_is_not_ascii_is_an_error_not_a_panic() {
    let odd_date = VERSION.replace("2025-01-01", "2025\u{2013}01-1");
    let (line, _) = failure(&[&odd_date]);
    assert_eq!(line, 2);
}
