//! The built-in schedule versions, the lookup by trade date, and how far the rates
//! are verified.

use crate::model::{days_from_civil, Day, Side, MAX_DAY, MIN_DAY};

use super::super::{
    builtin_schedules, Brokerage, CostError, CostModel, Rounding, Segment, SegmentRates, Trust,
};
use super::{
    accrual_model, assert_close, assert_paise, builtin_book, calculator_model, snapshot_day, CRORE,
};

const FNO_SEGMENTS: [Segment; 5] = [
    Segment::NfoFut,
    Segment::NfoOpt,
    Segment::BfoFut,
    Segment::BfoOpt,
    Segment::BfoOptOther,
];

/// The last day of the old F&O STT rates and the first day of the new ones.
fn fno_stt_boundary() -> (Day, Day) {
    (days_from_civil(2026, 3, 31), days_from_civil(2026, 4, 1))
}

fn trust(segment: Segment, day: Day) -> Trust {
    accrual_model().status(segment, day).unwrap().trust
}

#[test]
fn builtin_versions_cover_2024_10_01_onward_without_a_gap() {
    let versions = builtin_schedules().unwrap();
    let ranges: Vec<_> = versions
        .iter()
        .map(|v| (v.id.as_str(), v.valid_from, v.valid_to))
        .collect();
    assert_eq!(
        ranges,
        vec![
            (
                "IN-2024-10-01",
                days_from_civil(2024, 10, 1),
                Some(days_from_civil(2026, 3, 31))
            ),
            ("IN-2026-04-01", days_from_civil(2026, 4, 1), None),
        ]
    );
}

#[test]
fn the_older_version_has_no_verified_row() {
    let older = &builtin_schedules().unwrap()[0];
    assert_eq!(older.gst_verified_through, None);
    assert!(older
        .rows
        .iter()
        .flatten()
        .all(|row| row.verified_through.is_none()));
}

#[test]
fn the_current_version_is_verified_through_the_snapshot_day_except_bse_stock_options() {
    let current = &builtin_schedules().unwrap()[1];
    assert_eq!(current.gst_verified_through, Some(snapshot_day()));
    for segment in Segment::ALL {
        let verified = current.row(segment).unwrap().verified_through;
        let expected = if segment == Segment::BfoOptOther {
            None
        } else {
            Some(snapshot_day())
        };
        assert_eq!(verified, expected, "{segment:?}");
    }
}

#[test]
fn the_unverified_bse_options_row_names_its_conflicting_sources() {
    let current = &builtin_schedules().unwrap()[1];
    let notes = &current.row(Segment::BfoOptOther).unwrap().notes;
    assert!(
        notes.contains("Angel One") && notes.contains("Zerodha"),
        "{notes}"
    );
}

#[test]
fn every_builtin_version_records_its_source_and_caveats() {
    for version in builtin_schedules().unwrap() {
        assert!(
            !version.source.is_empty() && !version.notes.is_empty(),
            "{}",
            version.id
        );
    }
}

#[test]
fn every_builtin_version_has_a_row_for_every_segment() {
    for version in builtin_schedules().unwrap() {
        for segment in Segment::ALL {
            assert!(version.row(segment).is_some(), "{} {segment:?}", version.id);
        }
    }
}

#[test]
fn builtin_versions_differ_only_in_fno_stt() {
    let [old, new] = builtin_schedules().unwrap() else {
        panic!("expected two versions")
    };
    for segment in Segment::ALL {
        let (before, after) = (
            old.row(segment).unwrap().rates,
            new.row(segment).unwrap().rates,
        );
        let ignoring_stt = SegmentRates {
            stt_sell_rate: after.stt_sell_rate,
            stt_exercise_rate: after.stt_exercise_rate,
            ..before
        };
        assert_eq!(ignoring_stt, after, "{segment:?}");
        let changed = before.stt_sell_rate != after.stt_sell_rate;
        assert_eq!(changed, FNO_SEGMENTS.contains(&segment), "{segment:?}");
    }
}

#[test]
fn previous_fno_stt_rates_apply_before_2026_04_01() {
    // Section 1: "the previous F&O rates (0.02% on futures, 0.1% on options
    // premium)". Exercise was 0.125 percent of intrinsic value.
    let old = &builtin_schedules().unwrap()[0];
    let rows = [
        (Segment::NfoFut, 0.0002, 0.0),
        (Segment::BfoFut, 0.0002, 0.0),
        (Segment::NfoOpt, 0.001, 0.00125),
        (Segment::BfoOpt, 0.001, 0.00125),
        (Segment::BfoOptOther, 0.001, 0.00125),
    ];
    for (segment, stt_sell_rate, stt_exercise_rate) in rows {
        let rates = old.row(segment).unwrap().rates;
        assert_eq!(
            (rates.stt_sell_rate, rates.stt_exercise_rate),
            (stt_sell_rate, stt_exercise_rate)
        );
    }
}

#[test]
fn a_trade_is_charged_at_the_rates_in_force_on_its_trade_date() {
    // F&O STT on a Rs 1 crore sell, the day before and the day of the change.
    let (last_old, first_new) = fno_stt_boundary();
    let rows = [
        (Segment::NfoFut, 2_000.0, 5_000.0),
        (Segment::BfoFut, 2_000.0, 5_000.0),
        (Segment::NfoOpt, 10_000.0, 15_000.0),
        (Segment::BfoOpt, 10_000.0, 15_000.0),
    ];
    let model = accrual_model();
    for (segment, before, after) in rows {
        let stt = |day| {
            model
                .fill_cost_value(segment, Side::Sell, CRORE, day)
                .unwrap()
                .stt
        };
        assert_close(stt(last_old), before);
        assert_close(stt(first_new), after);
    }
}

#[test]
fn the_fno_stt_change_moves_nothing_but_stt() {
    let (last_old, first_new) = fno_stt_boundary();
    let model = accrual_model();
    for segment in FNO_SEGMENTS {
        let before = model
            .fill_cost_value(segment, Side::Sell, CRORE, last_old)
            .unwrap();
        let after = model
            .fill_cost_value(segment, Side::Sell, CRORE, first_new)
            .unwrap();
        assert_eq!(
            (
                before.brokerage,
                before.exchange_txn,
                before.sebi_fee,
                before.gst
            ),
            (
                after.brokerage,
                after.exchange_txn,
                after.sebi_fee,
                after.gst
            ),
            "{segment:?}"
        );
    }
}

#[test]
fn the_futures_worked_example_a_day_before_the_stt_change() {
    // Same trade as section 6, charged at 0.02 percent: STT 88 instead of 220, so
    // the total is 294.33 - 220 + 88 = 162.33.
    let (last_old, _) = fno_stt_boundary();
    let model = calculator_model();
    let cost = model.round_trip_cost(Segment::NfoFut, 1_000.0, 1_100.0, 400.0, last_old, last_old);
    let cost = cost.unwrap();
    assert_eq!(cost.stt, 88.0);
    assert_paise(model.reported_total(&cost), 162.33);
}

#[test]
fn a_round_trip_across_the_stt_change_charges_each_leg_at_its_own_rates() {
    // Buy Rs 1 crore of futures on the last old day, sell on the first new day: the
    // sell pays the new 0.05 percent, Rs 5,000. Sold a day earlier it would pay
    // Rs 2,000.
    let (last_old, first_new) = fno_stt_boundary();
    let model = accrual_model();
    let across = model.round_trip_cost(Segment::NfoFut, CRORE, CRORE, 1.0, last_old, first_new);
    let within = model.round_trip_cost(Segment::NfoFut, CRORE, CRORE, 1.0, last_old, last_old);
    assert_close(across.unwrap().stt, 5_000.0);
    assert_close(within.unwrap().stt, 2_000.0);
}

#[test]
fn rates_are_verified_only_through_the_snapshot_day() {
    // (trade date, trust): the current version is charged on every later date but
    // is verified only up to 2026-11-12.
    let rows = [
        (days_from_civil(2026, 4, 1), Trust::Verified),
        (snapshot_day(), Trust::Verified),
        (snapshot_day() + 1, Trust::PastVerification),
        (days_from_civil(2031, 1, 1), Trust::PastVerification),
        (days_from_civil(2025, 6, 2), Trust::Unverified),
    ];
    for (day, expected) in rows {
        assert_eq!(trust(Segment::NseEqDelivery, day), expected, "day {day}");
    }
}

#[test]
fn a_trade_after_the_verified_date_is_still_charged_at_the_current_rates() {
    let model = accrual_model();
    let later = days_from_civil(2031, 1, 1);
    assert_eq!(
        model.fill_cost_value(Segment::NfoFut, Side::Sell, CRORE, later),
        model.fill_cost_value(Segment::NfoFut, Side::Sell, CRORE, snapshot_day())
    );
}

#[test]
fn an_unverified_row_is_unverified_on_every_date() {
    assert_eq!(
        trust(Segment::BfoOptOther, snapshot_day()),
        Trust::Unverified
    );
}

#[test]
fn gst_status_follows_the_version_row() {
    let model = accrual_model();
    let rows = [
        (snapshot_day(), Trust::Verified),
        (snapshot_day() + 1, Trust::PastVerification),
    ];
    for (day, expected) in rows {
        assert_eq!(model.gst_status(day).unwrap().trust, expected);
    }
}

#[test]
fn a_trade_before_the_first_version_is_an_error() {
    let early = days_from_civil(2024, 9, 30);
    let error = accrual_model().fill_cost(Segment::NseEqDelivery, Side::Buy, 100.0, 10.0, early);
    assert_eq!(
        error,
        Err(CostError::NoScheduleForDay {
            day: early,
            first_valid_from: days_from_civil(2024, 10, 1)
        })
    );
}

#[test]
fn the_early_date_error_names_both_dates_and_the_remedy() {
    let early = days_from_civil(2024, 9, 30);
    let message = builtin_book().lookup(early).unwrap_err().to_string();
    for expected in ["2024-09-30", "2024-10-01", "supply a schedule", "backfill"] {
        assert!(message.contains(expected), "{message}");
    }
}

#[test]
fn every_dated_function_rejects_a_date_before_the_first_version() {
    let early = days_from_civil(2020, 1, 1);
    let model = accrual_model();
    assert!(model
        .round_trip_cost(Segment::NfoFut, 1.0, 2.0, 1.0, early, early)
        .is_err());
    assert!(model
        .fill_cost_value(Segment::NfoFut, Side::Buy, 1.0, early)
        .is_err());
    assert!(model
        .exercise_stt(Segment::NfoOpt, 1.0, 1.0, early)
        .is_err());
    assert!(model.dp_charge(early).is_err());
    assert!(model.status(Segment::NfoFut, early).is_err());
    assert!(model.gst_status(early).is_err());
}

#[test]
fn a_date_outside_the_supported_range_is_an_error_everywhere() {
    // Backfill is on, so only the range check can refuse these dates.
    let book = builtin_book().with_backfill(true);
    let model = CostModel::new(book, Brokerage::zerodha(), Rounding::ContractNote);
    for day in [Day::MIN, MIN_DAY - 1, MAX_DAY + 1, Day::MAX] {
        let unsupported = Err(CostError::UnsupportedDay { day });
        assert_eq!(
            model.fill_cost(Segment::NfoFut, Side::Buy, 1.0, 1.0, day),
            unsupported
        );
        assert_eq!(
            model.dp_charge(day).map(|charge| charge.fee),
            Err(CostError::UnsupportedDay { day })
        );
        assert_eq!(
            model.status(Segment::NfoFut, day).map(|s| s.schedule),
            Err(CostError::UnsupportedDay { day })
        );
    }
}

#[test]
fn backfill_is_off_by_default() {
    assert!(!builtin_book().backfill());
}

#[test]
fn backfill_charges_early_trades_at_the_first_version_rates_and_says_so() {
    let model = CostModel::new(
        builtin_book().with_backfill(true),
        Brokerage::zerodha(),
        Rounding::ContractNote,
    );
    let early = days_from_civil(2019, 6, 3);
    let first = days_from_civil(2024, 10, 1);
    assert_eq!(
        model.fill_cost(Segment::NfoFut, Side::Sell, 100.0, 10.0, early),
        model.fill_cost(Segment::NfoFut, Side::Sell, 100.0, 10.0, first)
    );
    assert_eq!(
        model.status(Segment::NfoFut, early).unwrap().trust,
        Trust::Backfilled
    );
}
