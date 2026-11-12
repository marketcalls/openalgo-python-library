//! User-supplied schedule versions take the same path as the built-in ones.

use crate::model::{days_from_civil, Day, Side, MAX_DAY};

use super::super::{
    builtin_schedules, Brokerage, CostError, CostModel, CostSchedule, Rounding, ScheduleBook,
    Segment, SegmentRates, SegmentRow, TaxBase, TaxUnit, Trust,
};
use super::{assert_close, snapshot_day};

/// Rates for one row whose sell tax and exchange charge are both `rate`.
fn rates(rate: f64) -> SegmentRates {
    SegmentRates {
        stt_buy_rate: 0.0,
        stt_sell_rate: rate,
        exchange_txn_rate: rate,
        ipft_rate: 0.0,
        sebi_rate: 0.000_001,
        stamp_buy_rate: 0.000_03,
        stt_exercise_rate: 0.0,
        calculator_tax_base: TaxBase::Sides,
        calculator_tax_unit: TaxUnit::Rupee,
    }
}

/// A version with one row (NSE intraday), never verified.
fn user_version(id: &str, from: Day, to: Option<Day>, rate: f64) -> CostSchedule {
    let mut rows = std::array::from_fn(|_| None);
    rows[Segment::NseEqIntraday.index()] = Some(SegmentRow {
        rates: rates(rate),
        verified_through: None,
        notes: String::new(),
    });
    CostSchedule {
        id: id.to_string(),
        valid_from: from,
        valid_to: to,
        gst_rate: 0.18,
        gst_verified_through: None,
        source: "test".to_string(),
        notes: String::new(),
        rows,
    }
}

/// The built-in versions plus a user version for 2019-04-01 to 2020-06-30, the
/// day before the first built-in version.
fn book_with_an_earlier_period() -> ScheduleBook {
    let from = days_from_civil(2019, 4, 1);
    let to = days_from_civil(2020, 6, 30);
    ScheduleBook::builtin_with(vec![user_version("user-2023", from, Some(to), 0.001)]).unwrap()
}

fn zero_brokerage_model(book: ScheduleBook) -> CostModel {
    CostModel::new(book, Brokerage::zero_brokerage(), Rounding::ContractNote)
}

fn version_id(book: &ScheduleBook, day: Day) -> Result<String, CostError> {
    book.lookup(day)
        .map(|(index, _)| book.versions()[index].id.clone())
}

#[test]
fn a_user_version_for_an_earlier_period_is_used_on_its_dates() {
    // Rs 1,00,000 sold at a 0.1 percent tax and exchange rate: Rs 100 each.
    let model = zero_brokerage_model(book_with_an_earlier_period());
    let day = days_from_civil(2020, 1, 15);
    let cost = model
        .fill_cost(Segment::NseEqIntraday, Side::Sell, 1_000.0, 100.0, day)
        .unwrap();
    assert_close(cost.stt, 100.0);
    assert_close(cost.exchange_txn, 100.0);
}

#[test]
fn a_user_version_is_unverified_unless_it_says_otherwise() {
    let model = zero_brokerage_model(book_with_an_earlier_period());
    let status = model
        .status(Segment::NseEqIntraday, days_from_civil(2020, 1, 15))
        .unwrap();
    assert_eq!((status.schedule, status.trust), (0, Trust::Unverified));
}

#[test]
fn builtin_versions_still_answer_for_their_own_dates() {
    // Built-in intraday STT is 0.025 percent: Rs 25 on Rs 1,00,000 sold.
    let model = zero_brokerage_model(book_with_an_earlier_period());
    let cost = model.fill_cost(
        Segment::NseEqIntraday,
        Side::Sell,
        1_000.0,
        100.0,
        snapshot_day(),
    );
    assert_close(cost.unwrap().stt, 25.0);
}

#[test]
fn a_segment_missing_from_a_version_is_an_error_on_its_dates() {
    let model = zero_brokerage_model(book_with_an_earlier_period());
    let day = days_from_civil(2020, 1, 15);
    assert_eq!(
        model.fill_cost(Segment::NfoFut, Side::Sell, 1_000.0, 100.0, day),
        Err(CostError::SegmentNotInSchedule {
            segment: Segment::NfoFut,
            day
        })
    );
}

#[test]
fn a_date_before_the_earliest_user_version_is_still_an_error() {
    let book = book_with_an_earlier_period();
    let day = days_from_civil(2019, 3, 31);
    assert!(matches!(
        book.lookup(day),
        Err(CostError::NoScheduleForDay { .. })
    ));
}

#[test]
fn a_later_user_version_takes_over_from_the_open_builtin_version() {
    let from = days_from_civil(2027, 4, 1);
    let book =
        ScheduleBook::builtin_with(vec![user_version("user-2027", from, None, 0.002)]).unwrap();
    assert_eq!(version_id(&book, from - 1).unwrap(), "IN-2026-04-01");
    assert_eq!(version_id(&book, from).unwrap(), "user-2027");
    let (index, _) = book.lookup(from - 1).unwrap();
    assert_eq!(book.versions()[index].valid_to, Some(from - 1));
}

#[test]
fn merging_user_versions_leaves_the_builtin_data_untouched() {
    let from = days_from_civil(2027, 4, 1);
    ScheduleBook::builtin_with(vec![user_version("user-2027", from, None, 0.002)]).unwrap();
    let last = builtin_schedules().unwrap().last().map(|v| v.valid_to);
    assert_eq!(last, Some(None));
}

#[test]
fn user_versions_are_sorted_by_start_date() {
    let first = days_from_civil(2025, 1, 1);
    let second = days_from_civil(2025, 7, 1);
    let book = ScheduleBook::from_versions(vec![
        user_version("second", second, None, 0.002),
        user_version("first", first, Some(second - 1), 0.001),
    ])
    .unwrap();
    let ids: Vec<&str> = book.versions().iter().map(|v| v.id.as_str()).collect();
    assert_eq!(ids, ["first", "second"]);
}

#[test]
fn a_date_between_two_versions_or_after_the_last_one_is_a_gap() {
    let first = days_from_civil(2025, 1, 1);
    let second = days_from_civil(2025, 7, 1);
    let end = days_from_civil(2025, 12, 31);
    let book = ScheduleBook::from_versions(vec![
        user_version("first", first, Some(second - 11), 0.001),
        user_version("second", second, Some(end), 0.002),
    ])
    .unwrap();
    for day in [second - 5, end + 1] {
        assert_eq!(book.lookup(day), Err(CostError::ScheduleGap { day }));
    }
}

#[test]
fn a_book_needs_at_least_one_version() {
    assert_eq!(
        ScheduleBook::from_versions(vec![]),
        Err(CostError::EmptyScheduleBook)
    );
}

#[test]
fn overlapping_versions_are_rejected() {
    let a = days_from_civil(2025, 1, 1);
    let b = days_from_civil(2025, 7, 1);
    let overlapping = [
        // The first still covers the day the second starts.
        vec![
            user_version("one", a, Some(b), 0.001),
            user_version("two", b, None, 0.001),
        ],
        // Both start on the same day.
        vec![
            user_version("one", a, None, 0.001),
            user_version("two", a, None, 0.001),
        ],
    ];
    for versions in overlapping {
        assert_eq!(
            ScheduleBook::from_versions(versions),
            Err(CostError::ScheduleOverlap {
                first: "one".to_string(),
                second: "two".to_string()
            })
        );
    }
}

#[test]
fn a_user_version_inside_a_builtin_range_is_rejected() {
    let from = days_from_civil(2025, 1, 1);
    let to = days_from_civil(2025, 7, 1);
    let clash = ScheduleBook::builtin_with(vec![user_version("clash", from, Some(to), 0.001)]);
    assert!(matches!(clash, Err(CostError::ScheduleOverlap { .. })));
}

#[test]
fn duplicate_version_ids_are_rejected() {
    let a = days_from_civil(2025, 1, 1);
    let duplicate = ScheduleBook::from_versions(vec![
        user_version("same", a, Some(a + 10), 0.001),
        user_version("same", a + 20, None, 0.001),
    ]);
    assert!(matches!(duplicate, Err(CostError::InvalidSchedule(_))));
}

/// An edit to a version and to the rates of its one row.
type Change = fn(&mut CostSchedule, &mut SegmentRates);

/// The error a version gives after `change` is applied to an otherwise valid one.
fn rejection(change: impl Fn(&mut CostSchedule, &mut SegmentRates)) -> String {
    let mut version = user_version("changed", days_from_civil(2025, 1, 1), None, 0.001);
    let mut row_rates = rates(0.001);
    change(&mut version, &mut row_rates);
    if let Some(row) = version.rows[Segment::NseEqIntraday.index()].as_mut() {
        row.rates = row_rates;
    }
    match ScheduleBook::from_versions(vec![version]) {
        Err(CostError::InvalidSchedule(why)) => why,
        other => panic!("expected InvalidSchedule, got {other:?}"),
    }
}

#[test]
fn a_rate_in_percent_units_is_rejected_for_every_kind_of_rate() {
    // Each value is a real rate written in percent instead of as a fraction:
    // 0.025 (intraday STT), 0.1 (delivery STT), 0.0001 (IPFT and SEBI fee),
    // 0.015 (delivery stamp duty), 0.15 (exercise STT), 18 (GST).
    let slips: [(&str, Change); 7] = [
        ("stt_sell_rate", |_, r| r.stt_sell_rate = 0.025),
        ("stt_buy_rate", |_, r| r.stt_buy_rate = 0.1),
        ("ipft_rate", |_, r| r.ipft_rate = 0.0001),
        ("sebi_rate", |_, r| r.sebi_rate = 0.0001),
        ("stamp_buy_rate", |_, r| r.stamp_buy_rate = 0.015),
        ("stt_exercise_rate", |_, r| r.stt_exercise_rate = 0.15),
        ("gst_rate", |version, _| version.gst_rate = 18.0),
    ];
    for (name, slip) in slips {
        let why = rejection(slip);
        assert!(why.contains(name), "{why}");
        assert!(
            why.contains("rates are fractions: 0.001 is 0.1 percent"),
            "{why}"
        );
    }
}

#[test]
fn the_bse_ss_group_exchange_charge_of_one_percent_is_accepted() {
    // The loosest real rate: Rs 1,00,000 per crore.
    let mut version = user_version("ss", days_from_civil(2025, 1, 1), None, 0.001);
    if let Some(row) = version.rows[Segment::NseEqIntraday.index()].as_mut() {
        row.rates.exchange_txn_rate = 0.01;
    }
    assert!(ScheduleBook::from_versions(vec![version]).is_ok());
}

#[test]
fn malformed_versions_are_rejected() {
    let rejections = [
        rejection(|_, r| r.stt_sell_rate = -0.001),
        rejection(|_, r| r.sebi_rate = f64::NAN),
        rejection(|version, _| version.valid_to = Some(version.valid_from - 1)),
        rejection(|version, _| version.valid_to = Some(MAX_DAY + 1)),
        rejection(|version, _| version.valid_from = Day::MIN),
        rejection(|version, _| version.id = " ".to_string()),
        rejection(|version, _| version.rows = std::array::from_fn(|_| None)),
        rejection(|_, r| {
            r.calculator_tax_base = TaxBase::Turnover;
        }),
    ];
    assert!(rejections.iter().all(|why| !why.is_empty()));
}
