//! The built-in versions before 2024-10-01: their dates, a contract note worked by
//! hand in each, every rate change on its effective date, and which rows are verified.
//!
//! Rates are read from the sources each version records: NSE circulars NSE/FA/46730,
//! NSE/FA/56129 and NSE/FATAX/56235, BSE notices 20210210-42, 20220425-2,
//! 20221109-7, 20231020-46 and 20240430-42, MCX circulars MCX/TRD/161/2021 and
//! MCX/TRD/252/2021, the SEBI fee notifications of 2019 and 2020, and the Finance Act
//! 2019 stamp duty table. A rupee figure "per crore" is the charge on a fill worth
//! Rs 1 crore.

use crate::model::{days_from_civil, Day, Side};

use super::super::{
    builtin_schedules, ContractNoteAggregator, CostBreakdown, CostError, Segment, Trust,
};
use super::{accrual_model, assert_close, CRORE};

fn d(year: i32, month: u32, day: u32) -> Day {
    days_from_civil(year, month, day)
}

#[test]
fn the_versions_before_2024_10_01_and_their_dates() {
    let listed: Vec<(&str, Day, Option<Day>)> = builtin_schedules()
        .unwrap()
        .iter()
        .take_while(|version| version.valid_from < d(2024, 10, 1))
        .map(|version| (version.id.as_str(), version.valid_from, version.valid_to))
        .collect();
    let expected = vec![
        ("IN-2020-07-01", d(2020, 7, 1), Some(d(2020, 12, 31))),
        ("IN-2021-01-01", d(2021, 1, 1), Some(d(2021, 2, 28))),
        ("IN-2021-03-01", d(2021, 3, 1), Some(d(2021, 3, 31))),
        ("IN-2021-04-01", d(2021, 4, 1), Some(d(2021, 4, 30))),
        ("IN-2021-05-01", d(2021, 5, 1), Some(d(2022, 5, 1))),
        ("IN-2022-05-02", d(2022, 5, 2), Some(d(2022, 11, 30))),
        ("IN-2022-12-01", d(2022, 12, 1), Some(d(2023, 3, 31))),
        ("IN-2023-04-01", d(2023, 4, 1), Some(d(2023, 10, 31))),
        ("IN-2023-11-01", d(2023, 11, 1), Some(d(2024, 5, 12))),
        ("IN-2024-05-13", d(2024, 5, 13), Some(d(2024, 9, 30))),
    ];
    assert_eq!(listed, expected);
}

/// The total of the contract note of one day holding `fills` NSE cash trades, each
/// (segment, side, price, quantity), under the default model.
fn nse_note_total(day: Day, fills: &[(Segment, Side, f64, f64)]) -> f64 {
    let model = accrual_model();
    let mut notes = ContractNoteAggregator::new();
    for &(segment, side, price, quantity) in fills {
        let cost = model
            .fill_cost(segment, side, price, quantity, day)
            .unwrap();
        notes.add(day, segment, &cost);
    }
    notes.total_charged()
}

/// One version's three notes: a delivery buy of 100 at 2,012.35 on one day, its sale
/// at 2,104.30 on another, and the worked example's intraday round trip (400 bought
/// at 1,000 and sold at 1,100) on a third.
struct VersionNotes {
    buy_day: Day,
    buy_note: f64,
    sell_day: Day,
    sell_note: f64,
    intraday_day: Day,
    intraday_note: f64,
}

/// Every version before 2024-10-01 with its notes, worked by hand.
///
/// Delivery (STT 0.1 percent each side, stamp 0.015 percent on the buy, no brokerage):
/// the buy of Rs 2,01,235 pays STT 201.235 -> 201 and stamp 30.18525 -> 30; the sale
/// of Rs 2,10,430 pays STT 210.43 -> 210. Intraday (STT 0.025 percent on the sell,
/// stamp 0.003 percent on the buy, brokerage Rs 20 a leg): STT 110, stamp 12,
/// brokerage 40. The exchange line is (charge + IPFT) x value, SEBI is its rate x
/// value, and GST is 18 percent of brokerage + exchange + SEBI, each rounded to the
/// paisa. The rates that move the totals:
///
/// - 2020-07-01: NSE Rs 3.25 per lakh + IPFT Rs 0.01 per crore, SEBI Rs 5 per crore.
///   Buy: exchange 6.540338735 -> 6.54, SEBI 0.1006175 -> 0.10, GST 1.1953721 -> 1.20,
///   total 201 + 6.54 + 0.10 + 30 + 1.20 = 238.84. Sell: 6.83918543 -> 6.84,
///   0.105215 -> 0.11, GST 1.2499921 -> 1.25, total 218.20. Intraday on Rs 8,40,000:
///   exchange 27.30084 -> 27.30, SEBI 0.42, GST 18% of 67.72084 = 12.1897512 -> 12.19,
///   total 40 + 110 + 27.30 + 0.42 + 12 + 12.19 = 201.91.
/// - 2021-01-01 and 2021-03-01: NSE Rs 3.45 per lakh. Buy: exchange 6.942808735 ->
///   6.94, GST 1.2678167 -> 1.27, total 239.31. Sell: 7.26004543 -> 7.26, GST
///   1.3257469 -> 1.33, total 218.70. Intraday: exchange 28.98084 -> 28.98, GST
///   12.4921512 -> 12.49, total 203.89.
/// - 2021-04-01 to 2022-12-01: SEBI back to Rs 10 per crore. Buy: SEBI 0.201235 ->
///   0.20, GST 1.2859279 -> 1.29, total 239.43. Sell: SEBI 0.21043 -> 0.21, GST
///   1.3446856 -> 1.34, total 218.81. Intraday: SEBI 0.84, GST 12.5677512 -> 12.57,
///   total 204.39.
/// - 2023-04-01 to 2024-05-13: NSE Rs 3.25 per lakh + IPFT Rs 10 per crore. Buy:
///   exchange 6.7413725 -> 6.74, GST 1.2496694 -> 1.25, total 239.19. Sell: 7.049405 ->
///   7.05, GST 1.3067703 -> 1.31, total 218.57. Intraday: exchange 28.14, GST
///   12.4164 -> 12.42, total 203.40.
fn version_notes() -> [(&'static str, VersionNotes); 10] {
    let notes = |buy: (Day, f64), sell: (Day, f64), intraday: (Day, f64)| VersionNotes {
        buy_day: buy.0,
        buy_note: buy.1,
        sell_day: sell.0,
        sell_note: sell.1,
        intraday_day: intraday.0,
        intraday_note: intraday.1,
    };
    [
        (
            "IN-2020-07-01",
            notes(
                (d(2020, 8, 3), 238.84),
                (d(2020, 11, 2), 218.20),
                (d(2020, 12, 1), 201.91),
            ),
        ),
        (
            "IN-2021-01-01",
            notes(
                (d(2021, 1, 4), 239.31),
                (d(2021, 2, 1), 218.70),
                (d(2021, 2, 15), 203.89),
            ),
        ),
        (
            "IN-2021-03-01",
            notes(
                (d(2021, 3, 2), 239.31),
                (d(2021, 3, 25), 218.70),
                (d(2021, 3, 15), 203.89),
            ),
        ),
        (
            "IN-2021-04-01",
            notes(
                (d(2021, 4, 5), 239.43),
                (d(2021, 4, 28), 218.81),
                (d(2021, 4, 15), 204.39),
            ),
        ),
        (
            "IN-2021-05-01",
            notes(
                (d(2021, 6, 1), 239.43),
                (d(2022, 1, 3), 218.81),
                (d(2021, 9, 1), 204.39),
            ),
        ),
        (
            "IN-2022-05-02",
            notes(
                (d(2022, 5, 4), 239.43),
                (d(2022, 11, 1), 218.81),
                (d(2022, 8, 1), 204.39),
            ),
        ),
        (
            "IN-2022-12-01",
            notes(
                (d(2022, 12, 1), 239.43),
                (d(2023, 3, 1), 218.81),
                (d(2023, 1, 2), 204.39),
            ),
        ),
        (
            "IN-2023-04-01",
            notes(
                (d(2023, 4, 3), 239.19),
                (d(2023, 10, 3), 218.57),
                (d(2023, 7, 3), 203.40),
            ),
        ),
        (
            "IN-2023-11-01",
            notes(
                (d(2023, 11, 1), 239.19),
                (d(2024, 5, 10), 218.57),
                (d(2024, 1, 1), 203.40),
            ),
        ),
        (
            "IN-2024-05-13",
            notes(
                (d(2024, 5, 13), 239.19),
                (d(2024, 9, 30), 218.57),
                (d(2024, 7, 1), 203.40),
            ),
        ),
    ]
}

/// The id of the built-in version that prices `day`.
fn version_on(day: Day) -> String {
    let model = accrual_model();
    let status = model.status(Segment::NseEqDelivery, day).unwrap();
    model.schedules().versions()[status.schedule].id.clone()
}

#[test]
fn each_version_charges_the_hand_worked_contract_notes_of_a_delivery_and_an_intraday_trade() {
    use Segment::{NseEqDelivery, NseEqIntraday};
    for (id, notes) in version_notes() {
        for day in [notes.buy_day, notes.sell_day, notes.intraday_day] {
            assert_eq!(version_on(day), id, "day {day} is not in {id}");
        }
        let buy = nse_note_total(
            notes.buy_day,
            &[(NseEqDelivery, Side::Buy, 2_012.35, 100.0)],
        );
        let sell = nse_note_total(
            notes.sell_day,
            &[(NseEqDelivery, Side::Sell, 2_104.30, 100.0)],
        );
        let intraday = nse_note_total(
            notes.intraday_day,
            &[
                (NseEqIntraday, Side::Buy, 1_000.0, 400.0),
                (NseEqIntraday, Side::Sell, 1_100.0, 400.0),
            ],
        );
        assert_eq!(
            (buy, sell, intraday),
            (notes.buy_note, notes.sell_note, notes.intraday_note),
            "{id}"
        );
    }
}

/// Which part of a fill's cost a boundary changes.
#[derive(Clone, Copy, Debug)]
enum Line {
    Stt,
    Exchange,
    Sebi,
}

impl Line {
    fn of(self, cost: &CostBreakdown) -> f64 {
        match self {
            Line::Stt => cost.stt,
            Line::Exchange => cost.exchange_txn,
            Line::Sebi => cost.sebi_fee,
        }
    }
}

/// (first day of the new rates, segment, side, line, rupees on Rs 1 crore the day
/// before, rupees on Rs 1 crore that day).
type Boundary = (Day, Segment, Side, Line, f64, f64);

fn boundaries() -> Vec<Boundary> {
    use Line::{Exchange, Sebi, Stt};
    use Segment::{
        BfoOpt, BfoOptOther, BseEqDelivery, McxFut, McxFutKapas, NfoFut, NfoOpt, NseEqDelivery,
    };
    use Side::{Buy, Sell};
    vec![
        // NSE/FA/46730: Rs 3.25 to 3.45 per lakh; IPFT Rs 0.01 per crore throughout.
        (d(2021, 1, 1), NseEqDelivery, Buy, Exchange, 325.01, 345.01),
        // BSE notice 20210210-42: groups A and B, Rs 300 to 345 per crore.
        (d(2021, 3, 1), BseEqDelivery, Buy, Exchange, 300.0, 345.0),
        // SEBI/LAD-NRO/GN/2020/011: half of Rs 10 per crore until 2021-03-31, also for
        // non-agri commodity futures.
        (d(2021, 4, 1), NseEqDelivery, Sell, Sebi, 5.0, 10.0),
        (d(2021, 4, 1), McxFut, Sell, Sebi, 5.0, 10.0),
        // MCX/TRD/252/2021: KAPAS from Rs 0.50 to the Group A Rs 2.60 per lakh.
        (d(2021, 5, 1), McxFutKapas, Buy, Exchange, 50.0, 260.0),
        // BSE notice 20220425-2: options pay Rs 500 per crore of premium, from nil.
        (d(2022, 5, 2), BfoOpt, Sell, Exchange, 0.0, 500.0),
        (d(2022, 5, 2), BfoOptOther, Sell, Exchange, 0.0, 500.0),
        // BSE notice 20221109-7: groups A and B, Rs 375 per crore flat.
        (d(2022, 12, 1), BseEqDelivery, Buy, Exchange, 345.0, 375.0),
        // NSE/FATAX/56235: futures STT 0.01 to 0.0125 percent, options 0.05 to 0.0625.
        (d(2023, 4, 1), NfoFut, Sell, Stt, 1_000.0, 1_250.0),
        (d(2023, 4, 1), NfoOpt, Sell, Stt, 5_000.0, 6_250.0),
        // NSE/FA/56129: Rs 3.45 to 3.25 per lakh, IPFT Rs 0.01 to 10 per crore.
        (d(2023, 4, 1), NseEqDelivery, Buy, Exchange, 345.01, 335.0),
        // BSE notice 20231020-46: nearest-expiry Sensex options, Rs 3,750 per crore.
        (d(2023, 11, 1), BfoOpt, Sell, Exchange, 500.0, 3_750.0),
        // BSE notice 20240430-42: Sensex and Bankex options, Rs 4,950 per crore.
        (d(2024, 5, 13), BfoOpt, Sell, Exchange, 3_750.0, 4_950.0),
        // NSE/FA/64323, MCX/F&A/631/2024 and the Finance (No. 2) Act 2024, into the
        // 2024-10-01 version.
        (d(2024, 10, 1), NseEqDelivery, Buy, Exchange, 335.0, 307.0),
        (d(2024, 10, 1), NfoFut, Sell, Stt, 1_250.0, 2_000.0),
        (d(2024, 10, 1), McxFut, Buy, Exchange, 260.0, 210.0),
    ]
}

#[test]
fn charges_start_on_2020_07_01_and_the_day_before_is_still_an_error() {
    // The uniform stamp duty of the Finance Act 2019 applies from 2020-07-01: Rs 30 on a
    // Rs 2,00,000 delivery buy.
    let model = accrual_model();
    let buy = |day| model.fill_cost(Segment::NseEqDelivery, Side::Buy, 2_000.0, 100.0, day);
    assert_close(buy(d(2020, 7, 1)).unwrap().stamp_duty, 30.0);
    assert_eq!(
        buy(d(2020, 6, 30)),
        Err(CostError::NoScheduleForDay {
            day: d(2020, 6, 30),
            first_valid_from: d(2020, 7, 1)
        })
    );
}

#[test]
fn every_rate_change_applies_from_its_effective_date_and_not_the_day_before() {
    let model = accrual_model();
    for (first_day, segment, side, line, before, after) in boundaries() {
        let on = |day| line.of(&model.fill_cost_value(segment, side, CRORE, day).unwrap());
        assert_close(on(first_day - 1), before);
        assert_close(on(first_day), after);
    }
}

/// The segments of each version before 2024-10-01 whose rates are not verified: BSE
/// groups X (ZP is charged otherwise) and SS (no dated source), MCX agri futures (CTT
/// exemption narrower than the row) and RBDPMOLEIN (no fee found), every MCX row in
/// 2020 (no fee circular found before 2020-12-18), and Sensex options while the
/// nearest expiry had its own slabs.
fn unverified_rows() -> [(&'static str, Vec<Segment>); 10] {
    use Segment::*;
    let always = [
        BseEqDeliveryX,
        BseEqIntradayX,
        BseEqDeliverySs,
        BseEqIntradaySs,
        McxFutAgri,
        McxFutRbdpmolein,
    ];
    let with = |extra: &[Segment]| -> Vec<Segment> {
        let mut rows = always.to_vec();
        rows.extend_from_slice(extra);
        rows.sort();
        rows
    };
    let all_mcx = [McxFut, McxFutCastorseed, McxFutPepper, McxFutKapas];
    [
        ("IN-2020-07-01", with(&all_mcx)),
        ("IN-2021-01-01", with(&[])),
        ("IN-2021-03-01", with(&[])),
        ("IN-2021-04-01", with(&[])),
        ("IN-2021-05-01", with(&[])),
        ("IN-2022-05-02", with(&[])),
        ("IN-2022-12-01", with(&[])),
        ("IN-2023-04-01", with(&[])),
        ("IN-2023-11-01", with(&[BfoOpt])),
        ("IN-2024-05-13", with(&[])),
    ]
}

#[test]
fn only_the_rows_without_a_primary_source_are_unverified() {
    let model = accrual_model();
    for (id, expected) in unverified_rows() {
        let version = builtin_schedules()
            .unwrap()
            .iter()
            .find(|version| version.id == id)
            .unwrap();
        let day = version.valid_from;
        let unverified: Vec<Segment> = Segment::ALL
            .into_iter()
            .filter(|&segment| model.status(segment, day).unwrap().trust != Trust::Verified)
            .collect();
        assert_eq!(unverified, expected, "{id}");
    }
}

#[test]
fn an_unverified_row_before_2024_10_01_says_why_in_its_notes() {
    let earlier = builtin_schedules()
        .unwrap()
        .iter()
        .filter(|version| version.valid_from < d(2024, 10, 1));
    for version in earlier {
        for segment in Segment::ALL {
            let row = version.row(segment).unwrap();
            if row.verified_through.is_none() {
                assert!(!row.notes.is_empty(), "{} {segment:?}", version.id);
            }
        }
    }
}

#[test]
fn every_version_before_2024_10_01_has_its_gst_verified_through_its_last_day() {
    for version in builtin_schedules().unwrap() {
        if version.valid_from < d(2024, 10, 1) {
            assert_eq!(
                version.gst_verified_through, version.valid_to,
                "{}",
                version.id
            );
        }
    }
}
