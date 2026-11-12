//! Error messages say what was wrong and what to do.

use crate::model::{Exchange, InstrumentKind, Product};

use super::super::{CostError, NameKind, Segment};

fn unknown(kind: NameKind) -> CostError {
    CostError::UnknownName {
        kind,
        name: "x".to_string(),
    }
}

fn every_error() -> Vec<CostError> {
    vec![
        unknown(NameKind::Segment),
        unknown(NameKind::Exchange),
        unknown(NameKind::Product),
        unknown(NameKind::Instrument),
        unknown(NameKind::BseGroup),
        unknown(NameKind::Commodity),
        unknown(NameKind::Preset),
        unknown(NameKind::Rounding),
        CostError::UnsupportedSegment("segment CDS_FUT".to_string()),
        CostError::UnknownSegmentId(77),
        CostError::UnknownSide(9),
        CostError::InvalidCombination {
            exchange: Exchange::Nfo,
            kind: InstrumentKind::Future,
            product: Product::Cnc,
            hint: "futures and options use NRML or MIS",
        },
        CostError::MissingBseGroup,
        CostError::NotTradable(Exchange::NseIndex),
        CostError::NotAnOption(Segment::NfoFut),
        CostError::InvalidBrokerage("the cap must be zero or more rupees, got -1".to_string()),
        CostError::InvalidInput {
            field: "price",
            value: -1.0,
        },
        CostError::UnsupportedDay { day: i32::MAX },
        CostError::LengthMismatch {
            name: "sides",
            expected: 2,
            got: 1,
        },
        CostError::NoScheduleForDay {
            day: 0,
            first_valid_from: 10,
        },
        CostError::ScheduleGap { day: 0 },
        CostError::SegmentNotInSchedule {
            segment: Segment::NfoFut,
            day: 0,
        },
        CostError::InvalidSchedule("version \"x\": it has no segment rates".to_string()),
        CostError::InvalidScheduleData {
            line: 3,
            reason: "expected 17 columns, found 16".to_string(),
        },
        CostError::ScheduleOverlap {
            first: "a".to_string(),
            second: "b".to_string(),
        },
        CostError::EmptyScheduleBook,
    ]
}

#[test]
fn every_message_is_plain_ascii() {
    for error in every_error() {
        let message = error.to_string();
        assert!(message.is_ascii(), "{message}");
    }
}

#[test]
fn every_message_is_a_full_sentence_not_a_bare_code() {
    for error in every_error() {
        let message = error.to_string();
        assert!(message.split_whitespace().count() >= 6, "{message}");
    }
}

#[test]
fn an_unknown_name_message_quotes_the_name_and_lists_what_to_use() {
    let rows = [
        (NameKind::Segment, "NSE_EQ_DELIVERY"),
        (NameKind::Exchange, "NSE, BSE, NFO, BFO or MCX"),
        (NameKind::Product, "CNC, MIS or NRML"),
        (NameKind::Instrument, "EQ, FUT or OPT"),
        (NameKind::BseGroup, "XT, Z, ZP"),
        (NameKind::Commodity, "MCX_FUT_AGRI"),
        (NameKind::Preset, "zerodha"),
        (NameKind::Rounding, "contract_note or calculator"),
    ];
    for (kind, advice) in rows {
        let message = unknown(kind).to_string();
        assert!(
            message.contains("\"x\"") && message.contains(advice),
            "{message}"
        );
    }
}

#[test]
fn schedule_date_messages_show_iso_dates() {
    let message = CostError::NoScheduleForDay {
        day: 0,
        first_valid_from: 10,
    }
    .to_string();
    assert!(
        message.contains("1970-01-01") && message.contains("1970-01-11"),
        "{message}"
    );
}

#[test]
fn the_unsupported_segment_message_says_what_is_out_of_scope() {
    let message = CostError::UnsupportedSegment("MCX options".to_string()).to_string();
    assert!(
        message.contains("MCX options") && message.contains("out of scope"),
        "{message}"
    );
}

#[test]
fn an_extreme_day_is_reported_without_panicking() {
    for day in [i32::MIN, i32::MAX] {
        let message = CostError::UnsupportedDay { day }.to_string();
        assert!(message.contains(&day.to_string()), "{message}");
    }
}

#[test]
fn a_data_file_error_names_the_line() {
    let error = CostError::InvalidScheduleData {
        line: 3,
        reason: "expected 17 columns".to_string(),
    };
    assert!(error.to_string().contains("line 3"));
}
