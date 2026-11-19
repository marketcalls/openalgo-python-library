//! A fill is charged on its traded value, units x multiplier x price, whatever the
//! contract's multiplier: a whole fill, the parts of an order and the sizers' estimates.
//!
//! The contracts carry the terms of `mcx_contracts.csv`: GOLDM, a 100-gram lot of 100
//! units quoted per 10 grams (multiplier 0.1); GOLD, a 1 kg lot of 1 unit quoted per 10
//! grams (multiplier 100); ZINC, a 5-tonne lot of 5 units quoted per kg (multiplier
//! 1,000); CRUDEOILM, a 10-barrel lot of 10 units quoted per barrel (multiplier 1).
//!
//! Every expected figure is worked by hand from the MCX futures (non-agri) rates of the
//! cost schedule IN-2026-04-01 on 2026-11-12: brokerage 0.03 percent capped at Rs 20 an
//! order (the Zerodha preset), CTT 0.01 percent on sells, exchange transaction charge
//! 0.0021 percent (MCX/F&A/631/2024), SEBI fee Rs 10 a crore (0.0001 percent), stamp duty
//! 0.002 percent on buys, GST 18 percent of brokerage, exchange charge and SEBI fee. The
//! day's MCX contract note rounds CTT and stamp duty to the rupee and the other lines to
//! the paisa.

use super::*;
use crate::exec::{InstrumentId, InstrumentSpec};
use crate::market::costs::{Brokerage, CostBreakdown, CostModel, Rounding, ScheduleBook};
use crate::market::facts::InstrumentMarket;
use crate::model::{days_from_civil, Day, Exchange, InstrumentKind, Product, Side};
use crate::portfolio::{FillEffect, FillKind, PositionChange, PositionKey};

/// The float error allowed on an unrounded accrual, rupees.
const ACCRUAL_TOLERANCE: f64 = 1e-9;

/// One MCX futures contract: its symbol, commodity, lot in units, tick in rupees and
/// multiplier.
struct Contract {
    symbol: &'static str,
    commodity: &'static str,
    lot_units: u32,
    tick_rupees: f64,
    multiplier: f64,
}

const GOLDM: Contract = Contract {
    symbol: "GOLDM05DEC26FUT",
    commodity: "GOLDM",
    lot_units: 100,
    tick_rupees: 1.0,
    multiplier: 0.1,
};
const GOLD: Contract = Contract {
    symbol: "GOLD05DEC26FUT",
    commodity: "GOLD",
    lot_units: 1,
    tick_rupees: 1.0,
    multiplier: 100.0,
};
const ZINC: Contract = Contract {
    symbol: "ZINC30NOV26FUT",
    commodity: "ZINC",
    lot_units: 5,
    tick_rupees: 0.05,
    multiplier: 1_000.0,
};
const CRUDEOILM: Contract = Contract {
    symbol: "CRUDEOILM17DEC26FUT",
    commodity: "CRUDEOILM",
    lot_units: 10,
    tick_rupees: 1.0,
    multiplier: 1.0,
};

impl Contract {
    fn spec(&self) -> InstrumentSpec {
        InstrumentSpec {
            symbol: self.symbol.to_string(),
            exchange: Exchange::Mcx,
            kind: InstrumentKind::Future,
            lot_size_units: self.lot_units,
            tick_size_rupees: self.tick_rupees,
            multiplier: self.multiplier,
            freeze_quantity_units: None,
            cost_segment: None,
            last_trading_day: None,
        }
    }

    fn facts(&self) -> InstrumentMarket {
        InstrumentMarket {
            underlying: Some(self.commodity.to_string()),
            ..InstrumentMarket::default()
        }
    }
}

fn model(rounding: Rounding) -> CostModel {
    match ScheduleBook::builtin() {
        Ok(book) => CostModel::new(book, Brokerage::zerodha(), rounding),
        Err(error) => panic!("{error}"),
    }
}

fn day() -> Day {
    days_from_civil(2026, 11, 12)
}

fn effect(kind: FillKind, change: PositionChange) -> FillEffect {
    FillEffect {
        kind,
        change,
        realized_rupees: 0.0,
        quantity_after: 0.0,
    }
}

/// A fill of one lot of the contract on 2026-11-12, NRML.
fn lot_fill<'a>(
    spec: &'a InstrumentSpec,
    facts: &'a InstrumentMarket,
    side: Side,
    price_rupees: f64,
    effect: FillEffect,
) -> FillToCharge<'a> {
    FillToCharge {
        fill: 0,
        spec,
        facts: Some(facts),
        key: PositionKey {
            instrument: InstrumentId(0),
            product: Product::Nrml,
        },
        side,
        units: f64::from(spec.lot_size_units),
        price_rupees,
        day: day(),
        effect,
    }
}

/// One lot bought at `buy` and sold at `sell` on 2026-11-12, the day's note settled.
fn round_trip(contract: &Contract, buy: f64, sell: f64) -> ChargesOutput {
    let (spec, facts) = (contract.spec(), contract.facts());
    let model = model(Rounding::ContractNote);
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    for fill in [
        lot_fill(
            &spec,
            &facts,
            Side::Buy,
            buy,
            effect(FillKind::Entry, PositionChange::Opened),
        ),
        lot_fill(
            &spec,
            &facts,
            Side::Sell,
            sell,
            effect(FillKind::Reduce, PositionChange::Closed),
        ),
    ] {
        if let Err(error) = book.charge(&model, &fill) {
            panic!("{error}");
        }
    }
    book.settle_day(day());
    book.into_output()
}

fn assert_lines(actual: &CostBreakdown, expected: [f64; 6], tolerance: f64) {
    let actual = actual.components();
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "line {index}: {got} vs {want}"
        );
    }
}

#[test]
fn a_goldm_buy_is_charged_on_its_traded_value_not_on_its_units() {
    // 100 units at 1,20,000.00 per 10 grams with multiplier 0.1: 100 grams worth
    // 12,00,000.00. Brokerage 0.03 percent is 360.00, capped at 20.00; exchange charge
    // 25.20; SEBI fee 1.20; stamp duty 24.00; GST 18 percent of 46.40 is 8.352; total
    // 78.752. On its units as the value (1,20,00,000.00) it would be 575.12.
    let output = round_trip(&GOLDM, 120_000.0, 121_000.0);
    let buy = &output.fills[0];
    // brokerage, CTT, exchange charge, SEBI fee, stamp duty, GST
    assert_lines(
        &buy.cost,
        [20.0, 0.0, 25.20, 1.20, 24.0, 8.352],
        ACCRUAL_TOLERANCE,
    );
    assert!(
        (buy.cost_rupees - 78.752).abs() <= ACCRUAL_TOLERANCE,
        "{}",
        buy.cost_rupees
    );
}

#[test]
fn a_goldm_round_trips_contract_note_rounds_the_charges_on_its_traded_value() {
    // Sell of 100 units at 1,21,000.00, worth 12,10,000.00: brokerage 20.00, CTT 121.00,
    // exchange charge 25.41, SEBI fee 1.21, GST 18 percent of 46.62 is 8.3916. With the
    // buy above the note's lines are brokerage 40.00, CTT 121, exchange charge 50.61,
    // SEBI fee 2.41, stamp duty 24 and GST 16.7436 rounded to 16.74: 254.76.
    let output = round_trip(&GOLDM, 120_000.0, 121_000.0);
    let note = &output.notes[0].note;
    assert_lines(&note.charged, [40.0, 121.0, 50.61, 2.41, 24.0, 16.74], 0.0);
    assert_eq!(note.charged_total, 254.76);
}

#[test]
fn every_contracts_round_trip_is_charged_on_its_traded_value() {
    // One lot bought and sold the same day. Note lines: brokerage, CTT, exchange charge,
    // SEBI fee, stamp duty, GST.
    // - GOLDM: worked in the test above, 254.76.
    // - GOLD, 1 unit at 1,20,000.00 then 1,21,000.00 with multiplier 100: worth
    //   1,20,00,000.00 and 1,21,00,000.00. Exchange charge 252.00 + 254.10, SEBI fee
    //   12.00 + 12.10, stamp duty 240.00, CTT 1,210.00, GST 18 percent of 284.00 and of
    //   286.20 is 51.12 + 51.516 = 102.636: 2,122.84.
    // - ZINC, 5 units at 280.00 then 282.50 per kg with multiplier 1,000: worth
    //   14,00,000.00 and 14,12,500.00. Exchange charge 29.40 + 29.6625 = 59.0625, SEBI
    //   fee 1.40 + 1.4125 = 2.8125, stamp duty 28.00, CTT 141.25, GST 18 percent of 50.80
    //   and of 51.075 is 9.144 + 9.1935 = 18.3375: 289.21.
    let rows = [
        (
            &GOLDM,
            120_000.0,
            121_000.0,
            [40.0, 121.0, 50.61, 2.41, 24.0, 16.74],
            254.76,
        ),
        (
            &GOLD,
            120_000.0,
            121_000.0,
            [40.0, 1_210.0, 506.10, 24.10, 240.0, 102.64],
            2_122.84,
        ),
        (
            &ZINC,
            280.0,
            282.5,
            [40.0, 141.0, 59.06, 2.81, 28.0, 18.34],
            289.21,
        ),
    ];
    for (contract, buy, sell, lines, total) in rows {
        let output = round_trip(contract, buy, sell);
        let note = &output.notes[0].note;
        assert_lines(&note.charged, lines, 0.0);
        assert_eq!(note.charged_total, total, "{}", contract.symbol);
        let paid: f64 = output.fills.iter().map(FillCharge::paid_rupees).sum();
        assert!((paid - total).abs() < 1e-9, "{}: {paid}", contract.symbol);
    }
}

#[test]
fn a_contract_with_multiplier_one_is_charged_as_before() {
    // CRUDEOILM, 10 units at 5,400.00 then 5,450.00 per barrel: worth 54,000.00 and
    // 54,500.00, below Rs 66,667, so the brokerage is 0.03 percent, 16.20 + 16.35.
    // Exchange charge 1.134 + 1.1445 = 2.2785, SEBI fee 0.054 + 0.0545 = 0.1085, stamp
    // duty 1.08 rounded to 1, CTT 5.45 rounded to 5, GST 18 percent of 17.388 and of
    // 17.549 is 3.12984 + 3.15882 = 6.28866: 47.23.
    let output = round_trip(&CRUDEOILM, 5_400.0, 5_450.0);
    let note = &output.notes[0].note;
    assert_lines(&note.charged, [32.55, 5.0, 2.28, 0.11, 1.0, 6.29], 0.0);
    assert_eq!(note.charged_total, 47.23);
}

#[test]
fn an_order_filled_in_parts_is_charged_on_its_whole_traded_value() {
    // A GOLDM buy of two lots filled 100 units at 1,20,000.00 and 100 at 1,20,010.00:
    // worth 12,00,000.00 + 12,00,100.00 = 24,00,100.00, costed once. Brokerage 20.00
    // (one order), exchange charge 50.4021, SEBI fee 2.4001, stamp duty 48.002, GST 18
    // percent of 72.8022 is 13.104396: 133.908596, posted as 133.91. The first part posts
    // 78.75 (its own 78.752) and the second the rest, 55.16.
    let (spec, facts) = (GOLDM.spec(), GOLDM.facts());
    let model = model(Rounding::ContractNote);
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let mut so_far = OrderCharges::default();
    let parts = [
        (120_000.0, effect(FillKind::Entry, PositionChange::Opened)),
        (120_010.0, effect(FillKind::Add, PositionChange::Changed)),
    ];
    let mut posted = Vec::new();
    for (price, effect) in parts {
        let fill = lot_fill(&spec, &facts, Side::Buy, price, effect);
        match book.charge_part(&model, &fill, &mut so_far) {
            Ok(rupees) => posted.push(rupees),
            Err(error) => panic!("{error}"),
        }
    }
    assert_eq!(posted, vec![78.75, 55.16]);
    let output = book.into_output();
    let accrued: f64 = output.fills.iter().map(|f| f.cost_rupees).sum();
    assert!(
        (accrued - 133.908596).abs() <= ACCRUAL_TOLERANCE,
        "{accrued}"
    );
    assert!((so_far.value_rupees - 2_400_100.0).abs() <= ACCRUAL_TOLERANCE);
}

#[test]
fn the_sizers_estimates_price_the_traded_value() {
    // One GOLDM lot at 1,20,000.00, worth 12,00,000.00. The entry is the buy of the first
    // test, 78.752. The round trip adds a sale at the same price: brokerage 20.00, CTT
    // 120.00, exchange charge 25.20, SEBI fee 1.20, GST 8.352, so 174.752 and 253.504 in
    // all.
    let (spec, facts) = (GOLDM.spec(), GOLDM.facts());
    let model = model(Rounding::ContractNote);
    let units = f64::from(GOLDM.lot_units);
    let entry = estimate_entry(
        &model,
        &spec,
        Some(&facts),
        Product::Nrml,
        Side::Buy,
        units,
        120_000.0,
        day(),
    );
    let trip = estimate_round_trip(
        &model,
        &spec,
        Some(&facts),
        Product::Nrml,
        units,
        120_000.0,
        day(),
    );
    let (entry, trip) = match (entry, trip) {
        (Ok(entry), Ok(trip)) => (entry, trip),
        (entry, trip) => panic!("{entry:?} {trip:?}"),
    };
    assert!((entry - 78.752).abs() <= ACCRUAL_TOLERANCE, "{entry}");
    assert!((trip - 253.504).abs() <= ACCRUAL_TOLERANCE, "{trip}");
}

#[test]
fn calculator_rounding_also_charges_the_traded_value() {
    // The GOLDM buy of the first test as the calculator rounds a one-sided trade:
    // brokerage 20.00, exchange charge 25.20, SEBI fee 1.20, GST 8.352 rounded to 8.35,
    // stamp duty 24: 78.75.
    let (spec, facts) = (GOLDM.spec(), GOLDM.facts());
    let model = model(Rounding::Calculator);
    let mut book = ChargeBook::new(Some(Rounding::Calculator));
    let buy = lot_fill(
        &spec,
        &facts,
        Side::Buy,
        120_000.0,
        effect(FillKind::Entry, PositionChange::Opened),
    );
    let posted = match book.charge(&model, &buy) {
        Ok(rupees) => rupees,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(posted, 78.75);
}
