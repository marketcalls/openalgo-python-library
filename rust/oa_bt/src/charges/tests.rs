//! The charge book against the cost model and the worked example of the reference
//! note (buy 400 at 1,000.00, sell 400 at 1,100.00 on 2026-11-12).

use super::*;
use crate::exec::{InstrumentId, InstrumentSpec};
use crate::market::costs::Brokerage;
use crate::market::costs::{CostModel, Rounding, ScheduleBook};
use crate::model::{days_from_civil, Day, Exchange, InstrumentKind, Product, Side};
use crate::portfolio::{FillEffect, FillKind, PositionChange, PositionKey};

fn sbin() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "SBIN".to_string(),
        exchange: Exchange::Nse,
        kind: InstrumentKind::Equity,
        lot_size_units: 1,
        tick_size_rupees: 0.05,
        multiplier: 1.0,
        freeze_quantity_units: None,
        cost_segment: None,
        last_trading_day: None,
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

fn opening() -> FillEffect {
    effect(FillKind::Entry, PositionChange::Opened)
}

fn closing() -> FillEffect {
    effect(FillKind::Reduce, PositionChange::Closed)
}

fn fill<'a>(
    spec: &'a InstrumentSpec,
    product: Product,
    side: Side,
    price: f64,
    on: Day,
    effect: FillEffect,
) -> FillToCharge<'a> {
    FillToCharge {
        fill: 0,
        spec,
        facts: None,
        key: PositionKey {
            instrument: InstrumentId(0),
            product,
        },
        side,
        units: 400.0,
        price_rupees: price,
        day: on,
        effect,
    }
}

fn charge(book: &mut ChargeBook, model: &CostModel, fill: &FillToCharge<'_>) -> f64 {
    match book.charge(model, fill) {
        Ok(rupees) => rupees,
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn a_fill_costs_what_the_cost_model_prices_it_at() {
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let buy = fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening());
    let expected = model.fill_cost(Segment::NseEqIntraday, Side::Buy, 1_000.0, 400.0, day());
    let expected = expected.map(|cost| cost.total()).unwrap_or(f64::NAN);
    assert_eq!(charge(&mut book, &model, &buy), expected);
}

#[test]
fn the_days_contract_note_charges_the_worked_example_total() {
    // Intraday round trip of the reference note: the note charges 200.62.
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let posted: f64 = [
        fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening()),
        fill(&spec, Product::Mis, Side::Sell, 1_100.0, day(), closing()),
    ]
    .iter()
    .map(|f| crate::portfolio::round_to_paise(charge(&mut book, &model, f)))
    .sum();
    let adjustments = book.settle_day(day());
    let total = posted + adjustments.iter().sum::<f64>();
    assert!((total - 200.62).abs() < 1e-9, "{total}");
    let output = book.into_output();
    assert_eq!(output.notes.len(), 1);
    assert_eq!(output.notes[0].note.charged_total, 200.62);
}

#[test]
fn a_delivery_round_trip_on_two_days_costs_the_round_trip_plus_the_dp_charge() {
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let next = day() + 1;
    let paid = charge(
        &mut book,
        &model,
        &fill(&spec, Product::Cnc, Side::Buy, 1_000.0, day(), opening()),
    ) + charge(
        &mut book,
        &model,
        &fill(&spec, Product::Cnc, Side::Sell, 1_100.0, next, closing()),
    );
    let trip = model.round_trip_cost(Segment::NseEqDelivery, 1_000.0, 1_100.0, 400.0, day(), next);
    let trip = trip.map(|cost| cost.total()).unwrap_or(f64::NAN);
    // Zerodha's DP charge: Rs 13 plus 18 percent GST, 15.34.
    assert!((paid - (trip + 15.34)).abs() < 1e-9, "{paid} vs {trip}");
}

#[test]
fn the_dp_charge_is_paid_once_per_scrip_per_sell_day() {
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let reduce = effect(FillKind::Reduce, PositionChange::Changed);
    charge(
        &mut book,
        &model,
        &fill(&spec, Product::Cnc, Side::Buy, 1_000.0, day(), opening()),
    );
    let next = day() + 1;
    charge(
        &mut book,
        &model,
        &fill(&spec, Product::Cnc, Side::Sell, 1_100.0, next, reduce),
    );
    charge(
        &mut book,
        &model,
        &fill(&spec, Product::Cnc, Side::Sell, 1_100.0, next, closing()),
    );
    let dp: Vec<bool> = book
        .into_output()
        .fills
        .iter()
        .map(|f| f.dp_charge.is_some())
        .collect();
    assert_eq!(dp, vec![false, true, false]);
}

#[test]
fn a_delivery_buy_and_sell_on_one_day_is_not_built() {
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    charge(
        &mut book,
        &model,
        &fill(&spec, Product::Cnc, Side::Buy, 1_000.0, day(), opening()),
    );
    let sell = fill(&spec, Product::Cnc, Side::Sell, 1_100.0, day(), closing());
    assert!(matches!(
        book.charge(&model, &sell),
        Err(ChargeError::NotBuilt { .. })
    ));
}

#[test]
fn calculator_rounding_charges_each_fill_and_settles_no_note() {
    let (model, spec) = (model(Rounding::Calculator), sbin());
    let mut book = ChargeBook::new(Some(Rounding::Calculator));
    let paid = charge(
        &mut book,
        &model,
        &fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening()),
    );
    let expected = model
        .fill_cost(Segment::NseEqIntraday, Side::Buy, 1_000.0, 400.0, day())
        .map(|cost| model.reported_total(&cost))
        .unwrap_or(f64::NAN);
    assert_eq!(paid, expected);
    assert!(book.settle_day(day()).is_empty());
}

#[test]
fn fills_carry_the_number_of_their_trade_once_it_closes() {
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    charge(
        &mut book,
        &model,
        &fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening()),
    );
    charge(
        &mut book,
        &model,
        &fill(&spec, Product::Mis, Side::Sell, 1_100.0, day(), closing()),
    );
    charge(
        &mut book,
        &model,
        &fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening()),
    );
    let trades: Vec<Option<u32>> = book.into_output().fills.iter().map(|f| f.trade).collect();
    assert_eq!(trades, vec![Some(1), Some(1), None]);
}

#[test]
fn a_note_whose_session_did_not_close_is_left_unsettled() {
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    charge(
        &mut book,
        &model,
        &fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening()),
    );
    let notes = book.into_output().notes;
    assert_eq!((notes.len(), notes[0].settled), (1, false));
}

#[test]
fn a_bse_equity_without_its_scrip_group_cannot_be_charged() {
    let model = model(Rounding::ContractNote);
    let spec = InstrumentSpec {
        exchange: Exchange::Bse,
        ..sbin()
    };
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let buy = fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening());
    assert!(matches!(
        book.charge(&model, &buy),
        Err(ChargeError::Cost(_))
    ));
}

#[test]
fn a_used_row_is_recorded_once() {
    let mut book = ChargeBook::new(None);
    let rows = crate::market::slippage::slippage_defaults().unwrap_or_default();
    book.record_use(MarketTable::Slippage, &rows[0].meta, day());
    book.record_use(MarketTable::Slippage, &rows[0].meta, day() + 1);
    let tables = book.into_output().tables;
    assert_eq!((tables.len(), tables[0].first_day), (1, day()));
}

#[test]
fn estimates_price_the_entry_and_the_round_trip() {
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let entry = estimate_entry(
        &model,
        &spec,
        None,
        Product::Mis,
        Side::Buy,
        400.0,
        1_000.0,
        day(),
    );
    let trip = estimate_round_trip(&model, &spec, None, Product::Mis, 400.0, 1_000.0, day());
    let buy = model.fill_cost(Segment::NseEqIntraday, Side::Buy, 1_000.0, 400.0, day());
    let both = model.round_trip_cost(
        Segment::NseEqIntraday,
        1_000.0,
        1_000.0,
        400.0,
        day(),
        day(),
    );
    assert_eq!(entry, Ok(buy.map(|c| c.total()).unwrap_or(f64::NAN)));
    assert_eq!(trip, Ok(both.map(|c| c.total()).unwrap_or(f64::NAN)));
    let none = estimate_entry(
        &model,
        &spec,
        None,
        Product::Mis,
        Side::Buy,
        0.0,
        1_000.0,
        day(),
    );
    assert_eq!(none, Ok(0.0));
}
