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
        positional_exit_day: None,
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

/// Half a paisa per rounded amount, and the float error of adding them.
const PAISA_ROUNDING: f64 = 0.005 + 1e-9;
/// What the lines rounded to the paisa can move a share: a few paise at most.
const FEW_PAISE: f64 = 0.03;

#[test]
fn a_fill_posts_what_the_cost_model_prices_it_at_rounded_to_the_paisa() {
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let buy = fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening());
    let expected = model.fill_cost(Segment::NseEqIntraday, Side::Buy, 1_000.0, 400.0, day());
    let expected = expected.map(|cost| cost.total()).unwrap_or(f64::NAN);
    let posted = charge(&mut book, &model, &buy);
    assert_eq!(posted, crate::portfolio::round_to_paise(posted));
    assert!(
        (posted - expected).abs() < PAISA_ROUNDING,
        "{posted} vs {expected}"
    );
}

#[test]
fn the_fills_of_a_settled_note_pay_what_the_note_charges() {
    // The intraday round trip of the reference note: 200.62 shared over its buy and
    // its sale, each a whole number of paise.
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    for trade in [
        fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening()),
        fill(&spec, Product::Mis, Side::Sell, 1_100.0, day(), closing()),
    ] {
        charge(&mut book, &model, &trade);
    }
    book.settle_day(day());
    let output = book.into_output();
    let paid: Vec<f64> = output.fills.iter().map(FillCharge::paid_rupees).collect();
    assert!(paid
        .iter()
        .all(|&rupees| rupees == crate::portfolio::round_to_paise(rupees)));
    assert!((paid.iter().sum::<f64>() - 200.62).abs() < 1e-9, "{paid:?}");
}

#[test]
fn each_line_rounding_of_an_intraday_note_goes_to_the_fill_that_paid_the_charge() {
    // Intraday STT is charged on the sale only and stamp duty on the buy only. The
    // sale's STT, 0.025 percent of 4,40,520, is Rs 110.13 and the note charges Rs 110.
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    for trade in [
        fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening()),
        fill(&spec, Product::Mis, Side::Sell, 1_101.3, day(), closing()),
    ] {
        charge(&mut book, &model, &trade);
    }
    book.settle_day(day());
    let output = book.into_output();
    let (buy, sale, note) = (&output.fills[0], &output.fills[1], &output.notes[0].note);
    assert_eq!(buy.cost.stt, 0.0);
    assert_eq!(sale.cost.stamp_duty, 0.0);
    // The buy alone paid stamp duty and the sale alone STT: each takes its line's
    // rounding, give or take the paise of the lines rounded to the paisa.
    let expected_buy = buy.cost_rupees + note.charged.stamp_duty - buy.cost.stamp_duty;
    let expected_sale = sale.cost_rupees + note.charged.stt - sale.cost.stt;
    for (record, expected) in [(buy, expected_buy), (sale, expected_sale)] {
        let off = (record.charged_rupees - expected).abs();
        assert!(off < FEW_PAISE, "{} vs {expected}", record.charged_rupees);
    }
    assert!(
        (note.charged.stt - sale.cost.stt).abs() > FEW_PAISE,
        "the STT line was rounded"
    );
}

#[test]
fn a_fill_of_a_note_not_settled_is_charged_its_cost_rounded_to_the_paisa() {
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let buy = fill(&spec, Product::Mis, Side::Buy, 1_000.0, day(), opening());
    charge(&mut book, &model, &buy);
    let output = book.into_output();
    let record = &output.fills[0];
    assert!(!output.notes[0].settled);
    assert_eq!(
        record.charged_rupees,
        crate::portfolio::round_to_paise(record.cost_rupees)
    );
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
    // Zerodha's DP charge: Rs 13 plus 18 percent GST, 15.34. Each fill posts its cost
    // rounded to the paisa.
    let rounding = 2.0 * PAISA_ROUNDING;
    assert!((paid - (trip + 15.34)).abs() < rounding, "{paid} vs {trip}");
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
fn a_delivery_buy_and_sell_on_one_day_is_charged_as_the_intraday_round_trip() {
    // The worked example on CNC: buy 400 at 1,000.00 and sell them at 1,100.00 the same
    // day. Brokers charge it as intraday: the note charges the intraday 200.62, and no
    // DP charge is paid (04-execution 2).
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let posted: f64 = [
        fill(&spec, Product::Cnc, Side::Buy, 1_000.0, day(), opening()),
        fill(&spec, Product::Cnc, Side::Sell, 1_100.0, day(), closing()),
    ]
    .iter()
    .map(|f| crate::portfolio::round_to_paise(charge(&mut book, &model, f)))
    .sum();
    let adjustments = book.settle_day(day());
    let total = posted + adjustments.iter().sum::<f64>();
    assert!((total - 200.62).abs() < 1e-9, "{total}");
    let output = book.into_output();
    let segments: Vec<Segment> = output.fills.iter().map(|f| f.segment).collect();
    assert_eq!(
        segments,
        vec![Segment::NseEqIntraday, Segment::NseEqIntraday]
    );
    assert!(output.fills.iter().all(|f| f.dp_charge.is_none()));
    let noted: f64 = output.notes.iter().map(|n| n.note.charged_total).sum();
    assert!((noted - 200.62).abs() < 1e-9, "{noted}");
}

#[test]
fn a_sale_from_holdings_bought_back_the_same_day_refunds_its_dp_charge() {
    // Sell 400 held from before at 1,100.00, then buy 400 back at 1,000.00 the same day:
    // the day nets to an intraday round trip, so the DP charge paid on the sale is
    // refunded with the buy.
    let (model, spec) = (model(Rounding::ContractNote), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let reduce = effect(FillKind::Reduce, PositionChange::Changed);
    let sell = fill(&spec, Product::Cnc, Side::Sell, 1_100.0, day(), reduce);
    let add = effect(FillKind::Add, PositionChange::Changed);
    let buy = fill(&spec, Product::Cnc, Side::Buy, 1_000.0, day(), add);
    let dp = model
        .dp_charge(day())
        .map(|dp| dp.total())
        .unwrap_or(f64::NAN);
    let on_sale = charge(&mut book, &model, &sell);
    let on_buy = charge(&mut book, &model, &buy);
    let intraday = model
        .round_trip_cost(
            Segment::NseEqIntraday,
            1_000.0,
            1_100.0,
            400.0,
            day(),
            day(),
        )
        .map(|cost| cost.total())
        .unwrap_or(f64::NAN);
    // Each posting is a whole number of paise.
    assert!(
        (on_sale + on_buy - intraday).abs() < 2.0 * PAISA_ROUNDING,
        "{on_sale} + {on_buy}"
    );
    assert!(on_sale > dp, "the sale paid the DP charge first");
    assert!(book
        .into_output()
        .fills
        .iter()
        .all(|f| f.dp_charge.is_none()));
}

fn standard_model() -> CostModel {
    match ScheduleBook::builtin() {
        Ok(book) => CostModel::new(book, Brokerage::standard(), Rounding::ContractNote),
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn a_cnc_buy_partly_sold_the_same_day_pays_the_brokerage_of_one_order() {
    // Standard preset: buy 400 at 1,000.00 on CNC and sell 200 at 1,000.00 the same day.
    // The buy is then 200 units intraday and 200 delivery, still one executed order: it
    // pays Rs 20 (not Rs 20 a part), and GST is 18 percent of brokerage 20 + exchange
    // 12.28 (0.00297 percent plus IPFT 0.0001 percent of 4,00,000) + SEBI 0.40 = 5.8824.
    let (model, spec) = (standard_model(), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let reduce = effect(FillKind::Reduce, PositionChange::Changed);
    let buy = fill(&spec, Product::Cnc, Side::Buy, 1_000.0, day(), opening());
    let sell = FillToCharge {
        units: 200.0,
        ..fill(&spec, Product::Cnc, Side::Sell, 1_000.0, day(), reduce)
    };
    charge(&mut book, &model, &buy);
    charge(&mut book, &model, &sell);
    let output = book.into_output();
    let (buy, sale) = (&output.fills[0].cost, &output.fills[1].cost);
    assert_eq!((buy.brokerage, sale.brokerage), (20.0, 20.0));
    assert!((buy.gst - 5.8824).abs() < 1e-9, "{}", buy.gst);
}

#[test]
fn a_sale_from_holdings_partly_netted_by_a_buy_of_the_day_pays_the_brokerage_of_one_order() {
    // Standard preset, shares held from before: buy 100 at 1,000.00 on CNC, then sell 400
    // at 1,000.00 the same day. The sale is 100 units intraday and 300 from holdings,
    // one executed order: Rs 20.
    let (model, spec) = (standard_model(), sbin());
    let mut book = ChargeBook::new(Some(Rounding::ContractNote));
    let add = effect(FillKind::Add, PositionChange::Changed);
    let reduce = effect(FillKind::Reduce, PositionChange::Changed);
    let buy = FillToCharge {
        units: 100.0,
        ..fill(&spec, Product::Cnc, Side::Buy, 1_000.0, day(), add)
    };
    charge(&mut book, &model, &buy);
    charge(
        &mut book,
        &model,
        &fill(&spec, Product::Cnc, Side::Sell, 1_000.0, day(), reduce),
    );
    assert_eq!(book.into_output().fills[1].cost.brokerage, 20.0);
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
