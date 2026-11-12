#![cfg(feature = "time")]
//! The futures rows of the dated margin table (04-execution 5) against the pages they
//! were read from: NSE stock futures and MCX futures (`fixtures/futures_margin_pages.csv`,
//! one contract per row with the page's rupee margin), plus the BSE and NSE index rows
//! whose pages gave a percent but no contract to check against, worked by hand.

use oa_bt::market::margin::{margin_defaults, MarginModel, MarginRequest};
use oa_bt::model::{days_from_civil, Exchange, InstrumentKind, Product, Side};

const PAGES: &str = include_str!("fixtures/futures_margin_pages.csv");

/// A two-decimal percent is off by at most half a hundredth of a percent of the value.
const PERCENT_ROUNDING: f64 = 0.005 / 100.0;
/// The page shows whole rupees.
const PAGE_ROUNDING_RUPEES: f64 = 1.0;

struct PageContract {
    exchange: Exchange,
    underlying: String,
    quantity_units: f64,
    price_rupees: f64,
    multiplier: f64,
    page_margin_rupees: f64,
}

fn number(text: &str) -> f64 {
    text.parse().unwrap_or(f64::NAN)
}

fn page_contracts() -> Vec<PageContract> {
    PAGES
        .lines()
        .filter(|line| !line.starts_with('#') && !line.starts_with("exchange,"))
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            PageContract {
                exchange: Exchange::from_code(fields[0]).unwrap_or(Exchange::Nfo),
                underlying: fields[1].to_string(),
                quantity_units: number(fields[2]),
                price_rupees: number(fields[3]),
                multiplier: number(fields[4]),
                page_margin_rupees: number(fields[5]),
            }
        })
        .collect()
}

fn margin(exchange: Exchange, underlying: &str, units: f64, price: f64, multiplier: f64) -> f64 {
    let request = MarginRequest {
        exchange,
        kind: InstrumentKind::Future,
        product: Product::Nrml,
        side: Side::Buy,
        quantity_units: units,
        price_rupees: price,
        multiplier,
        leverage: 1.0,
        day: days_from_civil(2026, 11, 12),
        underlying: Some(underlying),
    };
    match MarginModel.initial_margin(&request) {
        Ok(quote) => quote.rupees,
        Err(error) => panic!("{underlying}: {error}"),
    }
}

#[test]
fn every_page_row_reproduces_the_pages_rupee_margin() {
    let contracts = page_contracts();
    assert_eq!(
        contracts.len(),
        248,
        "218 NSE stock futures and 30 MCX futures"
    );
    for contract in contracts {
        let value = contract.quantity_units * contract.price_rupees * contract.multiplier;
        let rupees = margin(
            contract.exchange,
            &contract.underlying,
            contract.quantity_units,
            contract.price_rupees,
            contract.multiplier,
        );
        let tolerance = value * PERCENT_ROUNDING + PAGE_ROUNDING_RUPEES;
        assert!(
            (rupees - contract.page_margin_rupees).abs() <= tolerance,
            "{}: the table gives {rupees}, the page {}",
            contract.underlying,
            contract.page_margin_rupees
        );
    }
}

#[test]
fn a_bankex_future_blocks_eleven_point_two_three_percent() {
    // 30 units at 60,000.00 are 18,00,000; 11.23 percent of it is 2,02,140.00.
    let rupees = margin(Exchange::Bfo, "BANKEX", 30.0, 60_000.0, 1.0);
    assert!((rupees - 202_140.0).abs() < 1e-6, "{rupees}");
}

#[test]
fn a_niftyfpi_future_blocks_nine_point_three_six_percent() {
    // 50 units at 20,000.00 are 10,00,000; 9.36 percent of it is 93,600.00.
    let rupees = margin(Exchange::Nfo, "NIFTYFPI", 50.0, 20_000.0, 1.0);
    assert!((rupees - 93_600.0).abs() < 1e-6, "{rupees}");
}

#[test]
fn a_crude_oil_lot_blocks_its_rows_percent_of_contract_value() {
    // One lot of 100 barrels at 8,184.00 is 8,18,400; 31.25 percent of it is 2,55,750.
    let rupees = margin(Exchange::Mcx, "CRUDEOIL", 100.0, 8_184.0, 1.0);
    assert!((rupees - 255_750.0).abs() < 1e-6, "{rupees}");
}

#[test]
fn a_gold_lot_counts_its_multiplier() {
    // One lot of 1 kg quoted per 10 g at 1,39,716.00: value 1 x 139,716 x 100 =
    // 1,39,71,600; 9.25 percent of it is 12,92,373.00.
    let rupees = margin(Exchange::Mcx, "GOLD", 1.0, 139_716.0, 100.0);
    assert!((rupees - 1_292_373.0).abs() < 1e-6, "{rupees}");
}

#[test]
fn sensex_futures_have_no_row_and_are_not_free() {
    let request = MarginRequest {
        exchange: Exchange::Bfo,
        kind: InstrumentKind::Future,
        product: Product::Nrml,
        side: Side::Buy,
        quantity_units: 20.0,
        price_rupees: 80_000.0,
        multiplier: 1.0,
        leverage: 1.0,
        day: days_from_civil(2026, 11, 12),
        underlying: Some("SENSEX"),
    };
    assert!(MarginModel.initial_margin(&request).is_err());
}

#[test]
fn every_futures_row_names_its_source_and_is_unverified() {
    let rows = margin_defaults().map_err(|e| e.to_string());
    let rows = rows.as_deref().unwrap_or_default();
    let futures: Vec<_> = rows
        .iter()
        .filter(|row| row.kind == InstrumentKind::Future)
        .collect();
    assert_eq!(
        futures.len(),
        255,
        "5 NSE index rows, then NIFTYFPI, 218 stocks, 30 MCX, BANKEX"
    );
    for row in futures {
        assert!(!row.meta.source.is_empty(), "{}", row.meta.id);
        assert_eq!(
            row.meta.status,
            oa_bt::market::table_rows::RowStatus::Unverified,
            "{}",
            row.meta.id
        );
    }
}
