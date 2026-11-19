#![cfg(all(feature = "costs", feature = "time"))]
//! The option buffer against the exchanges' price bands (16-session-design unit M1, open
//! decision 8). Every limit the engine prices for an option lies
//! `OPTIONS_STOP_LIMIT_BUFFER_PERCENT` beyond a price, rounded to the tick away from the
//! trader: a leg's stop-limit beyond its trigger (`exec::segment::venue_order_type`), a GTT
//! child beyond its trigger (`exec::gtt::child_limit`), and the reprice of a stop that the
//! price has passed beyond the last price (16-session-design 4.6). NSE and BSE reject an
//! option limit beyond a band around a basis price; the bands, with their sources, are the
//! rows of the reference table `price_protection.csv`, read here as shipped.
//!
//! Each band is checked where the order meets it, with its basis at the price the limit
//! was priced from: a stop-limit meets limit price protection when it triggers, with the
//! market at its trigger, and the stop-limit range at entry, against its own trigger; a
//! GTT child and a reprice are plain limits, met by the bands measured from a market
//! price. How far that market price may drift from the order's price before the limit
//! leaves a band is stated in 04-execution 4.1, not tested here.
//!
//! One exception: a band with no rupee minimum (BSE stock options during a cooling-off)
//! is narrower than one tick at a premium of three ticks or less, where no limit off the
//! premium fits inside it whatever the buffer. Those premiums are left out of the check.

use oa_bt::exec::gtt::child_limit;
use oa_bt::exec::price::{is_on_tick, limit_beyond, PRICE_UNITS_PER_RUPEE};
use oa_bt::exec::segment::{venue_order_type, OPTIONS_STOP_LIMIT_BUFFER_PERCENT};
use oa_bt::exec::{GttPolicy, OrderType};
use oa_bt::model::{InstrumentKind, Side};

/// The reference table, as the Python package ships it.
const PRICE_PROTECTION_CSV: &str =
    include_str!("../../../openalgo/backtest/reference/data/price_protection.csv");

/// The option ticks of `tick_sizes.csv`: 0.05 rupees for index options and for stock
/// options of a share that closed at 250 or more, 0.01 for NSE stock options of a share
/// that closed below 250 (from 2025-11-03).
const OPTION_TICKS_RUPEES: [f64; 2] = [0.05, 0.01];

/// Every premium on the grid is tried up to here, so both sides of each band's switch
/// from its rupee minimum to its percent (25 to 100 rupees in the table) are covered tick
/// by tick.
const DENSE_UP_TO_RUPEES: f64 = 200.0;

/// Above `DENSE_UP_TO_RUPEES` premiums go up about one percent at a time to here, beyond
/// the dearest option traded on either exchange.
const TOP_PREMIUM_RUPEES: f64 = 100_000.0;

/// Growth of the premium per step above `DENSE_UP_TO_RUPEES`.
const SPARSE_STEP: f64 = 1.01;

/// Half a price unit: absorbs the binary error of comparing rupee prices, nothing more.
const HALF_PRICE_UNIT_RUPEES: f64 = 0.5 / PRICE_UNITS_PER_RUPEE;

/// The basis of a band measured from the order's own trigger.
const TRIGGER_BASIS: &str = "trigger_price";

/// How many failures an assertion message lists.
const FAILURES_SHOWN: usize = 20;

/// The premiums at or below which a band with no rupee minimum may be narrower than one
/// tick, in ticks: the BSE cooling-off band (30 percent) is narrower than a tick up to
/// 3 ticks.
const NARROW_UP_TO_TICKS: f64 = 3.0;

/// One row of `price_protection.csv`.
struct Band {
    exchange: String,
    segment: String,
    check: String,
    underlying: String,
    days: String,
    percent: f64,
    /// None for a band with no rupee minimum (an empty cell).
    min_rupees: Option<f64>,
    basis: String,
}

impl Band {
    /// The band in rupees around `basis_rupees`: the larger of the percent and the minimum.
    fn rupees(&self, basis_rupees: f64) -> f64 {
        let percent = self.percent / 100.0 * basis_rupees;
        self.min_rupees
            .map_or(percent, |minimum| percent.max(minimum))
    }

    /// True when the band around `basis_rupees` is at least one tick wide, so a limit one
    /// tick off the basis can lie inside it.
    fn holds_a_tick(&self, basis_rupees: f64, tick_rupees: f64) -> bool {
        self.rupees(basis_rupees) + HALF_PRICE_UNIT_RUPEES >= tick_rupees
    }

    /// True when a limit of `side` at `limit_rupees` lies inside the band around
    /// `basis_rupees`: a buy no higher than the basis plus the band, a sell no lower than
    /// the basis less the band.
    fn admits(&self, side: Side, limit_rupees: f64, basis_rupees: f64) -> bool {
        let band = self.rupees(basis_rupees) + HALF_PRICE_UNIT_RUPEES;
        match side {
            Side::Buy => limit_rupees <= basis_rupees + band,
            Side::Sell => limit_rupees >= basis_rupees - band,
        }
    }

    /// True when the band checks every limit order, not only a stop-limit's distance from
    /// its trigger.
    fn checks_plain_limits(&self) -> bool {
        self.basis != TRIGGER_BASIS
    }

    /// The premium at which the band stops being its rupee minimum and becomes its percent
    /// (None without a minimum).
    fn switch_rupees(&self) -> Option<f64> {
        self.min_rupees
            .map(|minimum| minimum / (self.percent / 100.0))
    }

    fn name(&self) -> String {
        format!(
            "{} {} {} {} {}",
            self.exchange, self.segment, self.check, self.underlying, self.days
        )
    }
}

/// The rows of `price_protection.csv`. The comment lines and the header are skipped; the
/// columns are found by name.
fn bands() -> Vec<Band> {
    let mut lines = PRICE_PROTECTION_CSV
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("a header row").split(',').collect();
    let column = |name: &str| {
        header
            .iter()
            .position(|cell| *cell == name)
            .unwrap_or_else(|| panic!("the column {name}"))
    };
    let at = [
        column("exchange"),
        column("segment"),
        column("check"),
        column("underlying"),
        column("days"),
        column("band_percent"),
        column("band_min_rupees"),
        column("basis"),
    ];
    lines
        .map(|line| {
            assert!(
                !line.contains('"'),
                "a quoted cell this reader cannot split: {line}"
            );
            let cells: Vec<&str> = line.split(',').collect();
            assert_eq!(cells.len(), header.len(), "{line}");
            let number = |index: usize| -> f64 { cells[index].parse().expect(line) };
            let minimum = at[6];
            Band {
                exchange: cells[at[0]].to_owned(),
                segment: cells[at[1]].to_owned(),
                check: cells[at[2]].to_owned(),
                underlying: cells[at[3]].to_owned(),
                days: cells[at[4]].to_owned(),
                percent: number(at[5]),
                min_rupees: (!cells[minimum].is_empty()).then(|| number(minimum)),
                basis: cells[at[7]].to_owned(),
            }
        })
        .collect()
}

/// Premiums on the grid of `tick_rupees`, lowest first: every tick up to
/// `DENSE_UP_TO_RUPEES`, then about one percent apart up to `TOP_PREMIUM_RUPEES`. Counted
/// in whole ticks, so every premium is on the grid.
fn premiums(tick_rupees: f64) -> Vec<f64> {
    let dense_ticks = (DENSE_UP_TO_RUPEES / tick_rupees).round() as i64;
    let top_ticks = (TOP_PREMIUM_RUPEES / tick_rupees).round() as i64;
    let mut ticks: Vec<i64> = (1..=dense_ticks).collect();
    let mut count = dense_ticks;
    while count < top_ticks {
        count = (((count as f64) * SPARSE_STEP).ceil() as i64).min(top_ticks);
        ticks.push(count);
    }
    ticks
        .into_iter()
        .map(|count| count as f64 * tick_rupees)
        .collect()
}

/// The limit of an option stop at `trigger_rupees`, as the venue takes it.
fn stop_limit(trigger_rupees: f64, side: Side, tick_rupees: f64) -> f64 {
    let asked = OrderType::StopMarket { trigger_rupees };
    match venue_order_type(InstrumentKind::Option, asked, side, tick_rupees) {
        Ok(OrderType::StopLimit { limit_rupees, .. }) => limit_rupees,
        other => panic!("an option stop at {trigger_rupees} became {other:?}"),
    }
}

/// The limit of an option GTT's child at `trigger_rupees` (decision GTT-1).
fn gtt_child_limit(trigger_rupees: f64, side: Side, tick_rupees: f64) -> f64 {
    let buffer = GttPolicy::default().child_buffer_percent(InstrumentKind::Option);
    child_limit(trigger_rupees, side, buffer, tick_rupees).expect("a child limit")
}

/// The limit a stop the price has passed is repriced to: the last price moved
/// `OPTIONS_STOP_LIMIT_BUFFER_PERCENT` against the trader by the crate's one rule for an
/// option limit (`limit_beyond`, 16-session-design 4.6, open decision 8). The session
/// prices the reprice (the stops supervisor) and the kernel only takes the price
/// (`Engine::reprice_exit`); the session's pricing must give the same limit.
fn reprice_limit(last_price_rupees: f64, side: Side, tick_rupees: f64) -> f64 {
    limit_beyond(
        last_price_rupees,
        side,
        OPTIONS_STOP_LIMIT_BUFFER_PERCENT,
        tick_rupees,
    )
    .expect("a reprice limit")
}

/// Every limit `price` gives at every premium of both ticks and both sides is a positive
/// price on the grid and lies inside each of `bands` that is at least one tick wide there,
/// measured from the premium it was priced from. Fails listing the first limits that do
/// not.
fn assert_inside(bands: &[&Band], price: fn(f64, Side, f64) -> f64) {
    assert!(!bands.is_empty(), "no band to check against");
    let mut failures = Vec::new();
    for tick in OPTION_TICKS_RUPEES {
        for side in [Side::Sell, Side::Buy] {
            for premium in premiums(tick) {
                let limit = price(premium, side, tick);
                if limit <= 0.0 || is_on_tick(limit, tick) != Ok(true) {
                    failures.push(format!(
                        "{side:?} at {premium} tick {tick}: {limit} off the grid"
                    ));
                }
                for band in bands.iter().filter(|band| {
                    band.holds_a_tick(premium, tick) && !band.admits(side, limit, premium)
                }) {
                    failures.push(format!(
                        "{}: {side:?} at {premium} tick {tick}: {limit} outside",
                        band.name()
                    ));
                }
            }
        }
    }
    let shown = &failures[..failures.len().min(FAILURES_SHOWN)];
    assert!(
        failures.is_empty(),
        "{} limits fail, first: {shown:#?}",
        failures.len()
    );
}

#[test]
fn the_table_holds_the_option_bands_of_both_exchanges_each_minimum_switching_inside_the_sweep() {
    let bands = bands();
    for wanted in [
        "NFO index_option lpp ALL every_day",
        "NFO stock_option lpp ALL every_day",
        "NFO index_option stop_limit_range ALL every_day",
        "NFO stock_option stop_limit_range ALL every_day",
        "NFO stock_option cooling_off ALL every_day",
        "BFO index_option lpp ALL every_day",
        // The narrowest BSE index option band: 40 percent with a 10 rupee minimum.
        "BFO index_option lpp SENSEX expiry_day",
        "BFO index_option lpp FOCIT every_day",
        "BFO stock_option lpp ALL every_day",
        "BFO stock_option cooling_off ALL every_day",
        "BFO index_option stop_limit_range ALL every_day",
    ] {
        assert!(bands.iter().any(|band| band.name() == wanted), "{wanted}");
    }
    for band in &bands {
        if let Some(switch) = band.switch_rupees() {
            assert!(
                switch > OPTION_TICKS_RUPEES[0] && switch < DENSE_UP_TO_RUPEES,
                "{}: switches at {switch}",
                band.name()
            );
        }
    }
}

#[test]
fn only_a_band_without_a_rupee_minimum_is_narrower_than_a_tick_and_only_up_to_three_ticks() {
    for band in bands() {
        for tick in OPTION_TICKS_RUPEES {
            for premium in premiums(tick) {
                let up_to_three_ticks =
                    premium <= NARROW_UP_TO_TICKS * tick + HALF_PRICE_UNIT_RUPEES;
                let expected = band.min_rupees.is_none() && up_to_three_ticks;
                assert_eq!(
                    !band.holds_a_tick(premium, tick),
                    expected,
                    "{} at {premium} tick {tick}",
                    band.name()
                );
            }
        }
    }
}

#[test]
fn an_option_stop_limit_lies_inside_every_band_at_every_premium_on_the_tick_grid() {
    let bands = bands();
    let all: Vec<&Band> = bands.iter().collect();
    assert_inside(&all, stop_limit);
}

#[test]
fn an_option_gtt_child_lies_inside_every_band_on_plain_limits_at_every_premium() {
    let bands = bands();
    let plain: Vec<&Band> = bands.iter().filter(|b| b.checks_plain_limits()).collect();
    assert_inside(&plain, gtt_child_limit);
}

#[test]
fn a_reprice_at_the_last_price_moved_by_the_buffer_lies_inside_every_band_on_plain_limits() {
    let bands = bands();
    let plain: Vec<&Band> = bands.iter().filter(|b| b.checks_plain_limits()).collect();
    assert_inside(&plain, reprice_limit);
}
