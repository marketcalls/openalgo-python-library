#![cfg(all(feature = "costs", feature = "time"))]
//! A kernel live session that spans several trading days (unit N4f item N4f-3): every
//! day's open is reached as the first day's is. The first tick at or after a later day's
//! open, or that day's first slice when no tick comes first, sends what waits for that
//! open: the expiry exit of a contract past its positional exit, and a rearm_daily leg's
//! day stop, placed again after the close ended the day before's.
//!
//! 1-minute slices of NSE and NFO sessions (09:15 to 15:30, 375 slices a day; a later
//! day's slice k is index 375 + k), bars flat at 1,000.00, no charges, no slippage. On the
//! first day a MARKET buy is decided at the 09:15 close (o1) and the venue fills it at
//! 1,000.00 at the 09:16 open.

mod venue_support;

use oa_bt::engine::{
    BaseBar, Driver, Engine, OrderRole, Pause, RiskExitReason, RunConfig, VenueCommand,
};
use oa_bt::exec::{InstrumentSpec, OrderId, OrderRequest, OrderType};
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{Distance, LegRules, PositionalStops, StrategyRiskConfig};

use venue_support::{config, order, sbin, session_of, Act, Script, NSE_OPEN_MINUTE, SBIN};

/// Slices of one session: 09:15 to 15:30.
const SLICES: usize = 375;
/// The NFO square-off of July 2025, 15:10: minute 355.
const SQUARE_OFF: usize = 355;
const ENTRY: OrderId = OrderId(1);
const SECOND: i64 = 1_000_000_000;
const EXPIRY_EXIT: OrderRole = OrderRole::RiskExit(RiskExitReason::ExpiryExit);

/// The open of minute `k` of `day`'s session, UTC nanoseconds.
fn minute_on(day: Day, k: u32) -> i64 {
    ts_from_ist(day, NSE_OPEN_MINUTE + k)
}

/// A session from `first` to `last` with the risk settings `risk`, trading `instrument`.
fn two_days(first: Day, last: Day, risk: StrategyRiskConfig, instrument: InstrumentSpec) -> Engine {
    let config = RunConfig {
        first_day: first,
        last_day: last,
        risk,
        ..config()
    };
    session_of(config, instrument)
}

/// Run `kernel` through the first day's slices, a flat bar at 1,000.00 in each, the
/// script acting at its decisions, until it waits before slice `until` of that day.
fn run_first_day_until(kernel: &mut Engine, day: Day, script: &mut Script, until: usize) {
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if next_slice >= until => return,
            Pause::Waiting { next_slice } => {
                let k = u32::try_from(next_slice).expect("a slice of the day");
                kernel
                    .add_bar(SBIN, flat_bar(minute_on(day, k)))
                    .expect("a bar of a closed slice");
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => panic!("the session finished on its first day"),
            pause => script.on_pause(kernel, &pause).expect("the script acts"),
        }
    }
}

fn flat_bar(ts_open: i64) -> BaseBar {
    BaseBar {
        ts_open,
        open: 1000.0,
        high: 1000.0,
        low: 1000.0,
        close: 1000.0,
        volume: 1_000.0,
    }
}

/// The first day's close: the rest of its slices, the venue confirming each
/// cancellation the close sends, until the session stands before the second day's first
/// slice. Returns the commands of the close.
fn close_first_day(kernel: &mut Engine, day: Day) -> Vec<VenueCommand> {
    run_first_day_until(kernel, day, &mut Script::new(Vec::new()), SLICES);
    let mut closed = Vec::new();
    loop {
        let commands = kernel.take_commands();
        if commands.is_empty() {
            return closed;
        }
        for command in &commands {
            if let VenueCommand::Cancel { order } = *command {
                kernel.apply_cancel(order).expect("the venue confirms");
            }
        }
        closed.extend(commands);
        let pause = kernel.advance().expect("the close completes");
        assert_eq!(pause, Pause::Waiting { next_slice: SLICES });
    }
}

/// The first day of `kernel`: a buy of `units` in `product` with `rules`, filled at
/// 1,000.00 at the 09:16 open; the commands of the fill taken and returned.
fn bought(
    kernel: &mut Engine,
    day: Day,
    (units, product, rules): (f64, Product, LegRules),
) -> Vec<VenueCommand> {
    let request = OrderRequest {
        leg_rules: rules,
        ..order(Side::Buy, units, OrderType::Market, product)
    };
    let mut script = Script::new(vec![(minute_on(day, 1), Act::Submit(request))]);
    run_first_day_until(kernel, day, &mut script, 1);
    kernel.take_commands();
    kernel
        .apply_fill(ENTRY, units, 1000.0, minute_on(day, 1))
        .expect("the entry fills");
    kernel.take_commands()
}

/// The places among `commands`: (side, units, order type, role).
fn places(commands: &[VenueCommand]) -> Vec<(Side, f64, OrderType, OrderRole)> {
    commands
        .iter()
        .filter_map(|command| match *command {
            VenueCommand::Place {
                side,
                units,
                order_type,
                role,
                ..
            } => Some((side, units, order_type, role)),
            _ => None,
        })
        .collect()
}

fn wednesday() -> Day {
    days_from_civil(2025, 7, 30)
}

fn thursday() -> Day {
    days_from_civil(2025, 7, 31)
}

/// A stock future of lot 75 whose positional exit day is Wednesday 07-30 and whose last
/// trading day is Thursday 07-31.
fn exiting_on_wednesday() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "SBIN31JUL25FUT".to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Future,
        lot_size_units: 75,
        last_trading_day: Some(thursday()),
        positional_exit_day: Some(wednesday()),
        ..sbin()
    }
}

/// A Wednesday-Thursday session that bought one lot in NRML on Wednesday. The expiry exit
/// the session made at Wednesday's 15:10 square-off was never answered, and Wednesday's
/// close cancelled it: the session stands before Thursday's first slice, still long.
fn expiry_exit_cancelled_by_the_first_close() -> Engine {
    let risk = StrategyRiskConfig::default();
    let mut kernel = two_days(wednesday(), thursday(), risk, exiting_on_wednesday());
    let unit = (75.0, Product::Nrml, LegRules::default());
    bought(&mut kernel, wednesday(), unit);
    let script = &mut Script::new(Vec::new());
    run_first_day_until(&mut kernel, wednesday(), script, SQUARE_OFF);
    let made = kernel.take_commands();
    assert_eq!(
        places(&made),
        vec![(Side::Sell, 75.0, OrderType::Market, EXPIRY_EXIT)],
        "the exit at Wednesday's square-off"
    );
    let closed = close_first_day(&mut kernel, wednesday());
    assert_eq!(closed, vec![VenueCommand::Cancel { order: OrderId(2) }]);
    let held = kernel
        .position(SBIN, Product::Nrml)
        .map(|p| p.quantity_units);
    assert_eq!(held, Some(75.0));
    kernel
}

#[test]
fn the_first_tick_at_a_later_days_open_sends_the_expiry_exit_that_waits_for_it() {
    let mut kernel = expiry_exit_cancelled_by_the_first_close();
    let first_tick = minute_on(thursday(), 0) + SECOND;
    kernel
        .push_tick(SBIN, 1000.0, first_tick)
        .expect("Thursday's first tick");
    assert_eq!(
        places(&kernel.take_commands()),
        vec![(Side::Sell, 75.0, OrderType::Market, EXPIRY_EXIT)]
    );
}

#[test]
fn with_no_tick_a_later_days_first_slice_makes_the_expiry_exit_at_its_open() {
    // The session reaches Thursday's open with the first slice: the exit is decided at
    // 09:15, as the first day's would be, not at the close of the slice a minute later.
    let mut kernel = expiry_exit_cancelled_by_the_first_close();
    kernel
        .add_bar(SBIN, flat_bar(minute_on(thursday(), 0)))
        .expect("Thursday's first bar");
    kernel.release(SLICES + 1);
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { .. } => break,
            Pause::Finished { .. } => panic!("the session finished"),
            _ => {}
        }
    }
    let exit = OrderId(3);
    assert_eq!(
        places(&kernel.take_commands()),
        vec![(Side::Sell, 75.0, OrderType::Market, EXPIRY_EXIT)]
    );
    let decided = kernel.order(exit).expect("the exit").decided_at;
    assert_eq!(decided, minute_on(thursday(), 0));
}

fn tuesday() -> Day {
    days_from_civil(2025, 7, 1)
}

fn next_wednesday() -> Day {
    days_from_civil(2025, 7, 2)
}

/// A Tuesday-Wednesday rearm_daily session that bought 100 SBIN in CNC with a stop of 50
/// points (950.00) on Tuesday: the day's stop (o2) placed with the fill and ended by
/// Tuesday's close. It stands before Wednesday's first slice.
fn rearm_leg_after_the_first_close() -> Engine {
    let risk = StrategyRiskConfig {
        positional_stops: PositionalStops::RearmDaily,
        ..StrategyRiskConfig::default()
    };
    let mut kernel = two_days(tuesday(), next_wednesday(), risk, sbin());
    let stop_50 = LegRules {
        stop: Some(Distance::Points(50.0)),
        ..LegRules::default()
    };
    let placed = bought(&mut kernel, tuesday(), (100.0, Product::Cnc, stop_50));
    let stop = OrderType::StopMarket {
        trigger_rupees: 950.0,
    };
    assert_eq!(
        places(&placed),
        vec![(Side::Sell, 100.0, stop, OrderRole::LegStop)]
    );
    let closed = close_first_day(&mut kernel, tuesday());
    assert_eq!(closed, vec![VenueCommand::Cancel { order: OrderId(2) }]);
    kernel
}

#[test]
fn a_rearm_daily_leg_waits_for_a_later_days_open_before_its_stop_goes_again() {
    let mut kernel = rearm_leg_after_the_first_close();
    let before_open = minute_on(next_wednesday(), 0) - 60 * SECOND;
    kernel
        .push_tick(SBIN, 1001.0, before_open)
        .expect("a pre-open tick");
    assert_eq!(kernel.take_commands(), Vec::new());
}

#[test]
fn the_first_tick_at_a_later_days_open_places_a_rearm_daily_legs_day_stop_again() {
    let mut kernel = rearm_leg_after_the_first_close();
    let first_tick = minute_on(next_wednesday(), 0) + SECOND;
    kernel
        .push_tick(SBIN, 1001.0, first_tick)
        .expect("Wednesday's first tick");
    let stop = OrderType::StopMarket {
        trigger_rupees: 950.0,
    };
    assert_eq!(
        places(&kernel.take_commands()),
        vec![(Side::Sell, 100.0, stop, OrderRole::LegStop)]
    );
}

#[test]
fn a_later_days_open_beyond_a_rearm_daily_stop_exits_the_leg_instead() {
    // Wednesday opens at 940.00, beyond the 950.00 stop: no stop goes, the leg is exited at
    // market, as on the first day of a carried leg.
    let mut kernel = rearm_leg_after_the_first_close();
    let first_tick = minute_on(next_wednesday(), 0) + SECOND;
    kernel
        .push_tick(SBIN, 940.0, first_tick)
        .expect("Wednesday's first tick");
    let crossed = OrderRole::RiskExit(RiskExitReason::StopCrossed);
    assert_eq!(
        places(&kernel.take_commands()),
        vec![(Side::Sell, 100.0, OrderType::Market, crossed)]
    );
}
