#![cfg(all(feature = "costs", feature = "time"))]
//! `positional_stops="rearm_daily"` in a live session (design 20-positional-live section
//! 6, unit N4 item N4-1): a CNC or NRML leg's stop is the DAY stop order an MIS leg's
//! broker-held stop is, placed again as each session reaches its open, ended by the
//! close's cancellation of the DAY orders, never a GTT; a session started before the open
//! waits for it, and an open already at or beyond the stop exits the leg at market,
//! flagged `stop_crossed`.
//!
//! SBIN on 1-minute slices on Tue 2025-07-01, Wed 07-02 and Thu 07-03, bars flat at
//! 1,000.00, no charges, no slippage. On day 1 a CNC buy of 100 at market is decided at
//! the 09:15 close (o1) and the venue fills it at 1,000.00 at the 09:16 open; stop 50
//! points: 950.00.

use oa_bt::engine::{
    BaseBar, Driver, Engine, KernelError, OrderRole, Pause, Protect, RiskExitReason, RunConfig,
    StateValue, VenueCommand,
};
use oa_bt::exec::{FillFlags, FillReason, InstrumentSpec, OrderId, OrderRequest, OrderType};
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{Distance, ExitReason, LegRules, PositionalStops, StrategyRiskConfig, TrailSpec};

mod venue_support;

use venue_support::{config, order, sbin, session_of, Act, Script, NSE_OPEN_MINUTE, SBIN};

const SLICES: usize = 375;
const ENTRY: OrderId = OrderId(1);
const SECOND: i64 = 1_000_000_000;

fn day(n: i32) -> Day {
    days_from_civil(2025, 6, 30) + n
}

/// The open of minute `k` of `day`'s session, UTC nanoseconds.
fn minute_on(day: Day, k: u32) -> i64 {
    ts_from_ist(day, NSE_OPEN_MINUTE + k)
}

fn rearm_config(on: Day) -> RunConfig {
    RunConfig {
        first_day: on,
        last_day: on,
        risk: StrategyRiskConfig {
            positional_stops: PositionalStops::RearmDaily,
            ..StrategyRiskConfig::default()
        },
        ..config()
    }
}

fn session_on(on: Day, instrument: InstrumentSpec) -> Engine {
    session_of(rearm_config(on), instrument)
}

fn stop_50() -> LegRules {
    LegRules {
        stop: Some(Distance::Points(50.0)),
        ..LegRules::default()
    }
}

fn entry(side: Side, units: f64, product: Product, rules: LegRules) -> OrderRequest {
    OrderRequest {
        leg_rules: rules,
        ..order(side, units, OrderType::Market, product)
    }
}

/// Run `kernel`, a session of `on`, adding a flat bar at 1,000.00 for each slice and
/// releasing it, the script acting at its decisions, until it waits before slice `until`
/// (`SLICES`: its close), or finishes. The commands are left for the test.
fn run_until(kernel: &mut Engine, on: Day, script: &mut Script, until: usize) {
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if next_slice >= until => return,
            Pause::Waiting { next_slice } => {
                let k = u32::try_from(next_slice).expect("a slice of the day");
                let bar = BaseBar {
                    ts_open: minute_on(on, k),
                    open: 1000.0,
                    high: 1000.0,
                    low: 1000.0,
                    close: 1000.0,
                    volume: 1_000.0,
                };
                kernel.add_bar(SBIN, bar).expect("a bar of a closed slice");
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => return,
            pause => script.on_pause(kernel, &pause).expect("the script acts"),
        }
    }
}

/// The rest of `kernel`'s session, a flat bar at 1,000.00 in each slice, the venue
/// confirming each cancellation the close sends. Returns the commands of the close.
fn close_day(kernel: &mut Engine, on: Day) -> Vec<VenueCommand> {
    let mut closed = Vec::new();
    loop {
        let pause = match kernel.advance() {
            Err(KernelError::RunFinished) => return closed,
            pause => pause.expect("the slice runs"),
        };
        match pause {
            Pause::Waiting { next_slice } if next_slice >= SLICES => {
                let commands = kernel.take_commands();
                for command in &commands {
                    if let VenueCommand::Cancel { order } = command {
                        kernel.apply_cancel(*order).expect("the venue confirms");
                    }
                }
                closed.extend(commands);
            }
            Pause::Waiting { next_slice } => {
                run_until(kernel, on, &mut Script::new(Vec::new()), next_slice + 1);
            }
            Pause::Finished { .. } => return closed,
            _ => {}
        }
    }
}

/// Day 1: the session that bought 100 SBIN in CNC with `rules` at 1,000.00 at the 09:16
/// open, waiting before 09:16 with the commands of the fill left to take.
fn bought_on_day_1(instrument: InstrumentSpec, request: OrderRequest) -> Engine {
    let mut kernel = session_on(day(1), instrument);
    let decided = minute_on(day(1), 1);
    let mut script = Script::new(vec![(decided, Act::Submit(request))]);
    run_until(&mut kernel, day(1), &mut script, 1);
    kernel.take_commands();
    let units = kernel.order(ENTRY).expect("the entry").quantity_units;
    kernel
        .apply_fill(ENTRY, units, 1000.0, minute_on(day(1), 1))
        .expect("the entry fills");
    kernel
}

/// The places among `commands`: (order, side, units, order type, role).
fn places(commands: &[VenueCommand]) -> Vec<(OrderId, Side, f64, OrderType, OrderRole)> {
    commands
        .iter()
        .filter_map(|command| match *command {
            VenueCommand::Place {
                order,
                side,
                units,
                order_type,
                role,
                ..
            } => Some((order, side, units, order_type, role)),
            _ => None,
        })
        .collect()
}

fn sl_m(trigger_rupees: f64) -> OrderType {
    OrderType::StopMarket { trigger_rupees }
}

/// Day 1 to its end, carried: the state carried to day 2 and the commands of day 1's
/// close.
fn carried_day_1() -> (StateValue, Vec<VenueCommand>) {
    let mut kernel = bought_on_day_1(sbin(), entry(Side::Buy, 100.0, Product::Cnc, stop_50()));
    kernel.take_commands();
    let closed = close_day(&mut kernel, day(1));
    (kernel.carried_state().expect("day 1 has closed"), closed)
}

/// A session of `on` that took `carried`, before its open.
fn restored(on: Day, carried: &StateValue) -> Engine {
    let mut kernel = session_on(on, sbin());
    kernel
        .restore_carried_state(carried)
        .expect("the session takes it");
    kernel
}

#[test]
fn the_fill_places_the_days_stop_as_a_day_sl_m_order_and_no_gtt() {
    let mut kernel = bought_on_day_1(sbin(), entry(Side::Buy, 100.0, Product::Cnc, stop_50()));
    let commands = kernel.take_commands();
    let expected = vec![(
        OrderId(2),
        Side::Sell,
        100.0,
        sl_m(950.0),
        OrderRole::LegStop,
    )];
    assert_eq!(places(&commands), expected);
    assert!(commands.iter().all(|command| command.gtt().is_none()));
    assert_eq!(kernel.order_role(OrderId(2)), Some(OrderRole::LegStop));
}

#[test]
fn the_close_ends_the_days_stop_and_carries_the_leg_with_no_stop_order() {
    let (carried, closed) = carried_day_1();
    assert_eq!(closed, vec![VenueCommand::Cancel { order: OrderId(2) }]);
    let StateValue::Map(parts) = &carried else {
        panic!("a carried state is a map");
    };
    let part = |name: &str| parts.iter().find(|(key, _)| key == name).map(|(_, v)| v);
    assert_eq!(part("orders"), Some(&StateValue::List(Vec::new())));
    assert_eq!(part("gtts"), Some(&StateValue::List(Vec::new())));
    let legs = part("legs").expect("the legs part");
    assert!(matches!(legs, StateValue::List(items) if items.len() == 1));
}

#[test]
fn a_carried_leg_sends_nothing_before_the_open_and_protect_waits_for_it() {
    let (carried, _) = carried_day_1();
    let mut kernel = restored(day(2), &carried);
    assert_eq!(kernel.take_commands(), Vec::new());
    assert_eq!(
        kernel.protect(SBIN, Product::Cnc),
        Ok(Protect::WaitsForOpen)
    );
    let before_open = minute_on(day(2), 0) - 60 * SECOND;
    kernel
        .push_tick(SBIN, 1001.0, before_open)
        .expect("a pre-open tick");
    assert_eq!(kernel.take_commands(), Vec::new());
}

#[test]
fn the_broker_ending_the_days_stop_in_the_closing_minute_is_its_expiry_not_a_loss_of_protection() {
    // The broker reports the day's stop cancelled before the session runs its closing
    // slice (15:29 to 15:30): that is the DAY order's end, so the stop is neither placed
    // again (a stop sent after the close would be refused, or kept by a broker for the
    // next day) nor the leg exited, and the close has nothing left to cancel.
    let (carried, _) = carried_day_1();
    let mut kernel = restored(day(2), &carried);
    kernel
        .push_tick(SBIN, 1001.0, minute_on(day(2), 0) + SECOND / 2)
        .expect("the first tick of the session");
    let (stop, ..) = places(&kernel.take_commands())[0];
    run_until(
        &mut kernel,
        day(2),
        &mut Script::new(Vec::new()),
        SLICES - 1,
    );
    kernel.take_commands();
    kernel.apply_cancel(stop).expect("the broker ended it");
    assert_eq!(kernel.take_commands(), Vec::new());
    assert_eq!(close_day(&mut kernel, day(2)), Vec::new());
    assert!(kernel.carried_state().is_ok());
}

#[test]
fn the_days_stop_is_placed_again_at_each_open_and_ended_at_each_close() {
    let (carried, _) = carried_day_1();
    let mut carried = carried;
    for n in [2, 3] {
        let mut kernel = restored(day(n), &carried);
        kernel
            .push_tick(SBIN, 1001.0, minute_on(day(n), 0) + SECOND / 2)
            .expect("the first tick of the session");
        let commands = kernel.take_commands();
        let stop = places(&commands);
        assert_eq!(stop.len(), 1, "day {n}: {commands:?}");
        let (id, side, units, order_type, role) = stop[0];
        assert_eq!(
            (side, units, order_type, role),
            (Side::Sell, 100.0, sl_m(950.0), OrderRole::LegStop)
        );
        let closed = close_day(&mut kernel, day(n));
        assert_eq!(closed, vec![VenueCommand::Cancel { order: id }], "day {n}");
        carried = kernel.carried_state().expect("the day has closed");
    }
}

#[test]
fn an_open_beyond_the_stop_exits_the_leg_at_market_flagged_stop_crossed() {
    let (carried, _) = carried_day_1();
    let mut kernel = restored(day(2), &carried);
    let at = minute_on(day(2), 0) + SECOND / 2;
    kernel.push_tick(SBIN, 940.0, at).expect("the open's tick");
    let commands = kernel.take_commands();
    let sent = places(&commands);
    assert_eq!(sent.len(), 1, "{commands:?}");
    let (id, side, units, order_type, role) = sent[0];
    let crossed = OrderRole::RiskExit(RiskExitReason::StopCrossed);
    assert_eq!(
        (side, units, order_type, role),
        (Side::Sell, 100.0, OrderType::Market, crossed)
    );
    kernel
        .apply_fill(id, 100.0, 940.0, at)
        .expect("the exit fills");
    run_until(
        &mut kernel,
        day(2),
        &mut Script::new(Vec::new()),
        usize::MAX,
    );
    let output = kernel.finish().expect("day 2 finished");
    let exit = &output.fills.last().expect("the exit's fill").fill;
    assert_eq!(exit.reason, FillReason::Exit(ExitReason::LegSl));
    assert!(exit.flags.contains(FillFlags::STOP_CROSSED));
}

#[test]
fn protect_at_the_open_beyond_the_stop_exits_flagged_stop_crossed() {
    let (carried, _) = carried_day_1();
    let mut kernel = restored(day(2), &carried);
    assert_eq!(
        kernel.protect(SBIN, Product::Cnc),
        Ok(Protect::WaitsForOpen)
    );
    run_until(&mut kernel, day(2), &mut Script::new(Vec::new()), 0);
    // A price at or below the stop when the session reaches its open: the first slice's
    // open phase exits the leg instead of placing the stop. With no tick at or after the
    // open, that price is the open of the slice's bar, 949.00 as the pre-open tick.
    let at = minute_on(day(2), 0) - SECOND;
    kernel.push_tick(SBIN, 949.0, at).expect("a pre-open tick");
    assert_eq!(
        kernel.take_commands(),
        Vec::new(),
        "nothing before the open"
    );
    let bar = BaseBar {
        ts_open: minute_on(day(2), 0),
        open: 949.0,
        high: 950.0,
        low: 948.0,
        close: 949.0,
        volume: 1_000.0,
    };
    kernel.add_bar(SBIN, bar).expect("the first slice's bar");
    kernel.release(1);
    run_until(&mut kernel, day(2), &mut Script::new(Vec::new()), 1);
    let sent = places(&kernel.take_commands());
    let crossed = OrderRole::RiskExit(RiskExitReason::StopCrossed);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].4, crossed);
}

#[test]
fn with_no_tick_before_the_first_slice_its_open_beyond_the_stop_exits_flagged_stop_crossed() {
    // No tick comes before the session runs its first slice, so the session's first price
    // is that slice's open, 940.00, beyond the 950.00 stop, not the carried close of
    // 1,000.00: the leg is exited at market, flagged stop_crossed, and no stop is placed.
    let (carried, _) = carried_day_1();
    let mut kernel = restored(day(2), &carried);
    assert!(matches!(
        kernel.advance(),
        Ok(Pause::Waiting { next_slice: 0 })
    ));
    let bar = BaseBar {
        ts_open: minute_on(day(2), 0),
        open: 940.0,
        high: 945.0,
        low: 935.0,
        close: 942.0,
        volume: 1_000.0,
    };
    kernel.add_bar(SBIN, bar).expect("the first slice's bar");
    kernel.release(1);
    run_until(&mut kernel, day(2), &mut Script::new(Vec::new()), 1);
    let crossed = OrderRole::RiskExit(RiskExitReason::StopCrossed);
    let sent: Vec<(Side, f64, OrderType, OrderRole)> = places(&kernel.take_commands())
        .into_iter()
        .map(|(_, side, units, order_type, role)| (side, units, order_type, role))
        .collect();
    assert_eq!(sent, vec![(Side::Sell, 100.0, OrderType::Market, crossed)]);
}

#[test]
fn a_trail_moves_the_days_stop_order() {
    // A continuous trail of 40 points arms once the price is 40 above the 1,000.00 fill;
    // at 1,050.00 the stop is 1,010.00, and the day's stop order is modified to it.
    let rules = LegRules {
        trail: Some(TrailSpec {
            trigger: Distance::Points(40.0),
            step: None,
        }),
        ..stop_50()
    };
    let mut kernel = bought_on_day_1(sbin(), entry(Side::Buy, 100.0, Product::Cnc, rules));
    kernel.take_commands();
    let at = minute_on(day(1), 1) + 10 * SECOND;
    kernel.push_tick(SBIN, 1050.0, at).expect("a tick");
    let commands = kernel.take_commands();
    let modify = VenueCommand::Modify {
        order: OrderId(2),
        units: 100.0,
        order_type: sl_m(1010.0),
    };
    assert_eq!(commands, vec![modify]);
}

#[test]
fn an_nrml_short_futures_stop_is_a_buy_above_the_price() {
    let future = InstrumentSpec {
        symbol: "SBIN31JUL25FUT".to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Future,
        lot_size_units: 750,
        last_trading_day: Some(days_from_civil(2025, 7, 31)),
        ..sbin()
    };
    let request = entry(Side::Sell, 750.0, Product::Nrml, stop_50());
    let mut kernel = bought_on_day_1(future, request);
    let sent = places(&kernel.take_commands());
    let expected = vec![(
        OrderId(2),
        Side::Buy,
        750.0,
        sl_m(1050.0),
        OrderRole::LegStop,
    )];
    assert_eq!(sent, expected);
}
