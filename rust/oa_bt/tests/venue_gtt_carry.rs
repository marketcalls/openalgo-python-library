#![cfg(all(feature = "costs", feature = "time"))]
//! A CNC position carried from one trading day's live session to the next (design
//! 20-positional-live section 4, unit N1 item N1-4): `carried_state` at the close of day
//! 1, `restore_carried_state` in day 2's session, and the trade list, positions and
//! equity of the two sessions equal to a backtest of the same bars and fills. Also the
//! same-day restart with GTTs, and the refusals of a carried state.
//!
//! SBIN on Tue 2025-07-01 and Wed 2025-07-02, 1-minute bars, no charges, no slippage. 100
//! are bought in CNC at the 09:15 close of day 1 and fill at the 09:16 open, 500.10; stop
//! 4 points (496.10), target 10 points (510.10). A day's bars are flat at
//! (500.10, 500.50, 499.70, 500.10) unless a test writes one.
//!
//! The venue of a live session is scripted here as the backtest's simulator fills: it
//! confirms every GTT request at once; before each slice runs it fills the MARKET orders
//! at the slice's open, then plays the slice's ticks (the bar's high, then its low), each
//! answered at once (a target's exit fills at the target, as the backtest fills it), then
//! fires a stop GTT the bar reaches from above, its child filling at the trigger.

mod venue_support;

use oa_bt::engine::{
    BarColumns, BaseBar, Engine, GttEnd, KernelError, LiveStateError, OrderRole, Pause,
    RiskExitReason, RunConfig, RunOutput, StateValue, VenueCommand,
};
use oa_bt::exec::{FillFlags, GttId, GttStatus, OrderId, OrderRequest, OrderType};
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Product, Side};
use oa_bt::rms::{Distance, LegRules, TrailSpec};
use venue_support::{config, order, run_backtest, Act, Ohlc, Script, CAPITAL, NSE_OPEN_MINUTE};

/// Slices of an NSE session on the 1-minute grid (09:15 to 15:30).
const SLICES: u32 = 375;
const FLAT: Ohlc = (500.1, 500.5, 499.7, 500.1);
const UNITS: f64 = 100.0;

fn day_1() -> Day {
    days_from_civil(2025, 7, 1)
}

fn day_2() -> Day {
    days_from_civil(2025, 7, 2)
}

/// The open of minute `k` of `day`'s session, UTC nanoseconds.
fn minute_on(day: Day, k: u32) -> i64 {
    ts_from_ist(day, NSE_OPEN_MINUTE + k)
}

/// A day's bars: `FLAT` unless written.
type DayBars = Vec<(u32, Ohlc)>;

fn bar_of(written: &DayBars, k: u32) -> Ohlc {
    written
        .iter()
        .find(|(m, _)| *m == k)
        .map_or(FLAT, |(_, bar)| *bar)
}

fn push_day(columns: &mut BarColumns, day: Day, written: &DayBars) {
    for k in 0..SLICES {
        let (open, high, low, close) = bar_of(written, k);
        columns.ts_open.push(minute_on(day, k));
        columns.open.push(open);
        columns.high.push(high);
        columns.low.push(low);
        columns.close.push(close);
        columns.volume.push(1_000.0);
    }
}

fn day_config(first: Day, last: Day) -> RunConfig {
    RunConfig {
        first_day: first,
        last_day: last,
        ..config()
    }
}

fn session_on(day: Day) -> Engine {
    venue_support::session(day_config(day, day))
}

/// No time exit.
fn no_rule(_: &mut RunConfig) {}

fn stop_and_target() -> LegRules {
    LegRules {
        stop: Some(Distance::Points(4.0)),
        target: Some(Distance::Points(10.0)),
        ..LegRules::default()
    }
}

fn stop_and_trail() -> LegRules {
    LegRules {
        stop: Some(Distance::Points(4.0)),
        trail: Some(TrailSpec {
            trigger: Distance::Points(2.0),
            step: None,
        }),
        ..LegRules::default()
    }
}

fn buy(rules: LegRules) -> OrderRequest {
    OrderRequest {
        leg_rules: rules,
        ..order(Side::Buy, UNITS, OrderType::Market, Product::Cnc)
    }
}

/// The scripted venue (module docs).
#[derive(Debug, Default)]
struct Venue {
    /// MARKET orders placed and not filled yet.
    orders: Vec<OrderId>,
    /// The GTTs resting, with (trigger, below).
    gtts: Vec<(GttId, f64)>,
    /// The price a target's exit fills at, once a tick reached it.
    target_price: Option<f64>,
    /// The minute at whose open the broker sells the whole position by itself, at the
    /// open; from then on the venue refuses every cancellation of a GTT.
    sells_all_at: Option<u32>,
    /// The venue refuses every cancellation of a GTT.
    refuses_cancels: bool,
    /// The minute at whose open the broker rejects the resting stop GTT after its
    /// placement.
    rejects_stop_at: Option<u32>,
}

impl Venue {
    fn answer(&mut self, kernel: &mut Engine) {
        loop {
            let commands = kernel.take_commands();
            if commands.is_empty() {
                return;
            }
            for command in commands {
                match command {
                    VenueCommand::Place { order, .. } => self.orders.push(order),
                    VenueCommand::PlaceGtt { gtt, terms, .. } => {
                        kernel.apply_gtt_placed(gtt).expect("the venue confirms");
                        self.gtts.push((gtt, terms.trigger_rupees));
                    }
                    VenueCommand::ModifyGtt { gtt, terms } => {
                        kernel.apply_gtt_modified(gtt).expect("the venue confirms");
                        self.gtts.retain(|(known, _)| *known != gtt);
                        self.gtts.push((gtt, terms.trigger_rupees));
                    }
                    VenueCommand::CancelGtt { gtt, .. } if self.refuses_cancels => {
                        kernel
                            .apply_gtt_cancel_refused(gtt)
                            .expect("the venue refuses");
                    }
                    VenueCommand::CancelGtt { gtt, .. } => {
                        kernel.apply_gtt_cancelled(gtt).expect("the venue confirms");
                        self.gtts.retain(|(known, _)| *known != gtt);
                    }
                    other => panic!("not scripted: {other:?}"),
                }
            }
        }
    }

    /// What the broker does by itself at the open of minute `k` (`sells_all_at`,
    /// `rejects_stop_at`), at `price`.
    fn act_alone(&mut self, kernel: &mut Engine, k: u32, price: f64, at: i64) {
        if self.sells_all_at == Some(k) {
            let held = kernel
                .position(venue_support::SBIN, Product::Cnc)
                .map_or(0.0, |p| p.quantity_units);
            let reason = oa_bt::rms::ExitReason::Manual;
            let sbin = venue_support::SBIN;
            let sold =
                kernel.apply_external_fill(sbin, Product::Cnc, Side::Sell, held, price, at, reason);
            assert!(sold.is_ok(), "{sold:?}");
            self.refuses_cancels = true;
        }
        if self.rejects_stop_at == Some(k) {
            let (gtt, _) = self.gtts.remove(0);
            let rejected = kernel.apply_gtt_ended(gtt, GttEnd::Rejected);
            assert!(rejected.is_ok(), "{rejected:?}");
        }
        self.answer(kernel);
    }

    fn fill_orders(&mut self, kernel: &mut Engine, price: f64, at: i64) {
        for order in std::mem::take(&mut self.orders) {
            let units = kernel.order(order).map_or(0.0, |o| o.quantity_units);
            kernel.apply_fill(order, units, price, at).expect("a fill");
        }
    }

    /// Before slice `k` of `day` runs (module docs).
    fn slice(&mut self, kernel: &mut Engine, day: Day, k: u32, bar: Ohlc) {
        let (open, high, low, _) = bar;
        let open_ns = minute_on(day, k);
        self.answer(kernel);
        self.fill_orders(kernel, open, open_ns);
        self.act_alone(kernel, k, open, open_ns + 1);
        for (second, price) in [(15, high), (30, low)] {
            let at = open_ns + second * 1_000_000_000;
            kernel
                .push_tick(venue_support::SBIN, price, at)
                .expect("a tick");
            self.answer(kernel);
            if !self.orders.is_empty() {
                let price = self.target_price.unwrap_or(price);
                self.fill_orders(kernel, price, at);
            }
        }
        let fired: Vec<(GttId, f64)> = self
            .gtts
            .iter()
            .filter(|(_, trigger)| low <= *trigger && open > *trigger)
            .copied()
            .collect();
        for (gtt, trigger) in fired {
            self.gtts.retain(|(known, _)| *known != gtt);
            let child = kernel
                .apply_gtt_triggered(gtt, open_ns + 1)
                .expect("the venue fired it");
            kernel
                .apply_fill(child, UNITS, trigger, open_ns + 1)
                .expect("the child fills at the trigger");
        }
    }
}

/// Run `kernel`, a session of `day`, to its end with the scripted venue on `written`, the
/// script acting at its decisions.
fn run_day(
    kernel: &mut Engine,
    day: Day,
    written: &DayBars,
    script: &mut Script,
    venue: &mut Venue,
) {
    use oa_bt::engine::Driver;
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if next_slice >= SLICES as usize => {
                venue.answer(kernel);
            }
            Pause::Waiting { next_slice } => {
                let k = u32::try_from(next_slice).expect("a slice of the day");
                let bar = bar_of(written, k);
                venue.slice(kernel, day, k, bar);
                let (open, high, low, close) = bar;
                let base = BaseBar {
                    ts_open: minute_on(day, k),
                    open,
                    high,
                    low,
                    close,
                    volume: 1_000.0,
                };
                kernel.add_bar(venue_support::SBIN, base).expect("a bar");
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => return,
            pause => script.on_pause(kernel, &pause).expect("the script acts"),
        }
    }
}

/// Day 1 and day 2 of a carried CNC long with `rules`: the two sessions' outputs, and the
/// backtest's of both days.
fn two_days(
    rules: LegRules,
    first: &DayBars,
    second: &DayBars,
) -> (RunOutput, RunOutput, RunOutput) {
    two_days_with(rules, (first, second), no_rule)
}

/// `two_days` with the run's configuration changed by `edit` in all three runs.
fn two_days_with(
    rules: LegRules,
    (first, second): (&DayBars, &DayBars),
    edit: fn(&mut RunConfig),
) -> (RunOutput, RunOutput, RunOutput) {
    let decided = minute_on(day_1(), 1);
    let mut columns = BarColumns::default();
    push_day(&mut columns, day_1(), first);
    push_day(&mut columns, day_2(), second);
    let acts = vec![(decided, Act::Submit(buy(rules.clone())))];
    let mut both = day_config(day_1(), day_2());
    edit(&mut both);
    let backtest = run_backtest(both, &columns, &mut Script::new(acts));
    let edited = |day| {
        let mut config = day_config(day, day);
        edit(&mut config);
        venue_support::session(config)
    };
    let mut monday = edited(day_1());
    let acts = vec![(decided, Act::Submit(buy(rules)))];
    let mut venue = Venue::default();
    run_day(
        &mut monday,
        day_1(),
        first,
        &mut Script::new(acts),
        &mut venue,
    );
    let carried = monday.carried_state().expect("day 1 has closed");
    let first_output = monday.finish().expect("day 1 finished");
    let mut tuesday = edited(day_2());
    let ids = tuesday
        .restore_carried_state(&carried)
        .expect("day 2 takes it");
    assert_eq!(ids.gtts.len(), 1, "the stop GTT carries");
    let mut venue = Venue {
        gtts: venue
            .gtts
            .iter()
            .map(|(_, trigger)| (ids.gtts[0].1, *trigger))
            .collect(),
        target_price: Some(510.1),
        ..Venue::default()
    };
    run_day(
        &mut tuesday,
        day_2(),
        second,
        &mut Script::new(Vec::new()),
        &mut venue,
    );
    (
        first_output,
        tuesday.finish().expect("day 2 finished"),
        backtest,
    )
}

/// What a trade traded: entry and exit instants and prices, units, gross P&L, reason,
/// fills.
type Traded = (i64, i64, f64, f64, f64, f64, &'static str, u32);

fn traded(output: &RunOutput) -> Vec<Traded> {
    output
        .trades
        .iter()
        .map(|t| {
            (
                t.entry_ts,
                t.exit_ts,
                t.entry_price_rupees,
                t.exit_price_rupees,
                t.entry_units,
                t.gross_pnl_rupees,
                t.exit_reason.code(),
                t.fills,
            )
        })
        .collect()
}

/// The equity rows of `output` on `day`: (close instant, equity, realized, unrealized).
fn equity_on(output: &RunOutput, day: Day) -> Vec<(i64, f64, f64, f64)> {
    let equity = &output.equity;
    (0..equity.ts.len())
        .filter(|row| equity.day[*row] == day)
        .map(|row| {
            (
                equity.ts[row],
                equity.equity_rupees[row],
                equity.realized_rupees[row],
                equity.unrealized_rupees[row],
            )
        })
        .collect()
}

fn final_units(output: &RunOutput) -> Vec<f64> {
    output
        .final_positions
        .iter()
        .map(|p| p.quantity_units)
        .collect()
}

fn assert_same_as_the_backtest(first: &RunOutput, second: &RunOutput, backtest: &RunOutput) {
    assert_eq!(traded(second), traded(backtest), "the trade list");
    assert_eq!(final_units(second), final_units(backtest), "the positions");
    assert_eq!(
        equity_on(first, day_1()),
        equity_on(backtest, day_1()),
        "day 1 equity"
    );
    assert_eq!(
        equity_on(second, day_2()),
        equity_on(backtest, day_2()),
        "day 2 equity"
    );
}

// ---------------------------------------------------------------- the acceptance paths

#[test]
fn a_carried_long_whose_stop_gtt_fires_on_day_2_trades_as_the_backtest() {
    // Day 2's minute 5 falls through the stop from 499.00: the GTT fires at 496.10 and its
    // child sells the 100 there. 100 x (496.10 - 500.10) = -400.00.
    let second = vec![(5, (499.0, 499.2, 495.0, 495.5))];
    let (first, second, backtest) = two_days(stop_and_target(), &Vec::new(), &second);
    let exit = (
        minute_on(day_1(), 1),
        minute_on(day_2(), 5),
        500.1,
        496.1,
        UNITS,
        -400.0,
        "leg_sl",
        2,
    );
    assert_eq!(traded(&second), [exit]);
    assert_same_as_the_backtest(&first, &second, &backtest);
}

#[test]
fn a_carried_long_whose_trail_moves_on_both_days_trades_as_the_backtest() {
    // Day 1's minute 3 reaches 503.00: the trail moves the stop to 501.00. Day 2's minute
    // 2 reaches 505.00: 503.00. Minute 6 falls to 502.00: the GTT sells at 503.00.
    // 100 x (503.00 - 500.10) = 290.00.
    let raised = (502.0, 502.4, 501.6, 502.0);
    let mut first: DayBars = (4..SLICES).map(|k| (k, raised)).collect();
    first.push((3, (501.0, 503.0, 500.8, 502.5)));
    let higher = (504.0, 504.4, 503.6, 504.0);
    let mut second: DayBars = (3..SLICES).map(|k| (k, higher)).collect();
    second.retain(|(k, _)| *k != 6);
    second.extend([
        (0, raised),
        (1, raised),
        (2, (503.0, 505.0, 502.8, 504.5)),
        (6, (504.0, 504.2, 502.0, 502.5)),
    ]);
    let (first, second, backtest) = two_days(stop_and_trail(), &first, &second);
    let exit = (
        minute_on(day_1(), 1),
        minute_on(day_2(), 6),
        500.1,
        503.0,
        UNITS,
        290.0,
        "leg_sl",
        2,
    );
    assert_eq!(traded(&second), [exit]);
    assert_same_as_the_backtest(&first, &second, &backtest);
}

#[test]
fn a_carried_long_whose_target_is_reached_on_day_2_cancels_the_gtt_then_exits() {
    // Day 2's minute 4 reaches 511.00: the target 510.10 cancels the stop GTT, and the
    // exit sells the 100 at 510.10. 100 x (510.10 - 500.10) = 1000.00.
    let second = vec![(4, (505.0, 511.0, 504.8, 510.5))];
    let (first, second, backtest) = two_days(stop_and_target(), &Vec::new(), &second);
    let exit = (
        minute_on(day_1(), 1),
        minute_on(day_2(), 4),
        500.1,
        510.1,
        UNITS,
        1000.0,
        "leg_target",
        2,
    );
    assert_eq!(traded(&second), [exit]);
    assert_same_as_the_backtest(&first, &second, &backtest);
}

#[test]
fn a_carried_position_counts_the_slices_it_was_held_before_for_its_time_exit() {
    // max_hold 380 bars: held from slice 1 of day 1, that is 374 slices of day 1, so the
    // time exit is due at the open of day 2's minute 6 (374 + 6 = 380), as the backtest's
    // grid across both days counts (381 - 1). It cancels the stop GTT, then sells the 100
    // at that open, 500.10: no gain.
    let hold = |config: &mut RunConfig| {
        config.risk = oa_bt::rms::StrategyRiskConfig {
            max_hold_bars: Some(380),
            ..oa_bt::rms::StrategyRiskConfig::default()
        };
    };
    let (first, second, backtest) = two_days_with(stop_4(), (&Vec::new(), &Vec::new()), hold);
    let exit = (
        minute_on(day_1(), 1),
        minute_on(day_2(), 6),
        500.1,
        500.1,
        UNITS,
        0.0,
        "time_exit",
        2,
    );
    assert_eq!(traded(&second), [exit]);
    assert_same_as_the_backtest(&first, &second, &backtest);
}

// ---------------------------------------------------------------- carried state

/// Day 1 run to its close holding 100 with `rules` and its stop GTT confirmed; the
/// session, not consumed.
fn day_1_closed(rules: LegRules) -> Engine {
    let decided = minute_on(day_1(), 1);
    let mut kernel = session_on(day_1());
    let acts = vec![(decided, Act::Submit(buy(rules)))];
    run_day(
        &mut kernel,
        day_1(),
        &Vec::new(),
        &mut Script::new(acts),
        &mut Venue::default(),
    );
    kernel
}

fn field<'a>(state: &'a StateValue, key: &str) -> &'a StateValue {
    let StateValue::Map(entries) = state else {
        panic!("not a map");
    };
    entries
        .iter()
        .find(|(known, _)| known == key)
        .map(|(_, value)| value)
        .expect("the field is there")
}

fn with_field(state: &StateValue, key: &str, value: StateValue) -> StateValue {
    let StateValue::Map(entries) = state else {
        panic!("not a map");
    };
    let entries = entries
        .iter()
        .map(|(known, old)| {
            let new = if known == key {
                value.clone()
            } else {
                old.clone()
            };
            (known.clone(), new)
        })
        .collect();
    StateValue::Map(entries)
}

#[test]
fn a_carried_state_holds_the_position_its_leg_and_its_stop_gtt_and_not_the_days_orders() {
    let state = day_1_closed(stop_and_target())
        .carried_state()
        .expect("closed");
    let count = |key: &str| match field(&state, key) {
        StateValue::List(items) => items.len(),
        other => panic!("{key} is {other:?}"),
    };
    assert_eq!(
        (
            count("positions"),
            count("legs"),
            count("gtts"),
            count("orders")
        ),
        (1, 1, 1, 0)
    );
    assert_eq!(field(&state, "day"), &StateValue::Int(i64::from(day_1())));
}

#[test]
fn a_carried_state_is_taken_only_once_the_session_has_closed() {
    let mut kernel = session_on(day_1());
    let mut script = Script::new(Vec::new());
    venue_support::wait_before(&mut kernel, &venue_support::bars(&[], &[]), &mut script, 2);
    let refused = kernel.carried_state();
    assert!(matches!(
        refused,
        Err(oa_bt::engine::KernelError::LiveState(
            oa_bt::engine::LiveStateError::NotRestorable { .. }
        ))
    ));
}

#[test]
fn a_carried_state_goes_only_to_a_later_trading_day() {
    let state = day_1_closed(stop_4()).carried_state().expect("closed");
    let mut same_day = session_on(day_1());
    assert!(matches!(
        same_day.restore_carried_state(&state),
        Err(oa_bt::engine::KernelError::LiveState(
            oa_bt::engine::LiveStateError::CarriedMismatch { .. }
        ))
    ));
}

#[test]
fn a_carried_state_needs_the_capital_it_was_saved_with() {
    let state = day_1_closed(stop_4()).carried_state().expect("closed");
    let mut other = venue_support::session(RunConfig {
        capital_rupees: CAPITAL / 2.0,
        ..day_config(day_2(), day_2())
    });
    assert!(matches!(
        other.restore_carried_state(&state),
        Err(oa_bt::engine::KernelError::LiveState(
            oa_bt::engine::LiveStateError::CarriedMismatch { .. }
        ))
    ));
}

#[test]
fn a_carried_stop_gtt_of_more_units_than_the_position_is_refused_whole() {
    let state = day_1_closed(stop_4()).carried_state().expect("closed");
    let StateValue::List(gtts) = field(&state, "gtts") else {
        panic!("gtts is a list");
    };
    let terms = field(&gtts[0], "terms");
    let more = with_field(terms, "units", StateValue::Float(200.0));
    let gtt = with_field(&gtts[0], "terms", more);
    let edited = with_field(&state, "gtts", StateValue::List(vec![gtt]));
    let mut tuesday = session_on(day_2());
    assert!(matches!(
        tuesday.restore_carried_state(&edited),
        Err(oa_bt::engine::KernelError::LiveState(
            oa_bt::engine::LiveStateError::Malformed { .. }
        ))
    ));
    assert!(tuesday
        .position(venue_support::SBIN, Product::Cnc)
        .is_none());
}

#[test]
fn a_carried_mis_position_is_refused() {
    // An MIS position the day-1 session stopped with (no square-off yet).
    let mut kernel = session_on(day_1());
    let mis = order(Side::Buy, UNITS, OrderType::Market, Product::Mis);
    let acts = vec![(minute_on(day_1(), 1), Act::Submit(mis))];
    venue_support::wait_before(
        &mut kernel,
        &venue_support::bars(&[], &[]),
        &mut Script::new(acts),
        1,
    );
    kernel.take_commands();
    let bought = kernel.apply_fill(OrderId(1), UNITS, 500.1, minute_on(day_1(), 1));
    assert!(bought.is_ok());
    assert!(kernel.stop_session().is_ok());
    let state = kernel.carried_state().expect("stopped");
    let mut tuesday = session_on(day_2());
    assert!(matches!(
        tuesday.restore_carried_state(&state),
        Err(oa_bt::engine::KernelError::LiveState(
            oa_bt::engine::LiveStateError::CarriedMismatch { .. }
        ))
    ));
}

fn stop_4() -> LegRules {
    LegRules {
        stop: Some(Distance::Points(4.0)),
        ..LegRules::default()
    }
}

#[test]
fn a_carried_leg_without_its_stop_gtt_crossed_before_the_open_exits_at_the_open() {
    // The carried state lost its stop GTT. At 09:10 the last price 495.00 is below the
    // stop 496.10: `protect` places no GTT and the exit at market waits for the open; the
    // first tick at 09:15 sends it, flagged stop_crossed.
    let state = day_1_closed(stop_4()).carried_state().expect("closed");
    let without = with_field(&state, "gtts", StateValue::List(Vec::new()));
    let mut tuesday = session_on(day_2());
    let ids = tuesday
        .restore_carried_state(&without)
        .expect("day 2 takes it");
    assert!(ids.gtts.is_empty());
    assert!(tuesday.take_commands().is_empty(), "no GTT by itself");
    let before_open = minute_on(day_2(), 0) - 5 * 60 * 1_000_000_000;
    let sbin = venue_support::SBIN;
    assert!(tuesday.push_tick(sbin, 495.0, before_open).is_ok());
    assert_eq!(
        tuesday.protect(sbin, Product::Cnc),
        Ok(oa_bt::engine::Protect::StopCrossed)
    );
    assert!(tuesday.take_commands().is_empty(), "it waits for the open");
    assert!(tuesday
        .push_tick(sbin, 495.0, minute_on(day_2(), 0) + 1)
        .is_ok());
    let placed = venue_support::placed_roles(&tuesday.take_commands());
    let exit = OrderRole::RiskExit(RiskExitReason::StopCrossed);
    assert_eq!(placed, [(OrderId(1), Side::Sell, UNITS, exit)]);
    // It fills at 494.90: the trade closes for the stop's reason, flagged stop_crossed.
    let filled = tuesday.apply_fill(OrderId(1), UNITS, 494.9, minute_on(day_2(), 0) + 2);
    assert!(filled.is_ok(), "{filled:?}");
    run_day(
        &mut tuesday,
        day_2(),
        &Vec::new(),
        &mut Script::new(Vec::new()),
        &mut Venue::default(),
    );
    let output = tuesday.finish().expect("day 2 finished");
    let fill = output.fills[0].fill;
    assert_eq!(fill.reason.code(), "leg_sl");
    assert!(fill.flags.contains(FillFlags::STOP_CROSSED));
}

#[test]
fn a_stop_gtt_left_resting_on_a_flat_position_at_the_close_is_carried_and_cancelled_at_once() {
    // The broker sells the 100 by itself at the open of minute 370 and refuses every
    // cancellation of g1 to the close, which completes with g1 resting on a flat
    // position. The carried state holds g1 marked to cancel, and day 2's session takes it
    // and asks the cancellation at once.
    let mut kernel = session_on(day_1());
    let acts = vec![(minute_on(day_1(), 1), Act::Submit(buy(stop_4())))];
    let mut venue = Venue {
        sells_all_at: Some(370),
        ..Venue::default()
    };
    run_day(
        &mut kernel,
        day_1(),
        &Vec::new(),
        &mut Script::new(acts),
        &mut venue,
    );
    let state = kernel.carried_state().expect("closed");
    let StateValue::List(gtts) = field(&state, "gtts") else {
        panic!("gtts is a list");
    };
    assert_eq!(field(&state, "positions"), &StateValue::List(Vec::new()));
    assert_eq!(gtts.len(), 1);
    assert_eq!(field(&gtts[0], "to_cancel"), &StateValue::Bool(true));
    let mut tuesday = session_on(day_2());
    let ids = tuesday
        .restore_carried_state(&state)
        .expect("day 2 takes it");
    let carried = ids.gtts[0].1;
    let commands = tuesday.take_commands();
    assert!(
        matches!(commands[..], [VenueCommand::CancelGtt { gtt, .. }] if gtt == carried),
        "{commands:?}"
    );
    assert!(tuesday.apply_gtt_cancelled(carried).is_ok());
    let state = tuesday.gtt_view(carried).map(|view| view.gtt.status);
    assert_eq!(state, Some(GttStatus::Canceled));
}

#[test]
fn a_carried_leg_whose_stop_gtt_is_rejected_again_on_day_2_is_exited_at_market() {
    // Day 1: the broker rejects g1 after its placement at minute 100, the leg's first
    // rejection, and g2 protects it again. Day 2: the broker rejects g2's successor, the
    // leg's second rejection: the leg is exited at market, stop_rejected (design 3.1, row
    // rejected), not protected a third time.
    let mut kernel = session_on(day_1());
    let acts = vec![(minute_on(day_1(), 1), Act::Submit(buy(stop_4())))];
    let mut venue = Venue {
        rejects_stop_at: Some(100),
        ..Venue::default()
    };
    run_day(
        &mut kernel,
        day_1(),
        &Vec::new(),
        &mut Script::new(acts),
        &mut venue,
    );
    let state = kernel.carried_state().expect("closed");
    let mut tuesday = session_on(day_2());
    let ids = tuesday
        .restore_carried_state(&state)
        .expect("day 2 takes it");
    let carried = ids.gtts[0].1;
    assert!(tuesday.apply_gtt_ended(carried, GttEnd::Rejected).is_ok());
    let sbin = venue_support::SBIN;
    assert!(tuesday
        .push_tick(sbin, 500.1, minute_on(day_2(), 0) + 1)
        .is_ok());
    let commands = tuesday.take_commands();
    assert!(gtt_free(&commands), "{commands:?}");
    let exit = OrderRole::RiskExit(RiskExitReason::StopRejected);
    let placed = venue_support::placed_roles(&commands);
    assert_eq!(placed, [(OrderId(1), Side::Sell, UNITS, exit)]);
}

/// True when `commands` hold no GTT's command.
fn gtt_free(commands: &[VenueCommand]) -> bool {
    commands.iter().all(|command| command.gtt().is_none())
}

#[test]
fn a_carried_run_keeps_counting_toward_the_daily_loss_measure() {
    // Decision 11.9. Day 1 closes with the 100 bought at 500.10 marked at 499.10: the run
    // is 100 x -1.00 = -100.00 down. Day 2 starts from that run, still open: -100.00, and
    // a tick at 498.60 makes it 100 x -1.50 = -150.00.
    let lower = (499.1, 499.1, 499.1, 499.1);
    let first: DayBars = (2..SLICES).map(|k| (k, lower)).collect();
    let mut kernel = session_on(day_1());
    let acts = vec![(minute_on(day_1(), 1), Act::Submit(buy(stop_4())))];
    run_day(
        &mut kernel,
        day_1(),
        &first,
        &mut Script::new(acts),
        &mut Venue::default(),
    );
    assert_eq!(kernel.daily_loss_measure(), -100.0);
    let state = kernel.carried_state().expect("closed");
    let mut tuesday = session_on(day_2());
    assert!(tuesday.restore_carried_state(&state).is_ok());
    assert_eq!(tuesday.daily_loss_measure(), -100.0);
    let sbin = venue_support::SBIN;
    assert!(tuesday
        .push_tick(sbin, 498.6, minute_on(day_2(), 0) + 1)
        .is_ok());
    assert_eq!(tuesday.daily_loss_measure(), -150.0);
}

#[test]
fn a_carried_state_goes_only_to_a_session_of_its_instruments() {
    let state = day_1_closed(stop_4()).carried_state().expect("closed");
    let config = day_config(day_2(), day_2());
    let mut other = venue_support::session_of(config, venue_support::nifty_call());
    let refused = other.restore_carried_state(&state);
    assert!(
        matches!(
            &refused,
            Err(KernelError::LiveState(LiveStateError::CarriedMismatch { reason }))
                if reason.contains("NIFTY31JUL2525000CE")
        ),
        "{refused:?}"
    );
    assert!(other.position(venue_support::SBIN, Product::Cnc).is_none());
}

#[test]
fn a_carried_state_refused_is_named_as_a_carried_state() {
    // The refusal names where the carried state breaks and says to pass what
    // carried_state() returned; it does not send the user to live_state().
    let state = day_1_closed(stop_4()).carried_state().expect("closed");
    let StateValue::List(gtts) = field(&state, "gtts") else {
        panic!("gtts is a list");
    };
    let terms = field(&gtts[0], "terms");
    let more = with_field(terms, "units", StateValue::Float(200.0));
    let gtt = with_field(&gtts[0], "terms", more);
    let edited = with_field(&state, "gtts", StateValue::List(vec![gtt]));
    let refused = session_on(day_2())
        .restore_carried_state(&edited)
        .expect_err("refused whole");
    let message = refused.to_string();
    assert!(message.contains("carried_state.gtts[0]"), "{message}");
    assert!(
        message.contains("pass the state carried_state() returned"),
        "{message}"
    );
    assert!(!message.contains("live_state"), "{message}");
}

/// Advance `kernel`, a session of `day`, adding the flat bars and releasing each slice
/// but filling nothing, until it waits before slice `slice`.
fn wait_on(kernel: &mut Engine, day: Day, slice: usize) {
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if next_slice >= slice => return,
            Pause::Waiting { next_slice } => {
                let k = u32::try_from(next_slice).expect("a slice of the day");
                let (open, high, low, close) = FLAT;
                let bar = BaseBar {
                    ts_open: minute_on(day, k),
                    open,
                    high,
                    low,
                    close,
                    volume: 1_000.0,
                };
                kernel.add_bar(venue_support::SBIN, bar).expect("a bar");
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => panic!("the session finished before slice {slice}"),
            _ => {}
        }
    }
}

#[test]
fn a_carried_stop_gtt_older_than_360_days_is_cancelled_then_placed_again_at_the_open() {
    // GTT-2 in a session: the carried g1 was created 361 days before day 2. At day 2's
    // session open it is cancelled, and only once the venue has taken that does the leg's
    // new stop GTT go, at its stop 496.10 (child 491.10), created on day 2.
    let state = day_1_closed(stop_4()).carried_state().expect("closed");
    let StateValue::List(gtts) = field(&state, "gtts") else {
        panic!("gtts is a list");
    };
    let created = i64::from(day_2()) - 361;
    let old = with_field(&gtts[0], "created_on", StateValue::Int(created));
    let edited = with_field(&state, "gtts", StateValue::List(vec![old]));
    let mut tuesday = session_on(day_2());
    let ids = tuesday
        .restore_carried_state(&edited)
        .expect("day 2 takes it");
    let carried = ids.gtts[0].1;
    assert!(tuesday.take_commands().is_empty());
    wait_on(&mut tuesday, day_2(), 1);
    let commands = tuesday.take_commands();
    assert!(
        matches!(commands[..], [VenueCommand::CancelGtt { gtt, .. }] if gtt == carried),
        "{commands:?}"
    );
    assert!(tuesday.apply_gtt_cancelled(carried).is_ok());
    let renewed = GttId(carried.0 + 1);
    let placed = tuesday.take_commands();
    assert!(
        matches!(placed[..], [VenueCommand::PlaceGtt { gtt, terms, .. }]
            if gtt == renewed && terms.trigger_rupees == 496.1 && terms.limit_rupees == 491.1),
        "{placed:?}"
    );
    let view = tuesday
        .gtt_view(renewed)
        .expect("the new GTT is the session's");
    assert_eq!(view.gtt.created_on, day_2());
}

#[test]
fn a_same_day_restart_keeps_the_gtts_and_their_ids() {
    // The session saved at the 09:17 close with g1 resting is restored into a restarted
    // session: g1 rests again under its id, and the venue can fire it there.
    let mut kernel = session_on(day_1());
    let acts = vec![(minute_on(day_1(), 1), Act::Submit(buy(stop_4())))];
    let bars = venue_support::bars(&[], &[]);
    venue_support::wait_before(&mut kernel, &bars, &mut Script::new(acts), 1);
    let mut venue = Venue::default();
    venue.slice(&mut kernel, day_1(), 1, FLAT);
    venue_support::pause_at_close(&mut kernel, &bars, 2);
    let saved = kernel.live_state();
    let mut restarted = session_on(day_1());
    venue_support::pause_at_close(&mut restarted, &bars, 2);
    assert!(restarted.restore_live_state(&saved).is_ok());
    let view = restarted.gtt_view(GttId(1)).expect("g1 is restored");
    assert_eq!(view.gtt.status, oa_bt::exec::GttStatus::Active);
    assert_eq!(view.terms.trigger_rupees, 496.1);
    venue_support::wait_before(&mut restarted, &bars, &mut Script::new(Vec::new()), 3);
    let child = restarted.apply_gtt_triggered(GttId(1), minute_on(day_1(), 3) + 1);
    assert!(child.is_ok());
}
