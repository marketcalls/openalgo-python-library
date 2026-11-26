#![cfg(all(feature = "costs", feature = "time"))]
//! The carried state of a live session that stopped mid-day without its close (a crash),
//! rebuilt from its last saved state and the venue's results after that save (design
//! 20-positional-live 4.2 step 2, review M7; unit N5c item N5c-1): `restore_stopped_state`
//! into a session built for the saved day, the results, `stop_session`, `carried_state`.
//!
//! The expected carried states are the closed-session path's: the same session stopped
//! right after its save (with the same results) and closed, then `carried_state`; or
//! figures worked by hand.
//!
//! SBIN on Tue 2025-07-01 (day 1), carried to Wed 2025-07-02 (day 2), 1-minute bars flat
//! at (500.10, 500.50, 499.70, 500.10) unless a test writes one, no charges, no slippage.
//! 100 are bought at market, decided at the 09:15 close, and fill at the 09:16 open at
//! 500.10; a stop of 4 points is 496.10, its GTT child's limit 491.10.

mod venue_support;

use oa_bt::engine::{
    BaseBar, Engine, KernelError, LiveStateError, Pause, Protect, RunConfig, StateValue,
    VenueCommand, VenueError,
};
use oa_bt::exec::{GttId, GttStatus, InstrumentSpec, OrderId, OrderRequest, OrderType};
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{Distance, LegRules, PositionalStops, StrategyRiskConfig, TrailSpec};
use venue_support::{config, order, sbin, Act, Ohlc, Script, CAPITAL, NSE_OPEN_MINUTE, SBIN};

const FLAT: Ohlc = (500.1, 500.5, 499.7, 500.1);
const UNITS: f64 = 100.0;
const ENTRY: OrderId = OrderId(1);
const G1: GttId = GttId(1);
const SECOND: i64 = 1_000_000_000;

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

fn day_config(day: Day) -> RunConfig {
    RunConfig {
        first_day: day,
        last_day: day,
        ..config()
    }
}

fn session_on(day: Day, instrument: InstrumentSpec) -> Engine {
    venue_support::session_of(day_config(day), instrument)
}

fn stop_4() -> LegRules {
    LegRules {
        stop: Some(Distance::Points(4.0)),
        ..LegRules::default()
    }
}

fn stop_and_trail() -> LegRules {
    LegRules {
        trail: Some(TrailSpec {
            trigger: Distance::Points(2.0),
            step: None,
        }),
        ..stop_4()
    }
}

fn buy(units: f64, product: Product, rules: LegRules) -> OrderRequest {
    OrderRequest {
        leg_rules: rules,
        ..order(Side::Buy, units, OrderType::Market, product)
    }
}

/// Run `kernel`, a session of `day`, adding a bar in each slice (`FLAT`, or the one
/// `written` for it) and releasing it, the script acting at its decisions, until it waits
/// before slice `until`. The commands are left for the test.
fn run_until(
    kernel: &mut Engine,
    day: Day,
    written: &[(u32, Ohlc)],
    script: &mut Script,
    until: usize,
) {
    use oa_bt::engine::Driver;
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if next_slice >= until => return,
            Pause::Waiting { next_slice } => {
                let k = u32::try_from(next_slice).expect("a slice of the day");
                let (open, high, low, close) = written
                    .iter()
                    .find(|(m, _)| *m == k)
                    .map_or(FLAT, |(_, bar)| *bar);
                let bar = BaseBar {
                    ts_open: minute_on(day, k),
                    open,
                    high,
                    low,
                    close,
                    volume: 1_000.0,
                };
                kernel.add_bar(SBIN, bar).expect("a bar of a closed slice");
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => panic!("the session finished before slice {until}"),
            pause => script.on_pause(kernel, &pause).expect("the script acts"),
        }
    }
}

/// Day 1's session of `instrument` waiting before 09:16 with the entry `request` of the
/// 09:15 close sent (its place taken), not filled.
fn entry_sent(
    instrument: InstrumentSpec,
    request: OrderRequest,
    edit: fn(&mut RunConfig),
) -> Engine {
    let mut config = day_config(day_1());
    edit(&mut config);
    let mut kernel = venue_support::session_of(config, instrument);
    let acts = vec![(minute_on(day_1(), 1), Act::Submit(request))];
    run_until(&mut kernel, day_1(), &[], &mut Script::new(acts), 1);
    let placed = kernel.take_commands();
    assert!(
        matches!(placed[..], [VenueCommand::Place { order, .. }] if order == ENTRY),
        "{placed:?}"
    );
    kernel
}

/// No change to the configuration.
fn as_is(_: &mut RunConfig) {}

/// `entry_sent` with the entry filled at the 09:16 open at `price`.
fn entry_filled(instrument: InstrumentSpec, request: OrderRequest, price: f64) -> Engine {
    let mut kernel = entry_sent(instrument, request.clone(), as_is);
    let units = kernel.order(ENTRY).expect("the entry").quantity_units;
    kernel
        .apply_fill(ENTRY, units, price, minute_on(day_1(), 1))
        .expect("the entry fills");
    kernel
}

/// `entry_filled` with the stop GTT's place taken and confirmed.
fn holding_with_stop_gtt(instrument: InstrumentSpec, request: OrderRequest, price: f64) -> Engine {
    let mut kernel = entry_filled(instrument, request, price);
    let placed = kernel.take_commands();
    assert!(
        matches!(placed[..], [VenueCommand::PlaceGtt { gtt, .. }] if gtt == G1),
        "{placed:?}"
    );
    kernel.apply_gtt_placed(G1).expect("the venue confirms");
    kernel
}

/// The carried state of `kernel` stopped where it stands.
fn stopped_and_carried(mut kernel: Engine) -> StateValue {
    kernel.stop_session().expect("the session stops");
    kernel.carried_state().expect("a stopped session carries")
}

/// A session of day 1 built for `instrument` given `saved`, carried to day 2, with
/// `unsent`: the session, standing where the save stood.
fn rebuilt(instrument: InstrumentSpec, saved: &StateValue, unsent: &[GttId]) -> Engine {
    let mut kernel = session_on(day_1(), instrument);
    kernel
        .restore_stopped_state(saved, day_2(), unsent)
        .expect("the stopped session takes its save");
    kernel
}

fn field<'a>(state: &'a StateValue, key: &str) -> &'a StateValue {
    let StateValue::Map(entries) = state else {
        panic!("not a map: {state:?}");
    };
    entries
        .iter()
        .find(|(known, _)| known == key)
        .map(|(_, value)| value)
        .unwrap_or_else(|| panic!("no field {key}"))
}

fn items<'a>(state: &'a StateValue, key: &str) -> &'a [StateValue] {
    match field(state, key) {
        StateValue::List(items) => items,
        other => panic!("{key} is {other:?}"),
    }
}

/// (units, average price) of each carried position.
fn positions(state: &StateValue) -> Vec<(f64, f64)> {
    items(state, "positions")
        .iter()
        .map(|p| {
            let number = |key| match field(p, key) {
                StateValue::Float(x) => *x,
                StateValue::Int(x) => *x as f64,
                other => panic!("{key} is {other:?}"),
            };
            (number("quantity_units"), number("average_price_rupees"))
        })
        .collect()
}

/// (trigger, child limit, units) of each carried GTT.
fn gtt_terms(state: &StateValue) -> Vec<(f64, f64, f64)> {
    items(state, "gtts")
        .iter()
        .map(|gtt| {
            let terms = field(gtt, "terms");
            let number = |key| match field(terms, key) {
                StateValue::Float(x) => *x,
                other => panic!("{key} is {other:?}"),
            };
            (
                number("trigger_rupees"),
                number("limit_rupees"),
                number("units"),
            )
        })
        .collect()
}

fn refusal(result: Result<Vec<(OrderId, OrderId)>, KernelError>) -> LiveStateError {
    match result {
        Err(KernelError::LiveState(error)) => error,
        other => panic!("not refused as a live state: {other:?}"),
    }
}

// ---------------------------------------------------------------- the closed-session path

#[test]
fn a_cnc_leg_with_its_stop_gtt_resting_is_carried_as_the_session_stopped_after_its_save() {
    let mut kernel = holding_with_stop_gtt(sbin(), buy(UNITS, Product::Cnc, stop_4()), 500.1);
    run_until(&mut kernel, day_1(), &[], &mut Script::new(Vec::new()), 5);
    let saved = kernel.live_state();
    let expected = stopped_and_carried(kernel);

    let carried = stopped_and_carried(rebuilt(sbin(), &saved, &[]));

    assert_eq!(carried, expected);
    assert_eq!(positions(&carried), [(UNITS, 500.1)]);
    assert_eq!(gtt_terms(&carried), [(496.1, 491.1, UNITS)]);
}

#[test]
fn an_nrml_futures_leg_is_carried_as_the_session_stopped_after_its_save() {
    let future = InstrumentSpec {
        symbol: "SBIN31JUL25FUT".to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Future,
        lot_size_units: 750,
        last_trading_day: Some(days_from_civil(2025, 7, 31)),
        ..sbin()
    };
    let request = buy(750.0, Product::Nrml, stop_4());
    let mut kernel = holding_with_stop_gtt(future.clone(), request, 500.1);
    run_until(&mut kernel, day_1(), &[], &mut Script::new(Vec::new()), 5);
    let saved = kernel.live_state();
    let expected = stopped_and_carried(kernel);

    let carried = stopped_and_carried(rebuilt(future, &saved, &[]));

    assert_eq!(carried, expected);
    assert_eq!(positions(&carried), [(750.0, 500.1)]);
    assert_eq!(gtt_terms(&carried), [(496.1, 491.1, 750.0)]);
}

fn rearm_daily(config: &mut RunConfig) {
    config.risk = StrategyRiskConfig {
        positional_stops: PositionalStops::RearmDaily,
        ..StrategyRiskConfig::default()
    };
}

#[test]
fn a_rearm_daily_leg_is_carried_with_its_day_stop_ended_as_the_session_stopped_after_its_save() {
    let mut kernel = entry_sent(sbin(), buy(UNITS, Product::Cnc, stop_4()), rearm_daily);
    kernel
        .apply_fill(ENTRY, UNITS, 500.1, minute_on(day_1(), 1))
        .expect("the entry fills");
    let placed = kernel.take_commands();
    assert!(
        matches!(
            placed[..],
            [VenueCommand::Place {
                order: OrderId(2),
                ..
            }]
        ),
        "the day stop is a DAY order: {placed:?}"
    );
    let saved = kernel.live_state();
    let expected = stopped_and_carried(kernel);
    let mut config = day_config(day_1());
    rearm_daily(&mut config);
    let mut session = venue_support::session_of(config, sbin());
    session
        .restore_stopped_state(&saved, day_2(), &[])
        .expect("the stopped session takes its save");

    let carried = stopped_and_carried(session);

    assert_eq!(carried, expected);
    assert!(
        items(&carried, "orders").is_empty(),
        "the DAY stop is not carried"
    );
    assert!(items(&carried, "gtts").is_empty());
    assert_eq!(positions(&carried), [(UNITS, 500.1)]);
}

// ---------------------------------------------------------------- the results after the save

#[test]
fn a_day_entry_working_at_the_save_and_filled_after_it_is_carried_with_its_units() {
    let kernel = entry_sent(sbin(), buy(UNITS, Product::Cnc, stop_4()), as_is);
    let saved = kernel.live_state();
    let mut stopped = kernel;
    stopped
        .apply_fill(ENTRY, UNITS, 500.1, minute_on(day_1(), 1) + 5 * SECOND)
        .expect("the entry fills");
    let expected = stopped_and_carried(stopped);
    let mut session = rebuilt(sbin(), &saved, &[]);
    session
        .apply_fill(ENTRY, UNITS, 500.1, minute_on(day_1(), 1) + 5 * SECOND)
        .expect("the fill after the save");

    let carried = stopped_and_carried(session);

    assert_eq!(carried, expected);
    assert_eq!(positions(&carried), [(UNITS, 500.1)]);
    // Its stop GTT was never placed (no session sent it): carried without one.
    assert!(items(&carried, "gtts").is_empty());
    assert_eq!(items(&carried, "legs").len(), 1);
}

#[test]
fn a_day_entry_working_at_the_save_with_no_fill_after_it_leaves_no_position() {
    let saved = entry_sent(sbin(), buy(UNITS, Product::Cnc, stop_4()), as_is).live_state();

    let carried = stopped_and_carried(rebuilt(sbin(), &saved, &[]));

    assert!(positions(&carried).is_empty());
    assert!(items(&carried, "legs").is_empty());
    assert!(items(&carried, "orders").is_empty());
}

#[test]
fn a_leg_carried_without_a_stop_gtt_gets_one_from_protect_the_next_day() {
    let saved = entry_sent(sbin(), buy(UNITS, Product::Cnc, stop_4()), as_is).live_state();
    let mut session = rebuilt(sbin(), &saved, &[]);
    session
        .apply_fill(ENTRY, UNITS, 500.1, minute_on(day_1(), 1) + 5 * SECOND)
        .expect("the fill after the save");
    let carried = stopped_and_carried(session);
    let mut wednesday = session_on(day_2(), sbin());
    let ids = wednesday
        .restore_carried_state(&carried)
        .expect("day 2 takes it");
    assert!(ids.gtts.is_empty());
    assert!(wednesday
        .push_tick(SBIN, 500.1, minute_on(day_2(), 0) - 60 * SECOND)
        .is_ok());

    assert_eq!(wednesday.protect(SBIN, Product::Cnc), Ok(Protect::Placed));

    let placed = wednesday.take_commands();
    assert!(
        matches!(placed[..], [VenueCommand::PlaceGtt { terms, .. }]
            if terms.trigger_rupees == 496.1 && terms.units == UNITS),
        "{placed:?}"
    );
}

#[test]
fn a_stop_gtt_place_sent_and_not_answered_is_carried_on_the_terms_asked() {
    let mut kernel = entry_filled(sbin(), buy(UNITS, Product::Cnc, stop_4()), 500.1);
    kernel.take_commands();
    let saved = kernel.live_state();

    let carried = stopped_and_carried(rebuilt(sbin(), &saved, &[]));

    assert_eq!(gtt_terms(&carried), [(496.1, 491.1, UNITS)]);
}

#[test]
fn a_stop_gtt_place_that_never_reached_the_venue_is_not_carried() {
    let mut kernel = entry_filled(sbin(), buy(UNITS, Product::Cnc, stop_4()), 500.1);
    kernel.take_commands();
    let saved = kernel.live_state();

    let carried = stopped_and_carried(rebuilt(sbin(), &saved, &[G1]));

    assert!(items(&carried, "gtts").is_empty());
    assert_eq!(positions(&carried), [(UNITS, 500.1)]);
}

#[test]
fn a_stop_gtt_place_the_save_holds_never_handed_out_is_dropped_as_the_closed_path_drops_it() {
    // The save was made with the stop GTT's place still among the commands.
    let kernel = entry_filled(sbin(), buy(UNITS, Product::Cnc, stop_4()), 500.1);
    let saved = kernel.live_state();
    let expected = stopped_and_carried(kernel);

    let carried = stopped_and_carried(rebuilt(sbin(), &saved, &[]));

    assert_eq!(carried, expected);
    assert!(items(&carried, "gtts").is_empty());
}

/// The trailed stop GTT's child limit: 1 percent below its 501.00 trigger (the NSE cash
/// segment's child buffer, GTT-1), 495.99, on the tick below: 495.95.
const TRAILED_LIMIT: f64 = 495.95;

/// Day 1 holding 100 with `stop_and_trail` and its stop GTT confirmed, a tick at 503.00 in
/// the 09:18 slice: the trail moves the stop to 501.00 and the modification is taken.
fn trail_sent() -> Engine {
    let request = buy(UNITS, Product::Cnc, stop_and_trail());
    let mut kernel = holding_with_stop_gtt(sbin(), request, 500.1);
    run_until(&mut kernel, day_1(), &[], &mut Script::new(Vec::new()), 3);
    kernel
        .push_tick(SBIN, 503.0, minute_on(day_1(), 3) + 15 * SECOND)
        .expect("a tick");
    let modified = kernel.take_commands();
    assert!(
        matches!(modified[..], [VenueCommand::ModifyGtt { gtt, terms }]
            if gtt == G1 && terms.trigger_rupees == 501.0),
        "{modified:?}"
    );
    kernel
}

#[test]
fn a_stop_gtt_modification_waiting_at_the_save_is_carried_on_the_terms_asked() {
    let saved = trail_sent().live_state();

    let carried = stopped_and_carried(rebuilt(sbin(), &saved, &[]));

    assert_eq!(gtt_terms(&carried), [(501.0, TRAILED_LIMIT, UNITS)]);
}

#[test]
fn a_stop_gtt_modification_that_never_reached_the_venue_is_carried_on_the_terms_that_rest() {
    let saved = trail_sent().live_state();

    let carried = stopped_and_carried(rebuilt(sbin(), &saved, &[G1]));

    assert_eq!(gtt_terms(&carried), [(496.1, 491.1, UNITS)]);
}

#[test]
fn a_stop_gtt_modified_after_the_save_is_carried_on_its_new_terms() {
    let saved = trail_sent().live_state();
    let mut session = rebuilt(sbin(), &saved, &[]);
    session
        .apply_gtt_modified(G1)
        .expect("the venue confirmed it");

    let carried = stopped_and_carried(session);

    assert_eq!(gtt_terms(&carried), [(501.0, TRAILED_LIMIT, UNITS)]);
}

#[test]
fn a_stop_gtt_that_fired_after_the_save_with_its_child_filled_leaves_nothing_carried() {
    let mut kernel = holding_with_stop_gtt(sbin(), buy(UNITS, Product::Cnc, stop_4()), 500.1);
    run_until(&mut kernel, day_1(), &[], &mut Script::new(Vec::new()), 5);
    let saved = kernel.live_state();
    let mut session = rebuilt(sbin(), &saved, &[]);
    let fired = minute_on(day_1(), 7);
    let child = session.apply_gtt_triggered(G1, fired).expect("it fired");
    session
        .apply_fill(child, UNITS, 495.0, fired + SECOND)
        .expect("its child sold");

    let carried = stopped_and_carried(session);

    assert!(positions(&carried).is_empty());
    assert!(items(&carried, "gtts").is_empty());
    assert!(items(&carried, "open_trades").is_empty());
}

// ---------------------------------------------------------------- refusals

fn cnc_save() -> StateValue {
    let mut kernel = holding_with_stop_gtt(sbin(), buy(UNITS, Product::Cnc, stop_4()), 500.1);
    run_until(&mut kernel, day_1(), &[], &mut Script::new(Vec::new()), 5);
    kernel.live_state()
}

#[test]
fn a_save_of_the_day_it_would_be_carried_to_is_refused() {
    let saved = cnc_save();
    let mut session = session_on(day_1(), sbin());

    let refused = refusal(session.restore_stopped_state(&saved, day_1(), &[]));

    assert!(
        matches!(refused, LiveStateError::OtherSession { .. }),
        "{refused:?}"
    );
    assert!(session.position(SBIN, Product::Cnc).is_none());
}

#[test]
fn a_save_of_other_instruments_is_refused() {
    let saved = cnc_save();
    let mut session = session_on(day_1(), venue_support::nifty_call());

    let refused = refusal(session.restore_stopped_state(&saved, day_2(), &[]));

    assert!(
        matches!(refused, LiveStateError::OtherSession { .. }),
        "{refused:?}"
    );
}

#[test]
fn a_save_of_another_capital_is_refused() {
    let saved = cnc_save();
    let config = RunConfig {
        capital_rupees: CAPITAL / 2.0,
        ..day_config(day_1())
    };
    let mut session = venue_support::session_of(config, sbin());

    let refused = refusal(session.restore_stopped_state(&saved, day_2(), &[]));

    assert!(
        matches!(refused, LiveStateError::OtherSession { .. }),
        "{refused:?}"
    );
}

#[test]
fn a_save_of_another_day_than_the_sessions_is_refused() {
    let saved = cnc_save();
    let mut session = session_on(day_2(), sbin());

    let refused = refusal(session.restore_stopped_state(&saved, days_from_civil(2025, 7, 3), &[]));

    assert!(
        matches!(refused, LiveStateError::OtherSession { .. }),
        "{refused:?}"
    );
}

#[test]
fn a_session_that_has_run_a_slice_takes_no_stopped_state() {
    let saved = cnc_save();
    let mut session = session_on(day_1(), sbin());
    run_until(&mut session, day_1(), &[], &mut Script::new(Vec::new()), 1);

    let refused = refusal(session.restore_stopped_state(&saved, day_2(), &[]));

    assert!(
        matches!(refused, LiveStateError::NotRestorable { .. }),
        "{refused:?}"
    );
}

#[test]
fn an_unsent_gtt_that_waits_for_no_answer_is_refused_and_nothing_is_restored() {
    let saved = cnc_save();
    let mut session = session_on(day_1(), sbin());

    let refused = session.restore_stopped_state(&saved, day_2(), &[G1]);

    assert!(
        matches!(
            refused,
            Err(KernelError::Venue(VenueError::GttState {
                gtt: G1,
                status: GttStatus::Active,
                ..
            }))
        ),
        "{refused:?}"
    );
    assert!(session.position(SBIN, Product::Cnc).is_none());
}

#[test]
fn a_state_of_another_layout_is_refused() {
    let saved = cnc_save();
    let StateValue::Map(entries) = &saved else {
        panic!("a map");
    };
    let older = StateValue::Map(
        entries
            .iter()
            .map(|(key, value)| {
                let value = if key == "layout" {
                    StateValue::Int(5)
                } else {
                    value.clone()
                };
                (key.clone(), value)
            })
            .collect(),
    );
    let mut session = session_on(day_1(), sbin());

    let refused = refusal(session.restore_stopped_state(&older, day_2(), &[]));

    assert_eq!(refused, LiveStateError::Version { found: 5 });
}
