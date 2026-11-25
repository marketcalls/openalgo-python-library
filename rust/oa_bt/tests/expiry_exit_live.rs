#![cfg(all(feature = "costs", feature = "time"))]
//! A derivatives position at expiry in a live session (design 20-positional-live section 9;
//! unit N4 item N4-2): a contract with a positional exit day is exited by the engine when
//! the session reaches the square-off time of that day (role `risk_exit`, reason
//! `expiry_exit`), its stop GTT cancelled first; an index future's stop GTT is cancelled at
//! the close of its last trading day and the position left to the exchange, and the next
//! day's session records its exit at the carried last close, reason `expiry`, flagged
//! `filled_at_last_close`, and never trades it.
//!
//! NFO contracts of lot 75 whose last trading day is Thu 2025-07-31, the NFO square-off
//! 15:10 (minute 355), 1-minute slices with bars flat at 1,000.00, no charges, no slippage.
//! A MARKET buy of one lot in NRML is decided at the 09:15 close (o1) and the venue fills it
//! at 1,000.00 at the 09:16 open.

use oa_bt::engine::{
    BaseBar, Driver, Engine, KernelError, LiveStateError, OrderRole, Pause, RiskExitReason,
    RunConfig, StateValue, Submission, VenueCommand,
};
use oa_bt::exec::{
    FillAt, FillFlags, FillReason, InstrumentSpec, OrderId, OrderRequest, OrderType, RejectCode,
};
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{Distance, ExitReason, LegRules, PositionalStops, StrategyRiskConfig};

mod venue_support;

use venue_support::{config, order, sbin, session_of, Act, Script, NSE_OPEN_MINUTE, SBIN};

const SLICES: usize = 375;
const SQUARE_OFF: usize = 355;
const ENTRY: OrderId = OrderId(1);

fn last_day() -> Day {
    days_from_civil(2025, 7, 31)
}

fn next_day() -> Day {
    days_from_civil(2025, 8, 1)
}

fn minute_on(day: Day, k: u32) -> i64 {
    ts_from_ist(day, NSE_OPEN_MINUTE + k)
}

fn contract(symbol: &str, exit_day: Option<Day>) -> InstrumentSpec {
    InstrumentSpec {
        symbol: symbol.to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Future,
        lot_size_units: 75,
        last_trading_day: Some(last_day()),
        positional_exit_day: exit_day,
        ..sbin()
    }
}

fn stock_future() -> InstrumentSpec {
    contract("SBIN31JUL25FUT", Some(last_day()))
}

fn index_future() -> InstrumentSpec {
    contract("NIFTY31JUL25FUT", None)
}

fn session_on(day: Day, instrument: InstrumentSpec) -> Engine {
    session_with(day, instrument, StrategyRiskConfig::default())
}

/// `session_on`, with the strategy's risk settings `risk`.
fn session_with(day: Day, instrument: InstrumentSpec, risk: StrategyRiskConfig) -> Engine {
    let config = RunConfig {
        first_day: day,
        last_day: day,
        risk,
        ..config()
    };
    session_of(config, instrument)
}

fn buy(rules: LegRules) -> OrderRequest {
    OrderRequest {
        leg_rules: rules,
        ..order(Side::Buy, 75.0, OrderType::Market, Product::Nrml)
    }
}

fn stop_50() -> LegRules {
    LegRules {
        stop: Some(Distance::Points(50.0)),
        ..LegRules::default()
    }
}

/// The close of the last bar of every session, 15:29 to 15:30: apart from the entry's
/// 1,000.00 and every other bar's, so a record at the wrong price shows.
const LAST_CLOSE: f64 = 988.0;

/// Run `kernel`, a session of `day`, adding bars flat at 1,000.00 (the last one closing at
/// `LAST_CLOSE`) and releasing each slice, until it waits before slice `until` or
/// finishes.
fn run_until(kernel: &mut Engine, day: Day, script: &mut Script, until: usize) {
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if next_slice >= until => return,
            Pause::Waiting { next_slice } => {
                let k = u32::try_from(next_slice).expect("a slice of the day");
                let close = match next_slice + 1 == SLICES {
                    true => LAST_CLOSE,
                    false => 1000.0,
                };
                let bar = BaseBar {
                    ts_open: minute_on(day, k),
                    open: 1000.0,
                    high: 1000.0,
                    low: close,
                    close,
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

/// The session of 2025-07-31 holding one lot of `instrument` bought at 1,000.00, with
/// `rules`, its stop GTT (if any) confirmed, waiting before 09:16 with no command left.
fn holding(instrument: InstrumentSpec, rules: LegRules) -> Engine {
    holding_on(last_day(), instrument, rules)
}

/// `holding`, in the session of `day`.
fn holding_on(day: Day, instrument: InstrumentSpec, rules: LegRules) -> Engine {
    holding_in(session_on(day, instrument), day, rules)
}

/// `holding_on`, in `kernel`, a session of `day` before its first slice.
fn holding_in(mut kernel: Engine, day: Day, rules: LegRules) -> Engine {
    let decided = minute_on(day, 1);
    let mut script = Script::new(vec![(decided, Act::Submit(buy(rules)))]);
    run_until(&mut kernel, day, &mut script, 1);
    kernel.take_commands();
    kernel
        .apply_fill(ENTRY, 75.0, 1000.0, minute_on(day, 1))
        .expect("the entry fills");
    for command in kernel.take_commands() {
        if let VenueCommand::PlaceGtt { gtt, .. } = command {
            kernel.apply_gtt_placed(gtt).expect("the venue confirms");
        }
    }
    kernel
}

/// The session's close: every command of it, the venue confirming each GTT cancel and
/// order cancel, until the session finishes.
fn close_day(kernel: &mut Engine, day: Day) -> Vec<VenueCommand> {
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
                    match *command {
                        VenueCommand::CancelGtt { gtt, .. } => {
                            kernel.apply_gtt_cancelled(gtt).expect("the venue confirms");
                        }
                        VenueCommand::Cancel { order } => {
                            kernel.apply_cancel(order).expect("the venue confirms");
                        }
                        _ => {}
                    }
                }
                closed.extend(commands);
            }
            Pause::Waiting { next_slice } => {
                run_until(kernel, day, &mut Script::new(Vec::new()), next_slice + 1);
            }
            Pause::Finished { .. } => return closed,
            _ => {}
        }
    }
}

#[test]
fn a_stock_future_is_exited_by_the_engine_at_the_square_off_of_its_exit_day() {
    // The session reaches the 15:10 open as the 15:09 slice closes (`venue_clock`).
    let mut kernel = holding(stock_future(), LegRules::default());
    run_until(
        &mut kernel,
        last_day(),
        &mut Script::new(Vec::new()),
        SQUARE_OFF - 1,
    );
    assert_eq!(kernel.take_commands(), Vec::new(), "nothing before 15:10");
    run_until(
        &mut kernel,
        last_day(),
        &mut Script::new(Vec::new()),
        SQUARE_OFF,
    );
    let expiry = OrderRole::RiskExit(RiskExitReason::ExpiryExit);
    let placed: Vec<VenueCommand> = kernel.take_commands();
    assert_eq!(
        placed,
        vec![VenueCommand::Place {
            order: OrderId(2),
            instrument: SBIN,
            product: Product::Nrml,
            side: Side::Sell,
            units: 75.0,
            order_type: OrderType::Market,
            role: expiry,
        }]
    );
    let at = minute_on(last_day(), SQUARE_OFF as u32);
    kernel
        .apply_fill(OrderId(2), 75.0, 1000.0, at)
        .expect("the exit fills");
    close_day(&mut kernel, last_day());
    let output = kernel.finish().expect("the session finished");
    assert_eq!(
        output.trades[0].exit_reason,
        FillReason::Exit(ExitReason::ExpiryExit)
    );
}

#[test]
fn the_expiry_exit_cancels_the_stop_gtt_first() {
    let mut kernel = holding(stock_future(), stop_50());
    run_until(
        &mut kernel,
        last_day(),
        &mut Script::new(Vec::new()),
        SQUARE_OFF + 1,
    );
    let commands = kernel.take_commands();
    assert_eq!(commands.len(), 1, "{commands:?}");
    let VenueCommand::CancelGtt { gtt, .. } = commands[0] else {
        panic!("the stop GTT's cancel comes first: {commands:?}");
    };
    kernel.apply_gtt_cancelled(gtt).expect("the venue confirms");
    let after: Vec<OrderRole> = kernel
        .take_commands()
        .iter()
        .filter_map(|command| match *command {
            VenueCommand::Place { role, .. } => Some(role),
            _ => None,
        })
        .collect();
    assert_eq!(after, vec![OrderRole::RiskExit(RiskExitReason::ExpiryExit)]);
}

/// The index future held over its last trading day's close with its stop GTT: the
/// commands of the close and the state carried to the next day.
fn index_future_carried() -> (Vec<VenueCommand>, StateValue) {
    let mut kernel = holding(index_future(), stop_50());
    let closed = close_day(&mut kernel, last_day());
    (closed, kernel.carried_state().expect("the day has closed"))
}

#[test]
fn an_index_futures_stop_gtt_is_cancelled_at_the_close_of_its_last_day_and_not_placed_again() {
    let (closed, _) = index_future_carried();
    assert_eq!(closed.len(), 1, "{closed:?}");
    assert!(
        matches!(closed[0], VenueCommand::CancelGtt { .. }),
        "{closed:?}"
    );
}

#[test]
fn the_next_days_session_records_the_expired_position_at_its_last_close() {
    let (_, carried) = index_future_carried();
    let mut kernel = session_on(next_day(), index_future());
    let ids = kernel
        .restore_carried_state(&carried)
        .expect("the next day takes it");
    assert_eq!(ids.expired, vec![SBIN]);
    assert!(
        ids.gtts.is_empty(),
        "the expired contract's GTTs are not restored"
    );
    assert_eq!(kernel.take_commands(), Vec::new(), "nothing is sent for it");
    run_until(
        &mut kernel,
        next_day(),
        &mut Script::new(Vec::new()),
        usize::MAX,
    );
    let output = kernel.finish().expect("the session finished");
    let exit = &output.fills.last().expect("the expiry's fill").fill;
    let expiry = FillReason::Exit(ExitReason::Expiry);
    assert_eq!(
        (exit.price_rupees, exit.at, exit.reason),
        (LAST_CLOSE, FillAt::Close, expiry)
    );
    assert!(exit.flags.contains(FillFlags::FILLED_AT_LAST_CLOSE));
    assert_eq!(output.trades.len(), 1);
    assert!(output
        .final_positions
        .iter()
        .all(|p| p.quantity_units == 0.0));
}

#[test]
fn the_expired_positions_result_belongs_to_its_last_day_not_to_the_next_days_loss_limit() {
    // Bought at 1,000.00 and carried at its 988.00 last close: a loss of 900.00 (75 x
    // 12.00) that the exchange's settlement ended on 07-31, as a backtest's settlement at
    // expiry ends it that day. The next day's daily loss measure starts at nothing, as a
    // backtest's next session does, rather than counting that loss again.
    let (_, carried) = index_future_carried();
    let mut kernel = session_on(next_day(), index_future());
    kernel
        .restore_carried_state(&carried)
        .expect("the next day takes it");
    assert_eq!(kernel.daily_loss_measure(), 0.0);
}

/// The commands that place an order: (side, units, order type, role).
fn placed(commands: &[VenueCommand]) -> Vec<(Side, f64, OrderType, OrderRole)> {
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

/// A stock future bought on Mon 07-28 with `rules`, in sessions with the risk settings
/// `risk`, whose exit day is Tue 07-29, when no session ran: the Wed 07-30 session, before
/// the contract's last trading day, has taken it and stands before its open.
fn carried_past_missed_exit_day(risk: StrategyRiskConfig, rules: LegRules) -> Engine {
    let monday = days_from_civil(2025, 7, 28);
    let instrument = contract("SBIN31JUL25FUT", Some(days_from_civil(2025, 7, 29)));
    let monday_session = session_with(monday, instrument.clone(), risk);
    let mut kernel = holding_in(monday_session, monday, rules);
    kernel.take_commands();
    close_day(&mut kernel, monday);
    let carried = kernel.carried_state().expect("Monday has closed");
    let mut kernel = session_with(wednesday(), instrument, risk);
    kernel
        .restore_carried_state(&carried)
        .expect("Wednesday takes it");
    kernel
}

const EXPIRY_EXIT: OrderRole = OrderRole::RiskExit(RiskExitReason::ExpiryExit);

#[test]
fn a_carried_position_whose_exit_day_was_missed_is_exited_at_the_next_sessions_first_open() {
    // No tick comes: the session reaches its open with its first slice, and exits it there.
    let risk = StrategyRiskConfig::default();
    let mut kernel = carried_past_missed_exit_day(risk, LegRules::default());
    assert_eq!(
        kernel.take_commands(),
        Vec::new(),
        "nothing before the open"
    );
    run_until(&mut kernel, wednesday(), &mut Script::new(Vec::new()), 1);
    assert_eq!(
        placed(&kernel.take_commands()),
        vec![(Side::Sell, 75.0, OrderType::Market, EXPIRY_EXIT)]
    );
}

/// The first tick of Wednesday's session, at 09:15:01, a second past its open.
fn first_tick_after_the_open(kernel: &mut Engine) {
    let at = minute_on(wednesday(), 0) + 1_000_000_000;
    kernel
        .push_tick(SBIN, 1000.0, at)
        .expect("the tick is taken");
}

#[test]
fn a_tick_at_the_open_exits_a_position_carried_past_its_exit_day_at_once() {
    // The tick reaches the open before the first slice runs: the exit goes on it, as the
    // backtest fills it at that open, not a base bar later; the first slice sends no other.
    let risk = StrategyRiskConfig::default();
    let mut kernel = carried_past_missed_exit_day(risk, LegRules::default());
    first_tick_after_the_open(&mut kernel);
    assert_eq!(
        placed(&kernel.take_commands()),
        vec![(Side::Sell, 75.0, OrderType::Market, EXPIRY_EXIT)]
    );
    run_until(&mut kernel, wednesday(), &mut Script::new(Vec::new()), 1);
    assert_eq!(placed(&kernel.take_commands()), Vec::new());
}

#[test]
fn a_tick_at_the_open_exits_a_rearm_daily_leg_carried_past_its_exit_day_with_no_days_stop() {
    // A rearm_daily leg's stop waits for the open too; the expiry exit is made first, so no
    // day's stop is placed for a leg that is being exited (and then cancelled by its exit).
    let risk = StrategyRiskConfig {
        positional_stops: PositionalStops::RearmDaily,
        ..StrategyRiskConfig::default()
    };
    let mut kernel = carried_past_missed_exit_day(risk, stop_50());
    first_tick_after_the_open(&mut kernel);
    let commands = kernel.take_commands();
    assert_eq!(
        placed(&commands),
        vec![(Side::Sell, 75.0, OrderType::Market, EXPIRY_EXIT)],
        "{commands:?}"
    );
}

#[test]
fn an_entry_submitted_after_the_square_off_of_the_exit_day_is_refused_and_sends_nothing() {
    // Decided at the 15:16 close of the exit day, after its 15:10 square-off.
    let mut kernel = session_on(last_day(), stock_future());
    let decided = minute_on(last_day(), 361);
    let mut script = Script::new(vec![(decided, Act::Submit(buy(LegRules::default())))]);
    run_until(&mut kernel, last_day(), &mut script, 362);
    assert!(
        matches!(
            script.submissions[..],
            [Submission::Rejected(_, RejectCode::ContractExpiring)]
        ),
        "{:?}",
        script.submissions
    );
    assert_eq!(kernel.take_commands(), Vec::new());
}

#[test]
fn a_resting_entry_still_working_at_the_square_off_of_the_exit_day_is_withdrawn() {
    // A BUY LIMIT at 990.00 placed at the 09:15 close rests below the market; the session
    // reaching the 15:10 open cancels it, as a backtest refuses it at a fill after then.
    let mut kernel = session_on(last_day(), stock_future());
    let request = OrderRequest {
        order_type: OrderType::Limit {
            limit_rupees: 990.0,
        },
        ..buy(LegRules::default())
    };
    let mut script = Script::new(vec![(minute_on(last_day(), 1), Act::Submit(request))]);
    run_until(&mut kernel, last_day(), &mut script, SQUARE_OFF - 1);
    assert_eq!(placed(&kernel.take_commands()).len(), 1, "the LIMIT rests");
    run_until(
        &mut kernel,
        last_day(),
        &mut Script::new(Vec::new()),
        SQUARE_OFF,
    );
    assert_eq!(
        kernel.take_commands(),
        vec![VenueCommand::Cancel { order: ENTRY }]
    );
}

#[test]
fn a_position_on_a_contract_settled_by_delivery_carried_past_its_last_day_is_refused() {
    // The stock future's expiry exit at 15:10 is never filled and the close cancels it, so
    // the session closes holding the position. On 08-01 the exchange has settled it by
    // delivery, which the engine cannot record: the restore is refused, nothing restored.
    let mut kernel = holding(stock_future(), LegRules::default());
    close_day(&mut kernel, last_day());
    let carried = kernel.carried_state().expect("the day has closed");
    let mut next = session_on(next_day(), stock_future());
    let refused = next.restore_carried_state(&carried);
    let Err(KernelError::LiveState(LiveStateError::MissedExpiryExit {
        symbol,
        last_trading_day,
    })) = refused
    else {
        panic!("refused for its missed exit: {refused:?}");
    };
    assert_eq!(
        (symbol.as_str(), last_trading_day),
        ("SBIN31JUL25FUT", last_day())
    );
    assert_eq!(next.take_commands(), Vec::new());
    assert!(next.position(SBIN, Product::Nrml).is_none());
}

/// `state` with `edit` applied to its part `name`.
fn edit_part(state: &StateValue, name: &str, edit: impl Fn(&mut StateValue)) -> StateValue {
    let mut state = state.clone();
    if let StateValue::Map(parts) = &mut state {
        for (key, value) in parts.iter_mut() {
            if key == name {
                edit(value);
            }
        }
    }
    state
}

#[test]
fn an_expired_position_with_no_last_close_and_no_mark_is_refused_as_malformed() {
    let (_, carried) = index_future_carried();
    let no_closes = edit_part(&carried, "last_closes", |closes| {
        *closes = StateValue::List(Vec::new());
    });
    let no_mark = edit_part(&no_closes, "positions", |positions| {
        if let StateValue::List(items) = positions {
            for item in items.iter_mut() {
                if let StateValue::Map(fields) = item {
                    for (key, value) in fields.iter_mut() {
                        if key == "mark_rupees" {
                            *value = StateValue::Null;
                        }
                    }
                }
            }
        }
    });
    let mut kernel = session_on(next_day(), index_future());
    let refused = kernel.restore_carried_state(&no_mark);
    let Err(KernelError::LiveState(LiveStateError::Malformed { path, reason })) = refused else {
        panic!("refused as malformed: {refused:?}");
    };
    assert_eq!(path, "carried_state.positions[0]");
    assert!(reason.contains("NIFTY31JUL25FUT"), "{reason}");
}

#[test]
fn the_next_days_session_never_trades_the_expired_contract() {
    let (_, carried) = index_future_carried();
    let mut kernel = session_on(next_day(), index_future());
    kernel
        .restore_carried_state(&carried)
        .expect("the next day takes it");
    let decided = minute_on(next_day(), 1);
    let mut script = Script::new(vec![(decided, Act::Submit(buy(LegRules::default())))]);
    run_until(&mut kernel, next_day(), &mut script, 2);
    assert!(
        matches!(
            script.submissions[..],
            [Submission::Rejected(_, RejectCode::MarketClosed)]
        ),
        "{:?}",
        script.submissions
    );
}
