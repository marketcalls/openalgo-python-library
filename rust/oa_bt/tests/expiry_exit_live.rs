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
    BaseBar, Driver, Engine, KernelError, OrderRole, Pause, RiskExitReason, RunConfig, StateValue,
    Submission, VenueCommand,
};
use oa_bt::exec::{
    FillAt, FillFlags, FillReason, InstrumentSpec, OrderId, OrderRequest, OrderType, RejectCode,
};
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{Distance, ExitReason, LegRules};

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
    let config = RunConfig {
        first_day: day,
        last_day: day,
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

/// Run `kernel`, a session of `day`, adding flat bars at 1,000.00 and releasing each
/// slice, until it waits before slice `until` or finishes; the venue confirms every GTT
/// request it takes on the way only when `confirm_gtts`.
fn run_until(kernel: &mut Engine, day: Day, script: &mut Script, until: usize) {
    loop {
        match kernel.advance().expect("the slice runs") {
            Pause::Waiting { next_slice } if next_slice >= until => return,
            Pause::Waiting { next_slice } => {
                let k = u32::try_from(next_slice).expect("a slice of the day");
                let bar = BaseBar {
                    ts_open: minute_on(day, k),
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

/// The session of 2025-07-31 holding one lot of `instrument` bought at 1,000.00, with
/// `rules`, its stop GTT (if any) confirmed, waiting before 09:16 with no command left.
fn holding(instrument: InstrumentSpec, rules: LegRules) -> Engine {
    let mut kernel = session_on(last_day(), instrument);
    let decided = minute_on(last_day(), 1);
    let mut script = Script::new(vec![(decided, Act::Submit(buy(rules)))]);
    run_until(&mut kernel, last_day(), &mut script, 1);
    kernel.take_commands();
    kernel
        .apply_fill(ENTRY, 75.0, 1000.0, minute_on(last_day(), 1))
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
        (1000.0, FillAt::Close, expiry)
    );
    assert!(exit.flags.contains(FillFlags::FILLED_AT_LAST_CLOSE));
    assert_eq!(output.trades.len(), 1);
    assert!(output
        .final_positions
        .iter()
        .all(|p| p.quantity_units == 0.0));
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
