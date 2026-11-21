//! The account's daily loss limit breached in one engine (18-multi-strategy 2.5, KR4):
//! the lock for the session day, the record stamped where the engine stands, the cancels
//! of its entries and adds, its exits by step 6's rule with the reason
//! `account_loss_limit`.
//!
//! SBIN on NSE in July 2025 with 10,00,000 of capital, no charges and no slippage. The
//! breach is the one a fund would hand over: a limit of 10,000.00 and a measure of
//! -12,500.00.

use crate::engine::account_test_kit::{
    daily_bars, end_paused, events, fills, flat, held, ist, july_2025, market, minute_bars,
    open_and_level, pause_at, run_to_end, slice, Setup, MINUTES_PER_SESSION,
};
use crate::engine::venue_test_kit::holding_100;
use crate::engine::{AccountBreach, AccountRule, Engine, Event, KernelError, Step, Submission};
use crate::exec::{FillAt, OrderId, OrderRequest, OrderType, RejectCode};
use crate::model::{Day, Product, Side};
use crate::rms::{RiskEvent, RiskEventName, RiskScope, StrategyRiskConfig};

const LIMIT_RUPEES: f64 = 10_000.0;
const MEASURE_RUPEES: f64 = -12_500.0;
/// The first slice of the second day of a 1-minute run.
const SECOND_DAY_FIRST_SLICE: usize = MINUTES_PER_SESSION as usize;
/// The last slice of the first day of a 1-minute run.
const FIRST_DAY_LAST_SLICE: usize = SECOND_DAY_FIRST_SLICE - 1;

fn breach(at: i64, day: Day) -> AccountBreach {
    AccountBreach {
        at,
        day,
        rule: AccountRule::Loss,
        limit_rupees: LIMIT_RUPEES,
        measure_rupees: MEASURE_RUPEES,
    }
}

/// The breach found at the round closing with slice `index` of `kernel`.
fn breach_at_close_of(kernel: &Engine, index: usize) -> AccountBreach {
    let slice = slice(kernel, index);
    breach(slice.close, slice.day)
}

/// The events of slice `index`, step `step`, by name.
fn events_at(kernel: &Engine, index: usize, step: u8) -> Vec<String> {
    events(kernel)
        .into_iter()
        .filter(|(slice, at_step, _)| (*slice as usize, *at_step) == (index, step))
        .map(|(_, _, name)| name)
        .collect()
}

/// The outcome of a buy of 10 of SBIN in `product` submitted now.
fn buy_now(kernel: &mut Engine, product: Product) -> String {
    match kernel.submit(market(Side::Buy, 10.0, product)) {
        Ok(Submission::Submitted(_)) => "submitted".to_string(),
        Ok(Submission::Rejected(_, code)) => format!("rejected {}", code.code()),
        other => panic!("an unexpected submission: {other:?}"),
    }
}

/// One day (2025-07-01) of 1-minute bars flat at 1,000.00 to minute 2, minute 3 falling to
/// close at 990.00, then flat at 990.00; an MIS buy of 100 decided at the close of minute 0
/// (filled at 1,000.00 at minute 1) and a resting MIS buy LIMIT of 50 at 900.00 decided at
/// the close of minute 1 (an add that works: no bar trades at 900.00). The engine stands
/// before minute 3.
fn mis_participant(risk: StrategyRiskConfig) -> Engine {
    let day = july_2025(1);
    let bars = minute_bars(&[day], |_, k| match k {
        0..=2 => flat(1_000.0),
        3 => [1_000.0, 1_000.0, 990.0, 990.0],
        _ => flat(990.0),
    });
    let mut kernel = Setup {
        risk,
        ..Setup::minutes(day, day)
    }
    .engine(bars);
    pause_at(&mut kernel, 0);
    assert!(kernel
        .submit(market(Side::Buy, 100.0, Product::Mis))
        .is_ok());
    pause_at(&mut kernel, 1);
    let resting = OrderRequest {
        order_type: OrderType::Limit {
            limit_rupees: 900.0,
        },
        ..market(Side::Buy, 50.0, Product::Mis)
    };
    assert_eq!(
        kernel.submit(resting),
        Ok(Submission::Submitted(OrderId(2)))
    );
    pause_at(&mut kernel, 2);
    end_paused(&mut kernel, 2);
    kernel
}

#[test]
fn a_participant_records_the_breach_at_its_slices_step_6_and_exits_at_the_next_open() {
    let mut kernel = mis_participant(StrategyRiskConfig::default());
    open_and_level(&mut kernel, 3);
    let found = breach_at_close_of(&kernel, 3);
    assert_eq!(kernel.account_breach(found), Ok(()));
    // The event, then the resting add o2 cancelled; the exit waits for minute 4's open.
    assert_eq!(events_at(&kernel, 3, 6), ["account_loss_hit", "Canceled"]);
    assert_eq!(held(&kernel, Product::Mis), 100.0);
    assert!(kernel.decision_phase(&slice(&kernel, 3)).is_some());
    assert_eq!(
        buy_now(&mut kernel, Product::Mis),
        "rejected account_loss_gate"
    );
    pause_at(&mut kernel, 4);
    let exit = (
        4,
        Step::OpenFills,
        Side::Sell,
        100.0,
        990.0,
        FillAt::Open,
        "account_loss_limit",
        false,
    );
    assert_eq!(fills(&kernel).last(), Some(&exit));
    assert_eq!(
        buy_now(&mut kernel, Product::Mis),
        "rejected account_loss_gate"
    );
}

#[test]
fn the_breach_event_is_the_accounts_with_minus_its_limit_and_its_measure() {
    let mut kernel = mis_participant(StrategyRiskConfig::default());
    open_and_level(&mut kernel, 3);
    assert_eq!(
        kernel.account_breach(breach_at_close_of(&kernel, 3)),
        Ok(())
    );
    let hit = kernel.record.events.iter().find_map(|event| match event {
        Event::Risk {
            instrument, event, ..
        } => Some((*instrument, *event)),
        _ => None,
    });
    let expected = RiskEvent {
        name: RiskEventName::AccountLossHit,
        scope: RiskScope::Account,
        level_rupees: Some(-10_000.0),
        target_index: None,
        observed_rupees: Some(-12_500.0),
    };
    assert_eq!(hit, Some((None, expected)));
}

#[test]
fn an_exit_already_waiting_keeps_its_reason_and_the_strategys_own_gate_comes_first() {
    // The strategy's own limit of 1,000.00: minute 3 marks at 990.00, (990.00 - 1,000.00) x
    // 100 = -1,000.00, which breaches it in step 6 (cancelling o2 and queueing the exit)
    // before the account's breach is taken.
    let own = StrategyRiskConfig {
        max_daily_loss_rupees: Some(1_000.0),
        ..StrategyRiskConfig::default()
    };
    let mut kernel = mis_participant(own);
    open_and_level(&mut kernel, 3);
    assert_eq!(
        kernel.account_breach(breach_at_close_of(&kernel, 3)),
        Ok(())
    );
    assert_eq!(
        events_at(&kernel, 3, 6),
        [
            "daily_loss_limit_hit",
            "state ACTIVE -> REDUCING",
            "Canceled",
            "account_loss_hit"
        ]
    );
    assert!(kernel.decision_phase(&slice(&kernel, 3)).is_some());
    assert_eq!(
        buy_now(&mut kernel, Product::Mis),
        "rejected daily_loss_gate"
    );
    pause_at(&mut kernel, 4);
    let exit = (
        4,
        Step::OpenFills,
        Side::Sell,
        100.0,
        990.0,
        FillAt::Open,
        "daily_loss_limit",
        false,
    );
    assert_eq!(fills(&kernel).last(), Some(&exit));
}

#[test]
fn a_participant_at_the_sessions_last_slice_exits_at_its_last_close_flagged() {
    // A CNC buy of 100 at 1,000.00; the session's last minute closes at 980.00.
    let day = july_2025(1);
    let last = FIRST_DAY_LAST_SLICE;
    let bars = minute_bars(&[day], |_, k| match k as usize == last {
        true => [1_000.0, 1_000.0, 980.0, 980.0],
        false => flat(1_000.0),
    });
    let mut kernel = Setup::minutes(day, day).engine(bars);
    pause_at(&mut kernel, 0);
    assert!(kernel
        .submit(market(Side::Buy, 100.0, Product::Cnc))
        .is_ok());
    pause_at(&mut kernel, last - 1);
    end_paused(&mut kernel, last - 1);
    open_and_level(&mut kernel, last);
    assert_eq!(
        kernel.account_breach(breach_at_close_of(&kernel, last)),
        Ok(())
    );
    assert_eq!(
        events_at(&kernel, last, 6),
        ["account_loss_hit", "PositionClosed"]
    );
    let slice_ = last as u32;
    let exit = (
        slice_,
        Step::AggregateRules,
        Side::Sell,
        100.0,
        980.0,
        FillAt::Close,
        "account_loss_limit",
        true,
    );
    assert_eq!(fills(&kernel).last(), Some(&exit));
}

#[test]
fn a_coarser_engine_records_the_breach_at_its_slices_step_6_and_locks_from_the_breach_on() {
    // Daily bars: 2025-07-01 flat at 1,000.00, 2025-07-02 opening at 1,000.00 and closing at
    // 970.00, 2025-07-03 flat at 970.00. A CNC buy of 100 decided at the first close fills
    // at the second open; the breach comes at 11:00 IST on 2025-07-02, inside that bar.
    let (first, second, third) = (july_2025(1), july_2025(2), july_2025(3));
    let bars = daily_bars(&[
        (first, flat(1_000.0)),
        (second, [1_000.0, 1_010.0, 960.0, 970.0]),
        (third, flat(970.0)),
    ]);
    let mut kernel = Setup::daily(first, third).engine(bars);
    pause_at(&mut kernel, 0);
    assert!(kernel
        .submit(market(Side::Buy, 100.0, Product::Cnc))
        .is_ok());
    end_paused(&mut kernel, 0);
    let day_bar = slice(&kernel, 1);
    assert_eq!(kernel.open_phase(&day_bar), Ok(None));
    assert_eq!(held(&kernel, Product::Cnc), 100.0);
    assert_eq!(
        kernel.account_breach(breach(ist(second, 11, 0), second)),
        Ok(())
    );
    assert!(
        events_at(&kernel, 1, 6).is_empty(),
        "recorded at its step 6"
    );
    assert_eq!(
        kernel.account_entry_gate(&day_bar),
        Err(RejectCode::AccountLossGate)
    );
    assert_eq!(kernel.levels_phase(&day_bar), Ok(()));
    assert_eq!(
        events_at(&kernel, 1, 6),
        ["account_loss_hit", "PositionClosed"]
    );
    let exit = (
        1,
        Step::AggregateRules,
        Side::Sell,
        100.0,
        970.0,
        FillAt::Close,
        "account_loss_limit",
        true,
    );
    assert_eq!(fills(&kernel).last(), Some(&exit));
    assert!(kernel.decision_phase(&day_bar).is_some());
    assert_eq!(
        buy_now(&mut kernel, Product::Cnc),
        "rejected account_loss_gate"
    );
    pause_at(&mut kernel, 2);
    assert_eq!(buy_now(&mut kernel, Product::Cnc), "submitted");
}

#[test]
fn a_breach_inside_a_coarser_engines_open_bar_cancels_its_resting_entries_before_its_levels() {
    // Daily bars: 2025-07-01 flat at 1,000.00, 2025-07-02 opening at 1,000.00 with a low of
    // 900.00 and closing at 970.00, 2025-07-03 flat at 970.00. A CNC buy LIMIT of 100 at
    // 950.00 decided at the first close would fill inside the second bar; the breach comes
    // at 11:00 IST on 2025-07-02, with that bar open. From the breach on no entry fills:
    // the LIMIT is cancelled at the bar's step 3, before its levels, and the breach is
    // recorded at its step 6 with nothing left to exit.
    let (first, second, third) = (july_2025(1), july_2025(2), july_2025(3));
    let bars = daily_bars(&[
        (first, flat(1_000.0)),
        (second, [1_000.0, 1_010.0, 900.0, 970.0]),
        (third, flat(970.0)),
    ]);
    let mut kernel = Setup::daily(first, third).engine(bars);
    pause_at(&mut kernel, 0);
    let resting = OrderRequest {
        order_type: OrderType::Limit {
            limit_rupees: 950.0,
        },
        ..market(Side::Buy, 100.0, Product::Cnc)
    };
    assert_eq!(
        kernel.submit(resting),
        Ok(Submission::Submitted(OrderId(1)))
    );
    end_paused(&mut kernel, 0);
    let day_bar = slice(&kernel, 1);
    assert_eq!(kernel.open_phase(&day_bar), Ok(None));
    assert_eq!(
        kernel.account_breach(breach(ist(second, 11, 0), second)),
        Ok(())
    );
    assert_eq!(kernel.levels_phase(&day_bar), Ok(()));
    assert_eq!(events_at(&kernel, 1, 3), ["Canceled"]);
    assert_eq!(events_at(&kernel, 1, 6), ["account_loss_hit"]);
    let filled = fills(&kernel);
    assert!(filled.is_empty(), "no fill: {filled:?}");
    assert_eq!(held(&kernel, Product::Cnc), 0.0);
}

/// Two days of 1-minute bars, 2025-07-01 flat at 1,000.00 and 2025-07-02 flat at 985.00, a
/// CNC buy of 100 filled at 1,000.00 at minute 1, and the engine standing between the two
/// sessions, after the first's last slice.
fn between_sessions() -> Engine {
    let (first, second) = (july_2025(1), july_2025(2));
    let bars = minute_bars(&[first, second], |day, _| {
        flat(if day == 0 { 1_000.0 } else { 985.0 })
    });
    let mut kernel = Setup::minutes(first, second).engine(bars);
    pause_at(&mut kernel, 0);
    assert!(kernel
        .submit(market(Side::Buy, 100.0, Product::Cnc))
        .is_ok());
    pause_at(&mut kernel, FIRST_DAY_LAST_SLICE);
    end_paused(&mut kernel, FIRST_DAY_LAST_SLICE);
    kernel
}

#[test]
fn an_engine_between_slices_records_the_breach_at_its_next_slices_step_1_and_exits_at_its_open() {
    // The breach comes at 21:00 IST on 2025-07-01 (another exchange's evening round): the
    // lock of that day is over by the next session.
    let mut kernel = between_sessions();
    let evening = breach(ist(july_2025(1), 21, 0), july_2025(1));
    assert_eq!(kernel.account_breach(evening), Ok(()));
    assert_eq!(held(&kernel, Product::Cnc), 100.0);
    let next = SECOND_DAY_FIRST_SLICE;
    pause_at(&mut kernel, next);
    assert_eq!(events_at(&kernel, next, 1), ["account_loss_hit"]);
    assert_eq!(events_at(&kernel, next, 2), ["PositionClosed"]);
    let exit = (
        next as u32,
        Step::OpenFills,
        Side::Sell,
        100.0,
        985.0,
        FillAt::Open,
        "account_loss_limit",
        false,
    );
    assert_eq!(fills(&kernel).last(), Some(&exit));
    assert_eq!(buy_now(&mut kernel, Product::Cnc), "submitted");
}

#[test]
fn a_breach_between_slices_on_the_engines_next_session_day_locks_that_day() {
    // The breach comes at 09:00 IST on 2025-07-02, before NSE opens at 09:15.
    let mut kernel = between_sessions();
    let morning = breach(ist(july_2025(2), 9, 0), july_2025(2));
    assert_eq!(kernel.account_breach(morning), Ok(()));
    pause_at(&mut kernel, SECOND_DAY_FIRST_SLICE);
    assert_eq!(
        events_at(&kernel, SECOND_DAY_FIRST_SLICE, 1),
        ["account_loss_hit"]
    );
    assert_eq!(
        buy_now(&mut kernel, Product::Cnc),
        "rejected account_loss_gate"
    );
}

#[test]
fn two_breaches_while_an_engine_stands_between_slices_are_both_recorded_and_it_exits_once() {
    // The evening of 2025-07-01 and the morning of 2025-07-02 before NSE opens (two
    // session days of another exchange's rounds) each breach the account's limit.
    let mut kernel = between_sessions();
    let evening = breach(ist(july_2025(1), 21, 0), july_2025(1));
    let morning = AccountBreach {
        measure_rupees: -15_000.0,
        ..breach(ist(july_2025(2), 9, 0), july_2025(2))
    };
    assert_eq!(kernel.account_breach(evening), Ok(()));
    assert_eq!(kernel.account_breach(morning), Ok(()));
    let next = SECOND_DAY_FIRST_SLICE;
    pause_at(&mut kernel, next);
    let observed: Vec<(u32, Step, Option<f64>)> = kernel
        .record
        .events
        .iter()
        .filter_map(|event| match event {
            Event::Risk { stamp, event, .. } => {
                Some((stamp.slice, stamp.step, event.observed_rupees))
            }
            _ => None,
        })
        .collect();
    let at_open = |measure| (next as u32, Step::SessionOpen, Some(measure));
    assert_eq!(observed, [at_open(-12_500.0), at_open(-15_000.0)]);
    assert_eq!(events_at(&kernel, next, 2), ["PositionClosed"]);
    let exits: Vec<_> = fills(&kernel)
        .into_iter()
        .filter(|fill| fill.6 == "account_loss_limit")
        .collect();
    assert_eq!(exits.len(), 1, "one exit of the 100 units: {exits:?}");
    assert_eq!(
        buy_now(&mut kernel, Product::Cnc),
        "rejected account_loss_gate"
    );
}

#[test]
fn an_engine_that_has_not_started_drops_a_breach_of_a_day_before_its_first_slice() {
    // A run of 2025-07-07 alone, with no position or order before its first slice. A
    // breach of an earlier day is over before it starts; one of its first day, before
    // NSE opens, is recorded at its first slice's step 1 and locks that day.
    let (earlier, first) = (july_2025(1), july_2025(7));
    let table = [
        (breach(ist(earlier, 15, 30), earlier), vec![], "submitted"),
        (
            breach(ist(first, 9, 0), first),
            vec!["account_loss_hit".to_string()],
            "rejected account_loss_gate",
        ),
    ];
    for (found, recorded, buy) in table {
        let bars = minute_bars(&[first], |_, _| flat(1_000.0));
        let mut kernel = Setup::minutes(first, first).engine(bars);
        assert_eq!(kernel.account_breach(found), Ok(()));
        pause_at(&mut kernel, 0);
        assert_eq!(events_at(&kernel, 0, 1), recorded, "{found:?}");
        assert_eq!(buy_now(&mut kernel, Product::Cnc), buy, "{found:?}");
    }
}

#[test]
fn an_engine_whose_run_has_ended_takes_nothing() {
    let day = july_2025(1);
    let mut kernel = Setup::minutes(day, day).engine(minute_bars(&[day], |_, _| flat(1_000.0)));
    run_to_end(&mut kernel);
    let before = events(&kernel);
    assert_eq!(kernel.account_breach(breach(ist(day, 21, 0), day)), Ok(()));
    assert_eq!(events(&kernel), before);
    assert!(kernel.finish().is_ok());
}

#[test]
fn a_breach_with_an_unusable_limit_or_measure_and_a_live_sessions_breach_are_refused() {
    let day = july_2025(1);
    let mut kernel = Setup::minutes(day, day).engine(minute_bars(&[day], |_, _| flat(1_000.0)));
    let unusable = [
        AccountBreach {
            limit_rupees: 0.0,
            ..breach(ist(day, 10, 0), day)
        },
        AccountBreach {
            measure_rupees: f64::NAN,
            ..breach(ist(day, 10, 0), day)
        },
    ];
    for refused in unusable {
        let outcome = kernel.account_breach(refused);
        let field = "account_max_daily_loss";
        assert!(
            matches!(outcome, Err(KernelError::InvalidConfig { field: f, .. }) if f == field),
            "{refused:?}"
        );
    }
    let mut session = holding_100();
    let live = session.account_breach(breach(ist(day, 10, 0), day));
    assert!(matches!(live, Err(KernelError::NotBuilt { .. })));
}
