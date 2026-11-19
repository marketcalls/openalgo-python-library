//! The phases of a slice (18-multi-strategy 2.3, KR1): the open phase is steps 1 and 2,
//! the levels phase steps 3 to 6, and a run driven a phase at a time, as a portfolio run's
//! fund drives each engine, is the run `advance` makes.
//!
//! SBIN on 2025-07-01, a 1-minute strategy: every minute is flat at 1,000.00 but minute 1
//! (open 1,000.00, high 1,001.00, low 990.00, close 995.00), so an MIS buy of 100 with a
//! stop 5 points below its fill, decided at the close of minute 0, fills at 1,000.00 at the
//! open of minute 1 and its stop at 995.00 is reached inside that same bar.

use crate::engine::account_test_kit::{
    end_paused, fills, held, july_2025, market, minute_bars, pause_at, slice, Setup,
};
use crate::engine::{Engine, KernelError, Pause, Step};
use crate::exec::{FillAt, OrderRequest};
use crate::model::{Product, Side};
use crate::rms::{Distance, LegRules};

/// The bars of the module docs.
fn engine() -> Engine {
    let day = july_2025(1);
    let bars = minute_bars(&[day], |_, k| match k {
        1 => [1_000.0, 1_001.0, 990.0, 995.0],
        _ => [1_000.0; 4],
    });
    Setup::minutes(day, day).engine(bars)
}

/// The buy of the module docs, with a stop 5 points below its fill.
fn buy_with_a_stop(engine: &mut Engine) {
    let request = OrderRequest {
        leg_rules: LegRules {
            stop: Some(Distance::Points(5.0)),
            ..LegRules::default()
        },
        ..market(Side::Buy, 100.0, Product::Mis)
    };
    assert!(engine.submit(request).is_ok());
}

#[test]
fn the_open_phase_fills_at_the_open_and_the_levels_phase_meets_the_stop_inside_the_bar() {
    let mut kernel = engine();
    pause_at(&mut kernel, 0);
    buy_with_a_stop(&mut kernel);
    end_paused(&mut kernel, 0);
    let minute_1 = slice(&kernel, 1);
    assert_eq!(kernel.open_phase(&minute_1), Ok(None));
    assert_eq!(held(&kernel, Product::Mis), 100.0);
    let entry = (
        1,
        Step::OpenFills,
        Side::Buy,
        100.0,
        1_000.0,
        FillAt::Open,
        "entry",
        false,
    );
    assert_eq!(fills(&kernel), [entry]);
    assert_eq!(kernel.levels_phase(&minute_1), Ok(()));
    assert_eq!(held(&kernel, Product::Mis), 0.0);
    let stop = (
        1,
        Step::Levels,
        Side::Sell,
        100.0,
        995.0,
        FillAt::Level,
        "leg_sl",
        false,
    );
    assert_eq!(fills(&kernel), [entry, stop]);
}

#[test]
fn an_opened_slice_is_no_decision_point() {
    let mut kernel = engine();
    pause_at(&mut kernel, 0);
    end_paused(&mut kernel, 0);
    let minute_1 = slice(&kernel, 1);
    assert_eq!(kernel.open_phase(&minute_1), Ok(None));
    assert_eq!(kernel.decision_instant(), None);
    let refused = kernel.submit(market(Side::Buy, 1.0, Product::Mis));
    assert_eq!(refused, Err(KernelError::NotAtDecisionPoint));
}

/// At the close of minute 0 the buy with the stop; at the close of minute 2, after the
/// stop has closed it, a CNC buy of 50 (no square-off closes it).
fn decide(kernel: &mut Engine, index: usize) {
    match index {
        0 => buy_with_a_stop(kernel),
        2 => assert!(kernel.submit(market(Side::Buy, 50.0, Product::Cnc)).is_ok()),
        _ => {}
    }
}

/// Run `kernel` to its end a phase at a time, deciding as `decide` says.
fn run_by_phases(kernel: &mut Engine) {
    let mut index = 0;
    while let Some(slice) = kernel.timeline.slice(index) {
        assert_eq!(kernel.open_phase(&slice), Ok(None));
        assert_eq!(kernel.levels_phase(&slice), Ok(()));
        if kernel.decision_phase(&slice).is_some() {
            decide(kernel, index);
        }
        assert_eq!(kernel.end_slice(&slice), Ok(None));
        index += 1;
    }
    assert!(matches!(kernel.advance(), Ok(Pause::Finished { .. })));
}

/// Run `kernel` to its end with `advance`, deciding as `decide` says.
fn run_by_advance(kernel: &mut Engine) {
    loop {
        match kernel.advance() {
            Ok(Pause::BarClose { stamp, .. }) => decide(kernel, stamp.slice as usize),
            Ok(Pause::Finished { .. }) => return,
            other => panic!("an unexpected pause: {other:?}"),
        }
    }
}

#[test]
fn a_run_driven_a_phase_at_a_time_is_the_run_advance_makes() {
    let mut by_phases = engine();
    run_by_phases(&mut by_phases);
    let mut by_advance = engine();
    run_by_advance(&mut by_advance);
    let phased = by_phases.finish().expect("finished");
    let advanced = by_advance.finish().expect("finished");
    // The entry and its stop at minute 1, the CNC buy at the open of minute 3.
    assert_eq!(phased.fills.len(), 3);
    assert_eq!(phased, advanced);
}
