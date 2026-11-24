#![cfg(all(feature = "costs", feature = "time"))]
//! The strategy's successive exits of a position in a live session add up as a backtest
//! fills them (`engine::venue_owner`): each is sized when it is sent, on the position less
//! what the earlier exits still have to fill, and capped at what is left.
//!
//! The bars rise 0.10 a minute from 500.00 (`venue_support::rising`); the square-off is at
//! 15:10 (slice 355).

mod venue_support;

use oa_bt::engine::{Engine, RunOutput, VenueCommand};
use oa_bt::exec::{OrderId, OrderRequest, OrderType, QuantitySpec};
use oa_bt::model::{Product, Side};
use venue_support::{
    at_close, bars, config, exit, market, minute, pause_at_close, run_backtest, run_echo, session,
    wait_before, Act, Echo, Script, SBIN,
};

/// A reduce-only MARKET sale of `units`: `reduce()` by units.
fn reduce_units(units: f64) -> OrderRequest {
    exit(Side::Sell, units, OrderType::Market)
}

/// A reduce-only MARKET sale of a fraction of the position: `reduce()`, or `exit()` at 1.
fn reduce_fraction(fraction: f64) -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::PositionFraction(fraction),
        ..reduce_units(1.0)
    }
}

/// A session holding 100 units bought in minute 1, paused at the close of minute 2 with
/// no command left to take.
fn holding_100() -> Engine {
    let mut kernel = session(config());
    let buy = at_close(0, Act::Submit(market(Side::Buy, 100.0)));
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(vec![buy]), 1);
    kernel
        .apply_fill(OrderId(1), 100.0, 500.1, minute(1))
        .expect("the buy fills");
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    kernel.take_commands();
    kernel
}

/// The (order, units) of each command, `None` units for a cancellation.
fn commands(kernel: &mut Engine) -> Vec<(OrderId, Option<f64>)> {
    kernel
        .take_commands()
        .into_iter()
        .map(|command| match command {
            VenueCommand::Place { order, units, .. }
            | VenueCommand::Modify { order, units, .. } => (order, Some(units)),
            VenueCommand::Cancel { order } => (order, None),
            other => panic!("an MIS position sends no GTT command: {other:?}"),
        })
        .collect()
}

fn held(kernel: &Engine) -> Option<f64> {
    kernel
        .position(SBIN, Product::Mis)
        .map(|position| position.quantity_units)
}

#[test]
fn successive_exits_in_one_bar_are_each_sized_on_what_the_earlier_leave() {
    // (exits, the units each is sent with)
    let cases: [(&[OrderRequest], &[f64]); 4] = [
        (&[reduce_fraction(0.5), reduce_fraction(0.5)], &[50.0, 25.0]),
        (&[reduce_units(50.0), reduce_units(10.0)], &[50.0, 10.0]),
        (&[reduce_fraction(0.5), reduce_fraction(1.0)], &[50.0, 50.0]),
        (&[reduce_units(80.0), reduce_units(50.0)], &[80.0, 20.0]),
    ];
    for (exits, units) in cases {
        let mut kernel = holding_100();
        for request in exits {
            assert!(kernel.submit(request.clone()).is_ok());
        }
        let sent: Vec<(OrderId, Option<f64>)> = units
            .iter()
            .zip(2..)
            .map(|(units, id)| (OrderId(id), Some(*units)))
            .collect();
        assert_eq!(commands(&mut kernel), sent, "{exits:?}");
    }
}

#[test]
fn a_later_exit_leaves_an_earlier_one_working() {
    // reduce() of 50 units, then of 10: the first fills whole as the second goes out,
    // and the strategy has closed the 60 it asked for.
    let mut kernel = holding_100();
    assert!(kernel.submit(reduce_units(50.0)).is_ok());
    assert!(kernel.submit(reduce_units(10.0)).is_ok());
    assert_eq!(
        commands(&mut kernel),
        [(OrderId(2), Some(50.0)), (OrderId(3), Some(10.0))]
    );
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        3,
    );
    assert!(kernel
        .apply_fill(OrderId(2), 50.0, 500.3, minute(3))
        .is_ok());
    assert!(kernel
        .apply_fill(OrderId(3), 10.0, 500.3, minute(3))
        .is_ok());
    assert!(commands(&mut kernel).is_empty());
    assert_eq!(held(&kernel), Some(40.0));
}

#[test]
fn a_later_exit_counts_what_an_earlier_one_has_filled() {
    // reduce() of 50 units fills 40, then reduce() of 10 units: 60 in all, the 10 the
    // first has left and the 10 of the second.
    let mut kernel = holding_100();
    assert!(kernel.submit(reduce_units(50.0)).is_ok());
    kernel.take_commands();
    let mut script = Script::new(vec![at_close(3, Act::Submit(reduce_units(10.0)))]);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    assert!(kernel
        .apply_fill(OrderId(2), 40.0, 500.3, minute(3))
        .is_ok());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 4);
    assert_eq!(commands(&mut kernel), [(OrderId(3), Some(10.0))]);
    assert!(kernel
        .apply_fill(OrderId(2), 10.0, 500.4, minute(4))
        .is_ok());
    assert!(kernel
        .apply_fill(OrderId(3), 10.0, 500.4, minute(4))
        .is_ok());
    assert_eq!(held(&kernel), Some(40.0));
}

/// The (slice, side, units, price, reason) of every fill.
fn fills(output: &RunOutput) -> Vec<(u32, Side, f64, f64, &'static str)> {
    output
        .fills
        .iter()
        .map(|record| {
            let fill = &record.fill;
            (
                record.stamp.slice,
                fill.side,
                fill.quantity_units,
                fill.price_rupees,
                fill.reason.code(),
            )
        })
        .collect()
}

#[test]
fn successive_exits_close_what_the_backtest_closes() {
    let cases: [&[OrderRequest]; 3] = [
        &[reduce_fraction(0.5), reduce_fraction(0.5)],
        &[reduce_units(50.0), reduce_units(10.0)],
        &[reduce_fraction(0.5), reduce_fraction(1.0)],
    ];
    let columns = bars(&[], &[]);
    for exits in cases {
        let mut acts = vec![at_close(0, Act::Submit(market(Side::Buy, 100.0)))];
        acts.extend(exits.iter().map(|r| at_close(10, Act::Submit(r.clone()))));
        let backtest = run_backtest(config(), &columns, &mut Script::new(acts.clone()));
        let echoed = run_echo(
            config(),
            &columns,
            &mut Script::new(acts),
            &mut Echo::default(),
        );
        assert_eq!(fills(&echoed), fills(&backtest), "{exits:?}");
    }
}
