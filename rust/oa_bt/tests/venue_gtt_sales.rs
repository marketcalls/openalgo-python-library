#![cfg(all(feature = "costs", feature = "time"))]
//! A strategy's sale of a CNC long in a live session (04-execution 2: delivery cannot be
//! shorted; design 20-positional-live 7.2 and decision 11.3: the strategy only ever sells
//! its own units). A CNC sale only closes, whatever its reduce-only flag says: beside the
//! leg's stop GTT, or beside another sale, one beyond the room the closing side leaves is
//! refused `exit_cover`, as a reduce-only one is; a MARKET sale beside the stop GTT goes
//! through the position's exit owner, as before. An NRML sale keeps its rule: one beyond
//! the room may open a short, as in a backtest.
//!
//! SBIN on 2025-07-01, 1-minute bars, no charges, no slippage; 100 bought in CNC at
//! 500.10 at the 09:16 open (`gtt_venue_support`).

mod gtt_venue_support;
mod venue_support;

use gtt_venue_support::{cnc_long, cnc_long_protected, nifty_future, points, stop_4, G1, UNITS};
use oa_bt::engine::{OrderRole, Submission, VenueCommand};
use oa_bt::exec::{OrderId, OrderRequest, OrderType, RejectCode};
use oa_bt::model::{Product, Side};
use oa_bt::rms::LegRules;
use venue_support::{
    at_close, bars, config, limit, minute, order, pause_at_close, placed_roles, session_of,
    stop_limit, stop_market, wait_before, Act, Ohlc, Script,
};

#[test]
fn a_resting_cnc_sale_of_the_long_beside_its_stop_gtt_is_refused_exit_cover() {
    // g1 covers the 100 held (trigger 496.10). A sale of the 100 that is not reduce-only
    // would sell them a second time once g1 fires; it can never open a short, so it is an
    // exit with no room left on the closing side, whatever its type.
    let rows = [
        ("a stop-loss SL-M at the stop", stop_market(496.1)),
        ("a stop-loss SL", stop_limit(497.0, 496.0)),
        ("a take-profit LIMIT", limit(520.0)),
    ];
    for (what, order_type) in rows {
        let mut kernel = cnc_long_protected(stop_4());
        pause_at_close(&mut kernel, &bars(&[], &[]), 1);
        let sale = order(Side::Sell, UNITS, order_type, Product::Cnc);
        let refused = Submission::Rejected(OrderId(2), RejectCode::ExitCover);
        assert_eq!(kernel.submit(sale), Ok(refused), "{what}");
        assert!(placed_roles(&kernel.take_commands()).is_empty(), "{what}");
    }
}

#[test]
fn a_cnc_sale_beyond_the_room_another_sale_leaves_is_refused_exit_cover() {
    // No leg rules. A take-profit LIMIT sale of the 100 (o2) rests as an exit. A second
    // sale of the 100 beside it (o3) would sell 200 of the 100 held if both filled: it is
    // an exit with no room left, not an entry kept out by the order in flight.
    let rows = [
        ("a stop-loss SL-M", stop_market(490.0)),
        ("a MARKET sale", OrderType::Market),
    ];
    for (what, order_type) in rows {
        let mut kernel = cnc_long(LegRules::default());
        pause_at_close(&mut kernel, &bars(&[], &[]), 1);
        let target = order(Side::Sell, UNITS, limit(520.0), Product::Cnc);
        let accepted = Submission::Submitted(OrderId(2));
        assert_eq!(kernel.submit(target), Ok(accepted), "{what}");
        let sale = order(Side::Sell, UNITS, order_type, Product::Cnc);
        let refused = Submission::Rejected(OrderId(3), RejectCode::ExitCover);
        assert_eq!(kernel.submit(sale), Ok(refused), "{what}");
        let target_placed = (OrderId(2), Side::Sell, UNITS, OrderRole::Exit);
        assert_eq!(
            placed_roles(&kernel.take_commands()),
            [target_placed],
            "{what}"
        );
    }
}

#[test]
fn a_market_cnc_sale_of_the_long_beside_its_stop_gtt_cancels_the_gtt_first() {
    // Unchanged: a MARKET sale of the 100 that is not reduce-only is the position's exit,
    // sent once the venue has cancelled g1.
    let mut kernel = cnc_long_protected(stop_4());
    let sale = order(Side::Sell, UNITS, OrderType::Market, Product::Cnc);
    let mut script = Script::new(vec![at_close(1, Act::Submit(sale))]);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 2);
    assert_eq!(script.submissions, [Submission::Submitted(OrderId(2))]);
    let commands = kernel.take_commands();
    assert!(placed_roles(&commands).is_empty());
    assert!(matches!(
        commands[..],
        [VenueCommand::CancelGtt { gtt: G1, .. }]
    ));
    assert!(kernel.apply_gtt_cancelled(G1).is_ok());
    let sold = (OrderId(2), Side::Sell, UNITS, OrderRole::Exit);
    assert_eq!(placed_roles(&kernel.take_commands()), [sold]);
}

/// A bar of the future near 25000.00.
const FUTURE_BAR: Ohlc = (25_000.0, 25_010.0, 24_990.0, 25_005.0);

#[test]
fn an_nrml_sale_beside_its_stop_gtt_beyond_the_room_is_still_an_entry() {
    // One lot of the future bought at 25000.00 with a stop 50 points below (g1 at
    // 24950.00, confirmed). A stop-loss SL-M sale of the lot beside g1 is an entry, as
    // before: a futures short is allowed, and a backtest of it ends short too.
    let future_bars = bars(&[(0, FUTURE_BAR), (1, FUTURE_BAR), (2, FUTURE_BAR)], &[]);
    let mut kernel = session_of(config(), nifty_future());
    let long = OrderRequest {
        leg_rules: LegRules {
            stop: Some(points(50.0)),
            ..LegRules::default()
        },
        ..order(Side::Buy, 75.0, OrderType::Market, Product::Nrml)
    };
    let acts = vec![at_close(0, Act::Submit(long))];
    wait_before(&mut kernel, &future_bars, &mut Script::new(acts), 1);
    kernel.take_commands();
    assert!(kernel
        .apply_fill(OrderId(1), 75.0, 25_000.0, minute(1))
        .is_ok());
    kernel.take_commands();
    assert!(kernel.apply_gtt_placed(G1).is_ok());
    let sale = order(Side::Sell, 75.0, stop_market(24_950.0), Product::Nrml);
    let mut script = Script::new(vec![at_close(1, Act::Submit(sale))]);
    wait_before(&mut kernel, &future_bars, &mut script, 2);
    assert_eq!(script.submissions, [Submission::Submitted(OrderId(2))]);
    let entry = (OrderId(2), Side::Sell, 75.0, OrderRole::Entry);
    assert_eq!(placed_roles(&kernel.take_commands()), [entry]);
}
