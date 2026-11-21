//! A portfolio run's account in one engine (18-multi-strategy 2.4, KR2 and KR3): the
//! second margin check of every opening fill and the cap the account puts on sizing.
//!
//! SBIN on 2025-07-01, every minute flat at 1,000.00, a 1-minute strategy with 10,00,000
//! of capital, no charges and no slippage: a buy of `u` units blocks `u` x 1,000.00 of
//! margin (cash equity at 1x), and a flat engine's own free cash is 10,00,000. The room is
//! what the account leaves beside it, so the account's free cash is the engine's own plus
//! the room.

use crate::engine::account_test_kit::{events, held, july_2025, market, minute_bars, pause_at};
use crate::engine::account_test_kit::{Setup, INFY, SBIN};
use crate::engine::venue_test_kit::holding_100;
use crate::engine::Submission;
use crate::engine::{AccountLink, Engine, KernelError, Pause, QuantityMode, RunOutput};
use crate::exec::{InstrumentId, OrderId, OrderRequest, OrderType, QuantitySpec, RejectCode};
use crate::model::{Product, Side};
use crate::sizing::{SizerSpec, SkipReason};

fn link(room_rupees: f64) -> AccountLink {
    AccountLink {
        room_rupees: Some(room_rupees),
        trades: None,
    }
}

/// The run of the module docs, in `quantity_mode`, linked to `link` before it starts.
fn engine_of(quantity_mode: QuantityMode, link: AccountLink) -> Engine {
    let day = july_2025(1);
    let setup = Setup {
        quantity_mode,
        ..Setup::minutes(day, day)
    };
    let mut kernel = setup.engine(minute_bars(&[day], |_, _| [1_000.0; 4]));
    assert_eq!(kernel.set_account_link(link), Ok(()));
    kernel
}

/// The run of the module docs in whole shares, linked to `link`.
fn linked(link: AccountLink) -> Engine {
    engine_of(QuantityMode::Venue, link)
}

/// The engine default's order of SBIN on CNC.
fn engine_default_buy() -> OrderRequest {
    engine_default_buy_of(SBIN)
}

/// The engine default's order of `instrument` on CNC.
fn engine_default_buy_of(instrument: InstrumentId) -> OrderRequest {
    let sized = QuantitySpec::Sizer(SizerSpec::EngineDefault);
    OrderRequest::market(instrument, Side::Buy, sized)
}

/// The exit of the whole MIS position.
fn exit_mis() -> OrderRequest {
    OrderRequest {
        product: Product::Mis,
        reduce_only: true,
        ..OrderRequest::market(SBIN, Side::Sell, QuantitySpec::PositionFraction(1.0))
    }
}

/// The events of minute `index`, step `step`, by name.
fn events_at(kernel: &Engine, index: u32, step: u8) -> Vec<String> {
    events(kernel)
        .into_iter()
        .filter(|(slice, at_step, _)| (*slice, *at_step) == (index, step))
        .map(|(_, _, name)| name)
        .collect()
}

/// Run `kernel` to its end: an MIS buy of 100 at the close of minute 0, the engine
/// default's CNC buy at the close of minute 5, the MIS position's exit at minute 10.
fn run_script(mut kernel: Engine) -> RunOutput {
    loop {
        let decided = match kernel.advance().expect("the slice runs") {
            Pause::BarClose { stamp, .. } => stamp.slice,
            Pause::Finished { .. } => break,
            _ => continue,
        };
        let request = match decided {
            0 => market(Side::Buy, 100.0, Product::Mis),
            5 => engine_default_buy(),
            10 => exit_mis(),
            _ => continue,
        };
        assert!(kernel.submit(request).is_ok());
    }
    kernel.finish().expect("finished")
}

#[test]
fn a_link_with_a_room_of_0_or_an_ample_room_changes_nothing() {
    let alone = run_script(linked(AccountLink::default()));
    // A room of 0: the account's free cash is the engine's own (a fund of one).
    assert_eq!(run_script(linked(link(0.0))), alone);
    // Ample: the account's free cash stays above the strategy's equity and free cash.
    assert_eq!(run_script(linked(link(2_000_000.0))), alone);
    assert_eq!(
        alone.fills.len(),
        3,
        "the MIS buy, the CNC buy and the MIS exit"
    );
}

/// The events at the open of minute 1 of an MIS buy of `units` decided at the close of
/// minute 0 under `link`.
fn opening_fill(link: AccountLink, units: f64) -> Vec<String> {
    let mut kernel = linked(link);
    pause_at(&mut kernel, 0);
    assert!(kernel
        .submit(market(Side::Buy, units, Product::Mis))
        .is_ok());
    pause_at(&mut kernel, 1);
    events_at(&kernel, 1, 2)
}

#[test]
fn an_opening_fill_is_checked_against_its_own_free_cash_then_the_accounts() {
    let filled = || vec!["Filled".to_string(), "PositionOpened".to_string()];
    let rejected = |code: &str| vec![format!("Rejected {code}")];
    // (link, units, the events of the fill): the engine's own free cash is 10,00,000, the
    // account's 10,00,000 plus the room.
    let table = [
        (AccountLink::default(), 1_100.0, rejected("margin")),
        (link(5_000_000.0), 1_100.0, rejected("margin")),
        (link(-950_000.0), 1_100.0, rejected("margin")),
        (link(-950_000.0), 100.0, rejected("account_margin")),
        (link(-950_000.0), 51.0, rejected("account_margin")),
        (link(-950_000.0), 50.0, filled()),
    ];
    for (link, units, expected) in table {
        assert_eq!(
            opening_fill(link, units),
            expected,
            "{link:?}, {units} units"
        );
    }
}

#[test]
fn a_closing_fill_is_never_checked_against_the_account() {
    let mut kernel = linked(AccountLink::default());
    pause_at(&mut kernel, 0);
    assert!(kernel
        .submit(market(Side::Buy, 100.0, Product::Mis))
        .is_ok());
    pause_at(&mut kernel, 1);
    // Own free cash 9,00,000: the account's is 9,00,000 - 20,00,000 = -11,00,000.
    assert_eq!(kernel.set_account_link(link(-2_000_000.0)), Ok(()));
    assert!(kernel.submit(exit_mis()).is_ok());
    pause_at(&mut kernel, 2);
    assert_eq!(events_at(&kernel, 2, 2), ["Filled", "PositionClosed"]);
    assert_eq!(held(&kernel, Product::Mis), 0.0);
}

#[test]
fn a_reversals_opening_part_beyond_the_account_is_refused_and_its_close_stands() {
    let mut kernel = linked(AccountLink::default());
    pause_at(&mut kernel, 0);
    assert!(kernel
        .submit(market(Side::Buy, 100.0, Product::Mis))
        .is_ok());
    pause_at(&mut kernel, 1);
    // The close frees the 1,00,000 of margin: own free cash 10,00,000, the account's
    // 50,000 against the 1,00,000 the short of 100 needs.
    assert_eq!(kernel.set_account_link(link(-950_000.0)), Ok(()));
    let reverse = OrderRequest {
        reverse: true,
        ..market(Side::Sell, 200.0, Product::Mis)
    };
    assert_eq!(
        kernel.submit(reverse),
        Ok(Submission::Submitted(OrderId(2)))
    );
    pause_at(&mut kernel, 2);
    assert_eq!(
        events_at(&kernel, 2, 2),
        ["Filled", "PositionClosed", "Rejected account_margin"]
    );
    assert_eq!(held(&kernel, Product::Mis), 0.0);
}

#[test]
fn a_sizer_sees_the_lesser_of_its_own_and_the_accounts_free_cash() {
    // (link, the free cash a sizer sees): flat with 10,00,000; its equity stays 10,00,000.
    let table = [
        (AccountLink::default(), 1_000_000.0),
        (link(500_000.0), 1_000_000.0),
        (link(-400_000.0), 600_000.0),
    ];
    for (link, free_cash) in table {
        let kernel = linked(link);
        let request = market(Side::Buy, 1.0, Product::Mis);
        let seen = kernel.sizing_inputs(&request).expect("sizing inputs");
        assert_eq!(seen.free_cash_rupees, free_cash, "{link:?}");
        assert_eq!(seen.equity_rupees, 1_000_000.0, "{link:?}");
    }
}

#[test]
fn the_engine_default_sizes_from_at_most_the_accounts_free_cash() {
    // At 1,000.00 with the venue cash buffer of 2 percent: 10,00,000 x 0.98 / 1,000.00 =
    // 980 units alone, 4,00,000 x 0.98 / 1,000.00 = 392 when the account has 4,00,000.
    for (link, units) in [(AccountLink::default(), 980.0), (link(-600_000.0), 392.0)] {
        let mut kernel = linked(link);
        pause_at(&mut kernel, 0);
        let submitted = kernel.submit(engine_default_buy());
        assert_eq!(submitted, Ok(Submission::Submitted(OrderId(1))));
        let order = kernel.order(OrderId(1)).expect("the order");
        assert_eq!(order.quantity_units, units, "{link:?}");
    }
}

#[test]
fn the_engine_default_of_two_instruments_sizes_its_share_from_at_most_the_accounts_free_cash() {
    // SBIN and INFY flat at 1,000.00. The engine default's CNC buy of SBIN at the close of
    // minute 0 takes half the capital: 10,00,000 / 2 x 0.98 / 1,000.00 = 490 units, filled
    // at minute 1 and blocking 4,90,000, which leaves 5,10,000 of the engine's own free
    // cash. INFY's at the close of minute 2, under the link given then, sizes from the
    // least of its share (5,00,000), its own free cash and the account's free cash
    // (5,10,000 plus the room), less the 2 percent buffer.
    let table = [
        (AccountLink::default(), 490.0),
        // A fund of one: the account's free cash is the engine's own.
        (link(0.0), 490.0),
        // The other strategies' free cash beside it (a second strategy of 10,00,000, flat).
        (link(1_000_000.0), 490.0),
        // The account's free cash 2,00,000: 2,00,000 x 0.98 / 1,000.00.
        (link(-310_000.0), 196.0),
    ];
    for (link, infy_units) in table {
        let day = july_2025(1);
        let flat = || minute_bars(&[day], |_, _| [1_000.0; 4]);
        let mut kernel = Setup::minutes(day, day).engine_over(vec![flat(), flat()]);
        pause_at(&mut kernel, 0);
        let sbin = kernel.submit(engine_default_buy_of(SBIN));
        assert_eq!(sbin, Ok(Submission::Submitted(OrderId(1))));
        pause_at(&mut kernel, 2);
        assert_eq!(kernel.set_account_link(link), Ok(()));
        let infy = kernel.submit(engine_default_buy_of(INFY));
        assert_eq!(infy, Ok(Submission::Submitted(OrderId(2))), "{link:?}");
        let units = |id| kernel.order(id).expect("the order").quantity_units;
        assert_eq!(
            (units(OrderId(1)), units(OrderId(2))),
            (490.0, infy_units),
            "{link:?}"
        );
    }
}

#[test]
fn an_engine_default_sized_at_the_decision_that_only_the_account_leaves_nothing_is_refused() {
    // Whole shares: an MIS buy decided at the close of minute 0 fills at minute 1, then
    // the engine default's CNC buy is sized at the close of minute 1 under the link. The
    // MIS buy of 100 leaves 9,00,000 of the engine's own free cash, the one of 1,000 none.
    // (MIS units, link, the engine default's submission)
    let table = [
        (
            100.0,
            AccountLink::default(),
            Submission::Submitted(OrderId(2)),
        ),
        // The account's free cash 0: only the other strategies leave it nothing.
        (
            100.0,
            link(-900_000.0),
            Submission::Rejected(OrderId(2), RejectCode::AccountMargin),
        ),
        // The engine's own free cash 0: its own reason stands, linked or not.
        (
            1_000.0,
            AccountLink::default(),
            Submission::Skipped(SkipReason::RoundedToZero),
        ),
        (
            1_000.0,
            link(-900_000.0),
            Submission::Skipped(SkipReason::RoundedToZero),
        ),
    ];
    for (mis_units, link, expected) in table {
        let mut kernel = linked(AccountLink::default());
        pause_at(&mut kernel, 0);
        assert!(kernel
            .submit(market(Side::Buy, mis_units, Product::Mis))
            .is_ok());
        pause_at(&mut kernel, 1);
        assert_eq!(kernel.set_account_link(link), Ok(()));
        let case = format!("{mis_units} MIS units, {link:?}");
        assert_eq!(kernel.submit(engine_default_buy()), Ok(expected), "{case}");
    }
}

#[test]
fn an_engine_default_refused_at_the_decision_is_an_order_rejected_account_margin() {
    let mut kernel = linked(link(-1_000_000.0));
    pause_at(&mut kernel, 0);
    assert_eq!(
        kernel.submit(engine_default_buy()),
        Ok(Submission::Rejected(OrderId(1), RejectCode::AccountMargin))
    );
    let order = kernel.order(OrderId(1)).expect("the order");
    assert_eq!(
        (order.reject_code, order.quantity_units),
        (Some(RejectCode::AccountMargin), 0.0)
    );
    assert_eq!(events_at(&kernel, 0, 7), ["Rejected account_margin"]);
}

#[test]
fn a_replacement_the_account_leaves_unsized_is_refused_and_the_order_works_on() {
    // A resting CNC buy LIMIT sized by the engine default at the close of minute 0: 980
    // units. At the close of minute 1 the account's free cash is 0.
    let mut kernel = linked(AccountLink::default());
    pause_at(&mut kernel, 0);
    let limit_at = |limit_rupees| OrderRequest {
        order_type: OrderType::Limit { limit_rupees },
        ..engine_default_buy()
    };
    assert_eq!(
        kernel.submit(limit_at(900.0)),
        Ok(Submission::Submitted(OrderId(1)))
    );
    pause_at(&mut kernel, 1);
    assert_eq!(kernel.set_account_link(link(-1_000_000.0)), Ok(()));
    assert_eq!(
        kernel.modify(OrderId(1), limit_at(905.0)),
        Ok(Submission::Rejected(OrderId(1), RejectCode::AccountMargin))
    );
    let order = kernel.order(OrderId(1)).expect("the order");
    let limit = OrderType::Limit {
        limit_rupees: 900.0,
    };
    assert_eq!(
        (
            order.quantity_units,
            order.request.order_type,
            order.status.is_working()
        ),
        (980.0, limit, true)
    );
}

#[test]
fn an_engine_default_sized_at_the_fill_that_only_the_account_leaves_nothing_is_refused() {
    // Fractional: sized at the fill at the open of minute 1, 1,000.00, with no buffer, on
    // the lesser free cash. With no account cash the strategy's own would have bought.
    let table = [
        (AccountLink::default(), "Filled 1000"),
        (link(-600_000.0), "Filled 400"),
        (link(-1_000_000.0), "Rejected account_margin"),
    ];
    for (link, expected) in table {
        let mut kernel = engine_of(QuantityMode::Fractional, link);
        pause_at(&mut kernel, 0);
        assert!(kernel.submit(engine_default_buy()).is_ok());
        pause_at(&mut kernel, 1);
        let order = kernel.order(OrderId(1)).expect("the order");
        let outcome = match order.reject_code {
            Some(code) => format!("Rejected {}", code.code()),
            None => format!("Filled {}", order.filled_units),
        };
        assert_eq!(outcome, expected, "{link:?}");
    }
}

#[test]
fn a_room_that_is_not_rupees_and_a_live_sessions_link_are_refused() {
    let mut kernel = linked(AccountLink::default());
    let nan = kernel.set_account_link(link(f64::NAN));
    assert!(matches!(
        nan,
        Err(KernelError::InvalidConfig {
            field: "account_link",
            ..
        })
    ));
    let mut session = holding_100();
    let live = session.set_account_link(link(0.0));
    assert!(matches!(
        live,
        Err(KernelError::InvalidConfig {
            field: "account_link",
            ..
        })
    ));
    assert_eq!(session.set_account_link(AccountLink::default()), Ok(()));
}
