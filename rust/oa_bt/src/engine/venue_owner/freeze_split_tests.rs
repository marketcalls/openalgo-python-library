//! An exit above the freeze quantity goes out in orders of at most it (`venue_owner`), each
//! a separate order with the exit's role: the strategy's own order and the engine's further
//! ones for `exit()`, the engine's orders for its own exit.
//!
//! A session builds no position above the freeze quantity (`venue_gates`), so these tests
//! hold 100 units of SBIN with no freeze quantity (`venue_test_kit::holding_100`) and then
//! lower it to 40, as a position restored from an earlier session or adopted from the
//! broker may stand above the freeze quantity the session reads.

use crate::engine::venue_test_kit::{holding_100, placed, SBIN};
use crate::engine::{Engine, EngineExit, ExitRequest, OrderRole, RiskExitReason, Submission};
use crate::exec::{OrderId, OrderRequest, QuantitySpec};
use crate::model::{Product, Side};
use crate::rms::ExitReason;

/// The freeze quantity the position is lowered to: 100 units leave as 40, 40 and 20.
const LOWERED_FREEZE_UNITS: u32 = 40;

/// Holding 100 with the freeze quantity lowered below it, paused at the close of minute 2.
fn holding_100_above_the_freeze_quantity() -> Engine {
    let mut kernel = holding_100();
    kernel.instruments[SBIN.index()].freeze_quantity_units = Some(LOWERED_FREEZE_UNITS);
    kernel
}

#[test]
fn an_engine_exit_above_the_freeze_quantity_goes_out_in_orders_with_its_role() {
    let mut kernel = holding_100_above_the_freeze_quantity();
    let kill = EngineExit::Risk(RiskExitReason::KillSwitch);
    let role = OrderRole::RiskExit(RiskExitReason::KillSwitch);
    assert_eq!(
        kernel.request_exit(SBIN, Product::Mis, kill),
        Ok(ExitRequest::Sent)
    );
    assert_eq!(
        placed(&kernel.take_commands()),
        [
            (OrderId(2), Side::Sell, 40.0, role),
            (OrderId(3), Side::Sell, 40.0, role),
            (OrderId(4), Side::Sell, 20.0, role),
        ]
    );
    for id in [2, 3, 4] {
        let order = kernel.order(OrderId(id)).expect("an exit order");
        let terms = (order.request.reduce_only, order.request.exit_reason);
        assert_eq!(terms, (true, Some(ExitReason::Manual)), "o{id}");
        assert_eq!(kernel.order_role(OrderId(id)), Some(role), "o{id}");
    }
}

#[test]
fn a_strategy_exit_above_the_freeze_quantity_is_its_own_order_and_more_with_its_role() {
    let mut kernel = holding_100_above_the_freeze_quantity();
    let whole = OrderRequest {
        product: Product::Mis,
        reduce_only: true,
        ..OrderRequest::market(SBIN, Side::Sell, QuantitySpec::PositionFraction(1.0))
    };
    assert_eq!(kernel.submit(whole), Ok(Submission::Submitted(OrderId(2))));
    assert_eq!(
        placed(&kernel.take_commands()),
        [
            (OrderId(2), Side::Sell, 40.0, OrderRole::Exit),
            (OrderId(3), Side::Sell, 40.0, OrderRole::Exit),
            (OrderId(4), Side::Sell, 20.0, OrderRole::Exit),
        ]
    );
    for id in [2, 3, 4] {
        let role = kernel.order_role(OrderId(id));
        assert_eq!(role, Some(OrderRole::Exit), "o{id}");
    }
}
