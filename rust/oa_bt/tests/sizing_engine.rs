#![cfg(all(feature = "costs", feature = "time"))]
//! Sizing through the engine: `submit_target` (FR-SIZE-08), the initial risk of a trade
//! for its R-multiple (FR-SIZE-09), and the caps and `reinvest` read from the run
//! configuration (FR-SIZE-07).
//!
//! One-minute SBIN bars on 2025-07-01, flat at 100.00 unless a test says otherwise; no
//! charges, no slippage, whole shares, capital 1,00,00,000. A decision at the close of
//! minute k fills at the open of minute k + 1.

mod order_support;

use oa_bt::engine::{
    run, Driver, Engine, KernelError, Pause, RunConfig, RunOutput, Submission, TargetRequest,
};
use oa_bt::exec::{OrderId, OrderRequest, OrderType, QuantitySpec};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{Distance, LegRules};
use oa_bt::sizing::{SizeCaps, SizerSpec, SizingControls, SkipReason, TargetSize};
use order_support::*;

const PRICE: f64 = 100.0;

/// The decision instant of minute k's bar: its close.
fn close_of(k: u32) -> i64 {
    minute(k + 1)
}

/// What the plan does at a decision.
#[derive(Clone, Debug)]
enum Step {
    Target(TargetSize),
    Submit(Box<OrderRequest>),
}

/// A driver that targets or submits at given decision instants and keeps, for each
/// target, its outcome and the orders working right after it.
struct Plan {
    steps: Vec<(i64, Step)>,
    targets: Vec<Option<Submission>>,
    submissions: Vec<Submission>,
    working_after_target: Vec<Vec<OrderId>>,
}

impl Driver for Plan {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { decided_at, .. } = pause else {
            return Ok(());
        };
        let due: Vec<Step> = self
            .steps
            .iter()
            .filter(|(instant, _)| instant == decided_at)
            .map(|(_, step)| step.clone())
            .collect();
        for step in due {
            match step {
                Step::Target(size) => {
                    let request = TargetRequest {
                        instrument: SBIN,
                        product: Product::Mis,
                        size,
                        leverage: None,
                    };
                    self.targets.push(engine.submit_target(request)?);
                    self.working_after_target.push(engine.working_orders());
                }
                Step::Submit(request) => self.submissions.push(engine.submit(*request)?),
            }
        }
        Ok(())
    }
}

fn run_plan_with(
    steps: Vec<(i64, Step)>,
    bars: oa_bt::engine::BarColumns,
    adjust: impl FnOnce(&mut RunConfig),
) -> (RunOutput, Plan) {
    let mut config = minute_config();
    adjust(&mut config);
    let mut kernel = engine(config, equity(), bars);
    let mut plan = Plan {
        steps,
        targets: Vec::new(),
        submissions: Vec::new(),
        working_after_target: Vec::new(),
    };
    run(&mut kernel, &mut plan).expect("the run completes");
    (kernel.finish().expect("the run finished"), plan)
}

fn run_plan(steps: Vec<(i64, Step)>) -> (RunOutput, Plan) {
    run_plan_with(steps, minute_bars(30, PRICE, &[]), |_| {})
}

/// (side, units, reason) of every fill, in order, without the session close's exit of
/// the MIS position (reason `eod`), which every run here ends with.
fn trades_of(output: &RunOutput) -> Vec<(Side, f64, &'static str)> {
    fills(output)
        .into_iter()
        .map(|(_, _, side, units, _, _, reason)| (side, units, reason))
        .filter(|(_, _, reason)| *reason != "eod")
        .collect()
}

fn mis(side: Side, units: f64, rules: LegRules) -> OrderRequest {
    OrderRequest {
        leg_rules: rules,
        ..order(side, units, OrderType::Market, Product::Mis)
    }
}

fn units(value: f64) -> Step {
    Step::Target(TargetSize::Units(value))
}

#[test]
fn a_target_from_flat_buys_the_target() {
    let (output, _) = run_plan(vec![(close_of(0), units(50.0))]);
    assert_eq!(trades_of(&output), vec![(Side::Buy, 50.0, "entry")]);
}

#[test]
fn a_larger_target_adds_the_difference_and_a_smaller_one_only_reduces() {
    let (output, plan) = run_plan(vec![
        (close_of(0), units(50.0)),
        (close_of(2), units(80.0)),
        (close_of(4), units(20.0)),
    ]);
    assert_eq!(
        trades_of(&output),
        vec![
            (Side::Buy, 50.0, "entry"),
            (Side::Buy, 30.0, "add"),
            (Side::Sell, 60.0, "signal"),
        ]
    );
    let reducing = output.orders.last().expect("three orders");
    assert!(reducing.request.reduce_only);
    assert_eq!(plan.targets.len(), 3);
}

#[test]
fn a_target_on_the_other_side_closes_and_reverses_in_one_order() {
    let (output, _) = run_plan(vec![
        (close_of(0), units(50.0)),
        (close_of(2), units(-30.0)),
    ]);
    assert_eq!(
        trades_of(&output),
        vec![
            (Side::Buy, 50.0, "entry"),
            (Side::Sell, 50.0, "signal"),
            (Side::Sell, 30.0, "entry"),
        ]
    );
    assert_eq!(output.orders.len(), 2);
    assert!(output.orders[1].request.reverse);
}

#[test]
fn a_target_of_zero_exits_the_position() {
    let (output, _) = run_plan(vec![(close_of(0), units(50.0)), (close_of(2), units(0.0))]);
    assert_eq!(
        trades_of(&output),
        vec![(Side::Buy, 50.0, "entry"), (Side::Sell, 50.0, "signal")]
    );
}

#[test]
fn a_target_already_held_sends_no_order() {
    let (output, plan) = run_plan(vec![(close_of(0), units(50.0)), (close_of(2), units(50.0))]);
    assert_eq!(plan.targets[1], None);
    assert_eq!(output.orders.len(), 1);
}

#[test]
fn a_target_counts_the_market_orders_still_queued() {
    // Three targets at one decision: 50, 50 again (nothing more to send), then 70 (20
    // more). Both orders fill at the next open.
    let (output, plan) = run_plan(vec![
        (close_of(0), units(50.0)),
        (close_of(0), units(50.0)),
        (close_of(0), units(70.0)),
    ]);
    assert_eq!(plan.targets[1], None);
    assert_eq!(
        trades_of(&output),
        vec![(Side::Buy, 50.0, "entry"), (Side::Buy, 20.0, "add")]
    );
}

#[test]
fn a_target_value_is_rounded_down_to_whole_shares_at_the_decision_price() {
    // 1,234 rupees at 100.00 is 12.34 shares: 12.
    let (output, _) = run_plan(vec![(
        close_of(0),
        Step::Target(TargetSize::Rupees(1_234.0)),
    )]);
    assert_eq!(trades_of(&output), vec![(Side::Buy, 12.0, "entry")]);
}

#[test]
fn a_target_percent_is_of_the_equity() {
    // 1 percent of 1,00,00,000 is 1,00,000 at 100.00: 1,000 shares.
    let percent = Step::Target(TargetSize::PercentOfEquity(1.0));
    let (output, _) = run_plan(vec![(close_of(0), percent)]);
    assert_eq!(trades_of(&output), vec![(Side::Buy, 1_000.0, "entry")]);
}

#[test]
fn a_target_leaves_resting_orders_working_and_does_not_count_them() {
    // A buy limit at 90.00 rests below the flat 100.00 market; the target of 50 still
    // buys 50, and the limit order is still working after it.
    let resting = order(Side::Buy, 10.0, limit(90.0), Product::Mis);
    let (output, plan) = run_plan(vec![
        (close_of(0), Step::Submit(Box::new(resting))),
        (close_of(0), units(50.0)),
    ]);
    assert_eq!(trades_of(&output), vec![(Side::Buy, 50.0, "entry")]);
    let Submission::Submitted(limit_id) = plan.submissions[0] else {
        panic!("the limit order was accepted");
    };
    assert!(plan.working_after_target[0].contains(&limit_id));
}

#[test]
fn an_add_by_target_keeps_the_positions_stop() {
    // The position opens with a stop 5.00 below; the target's add carries no rules and
    // is accepted (SPEC-14), so the leg keeps its stop.
    let stop = LegRules {
        stop: Some(Distance::Points(5.0)),
        ..LegRules::default()
    };
    let (output, _) = run_plan(vec![
        (
            close_of(0),
            Step::Submit(Box::new(mis(Side::Buy, 50.0, stop))),
        ),
        (close_of(2), units(80.0)),
    ]);
    assert_eq!(
        trades_of(&output),
        vec![(Side::Buy, 50.0, "entry"), (Side::Buy, 30.0, "add")]
    );
    let after_add = output
        .levels
        .iter()
        .find(|row| row.quantity_units == 80.0)
        .expect("a row after the add");
    assert_eq!(after_add.stop.map(|s| s.quantity_units), Some(80.0));
}

#[test]
fn a_target_outside_a_decision_point_is_an_error() {
    let mut kernel = engine(minute_config(), equity(), minute_bars(5, PRICE, &[]));
    let request = TargetRequest {
        instrument: SBIN,
        product: Product::Mis,
        size: TargetSize::Units(10.0),
        leverage: None,
    };
    assert_eq!(
        kernel.submit_target(request),
        Err(KernelError::NotAtDecisionPoint)
    );
}

fn stop_points(points: f64) -> LegRules {
    LegRules {
        stop: Some(Distance::Points(points)),
        ..LegRules::default()
    }
}

fn exit(held: Side) -> Step {
    Step::Submit(Box::new(exit_all(held, Product::Mis)))
}

#[test]
fn a_trade_records_the_loss_at_its_entry_stop_as_its_initial_risk() {
    // 10 shares at 100.00 with a stop 2.00 below: 20.00 of initial risk.
    let (output, _) = run_plan(vec![
        (
            close_of(0),
            Step::Submit(Box::new(mis(Side::Buy, 10.0, stop_points(2.0)))),
        ),
        (close_of(3), exit(Side::Buy)),
    ]);
    assert_eq!(output.trades[0].initial_risk_rupees, Some(20.0));
}

#[test]
fn a_short_trades_initial_risk_is_measured_to_the_stop_above() {
    // Short 10 at 100.00 with a stop 3.00 above: 30.00.
    let (output, _) = run_plan(vec![
        (
            close_of(0),
            Step::Submit(Box::new(mis(Side::Sell, 10.0, stop_points(3.0)))),
        ),
        (close_of(3), exit(Side::Sell)),
    ]);
    assert_eq!(output.trades[0].initial_risk_rupees, Some(30.0));
}

#[test]
fn a_trade_opened_without_a_stop_has_no_initial_risk() {
    let (output, _) = run_plan(vec![
        (
            close_of(0),
            Step::Submit(Box::new(mis(Side::Buy, 10.0, LegRules::default()))),
        ),
        (close_of(3), exit(Side::Buy)),
    ]);
    assert_eq!(output.trades[0].initial_risk_rupees, None);
}

#[test]
fn an_add_does_not_change_the_initial_risk_of_the_trade() {
    // The add doubles the position; the initial risk stays that of the opening fill.
    let (output, _) = run_plan(vec![
        (
            close_of(0),
            Step::Submit(Box::new(mis(Side::Buy, 10.0, stop_points(2.0)))),
        ),
        (
            close_of(2),
            Step::Submit(Box::new(mis(Side::Buy, 10.0, stop_points(2.0)))),
        ),
        (close_of(4), exit(Side::Buy)),
    ]);
    assert_eq!(output.trades.len(), 1);
    assert_eq!(output.trades[0].initial_risk_rupees, Some(20.0));
}

fn sized(units: f64) -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::Sizer(SizerSpec::Qty(units)),
        ..order(Side::Buy, 1.0, OrderType::Market, Product::Mis)
    }
}

#[test]
fn a_position_cap_counts_the_units_already_held() {
    // A cap of 60 shares: 40 held, so a sized add of 50 is cut to 20.
    let caps = SizeCaps {
        position_units: Some(60.0),
        ..SizeCaps::default()
    };
    let (output, _) = run_plan_with(
        vec![
            (close_of(0), Step::Submit(Box::new(sized(40.0)))),
            (close_of(2), Step::Submit(Box::new(sized(50.0)))),
        ],
        minute_bars(30, PRICE, &[]),
        |config| config.sizing.caps = caps,
    );
    assert_eq!(
        trades_of(&output),
        vec![(Side::Buy, 40.0, "entry"), (Side::Buy, 20.0, "add")]
    );
}

#[test]
fn a_cap_that_leaves_no_room_skips_the_order_for_the_cap() {
    let caps = SizeCaps {
        position_units: Some(40.0),
        ..SizeCaps::default()
    };
    let (_, plan) = run_plan_with(
        vec![
            (close_of(0), Step::Submit(Box::new(sized(40.0)))),
            (close_of(2), Step::Submit(Box::new(sized(10.0)))),
        ],
        minute_bars(30, PRICE, &[]),
        |config| config.sizing.caps = caps,
    );
    assert_eq!(plan.submissions[1], Submission::Skipped(SkipReason::Cap));
}

#[test]
fn without_reinvest_a_percent_sizer_ignores_the_profit_made() {
    // The first trade makes 1,000.00 (100 shares bought at 100.00, sold at 110.00), so
    // equity is 1,00,01,000. A 10 percent sizer at 110.00 then buys 10,00,100 / 110.00 =
    // 9,091.8 -> 9,091 shares reinvesting, and 10,00,000 / 110.00 = 9,090.9 -> 9,090
    // from the initial capital.
    let mut bars = minute_bars(30, PRICE, &[]);
    for k in 2..bars.open.len() {
        bars.open[k] = 110.0;
        bars.high[k] = 110.0;
        bars.low[k] = 110.0;
        bars.close[k] = 110.0;
    }
    let percent = OrderRequest {
        quantity: QuantitySpec::Sizer(SizerSpec::Percent {
            percent: 10.0,
            basis: oa_bt::sizing::SizingBasis::Equity,
        }),
        ..order(Side::Buy, 1.0, OrderType::Market, Product::Mis)
    };
    let steps = vec![
        (
            close_of(0),
            Step::Submit(Box::new(mis(Side::Buy, 100.0, LegRules::default()))),
        ),
        (close_of(3), exit(Side::Buy)),
        (close_of(5), Step::Submit(Box::new(percent))),
    ];
    let last_entry = |output: &RunOutput| output.fills.last().map(|r| r.fill.quantity_units);
    let (reinvesting, _) = run_plan_with(steps.clone(), bars.clone(), |_| {});
    let (initial, _) = run_plan_with(steps, bars, |config| {
        config.sizing = SizingControls {
            reinvest: false,
            ..SizingControls::default()
        }
    });
    assert_eq!(last_entry(&reinvesting), Some(9_091.0));
    assert_eq!(last_entry(&initial), Some(9_090.0));
}
