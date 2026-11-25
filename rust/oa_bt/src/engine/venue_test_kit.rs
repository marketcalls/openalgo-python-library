//! A harness for the unit tests of a live session: one NSE day (2025-07-01) of 1-minute
//! bars of SBIN, bar k opening at 500.00 + 0.10 k, a session holding 100 units, that
//! session restarted from its saved state (16-session-design R14) on an SBIN whose freeze
//! quantity has changed, and the broker selling part of the position by itself (R15):
//! changes to a position the session's own orders cannot make.

use crate::exec::{InstrumentId, InstrumentSpec, OrderId, OrderRequest, QuantitySpec};
use crate::market::calendar::{BaseInterval, Calendar};
use crate::market::facts::MarketFacts;
use crate::market::margin::MarginModel;
use crate::market::slippage::SlippageSpec;
use crate::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use crate::portfolio::PositionKey;
use crate::rms::{ExitReason, StrategyRiskConfig};
use crate::sizing::SizingControls;

use super::config::{
    BarColumns, CostConfig, FillTiming, IntradayOverrides, Perturbation, QuantityMode, RunConfig,
    RunSetup, StrategyInterval,
};
use super::kernel::Engine;
use super::market_data::BaseBar;
use super::pause::Pause;
use super::submit::Submission;
use super::venue::VenueCommand;
use super::venue_roles::OrderRole;

pub(super) const SBIN: InstrumentId = InstrumentId(0);
const MINUTE: i64 = 60_000_000_000;
/// 09:15 IST, the NSE session open, in minutes since midnight.
const NSE_OPEN_MINUTE: u32 = 9 * 60 + 15;

fn day() -> Day {
    days_from_civil(2025, 7, 1)
}

/// The open of minute `k` of the session (k = 0 opens at 09:15), UTC nanoseconds.
pub(super) fn minute(k: u32) -> i64 {
    ts_from_ist(day(), NSE_OPEN_MINUTE + k)
}

fn sbin() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "SBIN".to_string(),
        exchange: Exchange::Nse,
        kind: InstrumentKind::Equity,
        lot_size_units: 1,
        tick_size_rupees: 0.05,
        multiplier: 1.0,
        freeze_quantity_units: None,
        cost_segment: None,
        last_trading_day: None,
        positional_exit_day: None,
    }
}

/// A 1-minute strategy on 1-minute bars of the test day, with no costs and no slippage.
fn config() -> RunConfig {
    RunConfig {
        first_day: day(),
        last_day: day(),
        first_slice: None,
        last_slice: None,
        present: None,
        base: BaseInterval::Intraday(MINUTE),
        strategy_interval: StrategyInterval::Intraday(MINUTE),
        fill: FillTiming::NextOpen,
        quantity_mode: QuantityMode::Venue,
        capital_rupees: 10_000_000.0,
        costs: CostConfig::Off,
        slippage: SlippageSpec::Zero,
        margin: MarginModel,
        market: MarketFacts::default(),
        risk: StrategyRiskConfig::default(),
        sizing: SizingControls::default(),
        intraday: IntradayOverrides::default(),
        deliver_events: false,
        perturbation: Perturbation::default(),
    }
}

/// The bar of minute `k` (module docs): high and low 0.50 around its open, close 0.05
/// above it.
fn bar(k: u32) -> BaseBar {
    let open = 500.0 + 0.10 * f64::from(k);
    BaseBar {
        ts_open: minute(k),
        open,
        high: open + 0.50,
        low: open - 0.50,
        close: open + 0.05,
        volume: 1_000.0,
    }
}

/// Advance `kernel`, adding each slice's bar and releasing it, until a pause `until` takes.
fn advance_until(kernel: &mut Engine, until: impl Fn(&Pause) -> bool) {
    loop {
        let pause = kernel.advance().expect("the slice runs");
        if until(&pause) {
            return;
        }
        match pause {
            Pause::Waiting { next_slice } => {
                let k = u32::try_from(next_slice).expect("a slice of the test day");
                kernel
                    .add_bar(SBIN, bar(k))
                    .expect("a bar of a closed slice");
                kernel.release(next_slice + 1);
            }
            Pause::Finished { .. } => panic!("the session finished"),
            _ => {}
        }
    }
}

/// Advance `kernel` until it pauses at the close of minute `k`.
pub(super) fn pause_at_close(kernel: &mut Engine, k: u32) {
    advance_until(
        kernel,
        |pause| matches!(pause, Pause::BarClose { stamp, .. } if stamp.slice >= k),
    );
}

/// Advance `kernel` until it waits before slice `slice`, with the venue's commands left.
pub(super) fn wait_before(kernel: &mut Engine, slice: usize) {
    advance_until(
        kernel,
        |pause| matches!(pause, Pause::Waiting { next_slice } if *next_slice >= slice),
    );
}

/// An MIS MARKET order of `units` on `side`.
pub(super) fn market(side: Side, units: f64) -> OrderRequest {
    OrderRequest {
        product: Product::Mis,
        ..OrderRequest::market(SBIN, side, QuantitySpec::Units(units))
    }
}

/// A session holding 100 units of SBIN in MIS, bought by o1 at the close of minute 0 and
/// filled at 500.10 at the open of minute 1; paused at the close of minute 2 with no
/// command left to take.
pub(super) fn holding_100() -> Engine {
    let mut kernel = session_of(sbin());
    pause_at_close(&mut kernel, 0);
    let buy = kernel.submit(market(Side::Buy, 100.0));
    assert_eq!(buy, Ok(Submission::Submitted(OrderId(1))));
    wait_before(&mut kernel, 1);
    kernel.take_commands();
    let filled = kernel.apply_fill(OrderId(1), 100.0, 500.1, minute(1));
    assert!(filled.is_ok(), "{filled:?}");
    pause_at_close(&mut kernel, 2);
    kernel.take_commands();
    kernel
}

/// The signed units of SBIN held in MIS.
pub(super) fn held(kernel: &Engine) -> f64 {
    kernel.held_units(PositionKey {
        instrument: SBIN,
        product: Product::Mis,
    })
}

/// The (order, side, units, role) of the orders `commands` place.
pub(super) fn placed(commands: &[VenueCommand]) -> Vec<(OrderId, Side, f64, OrderRole)> {
    commands
        .iter()
        .filter_map(|command| match *command {
            VenueCommand::Place {
                order,
                side,
                units,
                role,
                ..
            } => Some((order, side, units, role)),
            _ => None,
        })
        .collect()
}

/// The broker sells `units` of the long at `price_rupees` at `at` (UTC nanoseconds) with
/// no order of the session's, as a sale made by hand at the broker does: the session
/// records it (`apply_external_fill`), then fits its closing orders and moves its exits
/// on, as after any result from its venue.
pub(super) fn sold_at_the_broker(kernel: &mut Engine, units: f64, price_rupees: f64, at: i64) {
    let sold = kernel.apply_external_fill(
        SBIN,
        Product::Mis,
        Side::Sell,
        units,
        price_rupees,
        at,
        ExitReason::Manual,
    );
    assert!(sold.is_ok(), "{sold:?}");
}

/// A live session of the test day trading `instrument`, before its first slice.
fn session_of(instrument: InstrumentSpec) -> Engine {
    let setup = RunSetup {
        config: config(),
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![instrument],
        bars: vec![BarColumns::default()],
    };
    Engine::new_session(setup).expect("a valid session")
}

/// `holding_100` saved (`live_state`) and restored into a session restarted at the close
/// of minute 2 on an SBIN whose freeze quantity is now `freeze_units`, as a restart reads
/// the day's freeze quantity again: it holds the 100 units and has no order (the buy had
/// ended), so its next order is o1.
pub(super) fn holding_100_restarted(freeze_units: Option<u32>) -> Engine {
    let saved = holding_100().live_state();
    let mut kernel = session_of(InstrumentSpec {
        freeze_quantity_units: freeze_units,
        ..sbin()
    });
    pause_at_close(&mut kernel, 2);
    let restored = kernel.restore_live_state(&saved);
    assert_eq!(restored, Ok(Vec::new()));
    kernel
}
