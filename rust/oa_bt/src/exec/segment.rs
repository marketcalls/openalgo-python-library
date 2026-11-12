//! The per-segment order table (04-execution 4.1): which stop order a segment accepts.
//! The same table drives the backtest's matching and the live adapter, so both send the
//! same order.
//!
//! | Segment | Stop order |
//! |---|---|
//! | Cash equity, futures | SL-M, as asked |
//! | Options | SL with the limit `OPTIONS_STOP_LIMIT_BUFFER_PERCENT` beyond the trigger (exchanges reject SL-M on NFO options) |

use crate::model::{InstrumentKind, Side};

use super::error::ExecError;
use super::order::OrderType;
use super::price::round_away_from_trader;

/// The limit of an options stop order beyond its trigger, percent (04-execution 4.1;
/// the GTT child of an option uses the same buffer, decision GTT-1).
pub const OPTIONS_STOP_LIMIT_BUFFER_PERCENT: f64 = 5.0;

/// The order a segment accepts for `order_type`: an SL-M on an option becomes an SL
/// whose limit lies `OPTIONS_STOP_LIMIT_BUFFER_PERCENT` beyond the trigger for the
/// order's side (above for a buy, below for a sell), rounded to the tick away from the
/// trader. Every other order is accepted as asked.
///
/// Errors: `InvalidPrice` or `InvalidTick` for an unusable trigger or tick.
pub fn venue_order_type(
    kind: InstrumentKind,
    order_type: OrderType,
    side: Side,
    tick_rupees: f64,
) -> Result<OrderType, ExecError> {
    match (kind, order_type) {
        (InstrumentKind::Option, OrderType::StopMarket { trigger_rupees }) => {
            let factor = 1.0 + side.sign() * OPTIONS_STOP_LIMIT_BUFFER_PERCENT / 100.0;
            let limit_rupees = round_away_from_trader(trigger_rupees * factor, tick_rupees, side)?;
            Ok(OrderType::StopLimit {
                trigger_rupees,
                limit_rupees,
            })
        }
        _ => Ok(order_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_options_stop_market_becomes_a_stop_limit_five_percent_beyond_the_trigger() {
        // A sell stop at 120.30: 5 percent below is 114.285, down to the tick 114.25.
        let asked = OrderType::StopMarket {
            trigger_rupees: 120.3,
        };
        let sent = venue_order_type(InstrumentKind::Option, asked, Side::Sell, 0.05);
        let expected = OrderType::StopLimit {
            trigger_rupees: 120.3,
            limit_rupees: 114.25,
        };
        assert_eq!(sent, Ok(expected));
    }

    #[test]
    fn a_buy_stop_on_an_option_takes_its_limit_above_the_trigger() {
        // 5 percent above 80.10 is 84.105, up to the tick 84.15.
        let asked = OrderType::StopMarket {
            trigger_rupees: 80.1,
        };
        let sent = venue_order_type(InstrumentKind::Option, asked, Side::Buy, 0.05);
        let expected = OrderType::StopLimit {
            trigger_rupees: 80.1,
            limit_rupees: 84.15,
        };
        assert_eq!(sent, Ok(expected));
    }

    #[test]
    fn cash_and_futures_keep_their_stop_market() {
        let asked = OrderType::StopMarket {
            trigger_rupees: 1460.0,
        };
        for kind in [InstrumentKind::Equity, InstrumentKind::Future] {
            assert_eq!(venue_order_type(kind, asked, Side::Sell, 0.05), Ok(asked));
        }
    }
}
