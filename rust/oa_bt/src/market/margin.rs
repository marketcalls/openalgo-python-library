//! Margin: what opening exposure blocks from a strategy's free cash (04-execution 5).
//!
//! The engine asks for the margin of the quantity an order opens (an entry or an add;
//! a reduction blocks nothing) at the fill bar's open, and rejects the order with
//! reject code `margin` when it is more than the free cash. The figures come from the
//! dated default table `margin_defaults.csv` (`margin_table`): delivery equity 100
//! percent of value, intraday equity value / leverage with leverage only when the
//! strategy asks for it (at most 5x), long options the premium, and every futures
//! contract, on any exchange and underlying, the default percent of its contract value
//! of decision D23. A run may give its own futures margin for a contract
//! (`FuturesMargin`, from the `margin=` argument resolved per underlying in Python): a
//! percent of the contract value, or rupees per lot of the contract traded. It is a
//! backtest assumption only: in live trading the broker computes margin. Zero defaults
//! are forbidden: a kind and product with no row is an error, never free. Short options
//! are Phase 3.

use std::fmt;

use crate::model::{Day, Exchange, InstrumentKind, Product, Side};

pub use super::margin_table::{margin_defaults, margin_row, MarginBasis, MarginRow};

/// Tolerance on the leverage cap, so that asking for exactly 5x of a 20 percent row
/// is allowed despite the binary representation of 100 / 20.
const LEVERAGE_TOLERANCE: f64 = 1.0e-9;

/// The largest futures margin in percent of contract value: the whole value.
const MOST_PERCENT_OF_VALUE: f64 = 100.0;

/// A run's own futures margin for a contract (`margin=`, decision D23), in place of the
/// default row: the figure a broker's margin calculator shows for its underlying.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FuturesMargin {
    /// Percent of the contract value (20.0 is 20 percent), above 0 and at most 100.
    PercentOfValue(f64),
    /// Rupees per lot of the contract traded, above 0: a position of n lots blocks n
    /// times this, whatever the lot size of its contract (a lot size changed at a roll
    /// is the next contract's own).
    RupeesPerLot(f64),
}

impl FuturesMargin {
    /// Check the figure.
    ///
    /// Errors: `InvalidRequest` for a percent that is not above 0 and at most 100, or
    /// rupees that are not a positive number.
    pub fn validate(self) -> Result<(), MarginError> {
        match self {
            FuturesMargin::PercentOfValue(percent)
                if !(percent.is_finite() && percent > 0.0 && percent <= MOST_PERCENT_OF_VALUE) =>
            {
                Err(invalid(format!(
                    "a futures margin in percent of contract value must be above 0 and at most \
                     100, got {percent}"
                )))
            }
            FuturesMargin::RupeesPerLot(rupees) if !(rupees.is_finite() && rupees > 0.0) => {
                Err(invalid(format!(
                    "a futures margin in rupees per lot must be a positive number, got {rupees}"
                )))
            }
            _ => Ok(()),
        }
    }
}

/// One margin question: open `quantity_units` at `price_rupees` on `day`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarginRequest {
    /// Exchange of the instrument.
    pub exchange: Exchange,
    /// Equity, future or option.
    pub kind: InstrumentKind,
    /// CNC, MIS or NRML.
    pub product: Product,
    /// Side of the exposure being opened.
    pub side: Side,
    /// Units opened, positive.
    pub quantity_units: f64,
    /// Price, rupees (the premium for an option).
    pub price_rupees: f64,
    /// Rupees per one-rupee move per unit.
    pub multiplier: f64,
    /// Units in one lot of the instrument, positive (rupees per lot are per lot of it).
    pub lot_size_units: f64,
    /// Intraday leverage asked for; 1.0 when none was asked for.
    pub leverage: f64,
    /// IST trade date, for the dated table.
    pub day: Day,
    /// The run's own margin for a futures contract; `None` uses the default table.
    pub futures_margin: Option<FuturesMargin>,
}

/// The margin of a request and the table row it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarginQuote {
    /// Rupees blocked.
    pub rupees: f64,
    /// The row of the default table that priced it; `None` when the run's own futures
    /// margin did.
    pub row: Option<&'static MarginRow>,
}

/// What went wrong while computing margin.
#[derive(Clone, Debug, PartialEq)]
pub enum MarginError {
    /// The request cannot be priced (non-positive quantity, price, multiplier or lot
    /// size, a leverage the table does not allow, an unusable own futures margin, or no
    /// rule for the kind and product).
    InvalidRequest { reason: String },
    /// The embedded default table is malformed (a defect of the release).
    InvalidTable { reason: String },
    /// A margin rule that is not built yet, and what the user can do instead.
    NotBuilt {
        feature: &'static str,
        instead: &'static str,
    },
}

impl fmt::Display for MarginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarginError::InvalidRequest { reason } => write!(f, "margin: {reason}"),
            MarginError::InvalidTable { reason } => {
                write!(f, "the built-in margin table is malformed: {reason}")
            }
            MarginError::NotBuilt { feature, instead } => {
                crate::model::write_not_supported(f, feature, instead)
            }
        }
    }
}

impl std::error::Error for MarginError {}

/// The margin rules of a run: the dated default table, or the run's own futures margin
/// where the request carries one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MarginModel;

impl MarginModel {
    /// Rupees of free cash that opening the request blocks, and the table row used.
    ///
    /// Errors: `InvalidRequest` for a non-positive quantity, price, multiplier or lot
    /// size, a leverage below 1, leverage on anything but a `leverage` row, leverage
    /// above the row's limit, an own futures margin on anything but a future or one
    /// `FuturesMargin::validate` refuses, or a kind and product with no rule;
    /// `NotBuilt` for short options; `InvalidTable` for a malformed table.
    pub fn initial_margin(&self, request: &MarginRequest) -> Result<MarginQuote, MarginError> {
        let value = notional(request)?;
        if request.kind == InstrumentKind::Option && request.side == Side::Sell {
            return Err(not_built(
                "selling an option to open a short position (short-option margin)",
                "buy options, or trade futures",
            ));
        }
        if let Some(own) = request.futures_margin {
            return own_futures_margin(own, request, value);
        }
        let row = margin_row(request.kind, request.product, request.day)?
            .ok_or_else(|| missing_row(request))?;
        let rupees = match row.basis {
            MarginBasis::Leverage { least_percent } => {
                let most = 100.0 / least_percent;
                if request.leverage > most * (1.0 + LEVERAGE_TOLERANCE) {
                    return Err(invalid(format!(
                        "leverage {}x is above the {most}x that row {} of the margin table allows",
                        request.leverage, row.meta.id
                    )));
                }
                value / request.leverage
            }
            MarginBasis::Percent { percent } => {
                if request.leverage != 1.0 {
                    return Err(invalid(format!(
                        "leverage applies to intraday (MIS) equity only; row {} blocks {percent} \
                         percent of the value",
                        row.meta.id
                    )));
                }
                value * percent / 100.0
            }
        };
        Ok(MarginQuote {
            rupees,
            row: Some(row),
        })
    }
}

/// The margin of a futures contract at the run's own figure: a percent of `value`, or
/// rupees per lot times the lots opened.
fn own_futures_margin(
    own: FuturesMargin,
    request: &MarginRequest,
    value: f64,
) -> Result<MarginQuote, MarginError> {
    if request.kind != InstrumentKind::Future {
        return Err(invalid(format!(
            "margin= sets the margin of futures only, not of {} instruments",
            request.kind.code()
        )));
    }
    if request.leverage != 1.0 {
        return Err(invalid(
            "leverage applies to intraday (MIS) equity only; a futures margin is set with \
             margin="
                .to_string(),
        ));
    }
    own.validate()?;
    let rupees = match own {
        FuturesMargin::PercentOfValue(percent) => value * percent / 100.0,
        FuturesMargin::RupeesPerLot(rupees) => {
            rupees * request.quantity_units / request.lot_size_units
        }
    };
    Ok(MarginQuote { rupees, row: None })
}

fn notional(request: &MarginRequest) -> Result<f64, MarginError> {
    let inputs = [
        ("quantity", request.quantity_units),
        ("price", request.price_rupees),
        ("multiplier", request.multiplier),
        ("lot size", request.lot_size_units),
    ];
    for (what, value) in inputs {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid(format!(
                "{what} must be a positive number, got {value}"
            )));
        }
    }
    if !(request.leverage.is_finite() && request.leverage >= 1.0) {
        return Err(invalid(format!(
            "leverage must be 1 or more, got {}",
            request.leverage
        )));
    }
    Ok(request.quantity_units * request.price_rupees * request.multiplier)
}

fn missing_row(request: &MarginRequest) -> MarginError {
    invalid(format!(
        "no margin rule for {} with {} on {} ({})",
        request.kind.code(),
        request.product.code(),
        request.exchange.code(),
        crate::market::calendar::format_day(request.day)
    ))
}

fn invalid(reason: String) -> MarginError {
    MarginError::InvalidRequest { reason }
}

fn not_built(feature: &'static str, instead: &'static str) -> MarginError {
    MarginError::NotBuilt { feature, instead }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::days_from_civil;

    fn request(kind: InstrumentKind, product: Product, leverage: f64) -> MarginRequest {
        MarginRequest {
            exchange: Exchange::Nse,
            kind,
            product,
            side: Side::Buy,
            quantity_units: 100.0,
            price_rupees: 500.0,
            multiplier: 1.0,
            lot_size_units: 1.0,
            leverage,
            day: days_from_civil(2025, 7, 1),
            futures_margin: None,
        }
    }

    fn quote(request: &MarginRequest) -> (f64, Option<String>) {
        match MarginModel.initial_margin(request) {
            Ok(quote) => (quote.rupees, quote.row.map(|row| row.meta.id.clone())),
            Err(error) => panic!("{error}"),
        }
    }

    fn row(id: &str) -> Option<String> {
        Some(id.to_string())
    }

    /// `lot` units of a future on `exchange` at `price`, one lot.
    fn future(exchange: Exchange, lot: f64, price: f64, multiplier: f64) -> MarginRequest {
        MarginRequest {
            exchange,
            quantity_units: lot,
            price_rupees: price,
            multiplier,
            lot_size_units: lot,
            ..request(InstrumentKind::Future, Product::Nrml, 1.0)
        }
    }

    fn own(request: MarginRequest, margin: FuturesMargin) -> MarginRequest {
        MarginRequest {
            futures_margin: Some(margin),
            ..request
        }
    }

    #[test]
    fn delivery_blocks_the_full_value() {
        // 100 x 500 = 50,000.
        let cnc = request(InstrumentKind::Equity, Product::Cnc, 1.0);
        assert_eq!(quote(&cnc), (50_000.0, row("CNC_EQ/1")));
    }

    #[test]
    fn delivery_cannot_be_leveraged() {
        let cnc = request(InstrumentKind::Equity, Product::Cnc, 2.0);
        let result = MarginModel.initial_margin(&cnc);
        assert!(matches!(result, Err(MarginError::InvalidRequest { .. })));
    }

    #[test]
    fn intraday_equity_is_unleveraged_unless_leverage_is_asked_for() {
        // 50,000 at 1x; at the 5x the row allows, 50,000 / 5 = 10,000.
        let plain = request(InstrumentKind::Equity, Product::Mis, 1.0);
        let five = request(InstrumentKind::Equity, Product::Mis, 5.0);
        assert_eq!(quote(&plain), (50_000.0, row("MIS_EQ/2")));
        assert_eq!(quote(&five), (10_000.0, row("MIS_EQ/2")));
    }

    #[test]
    fn intraday_leverage_above_five_times_is_refused() {
        let six = request(InstrumentKind::Equity, Product::Mis, 6.0);
        let result = MarginModel.initial_margin(&six);
        assert!(matches!(result, Err(MarginError::InvalidRequest { .. })));
    }

    #[test]
    fn intraday_before_the_peak_margin_rules_uses_the_stand_in_row() {
        let early = MarginRequest {
            day: days_from_civil(2021, 8, 31),
            ..request(InstrumentKind::Equity, Product::Mis, 5.0)
        };
        assert_eq!(quote(&early), (10_000.0, row("MIS_EQ/1")));
    }

    #[test]
    fn a_long_option_blocks_the_premium() {
        // 75 units at a premium of 120 = 9,000.
        let option = MarginRequest {
            exchange: Exchange::Nfo,
            quantity_units: 75.0,
            price_rupees: 120.0,
            ..request(InstrumentKind::Option, Product::Nrml, 1.0)
        };
        assert_eq!(quote(&option), (9_000.0, row("LONG_OPT/1")));
    }

    #[test]
    fn short_options_are_not_built() {
        let short = MarginRequest {
            side: Side::Sell,
            ..request(InstrumentKind::Option, Product::Nrml, 1.0)
        };
        let result = MarginModel.initial_margin(&short);
        assert!(matches!(result, Err(MarginError::NotBuilt { .. })));
    }

    #[test]
    fn equity_has_no_rule_for_a_carry_forward_product() {
        let nrml = request(InstrumentKind::Equity, Product::Nrml, 1.0);
        let result = MarginModel.initial_margin(&nrml);
        assert!(matches!(result, Err(MarginError::InvalidRequest { .. })));
    }

    /// Decision D23: 20 percent of the contract value, on every exchange, any underlying.
    #[test]
    fn every_future_blocks_twenty_percent_of_its_contract_value_by_default() {
        let cases = [
            // An NFO index future: 75 x 25,000.00 = 18,75,000; 20 percent 3,75,000.
            (future(Exchange::Nfo, 75.0, 25_000.0, 1.0), 375_000.0),
            // An NFO stock future: 750 x 800.00 = 6,00,000; 20 percent 1,20,000.
            (future(Exchange::Nfo, 750.0, 800.0, 1.0), 120_000.0),
            // A BFO index future: 20 x 80,000.00 = 16,00,000; 20 percent 3,20,000.
            (future(Exchange::Bfo, 20.0, 80_000.0, 1.0), 320_000.0),
            // An MCX future with a multiplier: 1 x 1,39,716.00 x 100 = 1,39,71,600;
            // 20 percent 27,94,320.
            (future(Exchange::Mcx, 1.0, 139_716.0, 100.0), 2_794_320.0),
        ];
        for (request, margin) in cases {
            let (rupees, id) = quote(&request);
            assert!((rupees - margin).abs() < 1e-6, "{request:?}: {rupees}");
            assert_eq!(id, row("FUT/1"));
        }
    }

    #[test]
    fn futures_cannot_be_leveraged() {
        let leveraged = MarginRequest {
            leverage: 2.0,
            ..future(Exchange::Nfo, 75.0, 25_000.0, 1.0)
        };
        let result = MarginModel.initial_margin(&leveraged);
        assert!(matches!(result, Err(MarginError::InvalidRequest { .. })));
    }

    #[test]
    fn an_own_percent_replaces_the_default_row() {
        // 75 x 25,000.00 = 18,75,000; 12.5 percent is 2,34,375.
        let nifty = own(
            future(Exchange::Nfo, 75.0, 25_000.0, 1.0),
            FuturesMargin::PercentOfValue(12.5),
        );
        assert_eq!(quote(&nifty), (234_375.0, None));
    }

    #[test]
    fn own_rupees_per_lot_count_the_lots_of_the_contract_traded() {
        // Three lots of 25 (75 units) at 1,50,000 a lot are 4,50,000; the same 75 units
        // as one lot of 75 are 1,50,000.
        let small_lots = MarginRequest {
            lot_size_units: 25.0,
            ..future(Exchange::Nfo, 75.0, 25_000.0, 1.0)
        };
        let per_lot = FuturesMargin::RupeesPerLot(150_000.0);
        assert_eq!(quote(&own(small_lots, per_lot)), (450_000.0, None));
        let large_lot = future(Exchange::Nfo, 75.0, 25_000.0, 1.0);
        assert_eq!(quote(&own(large_lot, per_lot)), (150_000.0, None));
    }

    #[test]
    fn an_own_futures_margin_on_anything_but_a_future_is_refused() {
        let equity = own(
            request(InstrumentKind::Equity, Product::Cnc, 1.0),
            FuturesMargin::PercentOfValue(20.0),
        );
        let result = MarginModel.initial_margin(&equity);
        assert!(matches!(result, Err(MarginError::InvalidRequest { .. })));
    }

    #[test]
    fn an_own_futures_margin_out_of_range_is_refused() {
        let unusable = [
            FuturesMargin::PercentOfValue(0.0),
            FuturesMargin::PercentOfValue(100.5),
            FuturesMargin::PercentOfValue(f64::NAN),
            FuturesMargin::RupeesPerLot(-1.0),
            FuturesMargin::RupeesPerLot(f64::INFINITY),
        ];
        for margin in unusable {
            let nifty = own(future(Exchange::Nfo, 75.0, 25_000.0, 1.0), margin);
            let result = MarginModel.initial_margin(&nifty);
            assert!(
                matches!(result, Err(MarginError::InvalidRequest { .. })),
                "{margin:?}"
            );
        }
        assert_eq!(FuturesMargin::PercentOfValue(100.0).validate(), Ok(()));
    }
}
