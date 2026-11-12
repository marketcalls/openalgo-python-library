//! Margin: what opening exposure blocks from a strategy's free cash (04-execution 5).
//!
//! The engine asks for the margin of the quantity an order opens (an entry or an add;
//! a reduction blocks nothing) at the fill bar's open, and rejects the order with
//! reject code `margin` when it is more than the free cash. The figures come from the
//! dated default table `margin_defaults.csv` (`margin_table`): delivery equity 100
//! percent of value, intraday equity value / leverage with leverage only when the
//! strategy asks for it (at most 5x), long options the premium, and futures a percent
//! of contract value per underlying (NSE index and stock futures, BSE BANKEX, MCX
//! commodities). Zero defaults are forbidden: a position for which no row exists is an
//! error, never free. Short options are Phase 3.

use std::fmt;

use crate::model::{Day, Exchange, InstrumentKind, Product, Side};

pub use super::margin_table::{margin_defaults, margin_row, MarginBasis, MarginRow};

/// Tolerance on the leverage cap, so that asking for exactly 5x of a 20 percent row
/// is allowed despite the binary representation of 100 / 20.
const LEVERAGE_TOLERANCE: f64 = 1.0e-9;

/// One margin question: open `quantity_units` at `price_rupees` on `day`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarginRequest<'a> {
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
    /// Intraday leverage asked for; 1.0 when none was asked for.
    pub leverage: f64,
    /// IST trade date, for the dated table.
    pub day: Day,
    /// The underlying of a derivative (futures margin is per underlying).
    pub underlying: Option<&'a str>,
}

/// The margin of a request and the table row it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarginQuote {
    /// Rupees blocked.
    pub rupees: f64,
    /// The row of the default table that priced it.
    pub row: &'static MarginRow,
}

/// What went wrong while computing margin.
#[derive(Clone, Debug, PartialEq)]
pub enum MarginError {
    /// The request cannot be priced (non-positive quantity, price or leverage, a
    /// leverage the table does not allow, or no rule for the kind and product).
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

/// The margin rules of a run: the dated default table.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MarginModel;

impl MarginModel {
    /// Rupees of free cash that opening the request blocks, and the table row used.
    ///
    /// Errors: `InvalidRequest` for a non-positive quantity, price or multiplier, a
    /// leverage below 1, leverage on anything but a `leverage` row, leverage above the
    /// row's limit, or a kind and product with no rule; `NotBuilt` for short options and
    /// for futures whose underlying has no row; `InvalidTable` for a malformed table.
    pub fn initial_margin(&self, request: &MarginRequest<'_>) -> Result<MarginQuote, MarginError> {
        let value = notional(request)?;
        if request.kind == InstrumentKind::Option && request.side == Side::Sell {
            return Err(not_built(
                "selling an option to open a short position (short-option margin)",
                "buy options, or trade futures",
            ));
        }
        let row = margin_row(
            request.kind,
            request.product,
            request.underlying,
            request.day,
        )?
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
        Ok(MarginQuote { rupees, row })
    }
}

fn notional(request: &MarginRequest<'_>) -> Result<f64, MarginError> {
    let inputs = [
        ("quantity", request.quantity_units),
        ("price", request.price_rupees),
        ("multiplier", request.multiplier),
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

fn missing_row(request: &MarginRequest<'_>) -> MarginError {
    if request.kind == InstrumentKind::Future {
        return not_built(
            "a future on an underlying the dated margin table does not list (it lists NSE \
             index and stock futures, BSE BANKEX and MCX commodities; not SENSEX or \
             SENSEX50)",
            "trade the futures of an underlying the table lists",
        );
    }
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

    fn request(kind: InstrumentKind, product: Product, leverage: f64) -> MarginRequest<'static> {
        MarginRequest {
            exchange: Exchange::Nse,
            kind,
            product,
            side: Side::Buy,
            quantity_units: 100.0,
            price_rupees: 500.0,
            multiplier: 1.0,
            leverage,
            day: days_from_civil(2025, 7, 1),
            underlying: None,
        }
    }

    fn quote(request: &MarginRequest<'_>) -> (f64, String) {
        match MarginModel.initial_margin(request) {
            Ok(quote) => (quote.rupees, quote.row.meta.id.clone()),
            Err(error) => panic!("{error}"),
        }
    }

    fn future(underlying: &'static str, lot: f64, price: f64) -> MarginRequest<'static> {
        MarginRequest {
            exchange: Exchange::Nfo,
            product: Product::Nrml,
            quantity_units: lot,
            price_rupees: price,
            underlying: Some(underlying),
            ..request(InstrumentKind::Future, Product::Nrml, 1.0)
        }
    }

    #[test]
    fn delivery_blocks_the_full_value() {
        // 100 x 500 = 50,000.
        let cnc = request(InstrumentKind::Equity, Product::Cnc, 1.0);
        assert_eq!(quote(&cnc), (50_000.0, "CNC_EQ/1".to_string()));
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
        assert_eq!(quote(&plain), (50_000.0, "MIS_EQ/2".to_string()));
        assert_eq!(quote(&five), (10_000.0, "MIS_EQ/2".to_string()));
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
        assert_eq!(quote(&early), (10_000.0, "MIS_EQ/1".to_string()));
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
        assert_eq!(quote(&option), (9_000.0, "LONG_OPT/1".to_string()));
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

    /// Each index-futures row: lot x price x the row's percent, worked by hand.
    #[test]
    fn index_futures_block_their_rows_percent_of_contract_value() {
        let cases = [
            ("NIFTY", 75.0, 25_000.0, 213_187.5, "FUT_NIFTY/1"),
            ("BANKNIFTY", 30.0, 60_000.0, 204_840.0, "FUT_BANKNIFTY/1"),
            ("FINNIFTY", 65.0, 27_000.0, 199_719.0, "FUT_FINNIFTY/1"),
            ("MIDCPNIFTY", 120.0, 14_000.0, 202_776.0, "FUT_MIDCPNIFTY/1"),
            ("NIFTYNXT50", 25.0, 70_000.0, 199_675.0, "FUT_NIFTYNXT50/1"),
        ];
        for (underlying, lot, price, margin, row) in cases {
            let (rupees, id) = quote(&future(underlying, lot, price));
            assert!((rupees - margin).abs() < 1e-6, "{underlying}: {rupees}");
            assert_eq!(id, row);
        }
    }

    #[test]
    fn futures_without_a_row_are_not_free() {
        let sensex = future("SENSEX", 20.0, 80_000.0);
        let result = MarginModel.initial_margin(&sensex);
        assert!(matches!(result, Err(MarginError::NotBuilt { .. })));
    }

    #[test]
    fn futures_cannot_be_leveraged() {
        let leveraged = MarginRequest {
            leverage: 2.0,
            ..future("NIFTY", 75.0, 25_000.0)
        };
        let result = MarginModel.initial_margin(&leveraged);
        assert!(matches!(result, Err(MarginError::InvalidRequest { .. })));
    }
}
