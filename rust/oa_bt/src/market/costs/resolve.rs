//! Resolving an instrument to its cost segment.
//!
//! The inputs are the shared model types (exchange, instrument kind, product) plus
//! the two facts only the cost model cares about: the BSE scrip group of an equity,
//! and the underlying of an MCX future or a BFO option. Anything that cannot be
//! resolved is an error; nothing defaults to a "nearest" segment.

use crate::model::{Exchange, InstrumentKind, Product};

use super::error::{CostError, NameKind};
use super::segment::Segment;

/// The BSE transaction-charge tier of a scrip group.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BseTier {
    /// A, B and the other non-exclusive groups (E, F, FC, G, GC, W, T).
    Standard,
    /// M, MT, TS, MS.
    M,
    /// X (formerly XC and XD), XT, Z, ZP.
    X,
    /// SS, ST.
    Ss,
}

/// BSE scrip groups and their tiers. Source: the BSE group table of the reference
/// note (from the Zerodha charges page).
const BSE_GROUPS: [(&str, BseTier); 21] = [
    ("A", BseTier::Standard),
    ("B", BseTier::Standard),
    ("E", BseTier::Standard),
    ("F", BseTier::Standard),
    ("FC", BseTier::Standard),
    ("G", BseTier::Standard),
    ("GC", BseTier::Standard),
    ("W", BseTier::Standard),
    ("T", BseTier::Standard),
    ("M", BseTier::M),
    ("MT", BseTier::M),
    ("TS", BseTier::M),
    ("MS", BseTier::M),
    ("X", BseTier::X),
    ("XC", BseTier::X),
    ("XD", BseTier::X),
    ("XT", BseTier::X),
    ("Z", BseTier::X),
    ("ZP", BseTier::X),
    ("SS", BseTier::Ss),
    ("ST", BseTier::Ss),
];

/// The scrip groups of the table, for error messages: "A, B, ..., ST".
pub(super) fn bse_group_names() -> String {
    let names: Vec<&str> = BSE_GROUPS.iter().map(|(code, _)| *code).collect();
    names.join(", ")
}

/// The tier of a BSE scrip group code (case and surrounding spaces do not matter).
/// An unknown group is an error, never the standard tier.
fn bse_tier(group: &str) -> Result<BseTier, CostError> {
    let group = group.trim();
    BSE_GROUPS
        .iter()
        .find(|(code, _)| code.eq_ignore_ascii_case(group))
        .map(|(_, tier)| *tier)
        .ok_or_else(|| CostError::unknown(NameKind::BseGroup, group))
}

// CTT is charged on the sale of a commodity derivative other than one on an
// agricultural commodity (Finance Act 2013 sections 116(7) and 117), and the
// agricultural commodities are exactly the 61 of rule 3 of the CTT Rules 2013 as
// substituted by notification 13/2015 of 2015-02-10 (reproduced in MCX master
// circular MCX/MCXCCL/443/2020, annexure 6). An MCX "agri" contract on anything else
// pays CTT like a metal. The agri lists below come from the reference note; which
// side of rule 3 each commodity falls on is noted on its list.

/// MCX agri commodities at the standard exchange rate whose futures pay no CTT and the
/// lower SEBI fee: cardamom and cotton are items (iii) and (viii) of rule 3.
const MCX_AGRI_CTT_EXEMPT: [&str; 2] = ["CARDAMOM", "COTTONCNDY"];

/// MCX agri commodities at the standard exchange rate whose futures pay the lower SEBI
/// fee but also CTT: crude palm oil, mentha oil, rubber, cotton seed wash oil and
/// sunflower oil are not in rule 3 (it lists cotton seed and sunflower seed, not
/// their oils).
const MCX_AGRI_CTT_PAYING: [&str; 5] = ["COTTONOIL", "CPO", "MENTHAOIL", "RUBBER", "SUNOIL"];

/// MCX agri commodities with a contract-specific exchange rate. Source: the
/// contract-specific rates of the reference note. Castor seed, pepper and kapas are
/// items (iv), (xiv) and (xii) of rule 3; RBD palmolein is not, and its schedule rows
/// charge CTT.
const MCX_AGRI_SPECIFIC: [(&str, Segment); 4] = [
    ("CASTORSEED", Segment::McxFutCastorseed),
    ("RBDPMOLEIN", Segment::McxFutRbdpmolein),
    ("PEPPER", Segment::McxFutPepper),
    ("KAPAS", Segment::McxFutKapas),
];

/// MCX non-agri commodities (bullion, base metals, energy, indices). Source: the
/// commodity list of the reference note, plus STEELREBAR, which the Zerodha
/// calculator script lists as non-agri.
const MCX_NON_AGRI: [&str; 26] = [
    "ALUMINI",
    "ALUMINIUM",
    "COPPER",
    "CRUDEOIL",
    "CRUDEOILM",
    "ELECDMBL",
    "GOLD",
    "GOLDGUINEA",
    "GOLDM",
    "GOLDPETAL",
    "GOLDTEN",
    "LEAD",
    "LEADMINI",
    "MCXBULLDEX",
    "MCXENRGDEX",
    "MCXMETLDEX",
    "NATGASMINI",
    "NATURALGAS",
    "NICKEL",
    "SILVER",
    "SILVER100",
    "SILVERM",
    "SILVERMIC",
    "STEELREBAR",
    "ZINC",
    "ZINCMINI",
];

/// Underlyings whose BSE options pay the higher exchange charge. Every other BSE
/// option (Sensex 50 and stock options) pays the lower one. Source: BSE notice of
/// 2024-09-27 as reported by Angel One on 2024-09-30.
const BFO_HIGHER_CHARGE_UNDERLYINGS: [&str; 2] = ["SENSEX", "BANKEX"];

const EQUITY_PRODUCT_HINT: &str = "equity trades use CNC (delivery) or MIS (intraday)";
const DERIVATIVE_PRODUCT_HINT: &str = "futures and options use NRML or MIS";
const COMMODITY_HINT: &str =
    "pass the commodity as the underlying (for example CRUDEOIL) so the agri or non-agri rates \
     can be chosen";
const BFO_UNDERLYING_HINT: &str =
    "pass the underlying (for example SENSEX, BANKEX or the stock symbol): Sensex and Bankex \
     options pay a different exchange charge from other BSE options";

/// What is being resolved, kept together so every error can name it.
#[derive(Clone, Copy)]
struct Request<'a> {
    exchange: Exchange,
    kind: InstrumentKind,
    product: Product,
    bse_group: Option<&'a str>,
    underlying: Option<&'a str>,
}

impl Request<'_> {
    fn invalid(&self, hint: &'static str) -> CostError {
        CostError::InvalidCombination {
            exchange: self.exchange,
            kind: self.kind,
            product: self.product,
            hint,
        }
    }

    /// True for an equity delivery trade, false for intraday.
    fn equity_delivery(&self) -> Result<bool, CostError> {
        match self.product {
            Product::Cnc => Ok(true),
            Product::Mis => Ok(false),
            Product::Nrml => Err(self.invalid(EQUITY_PRODUCT_HINT)),
        }
    }

    fn nse(&self) -> Result<Segment, CostError> {
        if self.kind != InstrumentKind::Equity {
            return Err(self.invalid("NSE futures and options trade on NFO"));
        }
        Ok(if self.equity_delivery()? {
            Segment::NseEqDelivery
        } else {
            Segment::NseEqIntraday
        })
    }

    fn bse(&self) -> Result<Segment, CostError> {
        if self.kind != InstrumentKind::Equity {
            return Err(self.invalid("BSE futures and options trade on BFO"));
        }
        let delivery = self.equity_delivery()?;
        let tier = bse_tier(self.bse_group.ok_or(CostError::MissingBseGroup)?)?;
        Ok(match (tier, delivery) {
            (BseTier::Standard, true) => Segment::BseEqDelivery,
            (BseTier::Standard, false) => Segment::BseEqIntraday,
            (BseTier::M, true) => Segment::BseEqDeliveryM,
            (BseTier::M, false) => Segment::BseEqIntradayM,
            (BseTier::X, true) => Segment::BseEqDeliveryX,
            (BseTier::X, false) => Segment::BseEqIntradayX,
            (BseTier::Ss, true) => Segment::BseEqDeliverySs,
            (BseTier::Ss, false) => Segment::BseEqIntradaySs,
        })
    }

    /// Reject a product or instrument kind that an equity derivatives exchange does
    /// not have.
    fn check_derivative(&self) -> Result<(), CostError> {
        if self.product == Product::Cnc {
            return Err(self.invalid(DERIVATIVE_PRODUCT_HINT));
        }
        match self.kind {
            InstrumentKind::Future | InstrumentKind::Option => Ok(()),
            _ => Err(self.invalid("this exchange lists futures and options only")),
        }
    }

    fn nfo(&self) -> Result<Segment, CostError> {
        self.check_derivative()?;
        Ok(if self.kind == InstrumentKind::Future {
            Segment::NfoFut
        } else {
            Segment::NfoOpt
        })
    }

    fn bfo(&self) -> Result<Segment, CostError> {
        self.check_derivative()?;
        if self.kind == InstrumentKind::Future {
            return Ok(Segment::BfoFut);
        }
        let underlying = self
            .underlying
            .ok_or_else(|| self.invalid(BFO_UNDERLYING_HINT))?
            .trim();
        let higher = BFO_HIGHER_CHARGE_UNDERLYINGS
            .iter()
            .any(|u| u.eq_ignore_ascii_case(underlying));
        Ok(if higher {
            Segment::BfoOpt
        } else {
            Segment::BfoOptOther
        })
    }

    fn mcx(&self) -> Result<Segment, CostError> {
        match self.kind {
            InstrumentKind::Option => Err(CostError::UnsupportedSegment("MCX options".to_string())),
            InstrumentKind::Future if self.product == Product::Cnc => {
                Err(self.invalid(DERIVATIVE_PRODUCT_HINT))
            }
            InstrumentKind::Future => Segment::mcx_future(
                self.underlying
                    .ok_or_else(|| self.invalid(COMMODITY_HINT))?,
            ),
            _ => Err(self.invalid("MCX lists commodity futures and options only")),
        }
    }
}

impl Segment {
    /// The MCX futures segment of a commodity.
    ///
    /// `commodity` is the commodity name as MCX lists it, for example "GOLDM" (the
    /// underlying of the contract, not the contract symbol); case and surrounding
    /// spaces do not matter.
    ///
    /// Errors: `UnknownName` for a commodity in neither the agri nor the non-agri
    /// lists. Guessing would either charge CTT to an exempt agri contract or skip it
    /// for one that owes it.
    pub fn mcx_future(commodity: &str) -> Result<Segment, CostError> {
        let name = commodity.trim();
        let listed = |list: &[&str]| list.iter().any(|known| known.eq_ignore_ascii_case(name));
        if let Some((_, segment)) = MCX_AGRI_SPECIFIC
            .iter()
            .find(|(known, _)| known.eq_ignore_ascii_case(name))
        {
            return Ok(*segment);
        }
        if listed(&MCX_AGRI_CTT_EXEMPT) {
            return Ok(Segment::McxFutAgri);
        }
        if listed(&MCX_AGRI_CTT_PAYING) {
            return Ok(Segment::McxFutAgriCtt);
        }
        if listed(&MCX_NON_AGRI) {
            return Ok(Segment::McxFut);
        }
        Err(CostError::unknown(NameKind::Commodity, name))
    }

    /// Resolve an instrument to its segment.
    ///
    /// - `bse_group` is the BSE scrip group from the instrument master. It is
    ///   required for BSE equity, where the exchange charge depends on it, and
    ///   ignored elsewhere.
    /// - `underlying` is required for an MCX future (the commodity, see
    ///   `mcx_future`) and for a BFO option (SENSEX, BANKEX, SENSEX50 or the stock
    ///   symbol), and ignored elsewhere.
    ///
    /// Errors: `MissingBseGroup` for BSE equity without a group; `UnknownName` for
    /// an unknown group or commodity; `InvalidCombination` when the exchange, kind
    /// and product do not go together or the underlying is missing;
    /// `UnsupportedSegment` for currency derivatives (CDS, BCD) and MCX options,
    /// which are out of scope; `NotTradable` for an index exchange.
    pub fn resolve(
        exchange: Exchange,
        kind: InstrumentKind,
        product: Product,
        bse_group: Option<&str>,
        underlying: Option<&str>,
    ) -> Result<Segment, CostError> {
        let request = Request {
            exchange,
            kind,
            product,
            bse_group,
            underlying,
        };
        match exchange {
            Exchange::Nse => request.nse(),
            Exchange::Bse => request.bse(),
            Exchange::Nfo => request.nfo(),
            Exchange::Bfo => request.bfo(),
            Exchange::Mcx => request.mcx(),
            Exchange::Cds | Exchange::Bcd => Err(CostError::UnsupportedSegment(format!(
                "exchange {} (currency derivatives)",
                exchange.code()
            ))),
            Exchange::NseIndex | Exchange::BseIndex => Err(CostError::NotTradable(exchange)),
        }
    }
}
