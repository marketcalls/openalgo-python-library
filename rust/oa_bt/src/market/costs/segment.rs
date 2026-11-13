//! Cost segments: one per row of the rate card.
//!
//! A segment is what decides which rates apply to a fill: the exchange, whether the
//! instrument is an equity, a future or an option, whether an equity trade is
//! delivery or intraday, the BSE scrip-group tier and the MCX commodity class. The
//! kernel resolves an instrument to a `Segment` once (see `resolve`) and passes the
//! small `Copy` value on every fill.

use crate::model::Exchange;

use super::error::{CostError, NameKind};

/// Number of segments, and the length of the rate table in a schedule.
pub const SEGMENT_COUNT: usize = 22;

/// One row of the rate card. The discriminant is the stable id used by the batch
/// API and the Python binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Segment {
    /// NSE equity, delivery (CNC).
    NseEqDelivery = 0,
    /// NSE equity, intraday (MIS).
    NseEqIntraday = 1,
    /// BSE equity delivery, groups A, B and the other non-exclusive groups.
    BseEqDelivery = 2,
    /// BSE equity intraday, groups A, B and the other non-exclusive groups.
    BseEqIntraday = 3,
    /// BSE equity delivery, groups M, MT, TS, MS.
    BseEqDeliveryM = 4,
    /// BSE equity intraday, groups M, MT, TS, MS.
    BseEqIntradayM = 5,
    /// BSE equity delivery, groups X, XT, Z, ZP.
    BseEqDeliveryX = 6,
    /// BSE equity intraday, groups X, XT, Z, ZP.
    BseEqIntradayX = 7,
    /// BSE equity delivery, groups SS, ST.
    BseEqDeliverySs = 8,
    /// BSE equity intraday, groups SS, ST.
    BseEqIntradaySs = 9,
    /// NSE index and stock futures.
    NfoFut = 10,
    /// NSE index and stock options (charges on premium value).
    NfoOpt = 11,
    /// BSE index and stock futures.
    BfoFut = 12,
    /// BSE Sensex and Bankex options (charges on premium value).
    BfoOpt = 13,
    /// MCX futures on a non-agri commodity (CTT on sells).
    McxFut = 14,
    /// MCX futures on an agri commodity exempt from CTT (one listed in rule 3 of the
    /// CTT Rules 2013), at the standard exchange rate.
    McxFutAgri = 15,
    /// MCX CASTORSEED futures (agri, own exchange rate until 2024-09-30).
    McxFutCastorseed = 16,
    /// MCX RBDPMOLEIN futures (agri, CTT on sells, own exchange rate until 2024-09-30).
    McxFutRbdpmolein = 17,
    /// MCX PEPPER futures (agri, own exchange rate until 2024-09-30).
    McxFutPepper = 18,
    /// MCX KAPAS futures (agri, own exchange rate until 2024-09-30).
    McxFutKapas = 19,
    /// BSE options other than Sensex and Bankex: Sensex 50 and stock options.
    BfoOptOther = 20,
    /// MCX futures on an agri commodity that is not in rule 3 of the CTT Rules 2013,
    /// so not exempt from CTT (CTT on sells), at the standard exchange rate.
    McxFutAgriCtt = 21,
}

/// Which rule of a brokerage preset applies to a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BrokerageClass {
    /// Equity delivery (CNC).
    Delivery = 0,
    /// Equity intraday (MIS).
    Intraday = 1,
    /// Futures on any exchange.
    Futures = 2,
    /// Options on any exchange.
    Options = 3,
}

impl BrokerageClass {
    /// The lower-case name used in listings: delivery, intraday, futures, options.
    pub fn code(self) -> &'static str {
        match self {
            BrokerageClass::Delivery => "delivery",
            BrokerageClass::Intraday => "intraday",
            BrokerageClass::Futures => "futures",
            BrokerageClass::Options => "options",
        }
    }
}

/// The contract note a segment's trades appear on. A broker issues one note per
/// trade day for each of these, so this is the key the contract-note aggregator
/// rounds by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum NoteGroup {
    /// NSE cash market (delivery and intraday together).
    NseCash = 0,
    /// BSE cash market (delivery and intraday together, all scrip groups).
    BseCash = 1,
    /// NSE futures and options.
    NseFo = 2,
    /// BSE futures and options.
    BseFo = 3,
    /// MCX commodity derivatives.
    Mcx = 4,
}

impl NoteGroup {
    /// Every note group, in id order.
    pub const ALL: [NoteGroup; 5] = [
        NoteGroup::NseCash,
        NoteGroup::BseCash,
        NoteGroup::NseFo,
        NoteGroup::BseFo,
        NoteGroup::Mcx,
    ];

    /// A short code for listings: NSE_EQ, BSE_EQ, NFO, BFO, MCX.
    pub fn code(self) -> &'static str {
        match self {
            NoteGroup::NseCash => "NSE_EQ",
            NoteGroup::BseCash => "BSE_EQ",
            NoteGroup::NseFo => "NFO",
            NoteGroup::BseFo => "BFO",
            NoteGroup::Mcx => "MCX",
        }
    }
}

/// The fixed facts about a segment.
struct SegmentInfo {
    segment: Segment,
    code: &'static str,
    exchange: Exchange,
    class: BrokerageClass,
    description: &'static str,
}

const fn info(
    segment: Segment,
    code: &'static str,
    exchange: Exchange,
    class: BrokerageClass,
    description: &'static str,
) -> SegmentInfo {
    SegmentInfo {
        segment,
        code,
        exchange,
        class,
        description,
    }
}

/// The segment table, in id order (a unit test checks the order).
#[rustfmt::skip]
const TABLE: [SegmentInfo; SEGMENT_COUNT] = {
    use BrokerageClass::{Delivery, Futures, Intraday, Options};
    use Exchange::{Bfo, Bse, Mcx, Nfo, Nse};
    [
        info(Segment::NseEqDelivery, "NSE_EQ_DELIVERY", Nse, Delivery, "NSE equity delivery (CNC)"),
        info(Segment::NseEqIntraday, "NSE_EQ_INTRADAY", Nse, Intraday, "NSE equity intraday (MIS)"),
        info(Segment::BseEqDelivery, "BSE_EQ_DELIVERY", Bse, Delivery, "BSE equity delivery, groups A, B, E, F, FC, G, GC, W, T"),
        info(Segment::BseEqIntraday, "BSE_EQ_INTRADAY", Bse, Intraday, "BSE equity intraday, groups A, B, E, F, FC, G, GC, W, T"),
        info(Segment::BseEqDeliveryM, "BSE_EQ_DELIVERY_M", Bse, Delivery, "BSE equity delivery, groups M, MT, TS, MS"),
        info(Segment::BseEqIntradayM, "BSE_EQ_INTRADAY_M", Bse, Intraday, "BSE equity intraday, groups M, MT, TS, MS"),
        info(Segment::BseEqDeliveryX, "BSE_EQ_DELIVERY_X", Bse, Delivery, "BSE equity delivery, groups X, XT, Z, ZP"),
        info(Segment::BseEqIntradayX, "BSE_EQ_INTRADAY_X", Bse, Intraday, "BSE equity intraday, groups X, XT, Z, ZP"),
        info(Segment::BseEqDeliverySs, "BSE_EQ_DELIVERY_SS", Bse, Delivery, "BSE equity delivery, groups SS, ST"),
        info(Segment::BseEqIntradaySs, "BSE_EQ_INTRADAY_SS", Bse, Intraday, "BSE equity intraday, groups SS, ST"),
        info(Segment::NfoFut, "NFO_FUT", Nfo, Futures, "NSE index and stock futures"),
        info(Segment::NfoOpt, "NFO_OPT", Nfo, Options, "NSE index and stock options (charges on premium)"),
        info(Segment::BfoFut, "BFO_FUT", Bfo, Futures, "BSE index and stock futures"),
        info(Segment::BfoOpt, "BFO_OPT", Bfo, Options, "BSE Sensex and Bankex options (charges on premium)"),
        info(Segment::McxFut, "MCX_FUT", Mcx, Futures, "MCX futures, non-agri commodity"),
        info(Segment::McxFutAgri, "MCX_FUT_AGRI", Mcx, Futures, "MCX futures, CTT-exempt agri commodity at the standard exchange rate (no CTT)"),
        info(Segment::McxFutCastorseed, "MCX_FUT_CASTORSEED", Mcx, Futures, "MCX CASTORSEED futures (agri, own exchange rate until 2024-09-30)"),
        info(Segment::McxFutRbdpmolein, "MCX_FUT_RBDPMOLEIN", Mcx, Futures, "MCX RBDPMOLEIN futures (agri, CTT on sells, own exchange rate until 2024-09-30)"),
        info(Segment::McxFutPepper, "MCX_FUT_PEPPER", Mcx, Futures, "MCX PEPPER futures (agri, own exchange rate until 2024-09-30)"),
        info(Segment::McxFutKapas, "MCX_FUT_KAPAS", Mcx, Futures, "MCX KAPAS futures (agri, own exchange rate until 2024-09-30)"),
        info(Segment::BfoOptOther, "BFO_OPT_OTHER", Bfo, Options, "BSE Sensex 50 and stock options (charges on premium)"),
        info(Segment::McxFutAgriCtt, "MCX_FUT_AGRI_CTT", Mcx, Futures, "MCX futures, agri commodity outside the CTT exemption, at the standard exchange rate (CTT on sells)"),
    ]
};

/// Codes of real segments that are deliberately not costed yet.
const NOT_COSTED_YET: [&str; 5] = ["CDS_FUT", "CDS_OPT", "BCD_FUT", "BCD_OPT", "MCX_OPT"];

impl Segment {
    /// Every segment, in id order.
    pub const ALL: [Segment; SEGMENT_COUNT] = {
        let mut all = [Segment::NseEqDelivery; SEGMENT_COUNT];
        let mut i = 0;
        while i < SEGMENT_COUNT {
            all[i] = TABLE[i].segment;
            i += 1;
        }
        all
    };

    /// Position in a schedule's rate table (the numeric id).
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    fn info(self) -> &'static SegmentInfo {
        &TABLE[self.index()]
    }

    /// The stable text code, for example "NSE_EQ_DELIVERY" or "NFO_OPT".
    pub fn code(self) -> &'static str {
        self.info().code
    }

    /// A one-line description for listings.
    pub fn description(self) -> &'static str {
        self.info().description
    }

    /// Parse a segment code (exact, upper case).
    ///
    /// Errors: `UnsupportedSegment` for a currency or MCX options code, which names a
    /// real segment this engine does not cost yet, and `UnknownName` for anything
    /// else that does not match.
    pub fn from_code(code: &str) -> Result<Segment, CostError> {
        if let Some(found) = TABLE.iter().find(|row| row.code == code) {
            return Ok(found.segment);
        }
        if NOT_COSTED_YET.contains(&code) {
            return Err(CostError::UnsupportedSegment(format!("segment {code}")));
        }
        Err(CostError::unknown(NameKind::Segment, code))
    }

    /// The segment with a numeric id, or None when the id is out of range.
    #[inline]
    pub fn from_u8(id: u8) -> Option<Segment> {
        Segment::ALL.get(id as usize).copied()
    }

    /// The OpenAlgo exchange the segment trades on.
    pub fn exchange(self) -> Exchange {
        self.info().exchange
    }

    /// Which rule of a brokerage preset applies.
    #[inline]
    pub fn brokerage_class(self) -> BrokerageClass {
        self.info().class
    }

    /// True for the options segments, whose charges are on premium value and which
    /// pay STT on exercise.
    #[inline]
    pub fn is_option(self) -> bool {
        self.brokerage_class() == BrokerageClass::Options
    }

    /// The contract note the segment's trades appear on.
    pub fn note_group(self) -> NoteGroup {
        match self.exchange() {
            Exchange::Nse => NoteGroup::NseCash,
            Exchange::Bse => NoteGroup::BseCash,
            Exchange::Nfo => NoteGroup::NseFo,
            Exchange::Bfo => NoteGroup::BseFo,
            // Every remaining segment is an MCX future.
            _ => NoteGroup::Mcx,
        }
    }
}
