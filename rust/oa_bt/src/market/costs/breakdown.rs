//! The itemised result types: what a fill costs, component by component.

use super::brokerage::DpFeeSource;

/// Names of the cost components, in the order `CostBreakdown::components` returns
/// them.
pub const COMPONENT_NAMES: [&str; 6] = [
    "brokerage",
    "stt",
    "exchange_txn",
    "sebi_fee",
    "stamp_duty",
    "gst",
];

/// The itemised cost of one fill (or a sum of fills). Every field is an amount in
/// rupees, never a rate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CostBreakdown {
    /// Brokerage, from the preset.
    pub brokerage: f64,
    /// Securities transaction tax, or commodities transaction tax on MCX.
    pub stt: f64,
    /// Exchange transaction charge, including the IPFT levy where there is one.
    pub exchange_txn: f64,
    /// SEBI turnover fee.
    pub sebi_fee: f64,
    /// Stamp duty.
    pub stamp_duty: f64,
    /// GST on brokerage plus exchange transaction charge plus SEBI fee.
    pub gst: f64,
    /// Calculator rounding only: an amount a line shows that the calculator leaves out
    /// of its GST base and its total (the IPFT levy inside the exchange line of NSE
    /// currency futures, see `rules::calculator_ipft`). Zero in contract-note rounding
    /// and in every other segment. Not a seventh component: `components` leaves it out.
    pub shown_not_charged: f64,
}

impl CostBreakdown {
    /// No cost at all.
    pub const ZERO: CostBreakdown = CostBreakdown {
        brokerage: 0.0,
        stt: 0.0,
        exchange_txn: 0.0,
        sebi_fee: 0.0,
        stamp_duty: 0.0,
        gst: 0.0,
        shown_not_charged: 0.0,
    };

    /// The plain float sum of the six components, less what is shown but not charged,
    /// in rupees. For the total a rounding mode reports, use `CostModel::reported_total`.
    #[inline]
    pub fn total(&self) -> f64 {
        self.brokerage + self.stt + self.exchange_txn + self.sebi_fee + self.stamp_duty + self.gst
            - self.shown_not_charged
    }

    /// The components in `COMPONENT_NAMES` order.
    #[inline]
    pub fn components(&self) -> [f64; 6] {
        [
            self.brokerage,
            self.stt,
            self.exchange_txn,
            self.sebi_fee,
            self.stamp_duty,
            self.gst,
        ]
    }

    /// A breakdown from components in `COMPONENT_NAMES` order, with nothing shown but
    /// not charged.
    #[inline]
    pub fn from_components(components: [f64; 6]) -> CostBreakdown {
        let [brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst] = components;
        CostBreakdown {
            brokerage,
            stt,
            exchange_txn,
            sebi_fee,
            stamp_duty,
            gst,
            shown_not_charged: 0.0,
        }
    }

    /// Add another breakdown to this one, component by component.
    #[inline]
    pub fn add(&mut self, other: &CostBreakdown) {
        self.brokerage += other.brokerage;
        self.stt += other.stt;
        self.exchange_txn += other.exchange_txn;
        self.sebi_fee += other.sebi_fee;
        self.stamp_duty += other.stamp_duty;
        self.gst += other.gst;
        self.shown_not_charged += other.shown_not_charged;
    }
}

/// The depository participant charge on a delivery sell, in rupees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DpCharge {
    /// Depository plus broker fee, per scrip per sell day.
    pub fee: f64,
    /// GST on the fee, rounded to the paisa.
    pub gst: f64,
    /// Whose fee this is: the preset's own, Zerodha's standing in, or the caller's.
    pub fee_source: DpFeeSource,
}

impl DpCharge {
    /// Fee plus GST: what is debited.
    #[inline]
    pub fn total(&self) -> f64 {
        self.fee + self.gst
    }
}
