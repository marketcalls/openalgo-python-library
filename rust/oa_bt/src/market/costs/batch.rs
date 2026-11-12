//! Costing many fills at once over slices, for the NumPy binding.

use crate::model::{Day, Side};

use super::book::RateStatus;
use super::breakdown::CostBreakdown;
use super::cost_model::{check_amount, CostModel};
use super::error::CostError;
use super::segment::Segment;

/// Mutable output columns for `CostModel::fill_cost_batch`: one value per fill in
/// each column. The amount columns are rupees.
pub struct CostColumns<'a> {
    /// Brokerage, from the preset.
    pub brokerage: &'a mut [f64],
    /// STT, or CTT on MCX.
    pub stt: &'a mut [f64],
    /// Exchange transaction charge, including the IPFT levy.
    pub exchange_txn: &'a mut [f64],
    /// SEBI turnover fee.
    pub sebi_fee: &'a mut [f64],
    /// Stamp duty.
    pub stamp_duty: &'a mut [f64],
    /// GST on brokerage, exchange charge and SEBI fee.
    pub gst: &'a mut [f64],
    /// The row's `CostModel::reported_total`.
    pub total: &'a mut [f64],
    /// Position of the schedule version that priced the row
    /// (`RateStatus::schedule`).
    pub schedule: &'a mut [u16],
    /// How far the row's rates can be trusted (`Trust as u8`).
    pub trust: &'a mut [u8],
}

impl CostColumns<'_> {
    fn lengths(&self) -> [(&'static str, usize); 9] {
        [
            ("out.brokerage", self.brokerage.len()),
            ("out.stt", self.stt.len()),
            ("out.exchange_txn", self.exchange_txn.len()),
            ("out.sebi_fee", self.sebi_fee.len()),
            ("out.stamp_duty", self.stamp_duty.len()),
            ("out.gst", self.gst.len()),
            ("out.total", self.total.len()),
            ("out.schedule", self.schedule.len()),
            ("out.trust", self.trust.len()),
        ]
    }

    /// Store one fill's result. The caller has checked that `row` is in range.
    fn write(&mut self, row: usize, cost: &CostBreakdown, total: f64, status: RateStatus) {
        self.brokerage[row] = cost.brokerage;
        self.stt[row] = cost.stt;
        self.exchange_txn[row] = cost.exchange_txn;
        self.sebi_fee[row] = cost.sebi_fee;
        self.stamp_duty[row] = cost.stamp_duty;
        self.gst[row] = cost.gst;
        self.total[row] = total;
        // A book never holds anywhere near 65,536 versions; saturate rather than wrap.
        self.schedule[row] = u16::try_from(status.schedule).unwrap_or(u16::MAX);
        self.trust[row] = status.trust as u8;
    }
}

impl CostModel {
    /// Cost many fills at once.
    ///
    /// `segments` holds segment ids (`Segment as u8`), `sides` holds 0 for a buy
    /// and 1 for a sell (`Side as u8`), `prices` are rupees per price unit,
    /// `quantities` are numbers of price units and `days` are IST trade dates. All
    /// inputs and all output columns must have the same length. Each row is one
    /// executed order, costed exactly as `fill_cost` would cost it.
    ///
    /// Errors: the index of the first bad row with its error (`LengthMismatch` is
    /// reported at index 0). Output rows before that index are filled.
    pub fn fill_cost_batch(
        &self,
        segments: &[u8],
        sides: &[u8],
        prices: &[f64],
        quantities: &[f64],
        days: &[Day],
        out: &mut CostColumns<'_>,
    ) -> Result<(), (usize, CostError)> {
        let rows = segments.len();
        let inputs = [
            ("sides", sides.len()),
            ("prices", prices.len()),
            ("quantities", quantities.len()),
            ("days", days.len()),
        ];
        for (name, got) in inputs.into_iter().chain(out.lengths()) {
            if got != rows {
                return Err((
                    0,
                    CostError::LengthMismatch {
                        name,
                        expected: rows,
                        got,
                    },
                ));
            }
        }
        for row in 0..rows {
            let (cost, status) = self
                .cost_row(
                    segments[row],
                    sides[row],
                    prices[row],
                    quantities[row],
                    days[row],
                )
                .map_err(|error| (row, error))?;
            out.write(row, &cost, self.reported_total(&cost), status);
        }
        Ok(())
    }

    /// Validate and cost one row of a batch.
    fn cost_row(
        &self,
        segment_id: u8,
        side_id: u8,
        price: f64,
        quantity: f64,
        day: Day,
    ) -> Result<(CostBreakdown, RateStatus), CostError> {
        let segment =
            Segment::from_u8(segment_id).ok_or(CostError::UnknownSegmentId(segment_id))?;
        let side = Side::from_u8(side_id).ok_or(CostError::UnknownSide(side_id))?;
        check_amount("price", price)?;
        check_amount("quantity", quantity)?;
        check_amount("traded value (price x quantity)", price * quantity)?;
        let priced = self.priced(segment, day)?;
        Ok((
            self.cost_under(&priced, segment, side, price, quantity),
            priced.status,
        ))
    }
}
