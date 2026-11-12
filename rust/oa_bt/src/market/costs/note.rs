//! Contract-note rounding: accrue unrounded per fill, round per day and note.
//!
//! # The rule assumed here (to be verified against real contract notes)
//!
//! `CONTRACT_NOTE_RULE` states it and travels with every result. In full: a broker
//! issues one contract note per client, per trade day, per exchange segment (NSE
//! cash, BSE cash, NSE F&O, BSE F&O and MCX: `NoteGroup`). Delivery and intraday
//! equity trades of one exchange share a note, and so do futures and options of one
//! exchange. On a note each charge is one line, the sum over that day's trades in
//! that segment; STT (CTT on MCX) and stamp duty are rounded to the nearest rupee,
//! the other lines to the nearest paisa, and the amount charged is the sum of the
//! rounded lines.
//!
//! Basis: Zerodha's STT support article (STT is rounded to the nearest rupee), the
//! Zerodha calculator (stamp duty to the rupee, everything else to two decimals)
//! and the reference note. Not confirmed: whether stamp duty is rounded to the rupee
//! or shown in paise on a real note, whether tax is rounded per note or per scrip
//! within a note, and whether GST is recomputed from the rounded lines (here the
//! accrued GST is rounded, which can differ by a paisa). The DP charge is not on a
//! contract note and is not handled here.
//!
//! # Exactness
//!
//! Accruals are summed as exact integers (`Picos`), not floats, so a line whose
//! exact value is Rs 1.50 is Rs 1.50 and rounds to Rs 2, whatever the float that
//! carried each fill looked like.

use std::collections::BTreeMap;

use crate::model::{is_supported_day, Day};

use super::breakdown::{CostBreakdown, COMPONENT_NAMES};
use super::cost_model::check_amount;
use super::error::CostError;
use super::rounding::{from_picos, round_half_up, to_picos, Grid, Picos, Unit};
use super::segment::{NoteGroup, Segment};

/// The rounding rule the aggregator applies, in one sentence, for result metadata.
/// It is an assumption: see the module documentation for what is not confirmed.
pub const CONTRACT_NOTE_RULE: &str =
    "assumed, not verified against a contract note: per trade day and exchange segment each \
     charge is summed over the day's trades, STT or CTT and stamp duty are rounded to the \
     nearest rupee (50 paise up) and brokerage, exchange charge, SEBI fee and GST to the \
     nearest paisa";

/// The grid and rounding unit of each component, in `COMPONENT_NAMES` order: tax
/// lines (STT or CTT, stamp duty) go to the rupee, the rest to the paisa.
const LINES: [(Grid, Unit); 6] = [
    (Grid::Charge, Unit::Paisa),
    (Grid::Charge, Unit::Rupee),
    (Grid::Charge, Unit::Paisa),
    (Grid::Charge, Unit::Paisa),
    (Grid::Charge, Unit::Rupee),
    (Grid::Gst, Unit::Paisa),
];

/// One contract note: an IST trade day and note group with its charges in rupees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteLine {
    /// The IST trade date of the note.
    pub day: Day,
    /// Which exchange segment's note it is.
    pub group: NoteGroup,
    /// Number of fills accrued into the note.
    pub fills: u32,
    /// The unrounded sum of the fills' costs.
    pub accrued: CostBreakdown,
    /// What the note charges: `accrued` after contract-note rounding.
    pub charged: CostBreakdown,
    /// The sum of the charged lines, a whole number of paise.
    pub charged_total: f64,
}

impl NoteLine {
    /// Charged total minus accrued total, in rupees: the amount to post to the
    /// ledger at the end of the day so the books match the note. It can be negative.
    pub fn adjustment(&self) -> f64 {
        self.charged_total - self.accrued.total()
    }
}

/// The exact accrued charges of one note, per component, and its fill count.
#[derive(Clone, Copy, Debug, Default)]
struct Accrual {
    picos: [Picos; 6],
    fills: u32,
}

impl Accrual {
    fn line(&self, day: Day, group: NoteGroup) -> NoteLine {
        let accrued = self.picos.map(from_picos);
        let mut charged = [0.0; 6];
        let mut charged_picos: Picos = 0;
        for (index, (_, unit)) in LINES.iter().enumerate() {
            charged[index] = round_half_up(self.picos[index], *unit);
            charged_picos = charged_picos.saturating_add(to_picos(charged[index], Grid::Charge));
        }
        NoteLine {
            day,
            group,
            fills: self.fills,
            accrued: CostBreakdown::from_components(accrued),
            charged: CostBreakdown::from_components(charged),
            charged_total: from_picos(charged_picos),
        }
    }
}

/// Accumulates unrounded per-fill costs and rounds them per trade day and contract
/// note.
///
/// The kernel adds every fill's `CostBreakdown` as it happens and reads the notes
/// in the session-close step. Adding a fill allocates only the first time a (day,
/// group) pair is seen.
#[derive(Clone, Debug, Default)]
pub struct ContractNoteAggregator {
    notes: BTreeMap<(Day, NoteGroup), Accrual>,
}

impl ContractNoteAggregator {
    /// An aggregator with no notes.
    pub fn new() -> ContractNoteAggregator {
        ContractNoteAggregator::default()
    }

    /// Accrue the unrounded cost of one fill (from `CostModel::fill_cost` in
    /// contract-note mode) into the note of its IST trade day and segment. The
    /// amounts are finite and not negative, as `fill_cost` returns them.
    pub fn add(&mut self, day: Day, segment: Segment, cost: &CostBreakdown) {
        let accrual = self.notes.entry((day, segment.note_group())).or_default();
        for (index, amount) in cost.components().into_iter().enumerate() {
            let exact = to_picos(amount, LINES[index].0);
            accrual.picos[index] = accrual.picos[index].saturating_add(exact);
        }
        accrual.fills += 1;
    }

    /// Correct the accrual of a note by `difference` (amounts may be negative) without
    /// counting a fill: a fill already accrued that is charged again at other rates of the
    /// same note (a delivery fill that a same-day sale turned into intraday).
    pub fn amend(&mut self, day: Day, segment: Segment, difference: &CostBreakdown) {
        let accrual = self.notes.entry((day, segment.note_group())).or_default();
        for (index, amount) in difference.components().into_iter().enumerate() {
            let exact = to_picos(amount, LINES[index].0);
            accrual.picos[index] = accrual.picos[index].saturating_add(exact);
        }
    }

    /// Accrue STT in rupees that is not tied to a fill, such as STT on an exercised
    /// option (from `CostModel::exercise_stt`), into the note of `day`.
    pub fn add_stt(&mut self, day: Day, segment: Segment, stt: f64) {
        self.add(
            day,
            segment,
            &CostBreakdown {
                stt,
                ..CostBreakdown::ZERO
            },
        );
    }

    /// The note of one day and group, if any fill was accrued into it.
    pub fn note(&self, day: Day, group: NoteGroup) -> Option<NoteLine> {
        self.notes
            .get(&(day, group))
            .map(|accrual| accrual.line(day, group))
    }

    /// Every note, ordered by day and then group.
    pub fn notes(&self) -> impl Iterator<Item = NoteLine> + '_ {
        self.notes
            .iter()
            .map(|(&(day, group), accrual)| accrual.line(day, group))
    }

    /// The sum of what every note charges, in rupees: a whole number of paise.
    pub fn total_charged(&self) -> f64 {
        let totals = self
            .notes()
            .map(|note| to_picos(note.charged_total, Grid::Charge));
        from_picos(totals.fold(0, Picos::saturating_add))
    }

    /// Number of notes.
    pub fn len(&self) -> usize {
        self.notes.len()
    }

    /// True when nothing has been accrued.
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }
}

/// Round a batch of per-fill costs into contract notes.
///
/// `days` (IST trade dates) and `segments` (segment ids) identify each fill and the
/// six `components` slices hold its unrounded costs in rupees, in `COMPONENT_NAMES`
/// order. All slices must have the same length. Returns one `NoteLine` per trade
/// day and note group, ordered by day and group.
///
/// Errors: the index of the first bad row with its error: `UnsupportedDay`,
/// `UnknownSegmentId`, `InvalidInput` for a negative, infinite or NaN amount, or
/// `LengthMismatch` (reported at index 0).
pub fn contract_notes(
    days: &[Day],
    segments: &[u8],
    components: [&[f64]; 6],
) -> Result<Vec<NoteLine>, (usize, CostError)> {
    let rows = days.len();
    let lengths = [("segments", segments.len())].into_iter().chain(
        COMPONENT_NAMES
            .into_iter()
            .zip(components.map(<[f64]>::len)),
    );
    for (name, got) in lengths {
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
    let mut aggregator = ContractNoteAggregator::new();
    for row in 0..rows {
        let amounts = components.map(|column| column[row]);
        let segment = checked_row(days[row], segments[row], amounts).map_err(|e| (row, e))?;
        aggregator.add(days[row], segment, &CostBreakdown::from_components(amounts));
    }
    Ok(aggregator.notes().collect())
}

/// Validate one row of a batch and return its segment.
fn checked_row(day: Day, segment_id: u8, amounts: [f64; 6]) -> Result<Segment, CostError> {
    if !is_supported_day(day) {
        return Err(CostError::UnsupportedDay { day });
    }
    for (name, amount) in COMPONENT_NAMES.into_iter().zip(amounts) {
        check_amount(name, amount)?;
    }
    Segment::from_u8(segment_id).ok_or(CostError::UnknownSegmentId(segment_id))
}
