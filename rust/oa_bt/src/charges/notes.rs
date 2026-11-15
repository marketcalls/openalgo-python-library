//! Contract-note settlement: what a day's note charges against what the ledger posted.
//!
//! Fills post their unrounded accruals rounded to the paisa, one by one. The note of a
//! trade day and note group charges its lines rounded once (`ContractNoteAggregator`,
//! rule `CONTRACT_NOTE_RULE`). At the session close the difference, a whole number of
//! paise that can be negative, is posted as the note's adjustment, so the ledger's
//! charges for the day equal the note's. The settled note is then shared over its
//! fills (`share_note`), so the fills' charges add up to it too.

use crate::market::costs::{apportion, CostBreakdown, NoteLine};
use crate::portfolio::round_to_paise;

use super::book::FillCharge;

/// One contract note of a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SettledNote {
    /// The note: day, group, fill count, accrued and charged lines, charged total.
    pub note: NoteLine,
    /// Rupees the ledger posted for the note's fills, whole paise.
    pub posted_rupees: f64,
    /// Charged total minus posted, rupees, whole paise; posted at the session close.
    pub adjustment_rupees: f64,
    /// False when the run ended before the note's session closed: no adjustment was
    /// posted for it.
    pub settled: bool,
}

/// Settle a note whose fills posted `posted_rupees`.
pub(super) fn settle(note: NoteLine, posted_rupees: f64) -> SettledNote {
    SettledNote {
        note,
        posted_rupees,
        adjustment_rupees: round_to_paise(note.charged_total - posted_rupees),
        settled: true,
    }
}

/// Give each fill of a settled note its share of what the note charges
/// (`apportion`). The note's fills are among the last of `fills`: fills are recorded
/// in time order and a note settles at the close of its day.
pub(super) fn share_note(fills: &mut [FillCharge], note: &NoteLine) {
    let of_day = fills
        .iter()
        .rposition(|record| record.day < note.day)
        .map_or(0, |last_earlier| last_earlier + 1);
    let on_note: Vec<usize> = (of_day..fills.len())
        .filter(|&index| {
            let record = &fills[index];
            record.day == note.day && record.segment.note_group() == note.group
        })
        .collect();
    let costs: Vec<CostBreakdown> = on_note.iter().map(|&index| fills[index].cost).collect();
    for (index, share) in on_note.into_iter().zip(apportion(note, &costs)) {
        fills[index].charged_rupees = share;
    }
}

impl SettledNote {
    /// A note left open at the end of the run.
    pub(super) fn unsettled(note: NoteLine, posted_rupees: f64) -> SettledNote {
        SettledNote {
            note,
            posted_rupees,
            adjustment_rupees: 0.0,
            settled: false,
        }
    }
}
