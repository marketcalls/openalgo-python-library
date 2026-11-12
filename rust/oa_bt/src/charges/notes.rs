//! Contract-note settlement: what a day's note charges against what the ledger posted.
//!
//! Fills post their unrounded accruals rounded to the paisa, one by one. The note of a
//! trade day and note group charges its lines rounded once (`ContractNoteAggregator`,
//! rule `CONTRACT_NOTE_RULE`). At the session close the difference, a whole number of
//! paise that can be negative, is posted as the note's adjustment, so the ledger's
//! charges for the day equal the note's.

use crate::market::costs::NoteLine;
use crate::portfolio::round_to_paise;

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
