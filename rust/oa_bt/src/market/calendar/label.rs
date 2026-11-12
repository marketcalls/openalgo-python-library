//! Detection of bars labelled by their close time but read as labelled by their open.
//!
//! A source that stamps each bar at its close, read as if it stamped the open, moves
//! every bar one bar later. The series then has the tell-tale shape:
//!
//! - no day has a bar at the first bar of its session (09:15 on NSE): that bar is
//!   stamped one interval later;
//! - most days have a bar that opens exactly at the session close: the last bar of
//!   the session, stamped at its close (the close of a bar cut at the session close is
//!   the session close itself, so this holds for every interval).
//!
//! The rule: the series is close-labelled when the offset check is aligned, more than
//! half of the session days with data have a bar at the session close, and no session
//! day has a bar at its first session bar.
//!
//! The offset check comes first because a series that is 5 hours 30 minutes late also
//! misses the session open and can have a bar at the close. Whenever a 5 hour 30
//! minute shift places the bars better (a shifted or an inconclusive outcome), the
//! series is never called close-labelled; and a close-labelled series is aligned,
//! because moving it 5 hours 30 minutes either way puts most bars outside the session.
//! The two findings therefore never coincide, and each comes with its own advice.
//!
//! Served as they are, close-labelled bars would each close one interval after they
//! really did, which is look-ahead, so `validate_intraday` rejects them.

use super::check::IntradayCheck;
use super::offset::OffsetCheck;

/// Whether the counts of a checked series show close-labelled bars (see the module
/// documentation for the rule).
pub(super) fn looks_close_labelled(check: &IntradayCheck) -> bool {
    check.offset == OffsetCheck::Aligned
        && check.days_with_bar_at_close.saturating_mul(2) > check.session_days
        && check.days_starting_at_open == 0
}
