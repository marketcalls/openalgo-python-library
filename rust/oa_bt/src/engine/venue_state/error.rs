//! Why a live session's saved state was not restored (`Engine::restore_live_state`,
//! `Engine::restore_carried_state`). Nothing of a refused state is restored.

use std::fmt;

use super::value::CARRIED_ROOT;

/// Why a saved state was not restored.
#[derive(Clone, Debug, PartialEq)]
pub enum LiveStateError {
    /// A field of the state is missing or cannot be read, or the parts of the state do not
    /// hold together (`checks`).
    Malformed {
        /// Where, as "live_state.positions[0].tranches" or "carried_state.gtts[0]": its
        /// first part names the call that returned the state.
        path: String,
        /// What is wrong with it.
        reason: String,
    },
    /// The state was written in another layout than this kernel reads.
    Version {
        /// The layout the state names.
        found: i64,
    },
    /// The state is of another session: another trading day, other instruments or another
    /// capital.
    OtherSession {
        /// What differs.
        reason: String,
    },
    /// The session cannot take this state now: it has traded or been restored already, its
    /// close has begun, or it stands before the instant the state was saved at; or the
    /// state was saved while the session closed, which no session resumes from.
    NotRestorable {
        /// Why, and what to do.
        reason: String,
    },
    /// A carried state does not fit the session it was given to: one of the same or a
    /// later trading day, of other instruments or another capital, or holding a position
    /// that does not carry overnight (`carry`).
    CarriedMismatch {
        /// What differs.
        reason: String,
    },
}

/// The call whose state `path` reads (`Malformed`): the path's first part.
fn call_of(path: &str) -> &str {
    path.split(['.', '[']).next().unwrap_or(path)
}

impl LiveStateError {
    /// True for an error in the state given (as opposed to a call at the wrong time).
    pub fn is_input_error(&self) -> bool {
        !matches!(self, LiveStateError::NotRestorable { .. })
    }
}

impl fmt::Display for LiveStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiveStateError::Malformed { path, reason } => {
                let call = call_of(path);
                let state = match call == CARRIED_ROOT {
                    true => "carried state",
                    false => "saved live state",
                };
                write!(
                    f,
                    "the {state} cannot be restored: {path} {reason}; pass the state {call}() \
                     returned, unchanged"
                )
            }
            LiveStateError::Version { found } => write!(
                f,
                "the saved live state has layout {found}, and this kernel reads layout {}: \
                 restore it with the version of openalgo that saved it",
                super::LIVE_STATE_LAYOUT
            ),
            LiveStateError::OtherSession { reason } => write!(
                f,
                "the saved live state is of another session ({reason}): restore it into a \
                 session of the same day and instruments"
            ),
            LiveStateError::NotRestorable { reason } => {
                write!(f, "the live state cannot be restored now: {reason}")
            }
            LiveStateError::CarriedMismatch { reason } => write!(
                f,
                "the carried state does not fit this session ({reason}): restore it into the \
                 session of a later trading day that trades its instruments with its capital"
            ),
        }
    }
}

impl std::error::Error for LiveStateError {}
