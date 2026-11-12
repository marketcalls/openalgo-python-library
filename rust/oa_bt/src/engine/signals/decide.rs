//! The decision of 14-strategy-api 4.1 at one strategy-bar close, as a pure function of
//! what is held and the four condition values of that bar.

/// What the strategy holds when it decides (step 7, after the fills of the slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holding {
    /// No position and no working entry order.
    Flat,
    /// No position, but an order that would open one is still queued (its fill bar was
    /// missing). It counts as not flat, so the entry is not placed again (SPEC-29).
    EntryPending,
    /// A long position with this many adds since it opened.
    Long { adds: u32 },
    /// A short position with this many adds since it opened.
    Short { adds: u32 },
}

/// The four conditions at one strategy bar; a condition that was not declared is false.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BarSignals {
    /// `entry_when`.
    pub entry: bool,
    /// `exit_when`.
    pub exit: bool,
    /// `short_when`.
    pub short: bool,
    /// `cover_when`.
    pub cover: bool,
}

/// What to do at the bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Place no order.
    Hold,
    /// Submit the `entry_when` order (buy): an entry when flat, an add when long.
    EnterLong,
    /// Submit the `short_when` order (sell): an entry when flat, an add when short.
    EnterShort,
    /// Exit the whole position (`exit()`, reason `signal`).
    Exit,
}

/// The `on_bar` of 14-strategy-api 4.1:
///
/// ```text
/// if flat:   entry -> buy; else short -> sell      (long entry has priority)
/// if long:   exit -> exit(); else entry and adds < pyramiding -> buy
/// if short:  cover -> exit(); else short and adds < pyramiding -> sell
/// ```
///
/// Conditions are level-triggered: the same values on the next bar decide again.
pub fn decide(holding: Holding, signals: BarSignals, pyramiding: u32) -> Decision {
    match holding {
        Holding::Flat if signals.entry => Decision::EnterLong,
        Holding::Flat if signals.short => Decision::EnterShort,
        Holding::Long { .. } if signals.exit => Decision::Exit,
        Holding::Long { adds } if signals.entry && adds < pyramiding => Decision::EnterLong,
        Holding::Short { .. } if signals.cover => Decision::Exit,
        Holding::Short { adds } if signals.short && adds < pyramiding => Decision::EnterShort,
        _ => Decision::Hold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENTRY: BarSignals = BarSignals {
        entry: true,
        exit: false,
        short: false,
        cover: false,
    };
    const EXIT: BarSignals = BarSignals {
        entry: false,
        exit: true,
        short: false,
        cover: false,
    };
    const SHORT: BarSignals = BarSignals {
        entry: false,
        exit: false,
        short: true,
        cover: false,
    };
    const COVER: BarSignals = BarSignals {
        entry: false,
        exit: false,
        short: false,
        cover: true,
    };
    const ALL: BarSignals = BarSignals {
        entry: true,
        exit: true,
        short: true,
        cover: true,
    };
    const NONE: BarSignals = BarSignals {
        entry: false,
        exit: false,
        short: false,
        cover: false,
    };

    #[test]
    fn each_branch_of_the_equivalent_on_bar() {
        let long = Holding::Long { adds: 0 };
        let short = Holding::Short { adds: 0 };
        // (holding, signals, pyramiding, decision), one row per line of 4.1.
        let rows = [
            (Holding::Flat, ENTRY, 0, Decision::EnterLong),
            (Holding::Flat, SHORT, 0, Decision::EnterShort),
            (Holding::Flat, ALL, 0, Decision::EnterLong),
            (Holding::Flat, EXIT, 0, Decision::Hold),
            (Holding::Flat, COVER, 0, Decision::Hold),
            (long, EXIT, 0, Decision::Exit),
            (long, ALL, 5, Decision::Exit),
            (long, ENTRY, 0, Decision::Hold),
            (long, ENTRY, 1, Decision::EnterLong),
            (long, SHORT, 1, Decision::Hold),
            (long, COVER, 1, Decision::Hold),
            (short, COVER, 0, Decision::Exit),
            (short, ALL, 5, Decision::Exit),
            (short, SHORT, 0, Decision::Hold),
            (short, SHORT, 1, Decision::EnterShort),
            (short, ENTRY, 1, Decision::Hold),
            (short, EXIT, 1, Decision::Hold),
        ];
        for (holding, signals, pyramiding, expected) in rows {
            assert_eq!(
                decide(holding, signals, pyramiding),
                expected,
                "{holding:?} {signals:?} pyramiding {pyramiding}"
            );
        }
    }

    #[test]
    fn adds_stop_at_the_pyramiding_limit() {
        assert_eq!(
            decide(Holding::Long { adds: 1 }, ENTRY, 2),
            Decision::EnterLong
        );
        assert_eq!(decide(Holding::Long { adds: 2 }, ENTRY, 2), Decision::Hold);
        assert_eq!(decide(Holding::Short { adds: 2 }, SHORT, 2), Decision::Hold);
    }

    #[test]
    fn a_queued_entry_is_not_placed_again() {
        assert_eq!(decide(Holding::EntryPending, ALL, 3), Decision::Hold);
    }

    #[test]
    fn no_condition_true_places_nothing() {
        for holding in [
            Holding::Flat,
            Holding::Long { adds: 0 },
            Holding::Short { adds: 0 },
        ] {
            assert_eq!(decide(holding, NONE, 3), Decision::Hold);
        }
    }
}
