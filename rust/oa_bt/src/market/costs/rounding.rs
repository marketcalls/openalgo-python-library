//! Rounding modes and the two rounding rules of the engine.
//!
//! - `round_half_up` rounds an amount held as exact integer pico-rupees. It is the
//!   one helper behind every rupee and paisa rounding of a sum (a contract-note
//!   line, a reported total, GST on the DP fee), so a line of exactly Rs 1.50
//!   rounds to Rs 2 even though the float that carried it was 1.4999999999999998.
//! - `calculator_paise` and `calculator_rupee` round a single product the way the
//!   Zerodha calculator's number formatting does, on the exact binary value. They
//!   are used only in calculator mode.

use super::error::{CostError, NameKind};

/// How per-fill costs are rounded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Rounding {
    /// Default. Every component is accrued unrounded on each fill; rounding happens
    /// once per trade day and contract note in `ContractNoteAggregator`. Rounding
    /// tax to the rupee on every fill would be biased for small trades (a small
    /// option sale would pay no STT at all).
    #[default]
    ContractNote,
    /// Reproduces the Zerodha brokerage calculator: every component is rounded on
    /// each fill or round trip the way the calculator rounds it. Used by the parity
    /// tests.
    Calculator,
}

impl Rounding {
    /// Both modes, the default first.
    pub const ALL: [Rounding; 2] = [Rounding::ContractNote, Rounding::Calculator];

    /// The mode's name: "contract_note" or "calculator".
    pub fn code(self) -> &'static str {
        match self {
            Rounding::ContractNote => "contract_note",
            Rounding::Calculator => "calculator",
        }
    }

    /// Parse a mode name. Errors: `UnknownName` for anything but the two codes.
    pub fn from_code(code: &str) -> Result<Rounding, CostError> {
        match code {
            "contract_note" => Ok(Rounding::ContractNote),
            "calculator" => Ok(Rounding::Calculator),
            other => Err(CostError::unknown(NameKind::Rounding, other)),
        }
    }
}

/// An amount as an integer number of pico-rupees (1e-12 rupee). Sums of these are
/// exact, which float sums are not.
pub(super) type Picos = i128;

const PICOS_PER_RUPEE: Picos = 1_000_000_000_000;
const PICOS_PER_PAISA: Picos = PICOS_PER_RUPEE / 100;

/// The decimal grid a float amount is snapped to when it becomes `Picos`.
#[derive(Clone, Copy)]
pub(super) enum Grid {
    /// Nine decimals: a charge is a value in paise times a rate of at most seven
    /// decimals, so its exact value has at most nine.
    Charge,
    /// Eleven decimals: GST is a two-decimal rate on top of charges.
    Gst,
}

impl Grid {
    fn picos(self) -> Picos {
        match self {
            Grid::Charge => 1_000,
            Grid::Gst => 10,
        }
    }
}

/// What a sum is rounded to.
#[derive(Clone, Copy)]
pub(super) enum Unit {
    /// Two decimals.
    Paisa,
    /// A whole rupee.
    Rupee,
}

/// The exact decimal amount a float stands for, as pico-rupees.
///
/// A float product such as 10,000 x 0.00015 is stored as 1.4999999999999998; its
/// exact decimal value is 1.5. Snapping to the grid the exact value must lie on
/// recovers it, because the float error (about 1e-16 of the amount) is far below
/// half a grid step for any single fill. Callers have checked the amount is finite.
pub(super) fn to_picos(rupees: f64, grid: Grid) -> Picos {
    let step = grid.picos();
    let steps = (rupees * (PICOS_PER_RUPEE / step) as f64).round();
    // The cast saturates on an absurd amount; so must the multiplication.
    (steps as Picos).saturating_mul(step)
}

/// Pico-rupees as a float number of rupees.
pub(super) fn from_picos(picos: Picos) -> f64 {
    picos as f64 / PICOS_PER_RUPEE as f64
}

/// Round an exact amount to the paisa or the rupee, half away from zero (50 paise
/// rounds up to the rupee, half a paisa up to the paisa). Returns rupees.
pub(super) fn round_half_up(picos: Picos, unit: Unit) -> f64 {
    let step = match unit {
        Unit::Paisa => PICOS_PER_PAISA,
        Unit::Rupee => PICOS_PER_RUPEE,
    };
    let half = step / 2;
    let nudged = if picos >= 0 {
        picos.saturating_add(half)
    } else {
        picos.saturating_sub(half)
    };
    from_picos(nudged / step * step)
}

/// Round a float sum of whole-paise amounts, or a single charge, to `unit`.
pub(super) fn round_sum(rupees: f64, unit: Unit) -> f64 {
    round_half_up(to_picos(rupees, Grid::Charge), unit)
}

// Layout of an IEEE 754 double: 1 sign bit, 11 exponent bits, 52 fraction bits.
const FRACTION_BITS: u32 = 52;
const FRACTION_MASK: u64 = (1 << FRACTION_BITS) - 1;
const EXPONENT_MASK: u64 = 0x7ff;
/// A normal double is `mantissa * 2^(exponent - 1075)` with the implicit leading bit
/// set in the mantissa; a subnormal one is `fraction * 2^-1074`.
const EXPONENT_OFFSET: i32 = 1075;
const SUBNORMAL_POWER: i32 = -1074;

const PAISE_PER_RUPEE: u128 = 100;
/// `mantissa * 100` is below 2^60, so a value shifted right by more than 60 bits is
/// below half a paisa.
const NEGLIGIBLE_SHIFT: u32 = 60;

/// Round rupees to two decimals the way the calculator's number formatting does:
/// half up on the exact binary value of the float.
///
/// `(x * 100.0).round() / 100.0` is not the same thing: the product is itself
/// rounded, so 1.115 (stored as 1.11499999999999999) becomes 111.50000000000001 and
/// rounds the wrong way. This function multiplies the mantissa by 100 in integer
/// arithmetic, so it decides on the number the double really holds.
///
/// The input is a non-negative finite amount (callers validate it); zero, negative,
/// NaN and infinite inputs are returned unchanged.
pub(super) fn calculator_paise(rupees: f64) -> f64 {
    if rupees.is_nan() || rupees.is_infinite() || rupees <= 0.0 {
        return rupees;
    }
    let bits = rupees.to_bits();
    let exponent = ((bits >> FRACTION_BITS) & EXPONENT_MASK) as i32;
    let fraction = bits & FRACTION_MASK;
    // rupees == mantissa * 2^power exactly.
    let (mantissa, power) = if exponent == 0 {
        (fraction, SUBNORMAL_POWER)
    } else {
        (fraction | (1 << FRACTION_BITS), exponent - EXPONENT_OFFSET)
    };
    if power >= 0 {
        // Already a whole number of rupees.
        return rupees;
    }
    let shift = power.unsigned_abs();
    if shift > NEGLIGIBLE_SHIFT {
        return 0.0;
    }
    let scaled = u128::from(mantissa) * PAISE_PER_RUPEE;
    let mut paise = scaled >> shift;
    let remainder = scaled & ((1u128 << shift) - 1);
    if remainder >= 1u128 << (shift - 1) {
        paise += 1;
    }
    paise as f64 / PAISE_PER_RUPEE as f64
}

/// Round a two-decimal amount to the rupee as the calculator does: 50 paise and
/// above rounds up.
#[inline]
pub(super) fn calculator_rupee(rupees: f64) -> f64 {
    rupees.round()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculator_paise_decides_ties_on_the_stored_binary_value() {
        // (input, expected). The first group is stored just below the decimal tie
        // and rounds down; the second just above and rounds up; the third are exact
        // binary ties and round up.
        let rows = [
            (1.005, 1.0),
            (1.115, 1.11),
            (2.675, 2.67),
            (0.285, 0.28),
            (8.345, 8.35),
            (0.005, 0.01),
            (0.125, 0.13),
            (0.375, 0.38),
        ];
        for (input, expected) in rows {
            assert_eq!(calculator_paise(input), expected, "{input}");
        }
    }

    #[test]
    fn calculator_paise_of_ordinary_amounts() {
        let rows = [
            (24.948, 24.95),
            (11.9934, 11.99),
            (0.084, 0.08),
            (0.004, 0.0),
            (2.5, 2.5),
        ];
        for (input, expected) in rows {
            assert_eq!(calculator_paise(input), expected, "{input}");
        }
    }

    #[test]
    fn calculator_paise_leaves_whole_and_huge_amounts_alone() {
        for amount in [0.0, 840_000.0, 1e18] {
            assert_eq!(calculator_paise(amount), amount);
        }
    }

    #[test]
    fn calculator_paise_of_tiny_amounts_is_zero() {
        for amount in [1e-300, f64::MIN_POSITIVE / 4.0, 1e-18] {
            assert_eq!(calculator_paise(amount), 0.0);
        }
    }

    #[test]
    fn calculator_rupee_sends_fifty_paise_up() {
        let rows = [
            (109.49, 109.0),
            (109.5, 110.0),
            (0.49, 0.0),
            (0.5, 1.0),
            (1.2, 1.0),
        ];
        for (input, expected) in rows {
            assert_eq!(calculator_rupee(input), expected, "{input}");
        }
    }

    #[test]
    fn a_float_just_below_a_decimal_tie_becomes_the_exact_decimal() {
        // 10,000 x 0.00015 is 1.5 exactly; the float product is 1.4999999999999998.
        let stamp_duty = 10_000.0 * 0.00015;
        assert!(stamp_duty < 1.5);
        assert_eq!(to_picos(stamp_duty, Grid::Charge), 1_500_000_000_000);
    }

    #[test]
    fn round_half_up_sends_exact_ties_up() {
        // (pico-rupees, to the rupee, to the paisa)
        let rows = [
            (1_500_000_000_000, 2.0, 1.5),
            (1_499_999_999_000, 1.0, 1.5),
            (2_500_000_000_000, 3.0, 2.5),
            (5_000_000_000, 0.0, 0.01),
            (4_999_999_000, 0.0, 0.0),
            (125_000_000_000, 0.0, 0.13),
            (0, 0.0, 0.0),
        ];
        for (picos, rupee, paisa) in rows {
            assert_eq!(round_half_up(picos, Unit::Rupee), rupee, "{picos}");
            assert_eq!(round_half_up(picos, Unit::Paisa), paisa, "{picos}");
        }
    }

    #[test]
    fn round_sum_clears_the_float_noise_of_adding_paise() {
        // 0.1 + 0.2 is 0.30000000000000004 as a float.
        assert_eq!(round_sum(0.1 + 0.2, Unit::Paisa), 0.3);
        assert_eq!(
            round_sum(40.0 + 110.0 + 25.79 + 0.84 + 12.0 + 11.99, Unit::Paisa),
            200.62
        );
    }

    #[test]
    fn rounding_mode_codes_round_trip() {
        for mode in [Rounding::ContractNote, Rounding::Calculator] {
            assert_eq!(Rounding::from_code(mode.code()), Ok(mode));
        }
    }

    #[test]
    fn contract_note_is_the_default_rounding_mode() {
        assert_eq!(Rounding::default(), Rounding::ContractNote);
    }

    #[test]
    fn an_unknown_rounding_mode_is_an_error() {
        assert_eq!(
            Rounding::from_code("nearest"),
            Err(CostError::UnknownName {
                kind: NameKind::Rounding,
                name: "nearest".to_string()
            })
        );
    }
}
