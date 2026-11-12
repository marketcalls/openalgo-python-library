//! A seeded pseudo-random generator for resampling, so that a Monte Carlo study gives
//! the same numbers for the same seed on every platform.
//!
//! The generator is SplitMix64 (Steele, Lea and Flood 2014, "Fast splittable
//! pseudorandom number generators", OOPSLA): a 64-bit counter advanced by a fixed odd
//! constant and passed through a bijective mixing function. Integers in a range come
//! from Lemire's multiply-and-reject method (Lemire 2019, "Fast random integer
//! generation in an interval", ACM Transactions on Modeling and Computer Simulation
//! 29(1)), which has no modulo bias. Only integer arithmetic is used.

/// The counter increment: the odd integer nearest 2^64 divided by the golden ratio.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// The two multipliers of the mixing function (Steele, Lea and Flood 2014).
const MIX_1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_2: u64 = 0x94D0_49BB_1331_11EB;

/// A SplitMix64 generator (module docs).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// A generator whose sequence is fixed by `seed`.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX_1);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_2);
        z ^ (z >> 31)
    }

    /// A uniform integer in `0..bound`; 0 when `bound` is 0 or 1.
    pub fn below(&mut self, bound: usize) -> usize {
        let range = bound as u64;
        if range <= 1 {
            return 0;
        }
        let mut product = u128::from(self.next_u64()) * u128::from(range);
        if (product as u64) < range {
            // Reject the few low words that would make some results more likely.
            let threshold = range.wrapping_neg() % range;
            while (product as u64) < threshold {
                product = u128::from(self.next_u64()) * u128::from(range);
            }
        }
        (product >> 64) as usize
    }
}
