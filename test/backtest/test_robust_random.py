"""The seeded draws of the robustness checks: SplitMix64 read as a counter, whole numbers
below a bound as the high word of a product, fractions of the top 53 bits, random orders
that are permutations, and runs laid out so the first runs never depend on how many are
asked for.

The reference outputs of SplitMix64 for seed 0 are the published first values of the
generator (Steele, Lea and Flood 2014): 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4,
0x06C45D188009454F.
"""

import numpy as np
import pytest

from openalgo.backtest._robust import _random

SEED_ZERO = (0xE220_A839_7B1D_CDAF, 0x6E78_9E6A_A1B9_65F4, 0x06C4_5D18_8009_454F)


def test_the_stream_of_seed_zero_starts_with_the_published_values():
    assert [int(value) for value in _random.outputs(0, 0, 3)] == list(SEED_ZERO)


def test_a_stretch_of_the_stream_is_where_its_counter_says():
    assert np.array_equal(_random.outputs(7, 5, 3), _random.outputs(7, 0, 8)[5:])


@pytest.mark.parametrize(
    ("bits", "bound", "drawn"),
    [
        (0, 10, 0),
        (2**63, 10, 5),  # half of 2^64 times 10, over 2^64
        (2**64 - 1, 7, 6),  # the largest output gives the largest number below the bound
        (2**62, 3, 0),  # 3 x 2^62 / 2^64 = 0.75
        (3 * 2**62, 4, 3),
    ],
)
def test_a_whole_number_draw_is_the_high_word_of_output_times_bound(bits, bound, drawn):
    assert _random.below(np.array([bits], dtype=np.uint64), bound)[0] == drawn


@pytest.mark.parametrize(
    ("bits", "fraction"),
    [(0, 0.0), (2**63, 0.5), (2**64 - 1, (2**53 - 1) / 2**53), (2**11, 2**-53)],
)
def test_a_fraction_is_the_top_53_bits_over_two_to_the_53(bits, fraction):
    assert _random.fractions(np.array([bits], dtype=np.uint64))[0] == fraction


def test_every_random_order_is_a_permutation_of_the_positions():
    orders = _random.orders(seed=4, first_run=0, runs=300, size=9)
    assert (np.sort(orders, axis=1) == np.arange(9)).all()


def test_random_orders_differ_from_run_to_run():
    orders = _random.orders(seed=4, first_run=0, runs=50, size=9)
    assert len({tuple(order) for order in orders}) > 40


def test_the_first_runs_are_the_same_whatever_the_number_of_runs():
    few = _random.orders(seed=11, first_run=0, runs=200, size=13)
    many = _random.orders(seed=11, first_run=0, runs=1_000, size=13)
    assert np.array_equal(few, many[:200])


def test_runs_drawn_in_chunks_equal_the_runs_drawn_at_once():
    whole = _random.block(seed=2, first_run=0, runs=10, width=6)
    chunks = [_random.block(2, first, size, 6) for first, size in _random.run_chunks(10, 6, 20)]
    assert np.array_equal(np.concatenate(chunks), whole)


def test_chunks_cover_every_run_in_order_with_at_most_the_draws_asked():
    assert _random.run_chunks(10, 6, 20) == [(0, 3), (3, 3), (6, 3), (9, 1)]


def test_a_chunk_holds_one_run_even_when_a_run_has_more_draws_than_a_chunk():
    assert _random.run_chunks(2, 50, 20) == [(0, 1), (1, 1)]


def test_each_check_reads_the_stream_of_the_seed_plus_its_number():
    assert _random.check_seed(0, 11) == 11
    assert _random.check_seed(5, 2) == 7
