"""The values the live journal keeps: each field of an intent and an update is checked
when it is written and read back as the type it was declared, so a NumPy integer is kept
as a whole number and a missing, non-finite or too large value is refused, naming its
field."""

import math

import numpy as np
import pytest
from journal_samples import (
    DAY_NS,
    an_intent,
    an_update,
    journal_path,
    open_journal,
    started_journal,
)

from openalgo.backtest.errors import JournalError

# SQLite keeps an INTEGER as a signed 64-bit number: 2**63 is one past the largest.
PAST_THE_LARGEST_INTEGER = 2**63


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


@pytest.fixture
def journal(path):
    with started_journal(path) as handle:
        yield handle


def test_numpy_integers_in_an_intent_are_kept_as_whole_numbers(journal):
    journal.record_intent(an_intent(quantity=np.int64(10), created_ns=np.int64(DAY_NS)))
    kept = journal.intent("i1")
    assert kept == an_intent()
    assert (type(kept.quantity), type(kept.created_ns)) == (int, int)


def test_numpy_integers_in_an_update_are_kept_as_whole_numbers(journal):
    numbers = {"quantity": 10, "filled_quantity": 4, "pending_quantity": 6}
    as_numpy = {name: np.int64(value) for name, value in numbers.items()}
    journal.record_update(an_update(**as_numpy, ts_recv_ns=np.int64(DAY_NS)))
    kept = journal.last_update("o1")
    assert kept == an_update(**numbers, ts_recv_ns=DAY_NS)
    assert {type(getattr(kept, name)) for name in [*numbers, "ts_recv_ns"]} == {int}


def test_a_numpy_float_price_is_kept_as_a_float(journal):
    journal.record_intent(an_intent(price=np.float64(812.35)))
    assert type(journal.intent("i1").price) is float


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"quantity": 10.5}, "OrderIntent.quantity must be a whole number"),
        ({"quantity": None}, "OrderIntent.quantity must be a whole number"),
        ({"quantity": True}, "OrderIntent.quantity must be a whole number"),
        ({"quantity": PAST_THE_LARGEST_INTEGER}, "OrderIntent.quantity must fit in a signed"),
        ({"created_ns": float(DAY_NS)}, "OrderIntent.created_ns must be a whole number"),
        ({"created_ns": -PAST_THE_LARGEST_INTEGER - 1}, "OrderIntent.created_ns must fit in"),
        ({"price": None}, "OrderIntent.price must be a finite float or int"),
        ({"price": "812.35"}, "OrderIntent.price must be a finite float or int"),
        ({"trigger_price": math.nan}, "OrderIntent.trigger_price must be a finite number"),
        ({"symbol": None}, "OrderIntent.symbol must be text"),
        ({"purpose": "entry"}, "OrderIntent.purpose must be a member of Purpose"),
    ],
    ids=[
        "fraction",
        "None",
        "bool",
        "too large",
        "float time",
        "too early",
        "no price",
        "text price",
        "nan",
        "no symbol",
        "plain purpose",
    ],
)
def test_an_intent_field_of_the_wrong_type_is_refused_naming_it(journal, changes, named):
    with pytest.raises(JournalError, match=named):
        journal.record_intent(an_intent(**changes))
    assert journal.intent("i1") is None


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"filled_quantity": None}, "OrderUpdate.filled_quantity must be a whole number"),
        ({"pending_quantity": 2**64}, "OrderUpdate.pending_quantity must fit in a signed"),
        ({"ts_recv_ns": np.uint64(2**64 - 1)}, "OrderUpdate.ts_recv_ns must fit in a signed"),
        ({"average_price": math.nan}, "OrderUpdate.average_price must be a finite number"),
        ({"status": "complete"}, "OrderUpdate.status must be a member of OrderStatus"),
    ],
    ids=["no fill", "too large", "numpy too large", "nan price", "plain status"],
)
def test_an_update_field_of_the_wrong_type_is_refused_naming_it(journal, changes, named):
    with pytest.raises(JournalError, match=named):
        journal.record_update(an_update(**changes))
    assert journal.updates("o1") == []


@pytest.mark.parametrize(
    ("clock", "said"),
    [
        (lambda: 1.5e18, "must be a whole number"),
        (lambda: PAST_THE_LARGEST_INTEGER, "must fit in a signed 64-bit integer"),
    ],
    ids=["float", "too large"],
)
def test_a_clock_that_gives_no_64_bit_whole_nanoseconds_is_refused(path, clock, said):
    with (
        open_journal(path, now=clock) as journal,
        pytest.raises(JournalError, match=rf"now\(\) {said}"),
    ):
        journal.record_intent(an_intent())
