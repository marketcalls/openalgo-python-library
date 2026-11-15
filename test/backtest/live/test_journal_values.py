"""The values the live journal keeps: each field of an intent and an update is checked
when it is written and read back as the type it was declared, so a NumPy integer is kept
as a whole number and a missing or non-finite value is refused, naming its field."""

import math

import numpy as np
import pytest
from journal_samples import DAY_NS, an_intent, an_update, journal_path, open_journal

from openalgo.backtest.errors import JournalError


@pytest.fixture
def path(tmp_path):
    return journal_path(tmp_path)


@pytest.fixture
def journal(path):
    with open_journal(path) as handle:
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
        ({"created_ns": float(DAY_NS)}, "OrderIntent.created_ns must be a whole number"),
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
        "float time",
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
        ({"average_price": math.nan}, "OrderUpdate.average_price must be a finite number"),
        ({"status": "complete"}, "OrderUpdate.status must be a member of OrderStatus"),
    ],
    ids=["no fill", "nan price", "plain status"],
)
def test_an_update_field_of_the_wrong_type_is_refused_naming_it(journal, changes, named):
    with pytest.raises(JournalError, match=named):
        journal.record_update(an_update(**changes))
    assert journal.updates("o1") == []


def test_a_clock_that_gives_no_whole_nanoseconds_is_refused(path):
    with (
        open_journal(path, now=lambda: 1.5e18) as journal,
        pytest.raises(JournalError, match=r"now\(\) must be a whole number"),
    ):
        journal.record_intent(an_intent())
