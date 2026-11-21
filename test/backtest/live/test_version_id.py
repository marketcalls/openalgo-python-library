"""A strategy's version id (`_live/_version_id.py`; 18-multi-strategy 12.1): 12 hexadecimal
characters, the start of the SHA-256 of the strategy's source hash and its parameter values
as canonical JSON (keys sorted, no spaces). The same code with the same parameters gives the
same id on every day and restart; a change to either gives a new one; a class with no
source (built at run time) hashes its compiled code in its place (`test_version_code`), or
its qualified name when its code cannot be read at all."""

from __future__ import annotations

import hashlib

import pytest

import openalgo.backtest as bt
from openalgo.backtest._live._version_id import (
    Started,
    StartVersions,
    parameter_values,
    version_of,
)


class Cross(bt.Strategy):
    """Two parameters; never trades."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    fast = 9
    slow = 21

    def on_bar(self, bar):
        pass


class SameCodeOtherName(Cross):
    """A subclass with nothing of its own but its name: its source differs."""


#: A class built at run time: `inspect` finds no source for it.
Built = type("Built", (bt.Strategy,), {"symbol": "SBIN", "exchange": "NSE", "fast": 5})


class Rule:
    """An object a class calls as a method: Python keeps no compiled code of it there."""

    def __call__(self, strategy, bar):
        return None


#: A class built at run time whose code cannot be read at all: its one callable is an object.
Opaque = type("Opaque", (Built,), {"on_bar": Rule()})


def by_hand(canonical: str) -> str:
    """The first 12 hexadecimal characters of the SHA-256 of `canonical`'s UTF-8 bytes."""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def test_a_class_with_no_readable_code_and_parameters_given_hashes_the_given_values():
    version = version_of(Opaque, {"fast": 7})
    assert version.version_id == by_hand('{"parameters":{"fast":7},"source":"Opaque"}')


def test_a_class_with_source_hashes_its_source_hash_and_parameters():
    from openalgo.backtest._manifest import strategy_source_hash

    source = strategy_source_hash(Cross)
    version = version_of(Cross, None)
    canonical = f'{{"parameters":{{"fast":9,"slow":21}},"source":"{source}"}}'
    assert version.version_id == by_hand(canonical)
    assert version.source_hashed


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        ((Cross, None), (Cross, None), True),
        ((Cross, None), (Cross, {"fast": 9}), True),
        ((Cross, None), (Cross, {"fast": 10}), False),
        ((Cross, None), (SameCodeOtherName, None), False),
        ((Built, None), (Built, {"fast": 5}), True),
    ],
    ids=[
        "the same code and parameters",
        "a parameter given at its default",
        "a parameter changed",
        "other code",
        "no source, the same parameters",
    ],
)
def test_a_version_id_changes_only_with_the_code_or_the_parameters(first, second, same):
    assert (version_of(*first).version_id == version_of(*second).version_id) is same


def test_a_version_id_is_12_lowercase_hexadecimal_characters():
    version_id = version_of(Cross, None).version_id
    assert len(version_id) == 12
    assert set(version_id) <= set("0123456789abcdef")


def test_the_parameter_values_are_the_defaults_with_the_given_ones_over_them():
    assert parameter_values(Cross, {"slow": 30}) == {"fast": 9, "slow": 30}


def test_a_set_parameter_is_hashed_in_sorted_order():
    # A set's own order depends on the process's hash seed: the id must not.
    from openalgo.backtest._manifest import strategy_source_hash

    source = strategy_source_hash(Cross)
    version = version_of(Cross, {"watch": {"TCS", "SBIN", "INFY", "HDFCBANK"}})
    canonical = (
        '{"parameters":{"fast":9,"slow":21,"watch":["HDFCBANK","INFY","SBIN","TCS"]},'
        f'"source":"{source}"}}'
    )
    assert version.version_id == by_hand(canonical)


class Model:
    """An object with no text of its own: its default text names its memory address."""


def test_an_object_known_only_by_its_address_is_hashed_by_its_type_and_named():
    # Two objects alive at once have two addresses, as one object has in two processes.
    one, other = Model(), Model()
    first, second = version_of(Cross, {"model": one}), version_of(Cross, {"model": other})
    assert first.version_id == second.version_id
    assert first.by_type == ("model",)
    assert version_of(Cross, None).by_type == ()


def test_the_banner_names_a_parameter_hashed_by_its_type():
    started = Started("Cross", "0123456789ab", True, {}, by_type=("model",))
    assert StartVersions((started,)).banner_lines()[-1] == (
        "Cross: model has no text of its value (an object shown by its memory address): its "
        "version id hashes its type only, so a change to its value does not change the id"
    )
