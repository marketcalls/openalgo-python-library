"""A strategy's version id (`_live/_version_id.py`; 18-multi-strategy 12.1): 12 hexadecimal
characters, the start of the SHA-256 of the strategy's source hash and its parameter values
as canonical JSON (keys sorted, no spaces). The same code with the same parameters gives the
same id on every day and restart; a change to either gives a new one; a class with no
source (built at run time) hashes its qualified name in its place."""

from __future__ import annotations

import hashlib

import pytest

import openalgo.backtest as bt
from openalgo.backtest._live._version_id import parameter_values, version_of


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


def by_hand(canonical: str) -> str:
    """The first 12 hexadecimal characters of the SHA-256 of `canonical`'s UTF-8 bytes."""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def test_a_class_with_no_source_hashes_its_qualified_name_and_parameters():
    version = version_of(Built, None)
    assert version.version_id == by_hand('{"parameters":{"fast":5},"source":"Built"}')
    assert not version.source_hashed


def test_a_class_with_no_source_and_parameters_given_hashes_the_given_values():
    version = version_of(Built, {"fast": 7})
    assert version.version_id == by_hand('{"parameters":{"fast":7},"source":"Built"}')


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
