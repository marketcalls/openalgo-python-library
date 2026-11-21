"""The version id of a strategy class whose source Python cannot find (`_live/_version_code.py`,
`_live/_version_id.py`; 18-multi-strategy 12.1, 2.4.0 item P5-3): a class defined in a
notebook cell, or built with `exec` or `type()`, has its compiled code hashed in place of
its source, so an edit of its code gives it a new version id, while the same code gives
the same id in every process of one Python version, whatever its hash seed. A class whose
code cannot be read either keeps the old rule: its qualified name and parameters.

The classes are `version_cells`': `Cell` of a notebook cell, and the cell's edits.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

import pytest
from version_cells import CELL, EDITS, KEPT, edited, from_cell

import openalgo.backtest as bt
from openalgo.backtest._live._version_id import Started, StartVersions, version_of
from openalgo.backtest._manifest import strategy_source_hash

HERE = pathlib.Path(__file__).parent
#: The Python the compiled code is hashed under, as the banner names it.
PYTHON = f"CPython {sys.version_info.major}.{sys.version_info.minor}"
#: The hash seeds the id is checked under, each in a fresh interpreter.
SEEDS = ("1", "2", "3")


def by_hand(canonical: str) -> str:
    """The SHA-256 of `canonical`'s UTF-8 bytes, in hexadecimal."""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cell_id(source: str = CELL) -> str:
    return version_of(from_cell(source), None).version_id


# ----- what the id follows ------------------------------------------------------------------


@pytest.mark.parametrize("edit", list(EDITS.values()), ids=list(EDITS))
def test_an_edit_of_a_class_built_at_run_time_gives_it_a_new_version_id(edit):
    assert cell_id(edited(edit)) != cell_id()


@pytest.mark.parametrize("edit", list(KEPT.values()), ids=list(KEPT))
def test_an_edit_that_changes_no_compiled_code_keeps_the_version_id(edit):
    assert cell_id(edited(edit)) == cell_id()


def test_the_same_code_built_twice_gets_the_same_version_id():
    first, second = from_cell(CELL), from_cell(CELL)
    assert first is not second
    assert version_of(first, None).version_id == version_of(second, None).version_id


def test_the_parameters_given_still_change_the_version_id():
    cell = from_cell(CELL)
    assert version_of(cell, {"fast": 10}).version_id != version_of(cell, None).version_id
    assert version_of(cell, {"fast": 9}).version_id == version_of(cell, None).version_id


def test_the_version_id_is_the_same_in_a_fresh_interpreter_under_every_hash_seed():
    # The cell holds a set of names, whose own order follows the hash seed; the parameters
    # given hold one too.
    code = (
        "import sys\n"
        f"sys.path[:0] = [{str(HERE.parents[2])!r}, {str(HERE)!r}]\n"
        "from version_cells import CELL, from_cell\n"
        "from openalgo.backtest._live._version_id import version_of\n"
        "watch = {'TCS', 'SBIN', 'INFY', 'HDFCBANK'}\n"
        "print(version_of(from_cell(CELL), {'watch': watch}).version_id)\n"
    )
    found = set()
    for seed in SEEDS:
        finished = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            timeout=300,
        )
        assert finished.returncode == 0, finished.stderr
        found.add(finished.stdout.strip().splitlines()[-1])
    here = version_of(from_cell(CELL), {"watch": {"TCS", "SBIN", "INFY", "HDFCBANK"}})
    assert found == {here.version_id}


# ----- what is hashed --------------------------------------------------------------------------


def test_a_class_built_at_run_time_is_hashed_by_its_compiled_code_under_its_python():
    version = version_of(from_cell(CELL), None)
    assert not version.source_hashed
    assert version.compiled_under == PYTHON


def test_a_class_of_plain_attributes_hashes_them_by_type_and_text():
    # Built has no function of its own; its attributes are hashed in name order, each as
    # its type's name and its text, with the Python version, and that hash stands in the
    # version id where a source hash would.
    built = type("Built", (bt.Strategy,), {"symbol": "SBIN", "exchange": "NSE", "fast": 5})
    code = (
        '{"classes":[{"attributes":{"exchange":["str","\'NSE\'"],"fast":["int","5"],'
        '"symbol":["str","\'SBIN\'"]},"functions":{},"name":"Built"}],'
        f'"python":"{PYTHON}"}}'
    )
    source = f"compiled:sha256:{by_hand(code)}"
    expected = by_hand(f'{{"parameters":{{"fast":5}},"source":"{source}"}}')[:12]
    assert version_of(built, None).version_id == expected


class Rule:
    """An object a class calls as a method: Python keeps no compiled code of it there."""

    def __call__(self, strategy, bar):
        return None


@pytest.mark.parametrize(
    "unreadable",
    [Rule(), staticmethod(len)],
    ids=["an object called as a method", "a builtin as a static method"],
)
def test_a_class_whose_code_cannot_be_read_hashes_its_qualified_name_and_parameters(unreadable):
    members = {"symbol": "SBIN", "exchange": "NSE", "fast": 5, "on_bar": unreadable}
    opaque = type("Opaque", (bt.Strategy,), members)
    version = version_of(opaque, None)
    assert version.version_id == by_hand('{"parameters":{"fast":5},"source":"Opaque"}')[:12]
    assert (version.source_hashed, version.compiled_under) == (False, None)


class Readable(bt.Strategy):
    """A class with its source in this file."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    fast = 9

    def on_bar(self, bar):
        pass


def test_a_class_with_readable_source_keeps_its_source_hash():
    source = strategy_source_hash(Readable)
    version = version_of(Readable, None)
    assert version.version_id == by_hand(f'{{"parameters":{{"fast":9}},"source":"{source}"}}')[:12]
    assert (version.source_hashed, version.compiled_under) == (True, None)


# ----- the banner ---------------------------------------------------------------------------


def test_the_banner_says_a_class_built_at_run_time_is_hashed_by_its_compiled_code():
    started = Started("Cell", "0123456789ab", False, {}, compiled_under="CPython 3.12")
    assert StartVersions((started,)).banner_lines()[1:] == [
        "Cell: its source could not be read (a class built at run time): its version id "
        "hashes its compiled code (its functions and plain class attributes) under CPython "
        "3.12 and its parameters, so the same code under another Python version gets "
        "another id"
    ]


def test_the_banner_keeps_its_words_for_a_class_whose_code_cannot_be_read_either():
    started = Started("Opaque", "0123456789ab", False, {})
    assert StartVersions((started,)).banner_lines()[1:] == [
        "Opaque: its source was not hashed (a class built at run time): its version id "
        "hashes its name and parameters only"
    ]
