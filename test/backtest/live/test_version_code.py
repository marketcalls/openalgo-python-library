"""The version id of a strategy class whose source Python cannot find (`_live/_version_code.py`,
`_live/_version_id.py`; 18-multi-strategy 12.1, 2.4.0 item P5-3): a class defined in a
notebook cell, or built with `exec` or `type()`, has its compiled code hashed in place of
its source, so an edit of its code gives it a new version id, while the same code gives
the same id in every process of one Python version, whatever its hash seed. A member whose
code Python does not keep (a builtin, an object called as a method) leaves the rest of the
class hashed, and a value known by its type alone is named for the start banner, as a
parameter is. A class whose code cannot be read at all (every callable it holds is such a
member) keeps the old rule: its qualified name and parameters.

The classes are `version_cells`': `Cell` of a notebook cell, and the cell's edits.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

import pytest
from version_cells import CELL, EDITS, KEPT, edited, from_cell, with_member

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


def test_a_class_of_plain_attributes_hashes_them_as_its_parameters_are_hashed():
    # Built has no function of its own; its attributes are hashed in name order, each by
    # the rule a parameter's value is hashed by, with the Python version, and that hash
    # stands in the version id where a source hash would.
    built = type("Built", (bt.Strategy,), {"symbol": "SBIN", "exchange": "NSE", "fast": 5})
    code = (
        '{"classes":[{"attributes":{"exchange":"NSE","fast":5,"symbol":"SBIN"},'
        f'"functions":{{}},"name":"Built"}}],"python":"{PYTHON}"}}'
    )
    source = f"compiled:sha256:{by_hand(code)}"
    expected = by_hand(f'{{"parameters":{{"fast":5}},"source":"{source}"}}')[:12]
    assert version_of(built, None).version_id == expected


def with_constant(constant: object) -> type:
    """`Cell` with `constant` added to its `on_bar`'s compiled constants."""
    cell = from_cell(CELL)
    code = cell.on_bar.__code__
    cell.on_bar.__code__ = code.replace(co_consts=(*code.co_consts, constant))
    return cell


#: The slices Python 3.14 keeps among a method's constants for `x[1:]`, `x[:5]`, `x[1:3]`
#: and `x[::2]`; Python 3.12 keeps their bounds instead, so the tests add them by hand.
SLICES = (slice(1, None, None), slice(None, 5, None), slice(1, 3, None), slice(None, None, 2))


def test_a_slice_among_a_methods_constants_is_hashed_by_its_text():
    versions = [version_of(with_constant(each), None) for each in SLICES]
    assert {version.compiled_under for version in versions} == {PYTHON}
    assert len({version.version_id for version in versions}) == len(SLICES)
    assert version_of(with_constant(SLICES[0]), None).version_id == versions[0].version_id


def test_a_cached_method_and_a_builtin_leave_the_class_hashed_by_its_compiled_code():
    # Cell holds a method cached with lru_cache and a static method cached with cache, each
    # read through the function its cache wraps, and math.floor as a static method, hashed
    # by its module and name: nothing in it is known by its type alone.
    version = version_of(from_cell(CELL), None)
    assert (version.compiled_under, version.by_type) == (PYTHON, ())


def test_an_object_called_as_a_method_is_named_and_the_rest_of_the_class_still_hashed():
    source = with_member("rule = Rule()")
    version = version_of(from_cell(source), None)
    assert (version.compiled_under, version.by_type) == (PYTHON, ("rule",))
    body = source.replace(*EDITS["a method's body"])
    assert version_of(from_cell(body), None).version_id != version.version_id


def test_an_object_called_as_a_method_gives_one_version_id_whatever_its_address():
    # Each build makes its own Rule at its own address; the id hashes its text without it.
    source = with_member("rule = Rule()")
    first, second = from_cell(source), from_cell(source)
    assert first.rule is not second.rule
    assert version_of(first, None).version_id == version_of(second, None).version_id


def test_a_class_attribute_known_by_its_type_alone_is_named_for_the_banner():
    version = version_of(from_cell(with_member("model = Model()")), None)
    assert version.by_type == ("model",)


def test_a_default_argument_known_by_its_type_alone_is_named_for_the_banner():
    member = "def sized(self, bar, model=Model()):\n        return model"
    version = version_of(from_cell(with_member(member)), None)
    assert version.by_type == ("sized's default model",)


# ----- values that can change as the strategy runs (2.4.0 item P6-3) -----------------------

#: The imports a cell of these tests adds before `Cell` (they change no compiled code of it).
PRELUDE = "import numpy as np\nimport pandas as pd\n"

#: Class attributes Python cannot hash, each with an edit of its contents only: (the
#: member, its value as first written, the value edited).
MUTABLE = {
    "a list": ("seen", "[500.0, 501.25]", "[500.0, 502.5]"),
    "a dict": ("rules", '{"stop": bt.points(3)}', '{"stop": bt.points(4)}'),
    "a set": ("names", '{"SBIN", "INFY"}', '{"SBIN", "TCS"}'),
    "a bytearray": ("buffer", 'bytearray(b"ab")', 'bytearray(b"ac")'),
    "a DataFrame": (
        "frame",
        'pd.DataFrame({"close": [500.0]})',
        'pd.DataFrame({"close": [501.0]})',
    ),
    "an ndarray": ("closes", "np.array([500.0, 501.25])", "np.array([500.0, 502.5])"),
    "a dataclass that is not frozen": ("state", "State(3)", "State(4)"),
}


def with_value(member: str, value: str) -> type:
    """`Cell` with the class attribute `member = value` added."""
    return from_cell(PRELUDE + with_member(f"{member} = {value}"))


def test_a_list_the_strategy_appends_to_keeps_the_version_id_and_is_named_for_the_banner():
    cell = with_value("seen", "[]")
    before = version_of(cell, None)
    cell.seen.append(1)
    after = version_of(cell, None)
    assert after.version_id == before.version_id
    assert "seen" in after.by_type


@pytest.mark.parametrize("value", list(MUTABLE.values()), ids=list(MUTABLE))
def test_a_value_python_cannot_hash_is_hashed_by_its_type_alone(value):
    member, first, edited_value = value
    version = version_of(with_value(member, first), None)
    assert version.by_type == (member,)
    assert version_of(with_value(member, edited_value), None).version_id == version.version_id


def test_a_value_of_another_type_gives_another_version_id():
    listed = version_of(with_value("seen", "[500.0]"), None)
    assert version_of(with_value("seen", "{500.0}"), None).version_id != listed.version_id


#: Immutable class attributes, each with an edit: (the member, its value as first
#: written, the value edited).
IMMUTABLE = {
    "a tuple": ("pair", '("SBIN", 5)', '("SBIN", 6)'),
    "a frozenset": ("names", 'frozenset({"SBIN", "INFY"})', 'frozenset({"SBIN", "TCS"})'),
    "bytes": ("raw", 'b"ab"', 'b"ac"'),
    "a complex number": ("gain", "1 + 2j", "1 + 3j"),
    "a date": ("start", "dt.date(2026, 11, 23)", "dt.date(2026, 11, 24)"),
    "a tuple of stop distances": (
        "stops",
        "(bt.points(3), bt.pct(1))",
        "(bt.points(3), bt.pct(2))",
    ),
}


@pytest.mark.parametrize("value", list(IMMUTABLE.values()), ids=list(IMMUTABLE))
def test_an_edit_of_an_immutable_value_still_changes_the_version_id(value):
    member, first, edited_value = value
    version = version_of(with_value(member, first), None)
    assert version.by_type == ()
    assert version_of(with_value(member, edited_value), None).version_id != version.version_id


def test_a_list_inside_a_tuple_is_hashed_by_its_type_and_the_rest_by_value():
    version = version_of(with_value("pair", '("SBIN", [5])'), None)
    assert version.by_type == ("pair",)
    assert version_of(with_value("pair", '("SBIN", [6])'), None).version_id == version.version_id
    assert version_of(with_value("pair", '("INFY", [5])'), None).version_id != version.version_id


def test_a_mutable_default_argument_is_hashed_by_its_type_and_named_for_the_banner():
    member = "def remember(self, bar, seen=[]):\n        seen.append(bar)\n        return seen"
    cell = from_cell(with_member(member))
    before = version_of(cell, None)
    cell.remember(None, 500.0)
    assert version_of(cell, None).version_id == before.version_id
    assert before.by_type == ("remember's default seen",)


# ----- classes nested in the strategy class (2.4.0 item P6-4) ------------------------------

#: A class nested in `Cell`: its levels above a price.
LEVELS = (
    "class Levels:\n"
    "        width = 2\n"
    "\n"
    "        def above(self, price):\n"
    "            return price + self.width\n"
    "\n"
    "        class Band:\n"
    "            def wide(self, price):\n"
    "                return price * 2\n"
)


def nested_id(source: str = LEVELS) -> str:
    """The version id of `Cell` holding the nested class of `source`."""
    return cell_id(with_member(source))


@pytest.mark.parametrize(
    "edit",
    [
        ("return price + self.width", "return price - self.width"),
        ("width = 2", "width = 3"),
        ("return price * 2", "return price * 3"),
        ("class Levels:", "class Levels(dict):"),
    ],
    ids=["its method's body", "its attribute", "a class nested in it", "its base"],
)
def test_an_edit_of_a_nested_class_gives_a_new_version_id(edit):
    old, new = edit
    assert LEVELS.count(old) == 1
    assert nested_id(LEVELS.replace(old, new)) != nested_id()


def test_the_same_nested_class_built_twice_gets_the_same_version_id():
    first, second = from_cell(with_member(LEVELS)), from_cell(with_member(LEVELS))
    assert first.Levels is not second.Levels
    assert version_of(first, None).version_id == version_of(second, None).version_id


def test_a_value_of_a_nested_class_known_by_its_type_is_named_with_its_class():
    nested = "class Levels:\n        seen = []\n        model = Model()\n"
    version = version_of(from_cell(with_member(nested)), None)
    assert version.by_type == ("Levels.model", "Levels.seen")


def test_a_nested_class_held_under_two_names_is_read_once():
    # `Alias` comes first in name order, so the class is read and named there.
    nested = "class Levels:\n        model = Model()\n    Alias = Levels\n"
    version = version_of(from_cell(with_member(nested)), None)
    assert version.by_type == ("Alias.model",)


def test_a_class_defined_outside_the_strategy_is_hashed_by_its_name_only():
    # Model is defined in the cell, not in Cell: an edit of it keeps Cell's id.
    source = with_member("kind = Model")
    edited_model = source.replace("class Model:\n    pass\n", "class Model:\n    size = 2\n")
    assert cell_id(edited_model) == cell_id(source)


class Rule:
    """An object a class calls as a method: Python keeps no compiled code of it there."""

    def __call__(self, strategy, bar):
        return None


@pytest.mark.parametrize(
    "unreadable",
    [Rule(), staticmethod(len)],
    ids=["an object called as a method", "a builtin as a static method"],
)
def test_a_class_whose_code_cannot_be_read_at_all_hashes_its_qualified_name_and_parameters(
    unreadable,
):
    # Every callable Opaque holds is one whose code Python does not keep.
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
        "hashes its compiled code (its functions and class attributes) under CPython 3.12 "
        "and its parameters, so the same code under another Python version gets another id"
    ]


def test_the_banner_keeps_its_words_for_a_class_whose_code_cannot_be_read_either():
    started = Started("Opaque", "0123456789ab", False, {})
    assert StartVersions((started,)).banner_lines()[1:] == [
        "Opaque: its source was not hashed (a class built at run time): its version id "
        "hashes its name and parameters only"
    ]
