"""Strategy classes as a notebook makes them, shared by the tests of the version id of a
class whose source Python cannot find (`test_version_code.py`; 2.4.0 item P5-3).

A notebook runs a cell by compiling its text and running it in its namespace: Python keeps
the classes' compiled code but no source file `inspect` could read. `from_cell` does the
same with a cell's text, compiled with no compiler flag of its caller (`dont_inherit`), so
a cell compiles alike in a test module and in a fresh interpreter. `CELL` is the cell the
tests edit: `Cell` buys 10 SBIN when its symbol is on a list (a set of names, which Python
keeps as a constant whose own order depends on the process's hash seed) and the bar closes
above 500.00, and holds what a notebook's strategy often holds: a method cached with
`functools.lru_cache`, a static method cached with `functools.cache`, a builtin as a
static method, and class attributes of times, a duration, a stop distance and the SDK's
clock rules, every one immutable (a mutable value, such as a list or a dict, is hashed by
its type alone). Each edit of `EDITS` changes one thing of it; `with_member` adds a member
to it. The cell also defines `Rule`, an object called as a method, `Model`, an object with
no text of its value (its text names its memory address), and `State`, a dataclass whose
values can change (Python cannot hash it), for the members the tests add.
"""

from typing import Any

#: The cell of `Cell` as first written.
CELL = """
import dataclasses
import datetime as dt
import functools
import math

import openalgo.backtest as bt


def logged(method):
    @functools.wraps(method)
    def wrapper(self, bar):
        return method(self, bar)

    return wrapper


class Rule:
    def __call__(self, bar):
        return None


class Model:
    pass


@dataclasses.dataclass
class State:
    count: int = 0


class Cell(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "1m"
    fast = 9
    watch = ("SBIN", "INFY")
    stop_distance = bt.points(3)
    entry_from = dt.time(9, 20)
    cooldown = dt.timedelta(minutes=5)
    window = bt.time_between("09:20", "15:00")
    days = bt.weekday(0, 1)
    floor = staticmethod(math.floor)

    def on_bar(self, bar):
        if self.symbol in {"SBIN", "INFY", "TCS", "HDFCBANK", "RELIANCE"} and bar.close > 500.0:
            self.buy(10)

    @logged
    def on_fill(self, bar):
        return self.helper(bar)

    @staticmethod
    def helper(bar, size=10):
        return size

    @property
    def level(self):
        return 500.0

    @functools.lru_cache(maxsize=None)
    def band(self, width):
        return width * 2

    @staticmethod
    @functools.cache
    def round_lot(units):
        return units - units % 1
"""

#: One edit each of `CELL`, by what it changes: (the text replaced, its replacement).
EDITS = {
    "a method's body": ("bar.close > 500.0", "bar.close >= 500.0"),
    "a constant": ("bar.close > 500.0", "bar.close > 505.0"),
    "a name a method calls": ("self.buy(10)", "self.sell(10)"),
    "a class attribute of a tuple": ('watch = ("SBIN", "INFY")', 'watch = ("SBIN", "TCS")'),
    "a stop distance in a class attribute": ("bt.points(3)", "bt.points(4)"),
    "a declaration": ('"1m"', '"5m"'),
    "a decorated method's body": ("return self.helper(bar)", "return self.helper(bar) + 1"),
    "a static method's default argument": ("size=10", "size=20"),
    "a property's body": ("return 500.0", "return 505.0"),
    "a cached method's body": ("return width * 2", "return width * 3"),
    "a cached static method's body": ("return units - units % 1", "return units - units % 5"),
    "a builtin held as a static method": ("math.floor", "math.ceil"),
    "a time of day in a class attribute": ("dt.time(9, 20)", "dt.time(9, 30)"),
    "a duration in a class attribute": ("dt.timedelta(minutes=5)", "dt.timedelta(minutes=6)"),
    "a time window in a class attribute": ('"09:20", "15:00"', '"09:30", "15:00"'),
    "a weekday rule in a class attribute": ("bt.weekday(0, 1)", "bt.weekday(0, 2)"),
}

#: Edits of `CELL` that change no compiled code: (the text replaced, its replacement).
KEPT = {
    "the class moved down its cell": ("\nimport functools", "\n\n\n# notes\nimport functools"),
    "a comment added in a method": (
        "            self.buy(10)",
        "            # the entry\n            self.buy(10)",
    ),
}


def edited(edit: tuple[str, str]) -> str:
    """`CELL` with one edit made (each replaced text stands once in it)."""
    old, new = edit
    assert CELL.count(old) == 1, old
    return CELL.replace(old, new)


def with_member(member: str, source: str = CELL) -> str:
    """`source` with `member`, a line of a class body, added to `Cell` after its
    declarations."""
    declarations = '    symbol, exchange, interval = "SBIN", "NSE", "1m"\n'
    assert source.count(declarations) == 1
    return source.replace(declarations, f"{declarations}    {member}\n")


def from_cell(source: str, name: str = "Cell") -> Any:
    """The class `name` a notebook cell of `source` defines (module docstring)."""
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec(compile(source, "<notebook cell>", "exec", dont_inherit=True), namespace)
    return namespace[name]


__all__ = ["CELL", "EDITS", "KEPT", "edited", "from_cell", "with_member"]
