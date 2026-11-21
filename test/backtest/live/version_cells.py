"""Strategy classes as a notebook makes them, shared by the tests of the version id of a
class whose source Python cannot find (`test_version_code.py`; 2.4.0 item P5-3).

A notebook runs a cell by compiling its text and running it in its namespace: Python keeps
the classes' compiled code but no source file `inspect` could read. `from_cell` does the
same with a cell's text, compiled with no compiler flag of its caller (`dont_inherit`), so
a cell compiles alike in a test module and in a fresh interpreter. `CELL` is the cell the
tests edit: `Cell` buys 10 SBIN when its symbol is on a list (a set of names, which Python
keeps as a constant whose own order depends on the process's hash seed) and the bar closes
above 500.00; each edit of `EDITS` changes one thing of it.
"""

from typing import Any

#: The cell of `Cell` as first written.
CELL = """
import functools

import openalgo.backtest as bt


def logged(method):
    @functools.wraps(method)
    def wrapper(self, bar):
        return method(self, bar)

    return wrapper


class Cell(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "1m"
    fast = 9
    watch = ("SBIN", "INFY")
    rules = {"stop": bt.points(3)}

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


def from_cell(source: str, name: str = "Cell") -> Any:
    """The class `name` a notebook cell of `source` defines (module docstring)."""
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec(compile(source, "<notebook cell>", "exec", dont_inherit=True), namespace)
    return namespace[name]


__all__ = ["CELL", "EDITS", "KEPT", "edited", "from_cell"]
