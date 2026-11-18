"""The live layer's modules keep to the package's size rule (CLAUDE.md, "Code standards
for the backtest package": split a file before it grows past roughly 400 lines;
16-session-design section 8: every module stays under about 400 lines). Counted as the
file's lines, docstrings and comments included.
"""

from __future__ import annotations

import pathlib

import pytest

import openalgo.backtest._live as live_package

#: The most lines a module of the live layer may have.
MODULE_LINES_MAX = 400
MODULES = sorted(pathlib.Path(live_package.__file__).parent.glob("*.py"))


@pytest.mark.parametrize("module", MODULES, ids=[module.name for module in MODULES])
def test_a_live_module_has_400_lines_or_fewer(module):
    lines = len(module.read_text(encoding="utf-8").splitlines())
    assert lines <= MODULE_LINES_MAX, f"{module.name} has {lines} lines: split it"
