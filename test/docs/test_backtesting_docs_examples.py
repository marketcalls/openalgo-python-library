"""The Python examples of the backtesting docs run, and print what the docs say they print.

Every fenced ```python block under `trading-platform/python/backtesting/` of the docs
repository (OPENALGO_DOCS_DIR) runs, except a block marked `<!-- requires-server -->`;
see `docs_examples` for the markers and how a script runs. Without OPENALGO_DOCS_DIR
these tests are skipped.

    OPENALGO_DOCS_DIR=../openalgo-docs pytest test/docs -q
"""

from pathlib import Path

import pytest
from docs_examples import DOCS_ENV, check, collect, docs_dir

SECTION = Path("trading-platform", "python", "backtesting")

DOCS = docs_dir()
if DOCS is None:
    pytest.skip(f"set {DOCS_ENV} to a checkout of openalgo-docs", allow_module_level=True)
EXAMPLES = collect(DOCS, SECTION)


def test_the_section_has_runnable_examples():
    assert EXAMPLES, f"no runnable Python block found under {DOCS / SECTION}"


@pytest.mark.parametrize("example", EXAMPLES, ids=[example.name for example in EXAMPLES])
def test_the_example_runs_and_prints_what_the_page_shows(example, tmp_path):
    check(example, tmp_path)
