"""The Python examples of the backtesting docs run, and print what the docs say they print.

Every fenced ```python block under `trading-platform/python/backtesting/` of the docs
repository (OPENALGO_DOCS_DIR) runs; see `docs_examples` for the markers and how a
script runs. A block marked `<!-- requires-server -->` runs against the OpenAlgo server
at the host the page names when OPENALGO_API_KEY is set, and is skipped otherwise; a
block marked `<!-- requires-network -->` runs only with OPENALGO_DOCS_NETWORK=1 (and
the Python in OPENALGO_DOCS_PYTHON, when set). Without OPENALGO_DOCS_DIR these tests
are skipped.

    OPENALGO_DOCS_DIR=../openalgo-docs pytest test/docs -q
"""

from pathlib import Path

import pytest
from docs_examples import DOCS_ENV, check, collect, docs_dir, skip_reason

SECTION = Path("trading-platform", "python", "backtesting")

DOCS = docs_dir()
if DOCS is None:
    pytest.skip(f"set {DOCS_ENV} to a checkout of openalgo-docs", allow_module_level=True)
EXAMPLES = collect(DOCS, SECTION)


def test_the_section_has_runnable_examples():
    assert EXAMPLES, f"no runnable Python block found under {DOCS / SECTION}"


@pytest.mark.parametrize("example", EXAMPLES, ids=[example.name for example in EXAMPLES])
def test_the_example_runs_and_prints_what_the_page_shows(example, tmp_path):
    reason = skip_reason(example)
    if reason is not None:
        pytest.skip(reason)
    check(example, tmp_path)
