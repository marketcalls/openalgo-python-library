"""The self-contained examples of the indicator docs run, and print what the docs show.

Most blocks of the indicator pages under `trading-platform/python/indicators/` read
`df` from an OpenAlgo server, so this section is opt-in: a ```python block runs only
when it is marked `<!-- runnable -->` (or `<!-- continues -->` after one); see
`docs_examples` for the markers and how a script runs. Without OPENALGO_DOCS_DIR these
tests are skipped.

    OPENALGO_DOCS_DIR=../openalgo-docs pytest test/docs -q
"""

from pathlib import Path

import pytest
from docs_examples import DOCS_ENV, check, collect, docs_dir, skip_reason

SECTION = Path("trading-platform", "python", "indicators")

DOCS = docs_dir()
if DOCS is None:
    pytest.skip(f"set {DOCS_ENV} to a checkout of openalgo-docs", allow_module_level=True)
EXAMPLES = collect(DOCS, SECTION, opt_in=True)


def test_the_section_has_runnable_examples():
    assert EXAMPLES, f"no block marked runnable under {DOCS / SECTION}"


@pytest.mark.parametrize("example", EXAMPLES, ids=[example.name for example in EXAMPLES])
def test_the_example_runs_and_prints_what_the_page_shows(example, tmp_path):
    reason = skip_reason(example)
    if reason is not None:
        pytest.skip(reason)
    check(example, tmp_path)
