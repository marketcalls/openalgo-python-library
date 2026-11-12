"""How the docs example runner treats the markers of a page.

These tests need neither the docs repository nor a server: each writes a small page to
a temporary folder and collects, skips or runs its blocks.
"""

import os
import sys
from pathlib import Path

import pytest
from docs_examples import (
    API_KEY_ENV,
    NETWORK_ENV,
    PYTHON_ENV,
    REQUIRES_NETWORK,
    REQUIRES_SERVER,
    check,
    interpreter,
    page_examples,
    run,
    skip_reason,
)

# A made-up key: the tests check that it reaches the script only through the environment.
FAKE_KEY = "0123456789abcdef" * 4
SERVER_PAGE = """\
<!-- requires-server -->
```python
key = "your_api_key"
print(len(key), key[:4])
```

<!-- output -->
```text
64 0123
```
"""


def examples_of(page_text, folder):
    page = folder / "page.md"
    page.write_text(page_text, encoding="utf-8")
    return page_examples(page, folder)


def test_a_block_without_markers_runs_and_needs_nothing(tmp_path):
    (example,) = examples_of('```python\nprint("hello")\n```\n', tmp_path)
    assert example.needs == set()
    assert skip_reason(example, environ={}) is None


def test_a_block_marked_requires_server_is_collected_as_needing_the_server(tmp_path):
    (example,) = examples_of(SERVER_PAGE, tmp_path)
    assert example.needs == {REQUIRES_SERVER}


def test_a_server_example_is_skipped_without_an_api_key(tmp_path):
    (example,) = examples_of(SERVER_PAGE, tmp_path)
    assert API_KEY_ENV in skip_reason(example, environ={})


def test_a_server_example_runs_when_an_api_key_is_set(tmp_path):
    (example,) = examples_of(SERVER_PAGE, tmp_path)
    assert skip_reason(example, environ={API_KEY_ENV: FAKE_KEY}) is None


def test_the_api_key_placeholder_is_read_from_the_environment_and_never_written(tmp_path):
    (example,) = examples_of(SERVER_PAGE, tmp_path)
    check(example, tmp_path, environ={**os.environ, API_KEY_ENV: FAKE_KEY})
    assert FAKE_KEY not in (tmp_path / "example.py").read_text(encoding="utf-8")


def test_a_failing_server_example_never_shows_the_api_key_in_the_report(tmp_path):
    page = "<!-- requires-server -->\n```python\nraise SystemExit(1)\n```\n"
    (example,) = examples_of(page, tmp_path)
    with pytest.raises(AssertionError) as failure:
        check(example, tmp_path, environ={**os.environ, API_KEY_ENV: FAKE_KEY})
    # The most verbose report pytest can print: arguments and locals, nothing shortened.
    report = failure.getrepr(
        showlocals=True, funcargs=True, truncate_locals=False, truncate_args=False
    )
    assert FAKE_KEY not in str(report)


def test_a_block_that_continues_a_server_block_needs_the_server_too(tmp_path):
    page = SERVER_PAGE + '\n<!-- continues -->\n```python\nprint("more")\n```\n'
    (example,) = examples_of(page, tmp_path)
    assert len(example.blocks) == 2
    assert example.needs == {REQUIRES_SERVER}


def test_an_output_varies_block_runs_but_its_output_is_not_compared(tmp_path):
    page = """\
<!-- output-varies -->
```python
print("printed today")
```

<!-- output -->
```text
printed on the day the page was written
```
"""
    (example,) = examples_of(page, tmp_path)
    assert example.outputs == [None]
    assert run(example, tmp_path) == ["printed today\n"]


@pytest.mark.parametrize(
    "markers",
    [
        "<!-- output-varies -->\n<!-- requires-server -->",
        "<!-- requires-server -->\n<!-- output-varies -->",
    ],
    ids=["varies-first", "server-first"],
)
def test_the_markers_before_a_block_apply_in_any_order(tmp_path, markers):
    page = (
        f'{markers}\n```python\nprint("today")\n```\n\n<!-- output -->\n```text\nyesterday\n```\n'
    )
    (example,) = examples_of(page, tmp_path)
    assert example.needs == {REQUIRES_SERVER}
    assert example.outputs == [None]


def test_an_output_varies_block_that_fails_still_fails_the_check(tmp_path):
    page = "<!-- output-varies -->\n```python\nraise SystemExit(3)\n```\n"
    (example,) = examples_of(page, tmp_path)
    with pytest.raises(AssertionError, match="exit code 3"):
        check(example, tmp_path)


NETWORK_PAGE = '<!-- requires-network -->\n```python\nprint("downloaded")\n```\n'


@pytest.mark.parametrize(
    ("flag", "runs"),
    [(None, False), ("0", False), ("1", True)],
    ids=["unset", "zero", "one"],
)
def test_a_network_example_runs_only_when_the_network_flag_is_one(tmp_path, flag, runs):
    (example,) = examples_of(NETWORK_PAGE, tmp_path)
    environ = {} if flag is None else {NETWORK_ENV: flag}
    assert example.needs == {REQUIRES_NETWORK}
    assert (skip_reason(example, environ=environ) is None) is runs


def test_a_network_example_runs_with_the_python_named_in_the_environment(tmp_path):
    (example,) = examples_of(NETWORK_PAGE, tmp_path)
    other_python = str(Path("elsewhere", "python.exe"))
    assert interpreter(example, environ={PYTHON_ENV: other_python}) == other_python
    assert interpreter(example, environ={}) == sys.executable


def test_an_example_without_the_network_marker_keeps_the_test_python(tmp_path):
    (example,) = examples_of(SERVER_PAGE, tmp_path)
    assert interpreter(example, environ={PYTHON_ENV: "elsewhere"}) == sys.executable
