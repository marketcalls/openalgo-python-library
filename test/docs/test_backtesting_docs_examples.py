"""The Python examples of the backtesting docs run, and print what the docs say they print.

The public docs live in a separate repository (openalgo-docs, published at
https://docs.openalgo.in). Point the environment variable OPENALGO_DOCS_DIR at a
checkout of it; without it these tests are skipped.

Every fenced ```python block under `trading-platform/python/backtesting/` is extracted
and run, except a block preceded by the HTML comment `<!-- requires-server -->` (it
needs a running OpenAlgo server or an API key). A block preceded by
`<!-- continues -->` runs after the runnable blocks before it on the same page, in the
same script; any other block starts a script of its own. Each script runs as a file in
a fresh interpreter, in an empty temporary folder, with a timeout, the way a reader runs
a copied example (a file, so that `bt.optimize` worker processes can import it).

A ```text block preceded by `<!-- output -->` right after a code block is what that
block prints: the test compares it with the real output, line by line, ignoring
trailing spaces.

    OPENALGO_DOCS_DIR=../openalgo-docs pytest test/docs -q
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

DOCS_ENV = "OPENALGO_DOCS_DIR"
SECTION = Path("trading-platform", "python", "backtesting")
REQUIRES_SERVER = "<!-- requires-server -->"
CONTINUES = "<!-- continues -->"
OUTPUT = "<!-- output -->"
TIMEOUT_SECONDS = 600
ROOT = Path(__file__).resolve().parents[2]
# The marker printed between the blocks of one script, to split its output by block.
SEPARATOR = "----- end of documented block -----"
FENCE = re.compile(r"^```(\w*)\s*$")


@dataclass
class Block:
    """One fenced block of a page: its language, its code, the comment markers on the
    lines just before it, and the line it starts on."""

    language: str
    code: str
    markers: set[str]
    line: int


@dataclass
class Example:
    """A script made of one or more runnable blocks of a page, with what each block is
    documented to print (None when the page shows no output for it)."""

    page: str
    line: int
    blocks: list[str] = field(default_factory=list)
    outputs: list[str | None] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.page}:{self.line}"

    def script(self) -> str:
        # Guarded: `bt.optimize` workers import the script under another name, and their
        # top-level code must not print markers into the shared output.
        marker = f'\nif __name__ == "__main__":\n    print({SEPARATOR!r}, flush=True)\n'
        return marker.join(self.blocks) + marker


def _blocks(text: str) -> list[Block]:
    """Every fenced block of a Markdown page, in order."""
    blocks: list[Block] = []
    lines = text.splitlines()
    markers: set[str] = set()
    index = 0
    while index < len(lines):
        opening = FENCE.match(lines[index])
        if opening is None:
            stripped = lines[index].strip()
            if stripped in (REQUIRES_SERVER, CONTINUES, OUTPUT):
                markers.add(stripped)
            elif stripped:
                markers = set()
            index += 1
            continue
        start = index
        index += 1
        body = []
        while index < len(lines) and not lines[index].startswith("```"):
            body.append(lines[index])
            index += 1
        blocks.append(Block(opening.group(1), "\n".join(body) + "\n", markers, start + 1))
        markers = set()
        index += 1
    return blocks


def _examples(page: Path, docs: Path) -> list[Example]:
    """The runnable scripts of one page."""
    name = page.relative_to(docs).as_posix()
    examples: list[Example] = []
    previous_was_code = False
    for block in _blocks(page.read_text(encoding="utf-8")):
        if block.language == "text" and OUTPUT in block.markers:
            if not previous_was_code:
                raise AssertionError(f"{name}:{block.line}: an output block must follow code")
            examples[-1].outputs[-1] = block.code
            previous_was_code = False
            continue
        previous_was_code = False
        if block.language != "python" or REQUIRES_SERVER in block.markers:
            continue
        if CONTINUES not in block.markers or not examples:
            examples.append(Example(name, block.line))
        examples[-1].blocks.append(block.code)
        examples[-1].outputs.append(None)
        previous_was_code = True
    return examples


def collect(docs: Path) -> list[Example]:
    """Every runnable script of the backtesting section of the docs at `docs`."""
    section = docs / SECTION
    if not section.is_dir():
        raise AssertionError(f"{section} does not exist: is {DOCS_ENV} the docs repository?")
    return [example for page in sorted(section.rglob("*.md")) for example in _examples(page, docs)]


def _lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.strip("\n").splitlines()]


def run(example: Example, folder: Path) -> list[str]:
    """Run an example as a file in `folder`; the output of each block. Raises
    AssertionError with the script's output when it fails."""
    script = folder / "example.py"
    script.write_text(example.script(), encoding="utf-8")
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    finished = subprocess.run(
        [sys.executable, script.name],
        cwd=folder,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=TIMEOUT_SECONDS,
        env=environment,
    )
    if finished.returncode != 0:
        raise AssertionError(
            f"{example.name} failed with exit code {finished.returncode}\n"
            f"--- stdout\n{finished.stdout}\n--- stderr\n{finished.stderr}"
        )
    parts = finished.stdout.split(SEPARATOR + "\n")
    return parts[: len(example.blocks)]


def _docs_dir() -> Path | None:
    value = os.environ.get(DOCS_ENV)
    return Path(value).resolve() if value else None


DOCS = _docs_dir()
if DOCS is None:
    pytest.skip(f"set {DOCS_ENV} to a checkout of openalgo-docs", allow_module_level=True)
EXAMPLES = collect(DOCS)


def test_the_section_has_runnable_examples():
    assert EXAMPLES, f"no runnable Python block found under {DOCS / SECTION}"


@pytest.mark.parametrize("example", EXAMPLES, ids=[example.name for example in EXAMPLES])
def test_the_example_runs_and_prints_what_the_page_shows(example, tmp_path):
    printed = run(example, tmp_path)
    for number, (expected, actual) in enumerate(zip(example.outputs, printed, strict=True)):
        if expected is None:
            continue
        assert _lines(actual) == _lines(expected), (
            f"{example.name}, block {number + 1}: the page shows\n{expected}\n"
            f"but the code prints\n{actual}"
        )
