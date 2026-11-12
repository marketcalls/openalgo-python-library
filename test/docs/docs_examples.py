"""Extract the Python examples of a docs section, run them, and compare what they print.

Shared by the tests in this folder. The public docs live in a separate repository
(openalgo-docs, published at https://docs.openalgo.in); the tests find it through the
environment variable OPENALGO_DOCS_DIR and are skipped without it.

Markers are HTML comments on the lines just before a fenced block (GitBook does not
render them):

    <!-- requires-server -->  the ```python block needs an OpenAlgo server or an API key
                              and is never run
    <!-- runnable -->         in a section collected with `opt_in=True`, the block runs;
                              other blocks of such a section are not run
    <!-- continues -->        the block runs after the runnable blocks before it on the
                              same page, in the same script
    <!-- output -->           the ```text block right after a runnable code block is what
                              that block prints

Each script runs as a file in a fresh interpreter, in an empty temporary folder, with
a timeout, the way a reader runs a copied example (a file, so that `bt.optimize` worker
processes can import it). Printed output is compared line by line, ignoring trailing
spaces.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DOCS_ENV = "OPENALGO_DOCS_DIR"
REQUIRES_SERVER = "<!-- requires-server -->"
RUNNABLE = "<!-- runnable -->"
CONTINUES = "<!-- continues -->"
OUTPUT = "<!-- output -->"
MARKERS = (REQUIRES_SERVER, RUNNABLE, CONTINUES, OUTPUT)
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


def blocks(text: str) -> list[Block]:
    """Every fenced block of a Markdown page, in order."""
    found: list[Block] = []
    lines = text.splitlines()
    markers: set[str] = set()
    index = 0
    while index < len(lines):
        opening = FENCE.match(lines[index])
        if opening is None:
            stripped = lines[index].strip()
            if stripped in MARKERS:
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
        found.append(Block(opening.group(1), "\n".join(body) + "\n", markers, start + 1))
        markers = set()
        index += 1
    return found


def _runs(block: Block, opt_in: bool) -> bool:
    if block.language != "python" or REQUIRES_SERVER in block.markers:
        return False
    return not opt_in or bool(block.markers & {RUNNABLE, CONTINUES})


def page_examples(page: Path, docs: Path, opt_in: bool = False) -> list[Example]:
    """The runnable scripts of one page. With `opt_in`, only blocks marked runnable
    (and the blocks that continue them) run."""
    name = page.relative_to(docs).as_posix()
    examples: list[Example] = []
    previous_ran = False
    for block in blocks(page.read_text(encoding="utf-8")):
        if block.language == "text" and OUTPUT in block.markers:
            if not previous_ran:
                raise AssertionError(f"{name}:{block.line}: an output block must follow run code")
            examples[-1].outputs[-1] = block.code
            previous_ran = False
            continue
        previous_ran = False
        if not _runs(block, opt_in):
            continue
        if CONTINUES not in block.markers or not examples:
            examples.append(Example(name, block.line))
        examples[-1].blocks.append(block.code)
        examples[-1].outputs.append(None)
        previous_ran = True
    return examples


def collect(docs: Path, section: Path, opt_in: bool = False) -> list[Example]:
    """Every runnable script of one section of the docs at `docs`."""
    folder = docs / section
    if not folder.is_dir():
        raise AssertionError(f"{folder} does not exist: is {DOCS_ENV} the docs repository?")
    return [
        example
        for page in sorted(folder.rglob("*.md"))
        for example in page_examples(page, docs, opt_in)
    ]


def lines(text: str) -> list[str]:
    """The lines of an output, without trailing spaces and surrounding blank lines."""
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


def check(example: Example, folder: Path) -> None:
    """Run an example and compare each documented output with what it prints."""
    printed = run(example, folder)
    for number, (expected, actual) in enumerate(zip(example.outputs, printed, strict=True)):
        if expected is None:
            continue
        assert lines(actual) == lines(expected), (
            f"{example.name}, block {number + 1}: the page shows\n{expected}\n"
            f"but the code prints\n{actual}"
        )


def docs_dir() -> Path | None:
    """The docs repository named by OPENALGO_DOCS_DIR, or None."""
    value = os.environ.get(DOCS_ENV)
    return Path(value).resolve() if value else None


__all__ = [
    "CONTINUES",
    "DOCS_ENV",
    "OUTPUT",
    "REQUIRES_SERVER",
    "RUNNABLE",
    "Example",
    "blocks",
    "check",
    "collect",
    "docs_dir",
    "lines",
    "page_examples",
    "run",
]
