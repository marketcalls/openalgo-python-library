"""Extract the Python examples of a docs section, run them, and compare what they print.

Shared by the tests in this folder. The public docs live in a separate repository
(openalgo-docs, published at https://docs.openalgo.in); the tests find it through the
environment variable OPENALGO_DOCS_DIR and are skipped without it.

Markers are HTML comments on the lines just before a fenced block (GitBook does not
render them):

    <!-- requires-server -->   the ```python block needs an OpenAlgo server; it runs only
                               when OPENALGO_API_KEY is set, with the literal
                               "your_api_key" read from that variable at run time
    <!-- requires-network -->  the block downloads data from a network data source; it
                               runs only when OPENALGO_DOCS_NETWORK=1, with the Python
                               named by OPENALGO_DOCS_PYTHON when set (an interpreter
                               that has the extra package the page installs)
    <!-- output-varies -->     the block runs, but what it prints is not compared (its
                               output depends on the day it runs)
    <!-- runnable -->          in a section collected with `opt_in=True`, the block runs;
                               other blocks of such a section are not run
    <!-- continues -->         the block runs after the runnable blocks before it on the
                               same page, in the same script, and shares their needs
    <!-- output -->            the ```text block right after a runnable code block is what
                               that block prints

Each script runs as a file in a fresh interpreter, in an empty temporary folder, with
a timeout, the way a reader runs a copied example (a file, so that `bt.optimize` worker
processes can import it). Printed output is compared line by line, ignoring trailing
spaces. The API key is never written to a file: the script reads it from the
environment of the interpreter that runs it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DOCS_ENV = "OPENALGO_DOCS_DIR"
API_KEY_ENV = "OPENALGO_API_KEY"
NETWORK_ENV = "OPENALGO_DOCS_NETWORK"
PYTHON_ENV = "OPENALGO_DOCS_PYTHON"
REQUIRES_SERVER = "<!-- requires-server -->"
REQUIRES_NETWORK = "<!-- requires-network -->"
OUTPUT_VARIES = "<!-- output-varies -->"
RUNNABLE = "<!-- runnable -->"
CONTINUES = "<!-- continues -->"
OUTPUT = "<!-- output -->"
MARKERS = (REQUIRES_SERVER, REQUIRES_NETWORK, OUTPUT_VARIES, RUNNABLE, CONTINUES, OUTPUT)
# The markers that say what a block needs before it can run.
NEEDS = (REQUIRES_SERVER, REQUIRES_NETWORK)
# The placeholder the docs show in `api(api_key=...)`, and the expression that replaces it
# in a script that runs against a server: the key stays in the environment.
API_KEY_PLACEHOLDER = '"your_api_key"'
API_KEY_FROM_ENVIRONMENT = f'__import__("os").environ["{API_KEY_ENV}"]'
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
    documented to print (None when the page shows no output for it, or when the block
    is marked output-varies) and what the script needs to run (requires-server,
    requires-network)."""

    page: str
    line: int
    blocks: list[str] = field(default_factory=list)
    outputs: list[str | None] = field(default_factory=list)
    needs: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return f"{self.page}:{self.line}"

    def script(self) -> str:
        # Guarded: `bt.optimize` workers import the script under another name, and their
        # top-level code must not print markers into the shared output.
        marker = f'\nif __name__ == "__main__":\n    print({SEPARATOR!r}, flush=True)\n'
        code = marker.join(self.blocks) + marker
        if REQUIRES_SERVER in self.needs:
            code = code.replace(API_KEY_PLACEHOLDER, API_KEY_FROM_ENVIRONMENT)
        return code


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
    if block.language != "python":
        return False
    return not opt_in or bool(block.markers & {RUNNABLE, CONTINUES})


def page_examples(page: Path, docs: Path, opt_in: bool = False) -> list[Example]:
    """The runnable scripts of one page. With `opt_in`, only blocks marked runnable
    (and the blocks that continue them) run."""
    name = page.relative_to(docs).as_posix()
    examples: list[Example] = []
    previous_ran = False
    previous_varies = False
    for block in blocks(page.read_text(encoding="utf-8")):
        if block.language == "text" and OUTPUT in block.markers:
            if not previous_ran:
                raise AssertionError(f"{name}:{block.line}: an output block must follow run code")
            if not previous_varies:
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
        examples[-1].needs |= block.markers & set(NEEDS)
        previous_ran = True
        previous_varies = OUTPUT_VARIES in block.markers
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


def skip_reason(example: Example, environ: Mapping[str, str] = os.environ) -> str | None:
    """Why `example` cannot run in an environment, or None when it can: a block that
    needs a server needs OPENALGO_API_KEY, one that needs the network needs
    OPENALGO_DOCS_NETWORK=1."""
    if REQUIRES_SERVER in example.needs and not environ.get(API_KEY_ENV):
        return f"needs an OpenAlgo server: set {API_KEY_ENV} to run it"
    if REQUIRES_NETWORK in example.needs and environ.get(NETWORK_ENV) != "1":
        return f"downloads from a network data source: set {NETWORK_ENV}=1 to run it"
    return None


def interpreter(example: Example, environ: Mapping[str, str] = os.environ) -> str:
    """The Python that runs `example`: OPENALGO_DOCS_PYTHON for a block that needs the
    network when it is set, otherwise the Python running the tests."""
    if REQUIRES_NETWORK in example.needs and environ.get(PYTHON_ENV):
        return environ[PYTHON_ENV]
    return sys.executable


def lines(text: str) -> list[str]:
    """The lines of an output, without trailing spaces and surrounding blank lines."""
    return [line.rstrip() for line in text.strip("\n").splitlines()]


def run(example: Example, folder: Path, environ: Mapping[str, str] = os.environ) -> list[str]:
    """Run an example as a file in `folder`; the output of each block. Raises
    AssertionError with the script's output when it fails."""
    # `environ` holds the API key: pytest leaves this frame, and its arguments, out of
    # a failure report.
    __tracebackhide__ = True
    script = folder / "example.py"
    script.write_text(example.script(), encoding="utf-8")
    environment = {**environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    finished = subprocess.run(
        [interpreter(example, environ), script.name],
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


def check(example: Example, folder: Path, environ: Mapping[str, str] = os.environ) -> None:
    """Run an example and compare each documented output with what it prints."""
    __tracebackhide__ = True  # as in `run`: keep the environment out of failure reports
    printed = run(example, folder, environ)
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
    "API_KEY_ENV",
    "CONTINUES",
    "DOCS_ENV",
    "NETWORK_ENV",
    "OUTPUT",
    "OUTPUT_VARIES",
    "PYTHON_ENV",
    "REQUIRES_NETWORK",
    "REQUIRES_SERVER",
    "RUNNABLE",
    "Example",
    "blocks",
    "check",
    "collect",
    "docs_dir",
    "interpreter",
    "lines",
    "page_examples",
    "run",
    "skip_reason",
]
