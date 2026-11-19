"""Golden tearsheet pages: the default page of three seeded runs (`tearsheet_runs`) in the
four report modes, pinned whole in `golden/tearsheets/<run>.<mode>.html` and compared
byte for byte, so any change to the default page shows as a diff.

The pages are regression pins, made from the tearsheet code as it was before custom
tearsheets were added. When a change to the default page is meant, regenerate them on
purpose, on the reference platform, and review the diff:

    python test/backtest/golden/tearsheets/regenerate.py --write

Platforms: on the reference platform (Windows x86_64) a page must equal its pin byte
for byte. Elsewhere the platform's math library may move the last bit of a figure, so
the page must equal its pin once every number in both is masked, and each number must
match its pinned one within `CROSS_PLATFORM_TOLERANCE` (relative, or absolute near
zero) or one unit of its last printed digit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from golden_runs import on_reference_platform
from tearsheet_runs import (
    MAX_TOTAL_BYTES,
    MODES,
    RUNS,
    default_page,
    page_path,
    pinned,
    read_page,
)

#: As the golden runs' (`test_golden.py`): far above a 1-ulp difference grown through a
#: ratio, far below any real change.
CROSS_PLATFORM_TOLERANCE = 1e-9
#: A number as a page prints it: digits, an optional fraction and exponent.
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")
#: Characters of a differing line shown in a failure message, each side.
SHOWN_CHARACTERS = 160


@pytest.fixture(scope="module")
def results() -> dict[str, object]:
    """Each golden run's result."""
    return {name: run() for name, run in RUNS.items()}


def _last_place(token: str) -> float:
    """One unit of the last printed digit of a number token ("12.35" is 0.01)."""
    mantissa, _, exponent = token.lower().partition("e")
    decimals = len(mantissa.partition(".")[2])
    return 10.0 ** (int(exponent or 0) - decimals)


def _numbers_agree(pinned_token: str, token: str) -> bool:
    a, b = float(pinned_token), float(token)
    allowed = max(
        CROSS_PLATFORM_TOLERANCE * max(abs(a), abs(b)),
        CROSS_PLATFORM_TOLERANCE,
        _last_place(pinned_token),
        _last_place(token),
    )
    return abs(a - b) <= allowed


def _first_difference(golden: str, page: str) -> str:
    """The first line where the two pages differ, both sides shortened."""
    golden_lines, lines = golden.split("\n"), page.split("\n")
    for number, (old, new) in enumerate(zip(golden_lines, lines, strict=False), start=1):
        if old != new:
            at = next(i for i, (x, y) in enumerate(zip(old, new, strict=False)) if x != y)
            start = max(0, at - SHOWN_CHARACTERS // 2)
            return (
                f"line {number}, character {at + 1}: pinned "
                f"{old[start : start + SHOWN_CHARACTERS]!r}, now "
                f"{new[start : start + SHOWN_CHARACTERS]!r}"
            )
    return f"{len(golden_lines)} pinned lines, {len(lines)} now"


def differences(golden: str, page: str, exact: bool) -> str | None:
    """Where `page` departs from its golden page (module docstring), or None."""
    if golden == page:
        return None
    if exact:
        return _first_difference(golden, page)
    if NUMBER.sub("#", golden) != NUMBER.sub("#", page):
        return _first_difference(NUMBER.sub("#", golden), NUMBER.sub("#", page))
    for old, new in zip(NUMBER.findall(golden), NUMBER.findall(page), strict=True):
        if not _numbers_agree(old, new):
            return f"a number moved: pinned {old}, now {new}"
    return None


def assert_golden(name: str, mode: str, page: str, folder: Path) -> None:
    """`page` (as pinned) equals the golden page of run `name` in `mode`; else the page
    is written to `folder` and the first difference named."""
    found = differences(read_page(name, mode), pinned(page), exact=on_reference_platform())
    if found is not None:
        written = folder / page_path(name, mode).name
        written.write_bytes(pinned(page).encode("utf-8"))
        pytest.fail(f"{name}.{mode} differs from its golden page at {found}; new page: {written}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", list(RUNS))
def test_the_default_page_equals_its_golden_page(name, mode, results, tmp_path):
    assert_golden(name, mode, default_page(results[name], mode), tmp_path)


def test_the_golden_pages_stay_small():
    total = sum(page_path(name, mode).stat().st_size for name in RUNS for mode in MODES)
    assert total < MAX_TOTAL_BYTES


@pytest.mark.parametrize("name", list(RUNS))
def test_a_golden_page_is_ascii_with_one_line_ending_kind(name):
    raw = page_path(name, "full").read_bytes()
    assert raw.isascii() and b"\r" not in raw


def test_the_comparison_names_the_first_differing_line():
    golden = "<a>\n<b class='x'>1.25</b>\n<c>"
    found = differences(golden, golden.replace("'x'", "'y'"), exact=True)
    assert found is not None and found.startswith("line 2, character 11")


@pytest.mark.parametrize(
    ("pinned_number", "number", "agree"),
    [
        ("12.35", "12.35", True),
        ("12.35", "12.34", True),  # one unit of the last printed digit
        ("12.35", "12.33", False),
        ("0.30000000000000004", "0.3", True),
        ("1.23457e+05", "1.23458e+05", True),
        ("1.23457e+05", "1.23459e+05", False),
    ],
)
def test_off_the_reference_platform_a_number_may_move_its_last_digit(pinned_number, number, agree):
    golden = f"<b>{pinned_number}</b>"
    assert (differences(golden, f"<b>{number}</b>", exact=False) is None) is agree


def test_off_the_reference_platform_the_text_around_the_numbers_must_not_move():
    assert differences("<b>1.5</b>", "<i>1.5</i>", exact=False) is not None
