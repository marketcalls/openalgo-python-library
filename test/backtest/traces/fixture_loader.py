"""Read the trace fixture files and resolve bars that are derived from another case.

A case either carries its own base rows or names another case of the same file in
`bars.derived_from_case`. A derived case gets the source rows aggregated to its own
`bars.interval`, so two variants of one market (a 1-minute and a 15-minute base) cannot
drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bars import resample, rows_from_bars
from .timegrid import Case

TRACES_DIR = Path(__file__).resolve().parent
TRACE_FILE_PATTERN = "trace-*.json"
SPEC_REVIEW_FILE = "spec_review.json"

TraceFile = dict[str, Any]


def read_json(path: Path) -> Any:
    """Parse one JSON file. The fixtures are ASCII, so that is the encoding required."""
    with path.open(encoding="ascii") as handle:
        return json.load(handle)


def trace_paths() -> list[Path]:
    """Every trace fixture file, in file-name order."""
    return sorted(TRACES_DIR.glob(TRACE_FILE_PATTERN))


def load_trace_files() -> list[TraceFile]:
    """Every trace file with derived bars resolved, in file-name order."""
    files = [read_json(path) for path in trace_paths()]
    for trace_file in files:
        resolve_derived_bars(trace_file)
    return files


def load_cases() -> list[Case]:
    """Every case of every trace file, in file and case order."""
    return [case for trace_file in load_trace_files() for case in trace_file["cases"]]


def load_spec_review() -> dict[str, Any]:
    """The specification review: the points the documents do not settle."""
    return read_json(TRACES_DIR / SPEC_REVIEW_FILE)


def resolve_derived_bars(trace_file: TraceFile) -> None:
    """Fill `bars.rows` of every derived case from its source case, in place.

    Raises ValueError when the source is missing, is itself derived, or when a case
    carries both rows and a source.
    """
    by_id = {case["id"]: case for case in trace_file["cases"]}
    for case in trace_file["cases"]:
        bars = case["bars"]
        source_id = bars["derived_from_case"]
        if source_id is None:
            if bars["rows"] is None:
                raise ValueError(f"{case['id']}: bars has neither rows nor derived_from_case")
            continue
        if bars["rows"] is not None:
            raise ValueError(f"{case['id']}: bars has both rows and derived_from_case")
        source = by_id.get(source_id)
        if source is None or source["bars"]["derived_from_case"] is not None:
            raise ValueError(
                f"{case['id']}: derived_from_case {source_id!r} is not a case with rows"
            )
        bars["rows"] = {
            instrument: rows_from_bars(resample(case, instrument, rows, bars["interval"]))
            for instrument, rows in source["bars"]["rows"].items()
        }
