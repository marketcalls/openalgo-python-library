"""Check the shape of a trace file and of a case's inputs against the fixture schema.

Every function returns a list of violations (empty when the object is well formed).
Each violation names the fixture and, where there is one, the slice. The expected
section of a case is checked by `schema_expected`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any

from . import vocabulary as vocab
from .timegrid import Case, interval_step

Violations = list[str]

INSTRUMENT_TYPES = ("equity", "future")
PRODUCTS = ("CNC", "MIS", "NRML")
ROLL_KEYS = ("sessions_before_expiry", "chain")
# Schema kinds whose objects may carry a depends_on tag.
TAGGABLE = (
    "expected_decision",
    "fill",
    "level_row",
    "order_event",
    "position_event",
    "risk_event",
    "final_position",
    "gross_pnl",
)


def key_violations(obj: Any, kind: str, where: str) -> Violations:
    """Missing and unknown keys of one object of a schema kind."""
    if not isinstance(obj, dict):
        return [f"{where}: {kind} must be an object, found {type(obj).__name__}"]
    required = set(vocab.KEYS[kind])
    allowed = required | ({vocab.DEPENDS_ON} if kind in TAGGABLE else set())
    found = set(obj)
    problems = [f"{where}: {kind} lacks the key {key!r}" for key in sorted(required - found)]
    problems += [f"{where}: {kind} has the unknown key {key!r}" for key in sorted(found - allowed)]
    return problems


def member_violation(value: Any, allowed: Collection[Any], what: str, where: str) -> Violations:
    """A violation unless `value` is one of the allowed values."""
    if value in allowed:
        return []
    return [f"{where}: {what} is {value!r}, expected one of {sorted(map(str, allowed))}"]


def each(items: Iterable[Any], kind: str, where: str) -> Violations:
    """Key violations for every object of a list, numbered from zero."""
    problems: Violations = []
    for index, item in enumerate(items):
        problems += key_violations(item, kind, f"{where}[{index}]")
    return problems


def validate_file(trace_file: dict[str, Any], file_name: str) -> Violations:
    """Shape of one trace file: its header and the uniqueness of its case ids."""
    problems = key_violations(trace_file, "file", file_name)
    if problems:
        return problems
    problems += member_violation(trace_file["format"], (vocab.FORMAT,), "format", file_name)
    if not isinstance(trace_file["roadmap_trace"], int):
        problems.append(f"{file_name}: roadmap_trace must be the trace number of the roadmap")
    case_ids = [case.get("id") for case in trace_file["cases"]]
    if not case_ids:
        problems.append(f"{file_name}: a trace file needs at least one case")
    if len(set(case_ids)) != len(case_ids):
        problems.append(f"{file_name}: case ids are not unique: {case_ids}")
    for case_id in case_ids:
        if not str(case_id).startswith(trace_file["id"]):
            problems.append(f"{file_name}: case id {case_id!r} does not start with the trace id")
    return problems


def validate_case_inputs(case: Case) -> Violations:
    """Shape of everything in a case except its expected section."""
    name = str(case.get("id", "<case without id>"))
    problems = key_violations(case, "case", name)
    if problems:
        return problems
    problems += _instrument_violations(case, name)
    problems += _calendar_violations(case, name)
    problems += _settings_violations(case, name)
    problems += _bars_violations(case, name)
    problems += _strategy_violations(case, name)
    problems += _derivation_violations(case, name)
    return problems


def _instrument_violations(case: Case, name: str) -> Violations:
    problems = each(case["instruments"], "instrument", f"{name} instruments")
    if problems:
        return problems
    for instrument in case["instruments"]:
        where = f"{name} instrument {instrument['symbol']}"
        problems += member_violation(instrument["type"], INSTRUMENT_TYPES, "type", where)
        problems += member_violation(instrument["product"], PRODUCTS, "product", where)
        if not instrument["tick_size_rupees"] > 0 or not instrument["lot_size_units"] > 0:
            problems.append(f"{where}: tick size and lot size must be positive")
        if (instrument["type"] == "future") != (instrument["expiry_date"] is not None):
            problems.append(f"{where}: a future has an expiry date and an equity has none")
    return problems


def _calendar_violations(case: Case, name: str) -> Violations:
    problems: Violations = []
    for instrument in case["instruments"]:
        if instrument["exchange"] not in case["calendar"]:
            problems.append(f"{name}: no calendar for exchange {instrument['exchange']}")
    for exchange, rows in case["calendar"].items():
        problems += each(rows, "session", f"{name} calendar {exchange}")
    return problems


def _settings_violations(case: Case, name: str) -> Violations:
    settings = case["settings"]
    problems = key_violations(settings, "settings", f"{name} settings")
    if problems:
        return problems
    for key in ("strategy_interval", "base"):
        try:
            interval_step(settings[key])
        except ValueError as error:
            problems.append(f"{name} settings: {key}: {error}")
    known = set(vocab.KEYS["settings"]) | {"product"}
    for key in settings["documented_defaults"]:
        if key not in known:
            problems.append(f"{name} settings: documented_defaults names the unknown key {key!r}")
    roll = settings["roll"]
    if roll is not None and set(roll) != set(ROLL_KEYS):
        problems.append(f"{name} settings: roll must have exactly the keys {ROLL_KEYS}")
    return problems


def _bars_violations(case: Case, name: str) -> Violations:
    bars = case["bars"]
    problems = key_violations(bars, "bars", f"{name} bars")
    if problems:
        return problems
    if bars["columns"] != vocab.BAR_COLUMNS:
        problems.append(f"{name} bars: columns must be {vocab.BAR_COLUMNS}")
    if bars["interval"] != case["settings"]["base"]:
        problems.append(f"{name} bars: interval must equal settings.base, the base interval")
    declared = {instrument["symbol"] for instrument in case["instruments"]}
    if set(bars["rows"]) != declared:
        problems.append(f"{name} bars: rows must cover exactly the declared instruments")
    for instrument, rows in bars["rows"].items():
        for row in rows:
            if len(row) != len(vocab.BAR_COLUMNS):
                problems.append(f"{name} {row[0]}: bar row of {instrument} has {len(row)} values")
    problems += each(case["missing_slices"], "missing_slice", f"{name} missing_slices")
    return problems


def _strategy_violations(case: Case, name: str) -> Violations:
    strategy = case["strategy"]
    problems = key_violations(strategy, "strategy", f"{name} strategy")
    if problems:
        return problems
    problems += each(strategy["decisions"], "decision", f"{name} strategy.decisions")
    if problems:
        return problems
    for index, decision in enumerate(strategy["decisions"]):
        where = f"{name} {decision['bar_open_ist']}: decision {index}"
        problems += member_violation(
            decision["if_position"], vocab.IF_POSITION, "if_position", where
        )
        problems += key_violations(decision["call"], "call", where)
        method = decision["call"].get("method")
        problems += member_violation(method, vocab.ORDER_METHODS, "call.method", where)
    problems += _conditions_violations(strategy["conditions"], name)
    return problems


def _conditions_violations(conditions: dict[str, Any] | None, name: str) -> Violations:
    if conditions is None:
        return []
    where = f"{name} strategy.conditions"
    problems = key_violations(conditions, "conditions", where)
    if problems:
        return problems
    bar_count = len(conditions["bar_opens_ist"])
    for key in ("entry_when", "exit_when", "short_when", "cover_when"):
        condition = conditions[key]
        if condition is not None and len(condition["values"]) != bar_count:
            problems.append(
                f"{where}: {key}.values needs one value for each of the {bar_count} bars"
            )
    return problems


def _derivation_violations(case: Case, name: str) -> Violations:
    steps = case["derivation"]
    problems = each(steps, "derivation_step", f"{name} derivation")
    if problems:
        return problems
    if [step["n"] for step in steps] != list(range(1, len(steps) + 1)):
        problems.append(f"{name} derivation: steps must be numbered 1, 2, 3, ... in order")
    for step in steps:
        where = f"{name} {step['slice_ist']}: derivation step {step['n']}"
        if not step["text"].strip():
            problems.append(f"{where} has no text")
        sequence_step = step["step"]
        if sequence_step is not None and not (vocab.FIRST_STEP <= sequence_step <= vocab.LAST_STEP):
            problems.append(f"{where} names step {sequence_step}; the sequence has steps 1 to 8")
    return problems
