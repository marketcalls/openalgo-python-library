"""What a live session's record holds and how its file is written (`session_record`,
`session_recorder`; 16-session-design section 8 unit 16), over the four fault-suite
sessions of `replay_samples`.

A record holds the session's ticks, news, kernel inputs with their results, decisions and
commands, the lines that define its scenario, the mode OpenAlgo was in, the `bt.live`
arguments it was given (a distance as a tagged object), and at its close the kernel's
whole run output. Its file is strict JSON lines: a number with no value (a live fill's
modelled slippage) is written null.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from replay_fixtures import *  # noqa: F403
from replay_samples import commands, recorded
from session_record import FORMAT, VERSION, ReplayError, SessionRecord, with_distances

from openalgo.backtest._live._types import LOGGER_NAME


def refuse(constant: str) -> None:
    raise ValueError(f"not strict JSON: {constant}")


def finished(record: SessionRecord) -> list[dict]:
    """The record's `finished` lines."""
    return [line for line in record.lines if line["kind"] == "finished"]


def test_a_record_holds_the_sessions_ticks_news_results_and_commands(session):
    held = set().union(*(record.kinds() for record in session.records))
    assert session.scenario.kinds | {"news", "history", "series", "bar", "release"} <= held
    assert all(record.header["format"] == FORMAT for record in session.records)


def test_a_record_holds_the_lines_that_define_its_session(session):
    session.scenario.defining(session.records)


def test_a_records_header_names_the_mode_openalgo_was_in(session):
    assert [record.header["mode"] for record in session.records] == [session.scenario.mode] * len(
        session.records
    )


def test_a_distance_argument_is_kept_as_a_tagged_object_and_read_back_as_given(session):
    for record in session.records:
        arguments = record.header["arguments"]
        for name, distance in session.scenario.distances.items():
            assert arguments[name] == {
                "tag": "distance",
                "unit": distance.unit,
                "value": distance.value,
                "period": distance.period,
            }
        assert with_distances(arguments) == arguments | session.scenario.distances


def test_a_record_file_is_strict_json_lines(session):
    for path in session.paths:
        with open(path, encoding="utf-8") as stream:
            for text in stream:
                json.loads(text, parse_constant=refuse)


def test_a_live_fills_missing_slippage_is_written_null(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # A live fill has no modelled slippage (NaN in the run output): the trailing
    # session's entry is written with null.
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    (path,) = recorded("trailing", make_fake_openalgo, monkeypatch, tmp_path).paths
    with open(path, encoding="utf-8") as stream:
        (last,) = [line for text in stream if (line := json.loads(text)).get("kind") == "finished"]
    assert last["output"]["fills"]["slippage_rupees"] == [None]


def test_the_finished_line_holds_the_runs_working_orders_and_open_position(session):
    # Each order working at the end has the terms of the kernel's last command for it,
    # and the position open is what the scenario leaves.
    terms: dict[int, dict] = {}
    for record in session.records:
        for command in commands(record):
            terms.setdefault(command["order_id"], {}).update(command)
    (last,) = finished(session.records[-1])
    orders, positions = last["output"]["orders"], last["output"]["final_positions"]
    working = [
        (orders["order_id"][at], orders["trigger_rupees"][at], orders["quantity_units"][at])
        for at, status in enumerate(orders["status"])
        if status == "Submitted"
    ]
    expected = [(i, terms[i]["trigger_price"], terms[i]["quantity_units"]) for i, *_ in working]
    assert working == expected
    assert bool(working) == bool(session.scenario.held_at_end)
    held = sum(position["quantity_units"] for position in positions)
    assert held == session.scenario.held_at_end


def test_a_record_of_another_format_or_version_is_refused(tmp_path):
    path = tmp_path / "other.jsonl"
    path.write_text(f'{{"format": "{FORMAT}", "version": {VERSION + 1}}}\n', encoding="utf-8")
    with pytest.raises(ReplayError, match="not a openalgo-live-session record of version 1"):
        SessionRecord.load(path)


def test_a_value_json_cannot_hold_is_refused_when_the_record_is_written(tmp_path):
    record = SessionRecord({"format": FORMAT, "version": VERSION}, [{"kind": "x", "v": 1e999}])
    with pytest.raises(ReplayError, match="JSON"):
        record.dump(tmp_path / "record.jsonl")
