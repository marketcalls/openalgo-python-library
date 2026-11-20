"""What a live session's record holds and how its file is written (`session_record`,
`session_recorder`; 16-session-design section 8 unit 16), over the four fault-suite
sessions of `replay_samples`.

A record holds the session's ticks, news, kernel inputs with their results, decisions and
commands, the lines that define its scenario, the mode OpenAlgo was in, the `bt.live`
arguments it was given (a distance, a brokerage preset or a cost model as a tagged
object, `record_objects`), and at its close the kernel's whole run output. Its file is
strict JSON lines: a number with no value (a live fill's modelled slippage) is written
null. An argument a record cannot keep stops the session at its start, named.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from record_objects import untagged
from replay_fixtures import *  # noqa: F403
from replay_samples import commands, recorded
from session_record import FORMAT, VERSION, ReplayError, SessionRecord, plain
from session_recorder import record_sessions
from session_samples import ist
from stops_samples import StopHold, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.units import Distance

#: Brokerage presets a `bt.live` call may give as `costs=`, by how they are built.
BROKERAGES = {
    "zerodha": bt.costs.zerodha(),
    "zero brokerage": bt.costs.zero_brokerage(),
    "zero brokerage, own DP fee": bt.costs.zero_brokerage(dp_fee=10),
    "flat": bt.costs.flat(20),
    "flat, own DP fee": bt.costs.flat(20, dp_fee=12),
    "percent with a cap": bt.costs.percent(0.03, 20),
    "percent with no cap": bt.costs.percent(0.07),
    "custom": bt.costs.custom(
        delivery=("percent", 0.1, 30), intraday=("percent", 0.03, 20), futures=20, options=20,
        dp_fee=15, name="my broker",
    ),
}  # fmt: skip
#: Cost models a `bt.live` call may give as `costs=`: on the built-in schedule versions,
#: and on versions of the user's own (the first two built-in ones, alone).
COST_MODELS = {
    "built-in schedules": lambda: bt.costs.CostModel(
        bt.costs.flat(20), "calculator", backfill=True
    ),
    "own schedules": lambda: bt.costs.CostModel(
        bt.costs.percent(0.03, 20), schedules=bt.costs.schedules()[:2], include_builtin=False
    ),
}


class HoldsFrom(StopHold):
    """`StopHold` with a parameter JSON cannot hold, a time of day, which it never reads."""

    hold_from = bt.param(dt.time(14, 41))


def strict(value: object) -> object:
    """`value` as a record keeps it, through strict JSON text and back."""
    return json.loads(json.dumps(plain(value), allow_nan=False))


def cost_fields(model: bt.costs.CostModel) -> tuple:
    """What a cost model charges by: its brokerage (and its name), rounding, backfill and
    schedule versions."""
    brokerage = model.brokerage
    return brokerage, brokerage.name, model.rounding, model.backfill, model.schedules()


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


def test_a_distance_argument_is_kept_as_a_tagged_object_and_read_back_as_given(
    session_with_objects,
):
    session = session_with_objects
    given = {n: v for n, v in session.scenario.objects.items() if isinstance(v, Distance)}
    assert given
    for record in session.records:
        arguments = record.header["arguments"]
        for name, distance in given.items():
            assert arguments[name] == {
                "tag": "distance",
                "unit": distance.unit,
                "value": distance.value,
                "period": distance.period,
            }
        assert untagged(arguments) == arguments | session.scenario.objects


def test_a_cost_preset_argument_is_kept_by_its_terms_and_read_back_as_given(session_with_objects):
    session = session_with_objects
    # The sandbox session's costs=bt.costs.flat(20): Rs 20 an executed order in each of
    # delivery, intraday, futures and options, Zerodha's DP fee standing in.
    given = {n: v for n, v in session.scenario.objects.items() if n == "costs"}
    assert given
    for record in session.records:
        arguments = record.header["arguments"]
        for name in given:
            assert arguments[name] == {
                "tag": "brokerage",
                "name": "flat(20)",
                "dp_fee": None,
                "rules": [["flat", 20.0, 0.0]] * 4,
            }
        back = untagged(arguments)
        assert [(back[n], back[n].name) for n in given] == [(v, v.name) for v in given.values()]


@pytest.mark.parametrize("brokerage", BROKERAGES.values(), ids=list(BROKERAGES))
def test_a_brokerage_preset_reads_back_equal_with_its_name_and_dp_fee(brokerage):
    kept = strict(brokerage)
    back = untagged(kept)
    assert kept["tag"] == "brokerage"
    assert (back, back.name, back.dp_fee, back.dp_fee_source) == (
        brokerage,
        brokerage.name,
        brokerage.dp_fee,
        brokerage.dp_fee_source,
    )


@pytest.mark.parametrize("model", COST_MODELS.values(), ids=list(COST_MODELS))
def test_a_cost_model_reads_back_with_its_brokerage_rounding_backfill_and_schedules(model):
    given = model()
    kept = strict(given)
    assert kept["tag"] == "cost_model"
    assert cost_fields(untagged(kept)) == cost_fields(given)


def test_an_argument_a_record_cannot_keep_stops_the_session_at_its_start_naming_it(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    record_sessions(monkeypatch)
    world.stop_at(ist(14, 44))
    with pytest.raises(ReplayError, match=r"bt\.live's params= cannot be kept .* pass JSON"):
        world.run(HoldsFrom, on_stop="keep", params={"hold_from": dt.time(14, 42)})
    assert world.orders() == []


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


def series_take(opened: int, open_: float, close: float) -> dict:
    """A record's line of a 1-minute bar SBIN's series took (prices rupees)."""
    bar = {"ts_open_ns": opened, "open": open_, "high": close, "low": open_, "close": close}
    return {"kind": "series", "strategy": "S", "call": "take", "args": [bar | {"volume": 1.0}]}


def test_a_slice_whose_series_took_another_bar_after_its_first_is_named_replaced():
    # 14:41's bar built from ticks (opening late, at 500.50, after a tick gap) is decided
    # on, then replaced by the broker's candle (opening at 500.00); 14:42's candle comes
    # again unchanged. `bars` gives the candle, the bar later decisions saw.
    minute = 60 * 10**9
    first, second = ist(14, 41), ist(14, 42)
    lines = [
        {"kind": "history", "strategy": "S", "base": [], "today": []},
        series_take(first, 500.5, 501.0),
        {"kind": "drive", "strategy": "S", "deciding": True, "decide_from_ns": first + minute},
        series_take(second, 501.0, 501.5),
        series_take(first, 500.0, 501.0),
        series_take(second, 501.0, 501.5),
    ]
    record = SessionRecord({"format": FORMAT, "version": VERSION}, lines)
    assert record.replaced("S") == [first]
    assert list(record.bars("S")["open"]) == [500.0, 501.0]


def test_a_record_of_another_format_or_version_is_refused(tmp_path):
    path = tmp_path / "other.jsonl"
    path.write_text(f'{{"format": "{FORMAT}", "version": {VERSION + 1}}}\n', encoding="utf-8")
    with pytest.raises(ReplayError, match="not a openalgo-live-session record of version 1"):
        SessionRecord.load(path)


def test_a_value_json_cannot_hold_is_refused_when_the_record_is_written(tmp_path):
    record = SessionRecord({"format": FORMAT, "version": VERSION}, [{"kind": "x", "v": 1e999}])
    with pytest.raises(ReplayError, match="JSON"):
        record.dump(tmp_path / "record.jsonl")
