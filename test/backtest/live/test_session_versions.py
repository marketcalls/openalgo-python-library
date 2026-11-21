"""The strategy's version on every order of a live session (18-multi-strategy 12.1; unit M11):
each start writes each strategy's version id and run manifest to the journal, every intent
carries the version id, the order OpenAlgo receives keeps the plain strategy name, and a
restart whose version differs from the day's earlier start says so.

The market and the scripts are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16 and SBIN trades 10 units every 10 s at 500.00. `BuyAndHold` buys 10 at
its first decision (14:42) and sells them three bars later (14:45); `qty` and `hold` are
its parameters. The expected version ids are worked by `version_of` from the class and its
parameters (test_version_id pins that rule by hand), never read from the session.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from types import SimpleNamespace

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from session_samples import BuyAndHold, StopFlag, ist, rig

from openalgo.backtest._live._session_loop import Session
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live._version_id import version_of
from openalgo.backtest._manifest import strategy_source_hash

#: The version of BuyAndHold with its own parameters, and with `hold` 4.
OWN = version_of(BuyAndHold, None).version_id
HOLD_4 = version_of(BuyAndHold, {"hold": 4}).version_id


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    return made


def events(world, kind: str) -> list[dict]:
    return [event for run in world.trace.runs for event in run.events if event["kind"] == kind]


def journal_rows(result, query: str) -> list[tuple]:
    connection = sqlite3.connect(result.paths["journal"])
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def test_every_order_of_the_result_carries_its_strategys_version_id(world):
    world.stop_at(ist(14, 47))
    result = live(world, BuyAndHold)
    assert len(result.orders) == 2
    assert set(result.orders["version_id"]) == {OWN}


def test_every_intent_of_the_journal_carries_the_version_id(world):
    world.stop_at(ist(14, 47))
    result = live(world, BuyAndHold)
    assert journal_rows(result, "SELECT strategy, version_id FROM intents") == [
        ("BuyAndHold", OWN),
        ("BuyAndHold", OWN),
    ]


def test_the_order_openalgo_receives_keeps_the_plain_strategy_name(world):
    world.stop_at(ist(14, 47))
    live(world, BuyAndHold)
    assert [order.strategy for order in world.fake.orders()] == ["BuyAndHold", "BuyAndHold"]


def test_the_result_holds_each_strategys_run_manifest_by_name(world):
    world.stop_at(ist(14, 41))
    result = live(world, BuyAndHold, params={"hold": 4})
    manifest = result.versions["BuyAndHold"]
    described = manifest["strategy"]
    assert (described["name"], described["parameters"]) == ("BuyAndHold", {"qty": 10, "hold": 4})
    assert described["source_hash"] == strategy_source_hash(BuyAndHold)
    assert manifest["arguments"]["capital"] == 100_000
    assert "api_key" not in json.dumps(manifest)


def test_each_start_writes_each_strategys_version_and_manifest_to_the_journal(world):
    world.stop_at(ist(14, 41))
    result = live(world, BuyAndHold)
    [(strategy, version_id, manifest)] = journal_rows(
        result, "SELECT strategy, version_id, manifest FROM versions"
    )
    assert (strategy, version_id) == ("BuyAndHold", OWN)
    assert json.loads(manifest) == result.versions["BuyAndHold"]


def test_the_banner_names_each_strategys_version_id(world):
    world.stop_at(ist(14, 41))
    live(world, BuyAndHold)
    assert f"Versions: BuyAndHold {OWN}" in world.lines


def test_the_daily_summary_names_each_strategys_version_id(world):
    world.stop_at(ist(14, 41))
    result = live(world, BuyAndHold)
    assert f"Strategies  : BuyAndHold (version {OWN})" in result.summary_text()


def two_starts(world, second_params=None):
    """A start stopped at 14:43 with BuyAndHold's own parameters, then a restart stopped
    at 14:45 with `second_params`."""
    world.stop_at(ist(14, 43))
    live(world, BuyAndHold)
    world.stop = StopFlag()
    world.stop_at(ist(14, 45))
    return live(world, BuyAndHold, params=second_params)


def test_a_restart_with_other_parameters_records_both_version_ids(world):
    result = two_starts(world, {"hold": 4})
    [changed] = events(world, "version_changed")
    assert (changed["strategy"], changed["earlier"], changed["now"]) == ("BuyAndHold", OWN, HOLD_4)
    assert journal_rows(result, "SELECT version_id FROM versions") == [(OWN,), (HOLD_4,)]


def test_a_restart_with_other_parameters_says_so_in_the_banner(world):
    two_starts(world, {"hold": 4})
    assert (
        f"BuyAndHold: its version changed since the day's earlier start, from {OWN} to "
        f"{HOLD_4} (its code or its parameters)"
    ) in world.lines


def test_a_restart_with_the_same_version_records_no_change(world):
    two_starts(world)
    assert events(world, "version_changed") == []
    assert not any("its version changed" in line for line in world.lines)


def test_a_class_with_no_source_is_named_in_the_banner(world):
    Built = type("Built", (BuyAndHold,), {})
    world.stop_at(ist(14, 41))
    live(world, Built)
    built = version_of(Built, None).version_id
    assert f"Versions: Built {built}" in world.lines
    assert (
        "Built: its source was not hashed (a class built at run time): its version id "
        "hashes its name and parameters only"
    ) in world.lines


def test_a_session_whose_start_has_not_recorded_versions_has_none():
    # The banner and the close read the versions; before the start records them they are
    # empty, as every other record of a start is.
    session = Session(
        SimpleNamespace(broker="zerodha"),
        now=lambda: 0,
        sleep=lambda seconds: None,
        threads=False,
        stop=StopFlag(),
        emit=print,
    )
    versions = session.versions
    assert (versions.banner_lines(), versions.manifests, versions.ids) == ([], {}, {})
