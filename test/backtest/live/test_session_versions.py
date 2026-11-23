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
import sys
from types import SimpleNamespace

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from session_samples import BuyAndHold, StopFlag, ist, rig

import openalgo.backtest as bt
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
    # BuyAndHold holds 3 bars; the restart is given 4.
    two_starts(world, {"hold": 4})
    assert (
        f"BuyAndHold: its version changed since the day's earlier start, from {OWN} to "
        f"{HOLD_4} (its code or its parameters; the parameters whose values changed: hold 3 "
        "to 4)"
    ) in world.lines


def deciding(self, bar):
    """BuyAndHold's decision, called from a function of its own: other compiled code."""
    BuyAndHold.on_bar(self, bar)


def test_a_restart_with_edited_code_says_so_in_the_banner_with_no_parameter_named(world):
    # A class built at run time whose on_bar differs between the starts, its parameters
    # the same: nothing in the run manifests says which part of the code changed, and its
    # id follows the values kept on its class too.
    first = type("Edited", (BuyAndHold,), {"on_bar": BuyAndHold.on_bar})
    second = type("Edited", (BuyAndHold,), {"on_bar": deciding})
    world.stop_at(ist(14, 43))
    live(world, first)
    world.stop = StopFlag()
    world.stop_at(ist(14, 45))
    live(world, second)
    before, after = version_of(first, None).version_id, version_of(second, None).version_id
    assert (
        f"Edited: its version changed since the day's earlier start, from {before} to "
        f"{after} (its code, its parameters or a value kept on its class; keep run state in "
        "self.state or on the instance)"
    ) in world.lines


def trading_once(self, bar):
    """BuyAndHold's decision, after noting on the class that the strategy has traded."""
    type(self).traded = True
    BuyAndHold.on_bar(self, bar)


def test_a_flag_the_strategy_sets_on_its_class_moves_its_version_at_a_restart(world):
    # The id follows the class's values when a start begins (2.4.1 item W-5): the first
    # start sets `traded`, a parameter, on the class, so the restart in the same process
    # runs another version and the banner names the value that moved. Run state belongs in
    # self.state or on the instance.
    Marks = type("Marks", (BuyAndHold,), {"traded": False, "on_bar": trading_once})
    before = version_of(Marks, None).version_id
    world.stop_at(ist(14, 43))
    live(world, Marks)
    assert Marks.traded is True
    after = version_of(Marks, None).version_id
    world.stop = StopFlag()
    world.stop_at(ist(14, 45))
    live(world, Marks)
    [changed] = events(world, "version_changed")
    assert (changed["earlier"], changed["now"]) == (before, after)
    assert (
        f"Marks: its version changed since the day's earlier start, from {before} to "
        f"{after} (its code or its parameters; the parameters whose values changed: traded "
        "false to true)"
    ) in world.lines


class Model:
    """A parameter value with no text of its own: its text names its memory address."""


class Modelled(BuyAndHold):
    """BuyAndHold with a parameter that takes an object, `model`."""

    model = bt.param(None)


def test_a_restart_names_no_parameter_whose_text_alone_differs_among_those_that_moved(world):
    # Each start is given its own object (both alive, so at two addresses) and the restart
    # holds 4 bars: the id leaves the object's address out, so the banner names the hold
    # alone (and its by-type line names the object).
    first, second = Model(), Model()
    world.stop_at(ist(14, 43))
    live(world, Modelled, params={"model": first})
    world.stop = StopFlag()
    world.stop_at(ist(14, 45))
    live(world, Modelled, params={"model": second, "hold": 4})
    before = version_of(Modelled, {"model": first}).version_id
    after = version_of(Modelled, {"model": second, "hold": 4}).version_id
    assert (
        f"Modelled: its version changed since the day's earlier start, from {before} to "
        f"{after} (its code or its parameters; the parameters whose values changed: hold 3 "
        "to 4)"
    ) in world.lines


def test_a_restart_with_the_same_version_records_no_change(world):
    two_starts(world)
    assert events(world, "version_changed") == []
    assert not any("its version changed" in line for line in world.lines)


def test_a_class_with_no_source_is_named_in_the_banner_with_the_python_its_code_is_hashed_under(
    world,
):
    Built = type("Built", (BuyAndHold,), {})
    world.stop_at(ist(14, 41))
    live(world, Built)
    built = version_of(Built, None).version_id
    python = f"CPython {sys.version_info.major}.{sys.version_info.minor}"
    assert f"Versions: Built {built}" in world.lines
    assert (
        "Built: its source could not be read (a class built at run time): its version id "
        f"hashes its compiled code (its functions and class attributes) under {python} and "
        "its parameters, so the same code under another Python version gets another id"
    ) in world.lines


class Rule:
    """An object a class calls as a method: Python keeps no compiled code of it there."""

    def __call__(self, bar):
        return None


def test_an_object_called_as_a_method_is_named_in_the_banner_by_its_type(world):
    # BuyAndHold's code is read; the Rule object's code is not, and the banner says so.
    Ruled = type("Ruled", (BuyAndHold,), {"rule": Rule()})
    world.stop_at(ist(14, 41))
    live(world, Ruled)
    assert (
        "Ruled: rule has no fixed text of its value (an object shown by its memory address, "
        "or one that can change as the strategy runs, such as a list, a dict or a DataFrame): "
        "its version id hashes its type only, so a change to its value does not change the id"
    ) in world.lines
    assert not any("its source was not hashed" in line for line in world.lines)


def remembering(self, bar):
    """BuyAndHold's decision, after noting the bar's close in the class's `seen` list."""
    self.seen.append(bar.close)
    BuyAndHold.on_bar(self, bar)


def test_a_list_the_strategy_appends_to_does_not_change_its_version_at_a_restart(world):
    # A class built at run time, as a notebook cell builds one: its first start appends a
    # close to `seen` at each decision, which the restart in the same process still holds.
    Remembers = type("Remembers", (BuyAndHold,), {"seen": [], "on_bar": remembering})
    world.stop_at(ist(14, 43))
    live(world, Remembers)
    assert Remembers.seen
    world.stop = StopFlag()
    world.stop_at(ist(14, 45))
    live(world, Remembers)
    assert events(world, "version_changed") == []
    assert (
        "Remembers: seen has no fixed text of its value (an object shown by its memory "
        "address, or one that can change as the strategy runs, such as a list, a dict or a "
        "DataFrame): its version id hashes its type only, so a change to its value does not "
        "change the id"
    ) in world.lines


def test_a_class_whose_code_cannot_be_read_at_all_is_named_in_the_banner_as_before(world):
    # Opaque's one callable is the Rule object it calls as its on_bar.
    members = {"symbol": "SBIN", "exchange": "NSE", "interval": "1m", "on_bar": Rule()}
    Opaque = type("Opaque", (bt.Strategy,), members)
    world.stop_at(ist(14, 41))
    live(world, Opaque)
    assert (
        "Opaque: its source was not hashed (a class built at run time): its version id "
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
