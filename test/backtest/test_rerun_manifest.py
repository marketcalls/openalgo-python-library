"""`Result.rerun(**overrides)` and the run manifest (14-strategy-api 5.2 and 6)."""

import json
import platform

import numpy as np
import pandas as pd
import pytest
from strategy_samples import CAPITAL, DAY, EmaCross, wave_bars, wave_with_history

import openalgo.backtest as bt
from openalgo.backtest._manifest import RunManifest, strategy_source_hash
from openalgo.backtest.data import load_bars
from openalgo.backtest.errors import ConfigError


def backtest(**settings):
    arguments = {"data": wave_bars(), "capital": CAPITAL, "costs": None, "slippage": 0}
    return bt.backtest(EmaCross, **{**arguments, **settings})


def assert_same(first, second):
    for name, table in first.to_dataframes().items():
        pd.testing.assert_frame_equal(table, second.to_dataframes()[name], check_exact=True)
    assert np.array_equal(first.equity.to_numpy(), second.equity.to_numpy(), equal_nan=True)
    assert first.assumptions == second.assumptions
    assert first.manifest == second.manifest


def test_a_rerun_without_overrides_reproduces_the_result_bit_for_bit():
    # A session of history: the strategy decides from the first bar of the run and
    # trades (on the test day alone its 10-bar EMA would keep it waiting most of it).
    result = backtest(data=wave_with_history(), start=DAY)
    assert len(result.fills) > 0
    assert_same(result, result.rerun())


def test_a_rerun_can_change_a_parameter():
    rerun = backtest().rerun(fast=5)
    assert rerun.manifest["strategy"]["parameters"] == {"fast": 5, "slow": 10}
    assert_same(rerun, backtest(params={"fast": 5}))


def test_a_rerun_with_params_replaces_the_parameters_of_the_original_run():
    # The original slow=15 is not carried over: slow is back at its default of 10.
    rerun = backtest(params={"slow": 15}).rerun(params={"fast": 5})
    assert rerun.manifest["strategy"]["parameters"] == {"fast": 5, "slow": 10}


def test_a_rerun_can_change_a_run_argument():
    rerun = backtest().rerun(quantity_mode="venue")
    assert rerun.assumptions["quantity_mode"] == "venue"


def test_a_rerun_refuses_an_unknown_name():
    with pytest.raises(ConfigError, match="neither an argument of bt.backtest nor a parameter"):
        backtest().rerun(speed=3)


def test_a_rerun_says_when_the_data_changed():
    frames = {"SBIN": wave_bars()}
    result = backtest(data=frames)
    frames["SBIN"].loc[frames["SBIN"].index[-1], "close"] += 0.05
    assert "not a reproduction" in result.rerun().warnings[0]


def test_the_manifest_round_trips_through_json():
    manifest = backtest().manifest
    text = json.dumps(manifest)
    assert RunManifest.from_json(RunManifest.from_dict(json.loads(text)).to_json()).to_dict() == (
        manifest
    )


def test_the_manifest_records_the_strategy_and_its_parameters():
    strategy = backtest(params={"slow": 15}).manifest["strategy"]
    assert strategy == {
        "module": "strategy_samples",
        "name": "EmaCross",
        "source_hash": strategy_source_hash(EmaCross),
        "parameters": {"fast": 3, "slow": 15},
    }
    assert strategy["source_hash"].startswith("sha256:")


def test_the_manifest_records_the_resolved_arguments():
    arguments = backtest().manifest["arguments"]
    assert arguments["base"] == "1m"
    assert arguments["product"] == "CNC"
    assert arguments["start"] == "2025-07-01"
    assert arguments["slippage"] == ["zero", 0.0, 0]
    assert arguments["calendar"] == "default"


def test_the_manifest_records_the_data_partition_hashes():
    loaded = load_bars("SBIN", "NSE", "1m", "2025-07-01", "2025-07-01", data=wave_bars())
    assert backtest().manifest["data"] == {"SBIN": loaded.meta.partition_hashes()}


def test_the_manifest_records_versions_seed_and_platform():
    manifest = backtest().manifest
    assert manifest["engine_version"].count(".") == 2
    assert manifest["reference_tables"] is None  # a generic instrument, no table used
    assert manifest["cost_schedules"] == []  # costs=None
    assert manifest["random_seed"] is None
    assert manifest["platform"]["python"] == platform.python_version()


def test_the_manifest_never_holds_the_client():
    class Client:
        api_key = "do-not-write-this-key"

    manifest = backtest(client=Client()).manifest
    assert manifest["arguments"]["client"] == "given"
    assert "do-not-write-this-key" not in json.dumps(manifest)
