"""A session plan's setup is the backtest's of the same day and strategy but for the live
terms (`_live/plan.py`, `_plan_setup.py`; 16-session-design 3.1 step 11, unit 7).

The backtest is prepared by the backtest's own pipeline (`_pipeline.prepare`) on the
same fake OpenAlgo, with the arguments `bt.backtest` passes it (its own defaults, broker
history, a 1-minute base, start and end today, `now` the present), and its history is
kept as its run keeps it. The plan is built at 10:15:05 IST on the market of
`plan_samples`; its instruments carry OpenAlgo's facts, equal there to the reference
tables' generic cash-equity terms, so only the facts tests change them.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import (
    DAY,
    EmaTrend,
    backtest_arguments,
    build,
    ist,
    market,
)

import openalgo.backtest as bt
from openalgo.backtest._live._preflight_plan import InstrumentFacts
from openalgo.backtest._pipeline import prepare
from openalgo.backtest._warm_up_probe import params_key
from openalgo.backtest.errors import ConfigError

BUILT_AT = ist(10, 15, 5)


@pytest.fixture
def fake(make_fake_openalgo, monkeypatch, tmp_path):
    return market(make_fake_openalgo, monkeypatch, tmp_path, at=BUILT_AT)


@pytest.fixture
def live(fake):
    """The plan's part of EmaTrend, built at 10:15:05."""
    return build(fake).strategies[0]


@pytest.fixture
def backtest(fake):
    """(the backtest of EmaTrend on the day prepared at 10:15:05, the history its run
    keeps)."""
    args = backtest_arguments(EmaTrend, fake.client(), fake.clock.now())
    prepared = prepare(EmaTrend, args["client"], args)
    needs = prepared.needs[params_key(None)]
    bars = {"SBIN": prepared.frames[0]}
    history = prepared.history.for_run(needs, prepared.plan, prepared.session_calendar(), bars)
    return prepared, history


def test_the_settings_are_the_backtests_of_the_same_day(live, backtest):
    prepared, _ = backtest
    assert live.setup.plan == prepared.plan


def test_the_settings_are_for_one_trading_day_on_a_1_minute_base_at_the_present(live):
    plan = live.setup.plan
    assert (plan.start, plan.end, plan.base, plan.present) == (DAY, DAY, "1m", BUILT_AT)
    assert (plan.product, plan.quantity_mode) == ("MIS", "venue")


def test_the_instruments_are_the_backtests_with_openalgos_facts(live, backtest):
    prepared, _ = backtest
    expected = tuple(dataclasses.replace(i, generic=False) for i in prepared.instruments)
    assert live.setup.instruments == expected


def test_the_kernels_config_is_the_backtests_with_the_same_cost_model(live, backtest):
    prepared, _ = backtest
    market = prepared.market_settings()
    config = prepared.plan.native_config(deliver_events=False) | market.config
    native = {name: value for name, value in config.items() if name != "costs"}
    assert {name: value for name, value in live.setup.config.items() if name != "costs"} == native
    assert live.setup.market.costs_label == market.costs_label


def test_the_market_settings_are_the_backtests(live, backtest):
    prepared, _ = backtest
    market = prepared.market_settings()
    assert live.setup.market.config["market"] == market.config["market"]
    assert live.setup.market.config["slippage"] == market.config["slippage"]
    assert live.setup.market.slippage_label == market.slippage_label


def test_todays_closed_candles_are_the_bars_the_backtest_runs_on(live, backtest):
    prepared, _ = backtest
    pd.testing.assert_frame_equal(live.setup.today, prepared.frames[0])


def test_the_history_before_today_is_the_one_the_backtests_run_keeps(live, backtest):
    _, history = backtest
    kept, expected = live.setup.history.symbols["SBIN"], history.symbols["SBIN"]
    pd.testing.assert_frame_equal(kept.base, expected.base)
    assert kept.note == expected.note


def test_the_indicator_needs_are_the_ones_the_backtest_learns(live, backtest):
    prepared, _ = backtest
    assert live.spec.needs == prepared.needs[params_key(None)]


def test_the_kernel_session_is_filled_by_an_external_venue(live):
    assert live.engine.venue() == "external"


def test_the_live_series_holds_the_history_the_backtest_keeps(live, backtest):
    _, history = backtest
    columns = live.series.history_columns("5m")
    assert columns["ts_close"].tolist() == history.symbols["SBIN"].strategy["ts_close"].tolist()
    assert columns["close"].tolist() == history.symbols["SBIN"].strategy["close"].tolist()


def test_after_the_catch_up_the_strategy_reads_what_the_backtest_reads_at_its_last_bar(fake, live):
    seen = []

    class Recording(EmaTrend):
        def on_bar(self, bar):
            seen.append((self._ctx.decided_at, len(self.close), self.close[-1], self.fast[-1]))

    args = backtest_arguments(Recording, fake.client(), fake.clock.now())
    bt.backtest(**args)
    strategy = live.strategy
    now = (live.context.decided_at, len(strategy.close), strategy.close[-1], strategy.fast[-1])
    assert now == seen[-1]


ODD_FACTS = InstrumentFacts("SBIN", "NSE", lot_size=1, tick_size=0.1, freeze_quantity=5_000)


def test_openalgos_tick_size_and_freeze_quantity_replace_the_reference_tables(fake):
    plan = build(fake, facts=(ODD_FACTS,)).strategies[0]
    instrument = plan.setup.instruments[0]
    assert (instrument.lot_size, instrument.tick_size, instrument.freeze_qty) == (1, 0.1, 5_000)
    assert plan.context.instruments["SBIN"] == instrument


def test_each_fact_that_differs_from_the_reference_tables_is_a_warning(fake):
    warnings = build(fake, facts=(ODD_FACTS,)).warnings
    assert (
        "SBIN: OpenAlgo's tick_size 0.1 replaces 0.05 of the reference tables for the session"
        in warnings
    )
    assert (
        "SBIN: OpenAlgo's freeze_qty 5000 replaces None of the reference tables for the session"
        in warnings
    )


def test_facts_equal_to_the_reference_tables_give_no_warning(fake):
    warnings = build(fake).warnings
    assert not [note for note in warnings if "replaces" in note]


def test_an_instrument_the_facts_do_not_cover_refuses_the_plan(fake):
    other = InstrumentFacts("INFY", "NSE", lot_size=1, tick_size=0.05, freeze_quantity=None)
    with pytest.raises(ConfigError, match="no preflight facts of NSE:SBIN"):
        build(fake, facts=(other,))
