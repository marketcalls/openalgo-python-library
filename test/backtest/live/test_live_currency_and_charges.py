"""What 2.4.1 changes at the start of `bt.live`, on the fake OpenAlgo (unit C of 2.4.1):
a strategy on a currency derivatives exchange is refused before any call, since currency
derivatives are backtest-only in this release, and the start banner names the charges
the session models, "standard" by default.

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16 and SBIN trades at 500.00 every 10 s.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from session_samples import BuyAndHold, rig

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import ConfigError


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    return made


CURRENCY = {
    "NSE currency future": ("USDINR29DEC26FUT", "CDS"),
    "NSE currency option": ("USDINR29DEC2688.5CE", "CDS"),
    "BSE currency future": ("USDINR29DEC26FUT", "BCD"),
}


@pytest.mark.parametrize(("symbol", "exchange"), CURRENCY.values(), ids=CURRENCY)
def test_a_currency_derivative_is_refused_before_any_call(world, symbol, exchange):
    strategy = type("Currency", (BuyAndHold,), {"symbol": symbol, "exchange": exchange})
    with pytest.raises(ConfigError, match=f"{exchange} currency derivatives are backtest-only"):
        live(world, strategy)
    assert world.fake.log.calls() == []


def test_the_banner_names_the_standard_charges_by_default(world):
    live(world, BuyAndHold)
    assert "Charges modelled with: standard" in world.lines


def test_the_banner_names_the_charges_given(world):
    live(world, BuyAndHold, costs="zerodha")
    assert "Charges modelled with: zerodha" in world.lines
