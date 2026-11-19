"""The price a backtest fills a MARKET order at, through the binding
(`Engine.market_fill_price`; rust/oa_py/src/backtest/engine/market.rs): what the day
summary's slippage sets each MARKET fill of a live session against.

The session is `test_kernel_session`'s: the NSE day 2025-07-01 on 1-minute slices from
09:15 IST, SBIN with a tick of 0.05. Expected prices are worked by hand.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_kernel_session import MINUTE_NS, NO_BARS, OPEN_NS, sbin, session_config

from openalgo.backtest._calendar_convert import native_attr
from openalgo.backtest.calendar import default_calendar

#: The open of the 09:18 slice, UTC nanoseconds.
AT_0918 = OPEN_NS + 3 * MINUTE_NS


def session_slipping(slippage: tuple[str, float, int]) -> Any:
    """A live session of the day whose run models `slippage` (kind, amount, min_ticks)."""
    config = session_config() | {"slippage": slippage}
    engine_class = native_attr("Engine")
    return engine_class.session(default_calendar()._cal, config, [sbin()], [NO_BARS])


@pytest.mark.parametrize(
    ("slippage", "side", "open_", "price"),
    [
        (("ticks", 2.0, 0), "BUY", 500.0, 500.1),
        (("ticks", 2.0, 0), "SELL", 500.0, 499.9),
        # 2 bps of 817.55 is 0.16351: 817.71351 rounded up to the tick, 817.38649 down.
        (("segment_default", 0.0, 0), "BUY", 817.55, 817.75),
        (("segment_default", 0.0, 0), "SELL", 817.55, 817.35),
        (("zero", 0.0, 0), "BUY", 500.0, 500.0),
    ],
    ids=["two ticks buy", "two ticks sell", "default buy", "default sell", "none"],
)
def test_a_market_fill_is_priced_by_the_runs_slippage_away_from_the_trader(
    slippage, side, open_, price
):
    engine = session_slipping(slippage)
    assert engine.market_fill_price(0, side, AT_0918, open_) == pytest.approx(price)


@pytest.mark.parametrize(
    ("instrument", "side", "at", "open_", "words"),
    [
        (0, "HOLD", AT_0918, 500.0, "unknown side"),
        (1, "BUY", AT_0918, 500.0, "no instrument"),
        (0, "BUY", AT_0918 + 1, 500.0, "no slice of the run opens"),
        (0, "BUY", AT_0918, 0.0, "above 0"),
    ],
    ids=["a side", "an instrument", "an instant off the grid", "an open of 0"],
)
def test_what_the_kernel_cannot_price_is_an_input_error(instrument, side, at, open_, words):
    engine = session_slipping(("ticks", 2.0, 0))
    with pytest.raises(native_attr("NativeEngineInputError"), match=words):
        engine.market_fill_price(instrument, side, at, open_)
