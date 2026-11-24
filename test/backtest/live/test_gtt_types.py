"""The GTT request's checks (`_live/_gtt_types.py`; 20-positional-live 3, the lead's ruling
N3a-1): a request the order worker cannot send as it stands is refused when it is made,
saying why. The valid request is a long CNC leg's stop on SBIN: sell 10 at a trigger of
790.00 with a child LIMIT at 780.00.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._gtt_types import Direction, GttAction, GttRequest
from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live._venue_requests import Priority
from openalgo.backtest.errors import LiveError

STOP = {
    "request_id": 1,
    "action": GttAction.PLACE,
    "purpose": Purpose.STOP,
    "strategy": "ema_cross",
    "gtt_id": 1,
    "symbol": "SBIN",
    "exchange": "NSE",
    "product": "CNC",
    "side": "SELL",
    "trigger_price": 790.0,
    "limit_price": 780.0,
    "quantity": 10,
    "direction": Direction.BELOW,
    "priority": Priority.PROTECT,
}


def test_a_stop_with_a_trigger_a_limit_and_units_is_a_request():
    assert GttRequest(**STOP).terms.limit_price == 780.0


@pytest.mark.parametrize(
    ("change", "words"),
    [
        # Every child is a LIMIT order: a limit of 0 would sell at any price.
        ({"limit_price": 0.0}, "its trigger and limit are above 0 rupees"),
        ({"trigger_price": 0.0}, "its trigger and limit are above 0 rupees"),
        ({"product": "MIS"}, "a GTT holds a CNC or NRML position"),
        ({"quantity": 0}, "its units are above 0"),
        ({"seq": 3}, "a place names no journal GTT"),
    ],
    ids=["a limit of 0", "a trigger of 0", "MIS", "no units", "a place naming a GTT"],
)
def test_a_request_that_cannot_be_sent_is_refused_saying_why(change, words):
    with pytest.raises(LiveError, match=words):
        GttRequest(**(STOP | change))
