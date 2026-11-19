"""Trade reshuffle (design 17, section 2.10): the closed trades in other orders, worked by
hand on a small trade list.

Trades -100, -100 and +300 on a capital of 1,000 have three distinct orders, equally
likely:

    -100, -100, +300   equity 900, 800, 1,100     deepest fall 800 / 1,000 - 1 = -20%
    -100, +300, -100   equity 900, 1,200, 1,100   deepest fall 900 / 1,000 - 1 = -10%
    +300, -100, -100   equity 1,300, 1,200, 1,100   1,100 / 1,300 - 1 = -15.38%

Every order ends at +100, a return of 10%. The run's own order is the first. With a
third of the orders at -20%, the drawdown 95% of orders stay within (the 5th
percentile) is -20%, and the median is -15.38%.
"""

import pytest
from robust_runs import blocks_of, check_of, trade_performance

from openalgo.backtest._robust._verdict import FAILED, INFORMATIONAL, NOT_JUDGED, PASSED

TRADES = [-100.0, -100.0, 300.0]
WORST, MIDDLE, MILDEST = 800 / 1_000 - 1, 1_100 / 1_300 - 1, 900 / 1_000 - 1
SMALL = {"min_trades": 3}


@pytest.fixture(scope="module")
def performance():
    return trade_performance(TRADES, capital=1_000.0)


def test_every_order_is_one_of_the_three_worked_by_hand_and_each_occurs(performance):
    table = check_of(performance, "reshuffle", runs=600, **SMALL).table
    found = sorted(set(table["max_drawdown"].round(10)))
    assert found == pytest.approx(sorted([WORST, MIDDLE, MILDEST]))


def test_every_order_ends_at_the_same_return(performance):
    table = check_of(performance, "reshuffle", runs=600, **SMALL).table
    assert table["final_return"].to_numpy() == pytest.approx(0.1)


def test_the_figures_are_the_drawdowns_shares_of_orders_stay_within(performance):
    summary = check_of(performance, "reshuffle", runs=600, **SMALL).summary
    assert summary["drawdown_95"] == pytest.approx(WORST)
    assert summary["drawdown_99"] == pytest.approx(WORST)
    assert summary["drawdown_50"] == pytest.approx(MIDDLE)
    assert summary["real_drawdown"] == pytest.approx(WORST)


def test_without_a_limit_the_check_is_informational(performance):
    result = check_of(performance, "reshuffle", runs=600, **SMALL)
    assert result.verdict == INFORMATIONAL
    assert "max_drawdown" in result.rule


@pytest.mark.parametrize(("limit", "verdict"), [(0.25, PASSED), (0.20, PASSED), (0.15, FAILED)])
def test_with_a_limit_the_drawdown_95_percent_stay_within_is_judged(performance, limit, verdict):
    result = check_of(performance, "reshuffle", runs=600, max_drawdown=limit, **SMALL)
    assert result.verdict == verdict


def test_another_level_judges_the_drawdown_that_share_stays_within(performance):
    # Two thirds of the orders stay within -15.38%, so at a level of 0.6 the limit of 16%
    # holds although the 95% drawdown is -20%.
    result = check_of(performance, "reshuffle", runs=600, max_drawdown=0.16, level=0.6, **SMALL)
    assert result.verdict == PASSED
    assert result.summary["drawdown_at_level"] == pytest.approx(MIDDLE)


def test_the_note_says_the_orders_end_at_the_closed_trades_net_pnl(performance):
    # A run's net profit also holds a position still open at its end, which no order of
    # the closed trades books: the words say whose end it is.
    result = check_of(performance, "reshuffle", runs=600, **SMALL)
    assert result.note == (
        "600 orders of the 3 closed trades, drawn without replacement and booked on Rs 1,000; "
        "every order of the closed trades ends at Rs 100 of net P&L."
    )


def test_the_final_pnl_tile_is_the_closed_trades(performance):
    (_, (tiles,)) = blocks_of(performance, "reshuffle", runs=600, **SMALL)
    shown = {tile.label: (tile.value, tile.sub) for tile in tiles.tiles}
    assert shown["Final P&L"] == ("Rs 100", "of the closed trades, every order")


def test_fewer_closed_trades_than_needed_are_not_judged():
    result = check_of(trade_performance([10.0] * 29), "reshuffle")
    assert result.verdict == NOT_JUDGED
    assert result.note == "Too few trades to judge: 29 closed trades, 30 needed."
    assert result.table.empty


def test_orders_that_lose_the_capital_first_are_counted_and_said(performance):
    # On 150 rupees, an order that starts -100, -100 is ruined before its +300.
    result = check_of(trade_performance(TRADES, capital=150.0), "reshuffle", runs=600, **SMALL)
    ruined = result.table["ruined"]
    assert ruined.any() and not ruined.all()
    assert "lose the whole capital first" in result.note


def test_the_same_seed_gives_the_same_orders(performance):
    first = check_of(performance, "reshuffle", runs=50, seed=3, **SMALL).table
    again = check_of(performance, "reshuffle", runs=50, seed=3, **SMALL).table
    assert first.equals(again)
