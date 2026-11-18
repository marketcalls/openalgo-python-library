"""The delayed-orders check (design 17, section 2.14) and a strategy that enters again while
its entry is held, with no server.

Under a delay a held entry is a working order (`self.orders`), not a position. A strategy
that decides its entries by its position alone sees none while its entry is held and
enters a second time, so its late run holds a bigger position than the run ever did, and
the check would judge that position, not a late entry.

`Twin` is a daily strategy that wants 10 shares at bars 4j and 4j + 1 and none at bars
4j + 2 and 4j + 3: at a wanted bar's close it buys 10 while flat, at an unwanted bar's
close it exits while it holds. On time its buy of bar 4j fills at bar 4j + 1's open, so
at bar 4j + 1 it holds and buys nothing more, and the exit of bar 4j + 2 sells at bar
4j + 3's open. `CarefulTwin` also buys only with no working order; on time the two trade
the same.

A bar late (`delay_entries=1`) the buy of bar 4j fills at bar 4j + 2's open. At bar 4j + 1
`Twin` is still flat and buys again, an entry held to bar 4j + 3's open, where it adds to
the position (20 units) before the exit of bar 4j + 2 sells all 20. `CarefulTwin` sees
its held entry in `self.orders` and buys nothing, holding 10 units as the run did.
"""

import numpy as np
import pytest
from robust_runs import FREE, opens_bars

import openalgo.backtest as bt
from openalgo.backtest._robust._verdict import NOT_JUDGED, PASSED

BLOCKS = 31
#: Opens 500, 500, 500, 503: the trade buys at 500 and sells at 503, on time (bars 4j + 1
#: and 4j + 3) and a bar late (bars 4j + 2 and 4j + 3).
BLOCK = [500.0, 500.0, 500.0, 503.0]


class Twin(bt.Strategy):
    """Wants 10 shares at bars 4j and 4j + 1 (module docstring), deciding by its position."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def init(self):
        self.bar_number = -1

    def may_enter(self) -> bool:
        return self.position.is_flat

    def on_bar(self, bar):
        self.bar_number += 1
        wanted = self.bar_number % 4 in (0, 1)
        if wanted and self.may_enter():
            self.buy(qty=10)
        elif not wanted and not self.position.is_flat:
            self.exit()


class CarefulTwin(Twin):
    """`Twin` that also buys only with no working order."""

    def may_enter(self) -> bool:
        return self.position.is_flat and not self.orders


def quiet(line: str) -> None:
    """A progress function that drops the plan."""


def run_of(strategy, **settings):
    """`strategy` on `BLOCKS` blocks of `BLOCK` and one more bar."""
    data = opens_bars(np.array(BLOCK * BLOCKS + BLOCK[:1]))
    return bt.backtest(strategy, data=data, **FREE, **settings)


def delayed_of(result):
    """The delayed check alone on `result`, in this process, measured by net profit."""
    report = result.robustness(
        checks=["delayed"], workers=1, objective="net_profit", progress=quiet
    )
    return report.check("delayed")


def test_the_twins_trade_the_same_on_time():
    # 31 trades of 10 shares bought at 500 and sold at 503: +30 each.
    plain, careful = run_of(Twin), run_of(CarefulTwin)
    assert plain.metrics["net_profit"] == careful.metrics["net_profit"] == 30.0 * BLOCKS
    assert plain.metrics["trades"] == careful.metrics["trades"] == BLOCKS


def test_a_strategy_deciding_by_its_position_alone_enters_again_a_bar_late():
    # Each block: the entry at bar 4j + 2's open, then at bar 4j + 3's open the second
    # entry adding 10 and the exit selling all 20.
    fills = run_of(Twin, delay_entries=1).fills
    assert (fills["reason"] == "add").sum() == BLOCKS
    assert set(fills.loc[fills["side"] == "SELL", "quantity_units"]) == {20.0}


def test_a_late_run_that_entered_again_leaves_the_check_not_judged_and_says_why():
    check = delayed_of(run_of(Twin))
    assert check.verdict == NOT_JUDGED
    assert check.note.startswith(
        "With entries one bar late the strategy entered again while its entry was held: "
        "it added to SBIN's position 31 times where the run added 0 times, and held up "
        "to 20 units where the run held at most 10. A held entry is a working order "
        "(self.orders), not a position"
    )
    assert "(self.position.is_flat and not self.orders)" in check.note


def test_the_twin_that_checks_its_working_orders_is_judged():
    # A bar late it buys at 500 (bar 4j + 2) and sells at 503 (bar 4j + 3): +30 a trade,
    # all of the run's kept.
    check = delayed_of(run_of(CarefulTwin))
    assert check.verdict == PASSED
    late = check.table.set_index(["delay_entries", "delay_exits"]).loc[(1, 0)]
    assert late["net_profit"] == 30.0 * BLOCKS and late["ratio"] == 1.0
    assert not check.table["entered_again"].any()


class EvenTwice(bt.Strategy):
    """Buys 10 shares while flat at bars 6j and 6j + 2, deciding by its position, and exits
    at bar 6j + 4. On time and a bar late its entry of bar 6j has filled by bar 6j + 2;
    two bars late it fills at bar 6j + 3's open, so at bar 6j + 2 it buys again."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def init(self):
        self.bar_number = -1

    def on_bar(self, bar):
        self.bar_number += 1
        step = self.bar_number % 6
        if step in (0, 2) and self.position.is_flat:
            self.buy(qty=10)
        elif step == 4 and not self.position.is_flat:
            self.exit()


def test_late_runs_that_entered_again_are_named_while_the_one_bar_late_run_is_judged():
    # Opens 500 at bars 6j to 6j + 4 and 503 at 6j + 5: on time the trade buys at 500
    # (bar 6j + 1) and sells at 503 (bar 6j + 5); a bar late it buys at 500 (bar 6j + 2):
    # +30 a trade either way.
    data = opens_bars(([500.0] * 5 + [503.0]) * BLOCKS + [500.0])
    check = delayed_of(bt.backtest(EvenTwice, data=data, **FREE))
    assert check.verdict == PASSED
    assert (
        "3 delayed runs entered again while an entry was held, so they hold more than the "
        "run ever did: entries 2 late, exits 0 late; entries 2 late, exits 1 late; entries "
        "2 late, exits 2 late. A held entry is a working order (self.orders), not a position."
    ) in check.note
    grew = check.table["entered_again"]
    assert list(grew) == [False, False, False, False, False, True, True, True]


@pytest.mark.parametrize(
    ("run", "late", "grew"),
    [
        # (symbol, adds, largest units) of the run's symbols and of one late symbol.
        # More adds and more units: it entered again.
        (((0, 0, 10.0),), (0, 1, 20.0), True),
        # More units but no more adds: a late entry sized from the equity bought more.
        (((0, 0, 10.0),), (0, 0, 11.0), False),
        # More adds but no more units: a strategy that adds added once more at other
        # prices, holding no more than the run's largest.
        (((0, 2, 30.0),), (0, 3, 30.0), False),
        # A symbol the run never held: any add and any unit is more.
        (((0, 0, 10.0),), (1, 1, 5.0), True),
        # Each symbol against its own: 20 units of symbol 1 are more than its 10, though
        # fewer than symbol 0's 100.
        (((0, 0, 100.0), (1, 0, 10.0)), (1, 1, 20.0), True),
    ],
)
def test_a_rerun_grew_a_symbols_position_when_it_added_more_and_held_more(run, late, grew):
    from openalgo.backtest._robust._position_sizes import PositionSize, grown

    sizes = (PositionSize(*late),)
    assert grown(sizes, tuple(PositionSize(*size) for size in run)) == (sizes if grew else ())
