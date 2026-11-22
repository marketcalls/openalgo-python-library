"""Every per-strategy setting of `bt.live` by strategy class name (`_live/_plan_args.py`,
`_strategy_args.py`; 18-multi-strategy 1.1, 1.3 and decision M5): a dict names each
strategy's own value, one value still applies to every strategy, and a strategy's class
attribute of the same name still wins, as in `bt.backtest`. The table of the rules both
verbs share is `test/backtest/test_strategy_args.py`; these tests follow the values into
each strategy's settings for the day (`plan.specify`).

`specify` runs on a clock at 10:15 IST of Monday 2026-11-16 and contacts nothing: each
strategy's `init` runs on the engine's reference bars only. Expected values are the
call's own.
"""

from __future__ import annotations

import pytest
from plan_samples import EmaTrend, specify

from openalgo.backtest.errors import ConfigError


class Fast(EmaTrend):
    """EmaTrend under a name of its own."""


class Slow(EmaTrend):
    """EmaTrend under a name of its own."""


class Early(EmaTrend):
    """Squares off at 14:30 whatever the run says: a class attribute."""

    square_off = "14:30"


def plans(strategies, **arguments) -> dict:
    """Each strategy's settings for the day, by name, of a session of `strategies`."""
    spec = specify(strategies, **arguments)
    return {each.name: each.plan for each in spec.strategies}


def test_each_strategy_takes_its_own_value_of_a_setting_given_by_name():
    found = plans(
        [Fast, Slow],
        entry_from={"Fast": "10:30", "Slow": "11:00"},
        entry_until={"Fast": "14:00", "Slow": "14:20"},
        square_off={"Fast": "14:45", "Slow": "15:00"},
    )
    times = {name: (p.entry_from, p.entry_until, p.square_off) for name, p in found.items()}
    assert times == {
        "Fast": ("10:30", "14:00", "14:45"),
        "Slow": ("11:00", "14:20", "15:00"),
    }


def test_each_strategy_takes_its_own_risk_rules_given_by_name():
    # after_risk_exit takes "lock_session" only in this version, so both name it.
    found = plans(
        [Fast, Slow],
        max_positions={"Fast": 1, "Slow": 2},
        cooldown={"Fast": 3, "Slow": 5},
        drawdown_halt={"Fast": 10, "Slow": 15},
        after_risk_exit={"Fast": "lock_session", "Slow": "lock_session"},
    )
    rules = ("max_positions", "cooldown_bars", "drawdown_halt", "after_risk_exit")
    assert {name: tuple(p.risk[rule] for rule in rules) for name, p in found.items()} == {
        "Fast": (1, 3, 10.0, "lock_session"),
        "Slow": (2, 5, 15.0, "lock_session"),
    }


def test_one_value_still_applies_to_every_strategy():
    found = plans([Fast, Slow], square_off="14:50", max_positions=1)
    assert {name: (p.square_off, p.risk["max_positions"]) for name, p in found.items()} == {
        "Fast": ("14:50", 1),
        "Slow": ("14:50", 1),
    }


def test_a_class_attribute_still_wins_over_the_value_given_by_name():
    found = plans([Early, Slow], square_off={"Early": "14:45", "Slow": "14:45"})
    assert {name: p.square_off for name, p in found.items()} == {
        "Early": "14:30",
        "Slow": "14:45",
    }


def test_a_setting_left_out_takes_the_backtests_default_for_every_strategy():
    found = plans([Fast, Slow])
    assert {name: (p.square_off, p.risk["cooldown_bars"]) for name, p in found.items()} == {
        "Fast": (None, None),
        "Slow": (None, None),
    }


@pytest.mark.parametrize(
    ("given", "words"),
    [
        (
            {"square_off": {"Fast": "14:45"}},
            "square_off as a dict must name every strategy of the run",
        ),
        (
            {"cooldown": {"Fast": 3, "Slow": 3, "Other": 3}},
            "cooldown names ['Other'], which this session does not run",
        ),
    ],
    ids=["a strategy left out", "a strategy the session does not run"],
)
def test_a_dict_by_name_is_refused_before_anything_is_called(given, words):
    with pytest.raises(ConfigError) as refused:
        specify([Fast, Slow], **given)
    assert words in str(refused.value)
