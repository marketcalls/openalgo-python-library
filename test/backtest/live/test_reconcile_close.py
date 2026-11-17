"""What the close says of a leg whose difference with OpenAlgo's books no later read will
confirm (`_live/_reconcile_close.py`; 16-session-design 5.4 "As built", rounds 7 and 8;
unit 12).

A comparison saw OpenAlgo's books differ from the engine for the first time (a
`Sighting`), and the session ended before a later read confirmed it. The engine's units
may have moved since by its own orders' fills, which the broker made too: OpenAlgo's
figure moves by the same units, and the close names the read's figures, the fills since
and what they leave. Expected values come from the scenarios: `StopHold` long 10 SBIN with
4 or 10 sold by hand, its stop of 10 filling or not after the read.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openalgo.backtest._live._reconcile_close import Differing, differing
from openalgo.backtest._live._reconcile_findings import Sighting

#: The leg as the close names it.
TRADER = SimpleNamespace(name="StopHold", symbol="SBIN", exchange="NSE")


@pytest.mark.parametrize(
    ("seen", "engine_now", "broker_now"),
    [
        (Sighting(10, 6), 10, 6),
        (Sighting(10, 6), 0, -4),
        (Sighting(10, 0), 10, 0),
        (Sighting(10, 6), 4, 0),
        (Sighting(0, -4), 0, -4),
        (Sighting(-5, -2), 0, 3),
    ],
    ids=[
        "4 sold by hand, nothing since",
        "4 sold by hand, the stop of 10 filled since",
        "all sold by hand, nothing since",
        "4 sold by hand, the engine sold 6 since",
        "short 4 after the stop filled, nothing since",
        "short 5, 3 bought by hand, the engine bought 5 since",
    ],
)
def test_openalgos_figure_moves_by_the_engines_own_fills_since_the_read(
    seen, engine_now, broker_now
):
    assert Differing(seen, engine_now).broker_units == broker_now


@pytest.mark.parametrize(
    ("seen", "engine_now", "flat"),
    [(Sighting(10, 0), 10, True), (Sighting(10, 6), 4, True), (Sighting(10, 6), 0, False)],
    ids=["all sold by hand", "the rest sold by the engine since", "short 4"],
)
def test_a_leg_is_flat_at_openalgo_when_its_figure_comes_to_no_position(seen, engine_now, flat):
    assert Differing(seen, engine_now).flat_at_openalgo is flat


def test_the_words_name_the_reads_figures_alone_when_no_fill_came_after_it():
    assert Differing(Sighting(10, 6), 10).words() == (
        "OpenAlgo's books showed 6 units against the engine's 10 at the last read, and the "
        "session ended before they agreed: nothing was sent for the difference"
    )


def test_the_words_name_no_position_for_a_flat_figure():
    assert Differing(Sighting(10, 0), 10).words() == (
        "OpenAlgo's books showed no position against the engine's 10 at the last read, and "
        "the session ended before they agreed: nothing was sent for the difference"
    )


@pytest.mark.parametrize(
    ("seen", "engine_now", "since"),
    [
        (
            Sighting(10, 6),
            0,
            "the engine's own fills (10 units sold) came after it, which would leave -4 units "
            "in OpenAlgo's books against the engine's 0",
        ),
        (
            Sighting(10, 6),
            4,
            "the engine's own fills (6 units sold) came after it, which would leave no "
            "position in OpenAlgo's books against the engine's 4",
        ),
        (
            Sighting(-5, -2),
            0,
            "the engine's own fills (5 units bought) came after it, which would leave 3 units "
            "in OpenAlgo's books against the engine's 0",
        ),
    ],
    ids=["the stop of 10 filled", "the engine sold 6", "the engine bought 5"],
)
def test_the_words_name_the_fills_after_the_read_and_what_they_leave(seen, engine_now, since):
    words = Differing(seen, engine_now).words()
    read = (
        f"OpenAlgo's books showed {seen.broker_units} units against the engine's "
        f"{seen.engine_units} at the last read"
    )
    assert words == (
        f"{read}; {since}, and the session ended before they agreed: nothing was sent for "
        "the difference"
    )


@pytest.mark.parametrize(
    ("seen", "engine_now", "summary"),
    [
        (Sighting(10, 6), 10, "StopHold SBIN NSE 6 units in OpenAlgo's books, the engine's 10"),
        (
            Sighting(10, 6),
            0,
            "StopHold SBIN NSE -4 units in OpenAlgo's books by the last read and the engine's "
            "own fills since, the engine's 0",
        ),
        (
            Sighting(10, 0),
            10,
            "StopHold SBIN NSE no position in OpenAlgo's books, the engine's 10",
        ),
    ],
    ids=["nothing since", "the stop of 10 filled since", "all sold by hand"],
)
def test_the_summary_names_openalgos_figure_against_the_engines_units_now(
    seen, engine_now, summary
):
    assert Differing(seen, engine_now).summary(TRADER) == summary


def test_a_leg_with_no_unconfirmed_difference_has_none_at_the_close():
    reconcile = SimpleNamespace(unconfirmed=lambda name: None)
    trader = SimpleNamespace(name="StopHold", open_units=lambda: 10.0)
    assert differing(reconcile, trader) is None


def test_the_close_takes_the_engines_units_now_beside_the_sighting():
    seen = Sighting(10, 6)
    reconcile = SimpleNamespace(unconfirmed={"StopHold": seen}.get)
    trader = SimpleNamespace(name="StopHold", open_units=lambda: 0.0)
    assert differing(reconcile, trader) == Differing(seen, 0)
