"""A live strategy's `bt.Result` adds up to its trades, on the fake OpenAlgo (the kernel
follow-up item 1; `engine::venue_session` in the kernel, `_live/_session_close.py`).

A kernel session's stop settles the day's contract notes and records the slice in progress,
as a backtest's step 8 does at a session close: the Result's equity then holds every fill
the broker made, the exit that ended the session included, and the notes' charges. So its
net profit is its trades' net, plus an open position's value at its mark less what its
fills were charged. The day summary's charges are each fill's charges rounded to the paisa
(they add up across the day's restarts); a contract note rounds its lines once, so the
Result's charges are the summary's plus the notes' rounding, which `Result.contract_notes`
names (its `adjustment` column).

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16 and SBIN trades every 10 s, at 500.00 until 14:44 and at 505.00 from
then on. `Hold` buys 10 SBIN at its first decision and never sells: the session's own exit
sells them, on Ctrl+C at 14:46:30 (on_stop="flatten", the default) or at NSE's square-off
at 15:00. Expected values come from the scenario: 10 bought at 500.00 and sold at 505.00
make 50.00 gross.
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from session_samples import Hold, ist, rig
from stops_samples import at_price

import openalgo.backtest as bt

#: SBIN's price from 14:44 on, rupees: the sale's price.
SOLD_AT = 505.0
#: 10 units bought at 500.00 and sold at 505.00, rupees.
GROSS = 50.0
#: How each round trip ends: the stop's cause and the session's word for it.
ENDS = {"ctrl_c": "ctrl_c", "square_off": "session_end"}


@pytest.fixture(params=list(ENDS))
def ended(request, make_fake_openalgo, monkeypatch, tmp_path) -> tuple[str, bt.SessionResult]:
    """`Hold`'s day ended by Ctrl+C or by the square-off: (how, the session's result)."""
    world = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(world, monkeypatch)
    at_price(world, ist(14, 44), SOLD_AT)
    if request.param == "ctrl_c":
        world.stop_at(ist(14, 46, 30))
    return request.param, live(world, Hold)


def note_rounding(result: bt.Result) -> float:
    """Rupees the contract notes of `result` changed the charges by, rounding their lines
    once per note."""
    return float(result.contract_notes["adjustment"].sum())


def test_the_round_trip_is_closed_by_the_sessions_exit(ended):
    how, session = ended
    trades = session.results["Hold"].trades
    assert (session.stopped_by, session.flat) == (ENDS[how], True)
    assert list(trades["exit_price_rupees"]) == [SOLD_AT]
    assert list(trades["gross_pnl_rupees"]) == [GROSS]


def test_a_live_results_net_profit_is_its_trades_net(ended):
    _, session = ended
    result = session.results["Hold"]
    assert result.metrics["net_profit"] == pytest.approx(result.trades["net_pnl_rupees"].sum())


def test_a_live_results_charges_are_its_settled_contract_notes(ended):
    _, session = ended
    result = session.results["Hold"]
    notes = result.contract_notes
    assert len(notes) and notes["settled"].all()
    assert result.metrics["total_costs"] == pytest.approx(notes["charged_total"].sum())
    assert result.trades["costs_rupees"].sum() == pytest.approx(notes["charged_total"].sum())


def test_the_day_summarys_charges_are_the_results_before_the_contract_note_rounding(ended):
    _, session = ended
    result = session.results["Hold"]
    summary = session.summary
    assert summary.gross_pnl == pytest.approx(GROSS)
    charged = result.trades["costs_rupees"].sum()
    assert summary.charges + note_rounding(result) == pytest.approx(charged)
    assert summary.net_pnl - note_rounding(result) == pytest.approx(result.metrics["net_profit"])


def test_a_position_kept_open_counts_its_mark_less_its_charges(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # Ctrl+C with on_stop="keep": the 10 bought at 500.00 stay open, marked at 505.00.
    world = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(world, monkeypatch)
    at_price(world, ist(14, 44), SOLD_AT)
    world.stop_at(ist(14, 46, 30))
    result = live(world, Hold, on_stop="keep").results["Hold"]
    assert result.trades.empty
    assert result.gross_pnl["unrealized"] == pytest.approx(GROSS)
    charged = result.costs["charged"].sum()
    assert charged > 0
    assert result.metrics["net_profit"] == pytest.approx(GROSS - charged)
