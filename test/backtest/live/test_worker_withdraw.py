"""What the order worker put back to send later is withdrawn once the session has printed
its last lines (`worker.OrderWorker.withdraw_put_back`, `_worker_queue`; 16-session-design
3.3 step 5 "As built", unit 12): the lines named it as held back and not sent, so it is
never sent after them, and the close does not wait for it.

A protective request is put back while OpenAlgo is silent: an INFY entry's placement times
out first, so the order path sends nothing until OpenAlgo answers a call (a probe) again.
The worker runs on the fake OpenAlgo's manual clock (`worker_samples`).
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from worker_samples import PRICE, WorkerRig

from openalgo.backtest._live._worker_queue import Probe


@pytest.fixture
def silent(fake_openalgo, tmp_path):
    """A worker whose INFY entry timed out: OpenAlgo is silent for its order path."""
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.add_instrument("INFY", "NSE")
    fake_openalgo.trade("SBIN", "NSE", PRICE)
    fake_openalgo.trade("INFY", "NSE", 1500.0)
    made = WorkerRig(fake_openalgo, tmp_path)
    made.messages()
    made.fake.faults.timeout("placeorder")
    made.run(made.place(symbol="INFY"))
    yield made
    made.worker.close()


def answers_again(rig: WorkerRig) -> None:
    """A probe that OpenAlgo answers, then a step a second later."""
    rig.run(Probe())
    rig.clock.advance(1)
    rig.worker.step()


def placed(rig: WorkerRig) -> list[str]:
    """The price type of each placement OpenAlgo was asked for."""
    return [entry.detail["request"]["pricetype"] for entry in rig.fake.log.calls("placeorder")]


def test_a_stop_put_back_before_the_withdrawal_is_not_sent_when_openalgo_answers(silent):
    silent.run(silent.stop())
    silent.worker.withdraw_put_back()
    silent.worker.step()
    answers_again(silent)
    assert placed(silent) == ["MARKET"]


def test_a_withdrawn_request_no_longer_counts_as_unfinished(silent):
    silent.run(silent.stop())
    put_back = silent.worker.unfinished
    silent.worker.withdraw_put_back()
    silent.worker.step()
    assert (put_back, silent.worker.unfinished) == (1, 0)


def test_a_stop_put_back_after_the_withdrawal_is_sent_when_openalgo_answers(silent):
    # Only what was put back before the lines is withdrawn: a request handed over after
    # them (a safe correction of the close's final read) is still tried.
    silent.worker.withdraw_put_back()
    silent.worker.step()
    silent.run(silent.stop())
    answers_again(silent)
    assert placed(silent) == ["MARKET", "SL-M"]


def test_a_stop_handed_over_and_not_tried_yet_is_not_withdrawn(silent):
    # Handed over before the lines and not tried by then, it was named as handed to the
    # order worker and maybe not sent: it is tried as any request.
    answers_again(silent)
    silent.worker.put(silent.stop())
    silent.worker.withdraw_put_back()
    silent.worker.step()
    assert placed(silent) == ["MARKET", "SL-M"]
