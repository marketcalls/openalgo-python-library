"""Speed of the live session's loop: a tick from its arrival at the loop to the kernel's
risk rules (16-session-design 2.5 step A.1 and 2.9; 15-phase2-plan section 4).

    python benchmark/live_loop_speed.py              # print the figures
    python benchmark/live_loop_speed.py --check      # also exit 1 when the target is missed
    python benchmark/live_loop_speed.py --check --slack 2   # a slower machine (CI)

The session runs in this process against the fake OpenAlgo of the test suite
(test/backtest/live/fake_openalgo): no server, no socket, every connection refused. Its
own loop (`Session.step`) runs as under the test seam: the order worker's and the data
worker's steps inline, the fake's direct feed handing each tick to the session's feed
callbacks, the fake's manual clock moved on by the script while the loop waits. Two
strategies with leg rules (a stop 3 points below the entry, trailing 3 points below the
best price, and a target 50 points above) buy 10 SBIN and 10 INFY on NSE at 14:42:01 IST
and hold them; both trade every `TRADE_EVERY_MS` of the fake's clock from 14:40:05 to
14:58, rising a paisa a second from 14:42, so every tick runs the legs' rules (the trail
ratchets, its modifies throttled and sent through the order worker to the fake).

Measured per tick on the wall clock (`time.perf_counter_ns`): from the moment the loop
begins step A.1 for it (the tick adapter is called) to the return of the kernel's
`push_tick` (tick mode: the trail, the target, the mark, the daily loss; R5), for every
tick that reached the kernel. The two stamps are taken by wrappers of the adapter and of
the kernel call, a fraction of a microsecond each. Printed: the ticks measured, the
median, the 99th percentile and the worst, in microseconds. The worst is shown, not
checked: one pause of the interpreter's garbage collector or of the machine lands there.

`--check` exits 1 when the 99th percentile reaches the target, or when the run measured
fewer than `MIN_TICKS` ticks or made no trailing modify (a check would then pass on
nothing). `--slack` multiplies the target by its value, for a machine slower than the
one it was set on.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Run against this checkout when it holds a built extension (a local build copied next
# to the package); otherwise against the installed wheel. The fake OpenAlgo lives with
# the live layer's tests, and its feed shares the feed tests' helpers.
ROOT = Path(__file__).resolve().parent.parent
if list((ROOT / "openalgo").glob("_oaindicators*")):
    sys.path.insert(0, str(ROOT))
sys.path[:0] = [str(ROOT / "test" / "backtest" / "live"), str(ROOT / "test")]

import pytest  # noqa: E402
from fake_openalgo import FakeOpenAlgo  # noqa: E402
from fake_openalgo.fixtures import guard_connections  # noqa: E402
from plan_samples import candles, ist, market  # noqa: E402
from session_samples import Rig, Script  # noqa: E402

import openalgo.backtest as bt  # noqa: E402
from openalgo.backtest._live import _session_feed, _stops_kernel  # noqa: E402

# ----- target ----------------------------------------------------------------------
# 15-phase2-plan section 4 ("a tick reaches the risk rules in well under a millisecond on
# the loop") and 16-session-design 2.9 ("a tick reaches push_tick in well under a
# millisecond of loop time"): the repository keeps 500 microseconds at the 99th
# percentile, half a millisecond.
TICK_TO_PUSH_TICK_P99_US_TARGET = 500.0
#: The fewest ticks a run must measure for its 99th percentile to stand on 100 of them.
MIN_TICKS = 10_000

# ----- the market --------------------------------------------------------------------
#: Milliseconds of the fake's clock between two trades of each symbol.
TRADE_EVERY_MS = 100
#: The session's start and the end of the trades (IST), and its Ctrl+C (keep the legs).
START = (14, 40, 5)
TRADES_END = (14, 58, 0)
STOP_AT = (14, 58, 0)
#: The price before 14:42, rupees, and its rise from then, rupees a second.
BASE_PRICE = 500.0
RISE_PER_S = 0.01
SYMBOLS = ("SBIN", "INFY")
NS_PER_US = 1_000


class LegsSbin(bt.Strategy):
    """Buys 10 SBIN with a stop, a trailing stop and a target when it holds none."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    rules: dict[str, Any] = {
        "stop": bt.points(3),
        "trail": bt.trail_points(3),
        "target": bt.points(50),
    }

    def on_bar(self, bar):
        if not self.position.qty and not self.orders:
            self.buy(10, **type(self).rules)


class LegsInfy(LegsSbin):
    """`LegsSbin` on INFY."""

    symbol = "INFY"


class Stamps:
    """The wall-clock stamps of each tick's way from step A.1 to `push_tick`'s return."""

    def __init__(self) -> None:
        self.latencies_ns: list[int] = []
        self._began: int | None = None

    def install(self, patch: pytest.MonkeyPatch) -> None:
        adapt, call = _session_feed.tick_from_feed, _stops_kernel.call

        def began(*args: Any, **kwargs: Any) -> Any:
            self._began = time.perf_counter_ns()
            return adapt(*args, **kwargs)

        def called(func: Callable[..., Any], *args: Any) -> Any:
            result = call(func, *args)
            if self._began is not None and getattr(func, "__name__", "") == "push_tick":
                self.latencies_ns.append(time.perf_counter_ns() - self._began)
                self._began = None
            return result

        patch.setattr(_session_feed, "tick_from_feed", began)
        patch.setattr(_stops_kernel, "call", called)


def price_at(now_ns: int) -> float:
    rising_s = max(0.0, (now_ns - ist(14, 42)) / 1e9)
    return round(BASE_PRICE + RISE_PER_S * rising_s, 2)


def session_run() -> tuple[Stamps, int]:
    """Run the session of the module docstring; (its stamps, the trailing modifies the
    fake took)."""
    with (
        pytest.MonkeyPatch.context() as patch,
        tempfile.TemporaryDirectory() as folder,
        warnings.catch_warnings(),
    ):
        # The session's own warnings (the cost schedule's rates verified only up to an
        # earlier date than the fake's day) say nothing of its speed.
        warnings.simplefilter("ignore")
        guard_connections(patch)
        fake = market(
            lambda **options: FakeOpenAlgo(**options),
            patch,
            Path(folder),
            at=ist(*START),
            today_until=START[:2],
        )
        try:
            fake.add_instrument("INFY", "NSE")
            fake.seed_candles("INFY", "NSE", candles(today_until=START[:2]))
            script = Script(fake)
            for symbol in SYMBOLS:
                fake.trade(symbol, "NSE", BASE_PRICE, volume=10.0)
            step_ns = TRADE_EVERY_MS * 1_000_000
            for instant in range(ist(*START) + step_ns, ist(*TRADES_END), step_ns):
                for symbol in SYMBOLS:
                    script.at(
                        instant,
                        lambda s=symbol, t=instant: fake.trade(s, "NSE", price_at(t), volume=10.0),
                    )
            stamps = Stamps()
            stamps.install(patch)
            rig = Rig(fake, script, Path(folder), trace=None)
            rig.stop_at(ist(*STOP_AT))
            rig.run([LegsSbin, LegsInfy], on_stop="keep")
            problems = fake.log.problems()
            if problems:
                raise RuntimeError(f"the fake OpenAlgo logged problems: {problems}")
            return stamps, len(fake.log.calls("modifyorder"))
        finally:
            fake.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when the target is missed")
    parser.add_argument("--slack", type=float, default=1.0, help="relax the target by this")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    stamps, modifies = session_run()
    took_s = time.perf_counter() - started
    latencies = np.array(stamps.latencies_ns, dtype=np.float64) / NS_PER_US
    median, p99 = np.percentile(latencies, [50, 99]) if len(latencies) else (np.nan, np.nan)
    worst = latencies.max() if len(latencies) else np.nan
    target = TICK_TO_PUSH_TICK_P99_US_TARGET * args.slack
    met = bool(p99 < target)
    print(f"live loop, tick from step A.1 to push_tick ({took_s:.1f} s for the run)")
    print(f"  ticks measured     : {len(latencies):>9,}")
    print(f"  trailing modifies  : {modifies:>9,}")
    print(f"  median             : {median:>9.1f} us")
    print(f"  99th percentile    : {p99:>9.1f} us   target < {target:.0f} us: "
          f"{'met' if met else 'MISSED'}")  # fmt: skip
    print(f"  worst              : {worst:>9.1f} us")
    if not args.check:
        return 0
    failures = []
    if not met:
        failures.append(f"99th percentile {p99:.1f} us reaches {target:.0f} us")
    if len(latencies) < MIN_TICKS:
        failures.append(f"only {len(latencies)} ticks measured, fewer than {MIN_TICKS}")
    if modifies == 0:
        failures.append("no trailing modify: the legs' rules did not run")
    for failure in failures:
        print(f"check failed (slack {args.slack}): {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
