"""The warning `bt.backtest` ends with when the strategy waited inside the run for its
indicators (decision D25, `_warm_up`).

Why the strategy waited and what to do is a line of `result.warnings`, but a reader who
prints only a few metrics sees none of it: own data that begins at `start` just shows
fewer trades. So a run in which any symbol waited ends with one `WarmUpWarning` that
gives the first decision and the strategy bars each symbol waited, and points to
`result.warnings` for the rest. Library code never prints; a warning reaches the screen
and can be filtered by its class.

It is issued once per run, by `bt.backtest` only (and so by `Result.rerun`, which calls
it): the combinations of `bt.optimize` run through `_pipeline.execute` directly and never
issue it, in this process or in its workers. It reads the warm-up the manifest records
(`_warm_up.Readiness.manifest`).
"""

from __future__ import annotations

import warnings
from typing import Any

from ._market import PACKAGE_DIR
from ._messages import counted
from ._warm_up import NEVER_DECIDED
from .errors import WarmUpWarning

SEE_DETAILS = "See result.warnings for why."


def _waits(record: dict[str, Any]) -> str | None:
    """The sentence of a wait inside the run, None without one."""
    waited = [(name, bars) for name, bars in record["waited"].items() if bars]
    if not waited:
        return None
    counts = ", ".join(f"{name} waited {counted(bars, 'strategy bar')}" for name, bars in waited)
    if record["first_decision"] == NEVER_DECIDED:
        return (
            "The strategy made no decision: its indicators did not finish their warm-up "
            f"within the run's bars ({counts})."
        )
    return (
        "The strategy waited inside the run for its indicators to warm up and made its "
        f"first decision at {record['first_decision']} ({counts}): signals before then were "
        "not taken."
    )


def wait_notice(record: dict[str, Any]) -> str | None:
    """The text of the warning for the warm-up a manifest records, or None when the
    strategy decided from `start` for every symbol. For example "The strategy waited
    inside the run for its indicators to warm up and made its first decision at
    2026-01-22 15:30 IST (SBIN waited 15 strategy bars): signals before then were not
    taken. See result.warnings for why." A symbol left out (`no_decision`) adds "The
    strategy made no decision for BBB: its indicators were never ready within its
    bars." """
    told = [_waits(record)]
    left_out = record.get("no_decision", [])
    if left_out:
        whose = "its" if len(left_out) == 1 else "their"
        told.append(
            f"The strategy made no decision for {', '.join(left_out)}: {whose} indicators "
            f"were never ready within {whose} bars."
        )
    sentences = [sentence for sentence in told if sentence]
    return " ".join([*sentences, SEE_DETAILS]) if sentences else None


def announce_wait(result: Any) -> None:
    """Issue the notice of a finished run (`wait_notice`) as one WarmUpWarning
    attributed to the caller's line, or nothing when the strategy did not wait.
    `result` is the run's `Result`."""
    notice = wait_notice(result.manifest["arguments"]["warm_up"])
    if notice is not None:
        warnings.warn(notice, WarmUpWarning, skip_file_prefixes=(PACKAGE_DIR,))


__all__ = ["announce_wait", "wait_notice"]
