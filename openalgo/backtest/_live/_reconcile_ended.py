"""What the reconciler (`reconcile`) tells the close once it has printed its last lines
(16-session-design 5.4 "As built", rounds 7 to 9, the unit 12 and unit 13 follow-ups):
the queries the close, the stops supervisor and the final read ask of a leg's difference
with OpenAlgo's books that no later read will confirm.

`EndQueries` is a part of `reconcile.Reconciler`, which keeps the findings (`_findings`, a
`_reconcile_findings.Findings`), the exit guard (`guard`) and whether the close has ended
it (`_ended`). Before `end` none of the close's queries names anything: a comparison may
still confirm what a leg's last one saw.

- `unconfirmed`: the position difference a leg's last comparison saw for the first time,
  carried to the close by the engine's own fills since (`_reconcile_close`).
- `unjudged`: a leg no comparison judged after a fill of its own orders landed during a
  read.
- `late`: the difference a leg's last comparison saw when it was late (the leg in flight
  past the in-flight limit), named as that read saw it; `late_counted` when comparisons in
  time had confirmed it and acted on it while it lasted, so the close names it and counts
  it no more.
- `broker_units`: OpenAlgo's units of a leg while they differ from the engine's with
  nothing taking them in.

Quantities are signed units. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from ._reconcile_findings import Sighting


class EndQueries:
    """The close's queries of a `reconcile.Reconciler` (module docstring)."""

    _findings: Any
    _ended: bool
    guard: Any

    def end(self) -> None:
        """The close has printed its last lines: the exit guard ends (`ExitGuard.end`),
        and no later comparison will confirm a difference first seen (module docstring)."""
        self._ended = True
        self._findings.end()
        self.guard.end()

    def unconfirmed(self, name: str) -> Sighting | None:
        """Once the close has printed its last lines, the position difference strategy
        `name`'s last comparison saw for the first time, else None; None too when its last
        comparison was late with a difference (`late` names the leg then)."""
        if not self._ended or self._findings.late(name) is not None:
            return None
        return self._findings.unconfirmed(name)

    def unjudged(self, name: str) -> bool:
        """Once the close has printed its last lines, True when no comparison judged
        strategy `name`'s position since a fill of its own orders landed during a read,
        unless a late comparison since saw a difference (`late` names the leg then)."""
        findings = self._findings
        return self._ended and findings.unjudged(name) and findings.late(name) is None

    def late(self, name: str) -> Sighting | None:
        """Once the close has printed its last lines, the position difference strategy
        `name`'s last comparison saw when it was late (the leg in flight past the limit):
        named at the close as that read saw it, neither carried nor confirmed; else None."""
        return self._findings.late(name) if self._ended else None

    def late_counted(self, name: str) -> bool:
        """True when strategy `name`'s late difference (`late`) is one comparisons in time
        confirmed and acted on while it lasted: a mismatch of the day already, which the
        close names but does not count again."""
        return self.late(name) is not None and self._findings.late_counted(name)

    def broker_units(self, name: str) -> int | None:
        """OpenAlgo's signed units of strategy `name`'s leg at the last read while they
        differ from the engine's with nothing taking them in: while the exit guard holds
        the leg (`ExitGuard.broker_units`), and after the last lines when no later
        comparison will confirm the difference (`unconfirmed`); else None."""
        guarded, seen = self.guard.broker_units(name), self.unconfirmed(name)
        return guarded if guarded is not None or seen is None else seen.broker_units


__all__ = ["EndQueries"]
