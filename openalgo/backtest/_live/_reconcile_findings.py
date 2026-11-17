"""What the reconciler (`reconcile`) does with the findings of its comparisons
(16-session-design 5.3; unit 12): a finding is noted when first seen, acted on once when
seen in two comparisons in a row (`_reconcile_act.Corrections`), and is a row of the
day's reconciliations (`Reconciliation`) at those two sightings only, not at each read
while it lasts; once it has been absent from a comparison, a sighting is a first one
again. An unexplained difference that lasts past the in-flight limit is an ERROR alert,
once, to check the account (5.3 row 3).

Quantities are signed units, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._reconcile_act import Corrections, Finding
from ._reconcile_rules import Kind, LegBooks
from ._session_trader import Trader
from ._types import AlertKind


@dataclass(frozen=True)
class Reconciliation:
    """A difference found: when, the strategy, what differed (`kind`, a
    `_reconcile_rules.Kind` value or "orphan", "foreign", "stop_terms"), the engine's and
    the broker's units (None: not read), and what was done ("noted" at first sight)."""

    at_ns: int
    strategy: str
    kind: str
    engine_units: int
    broker_units: int | None
    action: str


class Findings:
    """The findings of one session's comparisons (module docstring). `session` is the
    `_session_loop.Session`, `corrections` the reconciler's safe corrections, and
    `lag_alert_after_s` the seconds an unexplained difference lasts before its ERROR alert;
    `rows` are the day's reconciliations."""

    def __init__(self, session: Any, corrections: Corrections, lag_alert_after_s: float) -> None:
        self._session = session
        self._corrections = corrections
        self._lag_after_ns = round(lag_alert_after_s * NANOS_PER_SECOND)
        self.rows: list[Reconciliation] = []
        self._last: dict[str, set[Finding]] = {}
        # The findings of each leg seen twice in a row since they were last absent.
        self._confirmed: dict[str, set[Finding]] = {}
        self._acted: set[tuple[str, Finding]] = set()
        self._lag_since: dict[str, int] = {}
        self._lag_alerted: set[str] = set()

    def at_start(
        self, trader: Trader, kind: Kind, journal_units: int, books: LegBooks, own: frozenset[str]
    ) -> None:
        """A restart's difference of `trader`'s journal position (`journal_units`) from the
        broker's: acted on at once, as two comparisons in a row would be (5.5)."""
        finding = (kind, journal_units, books.units)
        self._acted.add((trader.name, finding))
        action = self._corrections.act(trader, finding, books, own)
        self._row(trader.name, kind.value, journal_units, books.units, action)

    def judged(
        self,
        trader: Trader,
        findings: set[Finding],
        books: LegBooks,
        own: frozenset[str],
        *,
        late: bool,
    ) -> bool:
        """Take one comparison's `findings` of `trader`'s leg (module docstring); `late` for
        a leg compared while in flight past the limit (alerted, nothing done). True when a
        finding was seen for the first time: a read soon confirms it."""
        name = trader.name
        last, self._last[name] = self._last.get(name, set()), findings
        confirmed = self._confirmed.setdefault(name, set())
        confirmed &= findings
        if not any(finding[0] is Kind.UNEXPLAINED for finding in findings):
            self._lag_since.pop(name, None)
        for finding in sorted(findings, key=repr):
            if finding[0] is Kind.UNEXPLAINED:
                self._lag(trader, books)
            if finding in confirmed:
                continue
            action = "noted"
            if finding in last:
                confirmed.add(finding)
                action = "acted"
                if (name, finding) not in self._acted:
                    self._acted.add((name, finding))
                    corrections = self._corrections
                    act = corrections.alert_only if late else corrections.act
                    action = act(trader, finding, books, own)
            kind = finding[0].value if isinstance(finding[0], Kind) else finding[0]
            engine = finding[1] if isinstance(finding[0], Kind) else round(trader.open_units())
            self._row(name, kind, engine, books.units, action)
        return bool(findings - last)

    def _row(
        self, name: str, kind: str, engine_units: int, broker_units: int | None, action: str
    ) -> None:
        self.rows.append(
            Reconciliation(self._session.now(), name, kind, engine_units, broker_units, action)
        )
        self._session.event("mismatch", strategy=name, found=kind, action=action)

    def _lag(self, trader: Trader, books: LegBooks) -> None:
        """An unexplained difference that lasts past the in-flight limit: an ERROR alert,
        once, to check the account (5.3 row 3)."""
        now_ns = self._session.now()
        since = self._lag_since.setdefault(trader.name, now_ns)
        if now_ns - since > self._lag_after_ns and trader.name not in self._lag_alerted:
            self._lag_alerted.add(trader.name)
            self._session.alert(
                AlertKind.ERROR,
                f"{trader.name}: OpenAlgo's books have shown {books.units} units of "
                f"{trader.symbol} {trader.exchange} against the engine's "
                f"{round(trader.open_units())} for over "
                f"{self._lag_after_ns / NANOS_PER_SECOND:g} s with no trade to explain it: "
                "check the account in OpenAlgo",
            )


__all__ = ["Findings", "Reconciliation"]
