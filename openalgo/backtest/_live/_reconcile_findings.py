"""What the reconciler (`reconcile`) does with the findings of its comparisons
(16-session-design 5.3; unit 12): a finding is noted when first seen, acted on once when
seen in two comparisons in a row (`_reconcile_act.Corrections`), and is a row of the
day's reconciliations (`Reconciliation`) at those two sightings only, not at each read
while it lasts; once it has been absent from a comparison, a sighting is a first one
again. An unexplained difference that lasts past the in-flight limit is an ERROR alert,
once, to check the account (5.3 row 3). A leg whose last comparison saw a position
difference for the first time has it as its `Sighting` until a later comparison judges
the position again (`unconfirmed`): at the close no later read will confirm it, so the
close reports it (16-session-design 5.4 "As built", round 7).

Only a comparison that judges the position takes or ends a sighting: one that read both
books, attributed the trade book's own fills first and found none new (5.4, round 9). A
read whose trade book gave the kernel a fill of the leg's own orders took its position
book first, which may predate the fill: it judges nothing, keeps the sighting the round 8
carry then moves by the fill, and leaves the leg unjudged (`fill_during_read`,
`unjudged`) until a comparison judges it. A late comparison (the leg in flight past the
in-flight limit, its trades not attributed) only alerts, at once when it first sees a
difference and once per difference (a comparison in time still acts on it): it takes no
sighting, so no later carry can count a fill its position book already held, and judges
nothing. A leg whose last comparison that read the position book was late and found a
position difference keeps it (`late`) for the close, which names its figures as that read
saw them, neither carried nor confirmed (16-session-design 5.4, the unit 12 follow-up).
One that comparisons in time confirmed and acted on while it lasted is kept and named the
same way, but marked counted (`late_counted`): it was a mismatch of the day then, and the
close counts it no more (the unit 13 follow-up, lead's rule of 2026-11-18). A member of
an instrument several strategies of the session share keeps, with its sighting, its group
as that comparison judged it (18-multi-strategy 4.11): OpenAlgo's figure is the group's
net, which the close weighs against the members' sum.

Quantities are signed units, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._group_rules import GroupFigure, and_join
from ._reconcile_act import Corrections
from ._reconcile_rules import Kind, LegBooks
from ._reconcile_steps import Finding
from ._session_trader import Trader
from ._types import AlertKind

#: What a row says was done of a late comparison's finding after the close's last lines.
NAMED_AT_THE_CLOSE = "named at the close"
#: What the "reconcile" event of a leg says it found when a fill of the leg's own orders
#: landed during the read: nothing judged (module docstring).
FILL_DURING_READ = "fill_during_read"


@dataclass(frozen=True)
class Sighting:
    """A position difference a comparison saw for the first time: the engine's signed
    units as compared and OpenAlgo's books'; for a member of a shared instrument, its
    `group` as the comparison judged it, whose net OpenAlgo's books showed (None alone)."""

    engine_units: int
    broker_units: int
    group: GroupFigure | None = None


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
        # By strategy: the position difference its last judged comparison saw first.
        self._first: dict[str, Sighting] = {}
        # The strategies no comparison judged since a fill landed during a read.
        self._unjudged: set[str] = set()
        # By strategy: the position difference its last comparison saw, when it was late.
        self._late: dict[str, Sighting] = {}
        # The strategies whose late difference comparisons in time acted on and counted.
        self._late_counted: set[str] = set()
        self._late_alerted: set[tuple[str, Finding]] = set()
        self._ended = False

    def end(self) -> None:
        """The close has printed its last lines: a late comparison from now on alerts
        nothing at once, since the close's own alert names its difference."""
        self._ended = True

    def unconfirmed(self, name: str) -> Sighting | None:
        """The position difference strategy `name`'s last comparison that judged the
        position saw for the first time, or None (module docstring)."""
        return self._first.get(name)

    def late(self, name: str) -> Sighting | None:
        """The position difference strategy `name`'s last comparison that read the
        position book saw, when that comparison was late; else None (module docstring)."""
        return self._late.get(name)

    def late_counted(self, name: str) -> bool:
        """True when strategy `name`'s late difference (`late`) was confirmed and acted on
        by comparisons in time while it lasted: a mismatch counted then (module
        docstring)."""
        return name in self._late and name in self._late_counted

    def fill_during_read(self, name: str) -> None:
        """A read's trade book gave the kernel a fill of strategy `name`'s own orders: that
        read judges nothing of the leg, which stays unjudged until a comparison judges it
        (module docstring). A late difference seen before it is no longer the last word."""
        self._unjudged.add(name)
        self._late.pop(name, None)

    def unjudged(self, name: str) -> bool:
        """True when no comparison has judged strategy `name`'s position since a fill of
        its own orders landed during a read."""
        return name in self._unjudged

    def at_start(
        self,
        trader: Trader,
        kind: Kind,
        journal_units: int,
        books: LegBooks,
        own: frozenset[str],
        *,
        stale: bool = False,
    ) -> None:
        """A restart's difference of `trader`'s journal position (`journal_units`) from the
        broker's: acted on at once, as two comparisons in a row would be (5.5); `stale`
        when the restored engine holds other units, so it is only locked and alerted
        (`Corrections.stale`) and the next comparisons judge the leg."""
        finding = (kind, journal_units, books.units)
        if stale:
            action = self._corrections.stale(trader, finding, books, own)
        else:
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
        group: GroupFigure | None = None,
    ) -> bool:
        """Take one comparison's `findings` of `trader`'s leg (module docstring); `late` for
        a leg compared while in flight past the limit (alerted at once, nothing done, no
        sighting taken, its position difference kept for the close); `group` the leg's
        group on a shared instrument, as the comparison judged it. True when a finding was
        seen for the first time: a read soon confirms it."""
        name = trader.name
        last, self._last[name] = self._last.get(name, set()), findings
        confirmed = self._confirmed.setdefault(name, set())
        confirmed &= findings
        if not any(finding[0] is Kind.UNEXPLAINED for finding in findings):
            self._lag_since.pop(name, None)
        if books.units is not None and books.trades is not None and not late:
            self._sighted(name, findings, last, group)
            self._unjudged.discard(name)
        if books.units is not None:
            self._late_difference(name, findings, confirmed, late=late, group=group)
        for finding in sorted(findings, key=repr):
            if finding[0] is Kind.UNEXPLAINED:
                self._lag(trader, books)
            if finding in confirmed:
                continue
            if late:
                if finding in last:
                    confirmed.add(finding)
                action = self._alert_once(trader, finding, books, own)
            else:
                action = self._act_when_seen_twice(trader, finding, books, own, last)
            kind = finding[0].value if isinstance(finding[0], Kind) else finding[0]
            engine = finding[1] if isinstance(finding[0], Kind) else round(trader.open_units())
            self._row(name, kind, engine, books.units, action)
        return bool(findings - last)

    def _act_when_seen_twice(
        self,
        trader: Trader,
        finding: Finding,
        books: LegBooks,
        own: frozenset[str],
        last: set[Finding],
    ) -> str:
        """A finding of a comparison in time: noted when first seen, acted on once when the
        comparison before saw it too. What was done."""
        if finding not in last:
            return "noted"
        self._confirmed[trader.name].add(finding)
        if (trader.name, finding) in self._acted:
            return "acted"
        self._acted.add((trader.name, finding))
        return self._corrections.act(trader, finding, books, own)

    def _alert_once(
        self, trader: Trader, finding: Finding, books: LegBooks, own: frozenset[str]
    ) -> str:
        """A finding of a late comparison: alerted at once, once; after the last lines
        left to the close's alert (module docstring)."""
        key = (trader.name, finding)
        if self._ended:
            return NAMED_AT_THE_CLOSE
        if key not in self._late_alerted:
            self._late_alerted.add(key)
            return self._corrections.alert_only(trader, finding, books, own)
        return "alerted"

    def _late_difference(
        self,
        name: str,
        findings: set[Finding],
        confirmed: set[Finding],
        *,
        late: bool,
        group: GroupFigure | None,
    ) -> None:
        """Keep the position difference of a late comparison for the close; a comparison
        in time, or a late one that agrees, ends it. One that comparisons in time confirmed
        and acted on while it lasted (`confirmed`) is kept too, for the close to name, and
        marked counted: it was counted then, and the close must not count it again (the
        in-time path takes no sighting of it either)."""
        position = next((f for f in findings if isinstance(f[0], Kind)), None)
        if late and position is not None:
            self._late[name] = Sighting(position[1], position[2], group)
            if position in confirmed and (name, position) in self._acted:
                self._late_counted.add(name)
            else:
                self._late_counted.discard(name)
        else:
            self._late.pop(name, None)
            self._late_counted.discard(name)

    def _sighted(
        self, name: str, findings: set[Finding], last: set[Finding], group: GroupFigure | None
    ) -> None:
        """A comparison that judged the position: its difference, when the comparison
        before did not show it, is the leg's first sighting; none otherwise."""
        position = next((f for f in findings if isinstance(f[0], Kind)), None)
        if position is None or position in last:
            self._first.pop(name, None)
        else:
            self._first[name] = Sighting(position[1], position[2], group)

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
            engine = f"the engine's {round(trader.open_units())}"
            group = self._corrections.figures.get(trader.name)
            if group is not None:
                engine = f"the {group.units} {and_join(group.members)} hold together"
            self._session.alert(
                AlertKind.ERROR,
                f"{trader.name}: OpenAlgo's books have shown {books.units} units of "
                f"{trader.symbol} {trader.exchange} against {engine} for over "
                f"{self._lag_after_ns / NANOS_PER_SECOND:g} s with no trade to explain it: "
                "check the account in OpenAlgo",
            )


__all__ = ["FILL_DURING_READ", "NAMED_AT_THE_CLOSE", "Findings", "Reconciliation", "Sighting"]
