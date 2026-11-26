"""What a strategy's GTT map (`_venue_gtts.GttMap`) knows of each of the kernel's GTTs
(`KernelGtt`), and its restart and take-back handling (`GttTakeBack`), split from
`_venue_gtts` so each module keeps one job (20-positional-live 3, 4.2; the lead's rulings
N3b-2, N5a-1 and N5a-5).

- **A restart** (`take_back`): each GTT the kernel restores is taken back with the
  journal's GTT and the broker's trigger ID it is bound to (`_session_restore_gtts`,
  `_carry_restore`); one whose state the restart cannot tell is unsettled: the kernel's
  later commands about it are not sent and its findings apply nothing (the session has
  locked the strategy's entries and alerted).
- **A GTT carried from an earlier day** keeps its journal GTT under the new kernel id the
  restore gave it, so a finding of the follower, which names the kernel id it was placed
  with, is applied to the GTT holding its journal GTT now (`local`).
- **A place a restart settled** (`placed`): answered PLACED, it is applied as the kernel
  holds the GTT: none for one restored resting (a place carried under way), and
  `apply_gtt_modified` for one restored with its modify pending (a replace's place).

Quantities are whole units, prices rupees, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from ..errors import LiveError
from ._gtt_types import GttEvent, GttOutcome
from ._journal_gtt_rows import GttTerms
from ._types import Purpose


@dataclass
class KernelGtt:
    """What the session knows of one of the kernel's GTTs: what it is for, the journal's
    GTT and the broker's trigger ID once a place was answered, the commands waiting for
    them, the terms last asked for, since when (UTC ns) a place handed to the order worker
    waits for its answer (None before the hand-over and once answered), the kernel order
    of its child once it fired, whether a cancel of it was handed to the order worker and
    not answered (`cancel_sent`), whether a restart could not tell where it stands
    (`unsettled`, module docstring), the GTT follower's latest finding about it (`finding`,
    None for none: a finding other than RESTING that the kernel did not end it by leaves
    it not confirmed at the close, `_session_carry`), and why the close did not confirm
    it, once the kernel's session had closed and could hear of it no more (`unconfirmed`,
    `_session_carry`; "" for none)."""

    purpose: Purpose
    asked: GttTerms
    seq: int | None = None
    trigger_id: str = ""
    deferred: list[Mapping[str, Any]] = field(default_factory=list)
    placing_ns: int | None = None
    child: int | None = None
    cancel_sent: bool = False
    unsettled: bool = False
    finding: GttEvent | None = None
    unconfirmed: str = ""


class GttTakeBack:
    """The restart and take-back handling of a GTT map (module docstring), mixed into
    `_venue_gtts.GttMap`, which holds the kernel session (`_engine`) and what it knows of
    each kernel GTT by id (`_gtts`)."""

    _engine: Any
    _gtts: dict[int, KernelGtt]

    def take_back(
        self, view: Mapping[str, Any], *, seq: int | None, trigger_id: str, unsettled: bool
    ) -> None:
        """Take back at a restart the kernel's GTT `view` (an `Engine.gtt` dict): the
        journal's GTT `seq` bound to the broker's `trigger_id`, or `unsettled` (module
        docstring)."""
        purpose = Purpose.STOP if view["purpose"] == "stop" else Purpose.ENTRY
        known = KernelGtt(purpose, gtt_terms(view), seq=seq, trigger_id=trigger_id)
        known.unsettled = unsettled
        self._gtts[int(view["gtt_id"])] = known

    def local(self, outcome: GttOutcome) -> GttOutcome:
        """`outcome` naming the kernel GTT that holds its journal GTT now (module
        docstring)."""
        for gtt_id, known in self._gtts.items():
            if known.seq == outcome.seq and gtt_id != outcome.gtt_id:
                return replace(outcome, gtt_id=gtt_id)
        return outcome

    def placed(self, gtt_id: int) -> None:
        """Apply a PLACED answer to kernel GTT `gtt_id` as the kernel holds it (module
        docstring)."""
        state = self._engine.gtt(gtt_id)["state"]
        if state == "ModifyPending":
            self._engine.apply_gtt_modified(gtt_id)
        elif state != "Active":
            self._engine.apply_gtt_placed(gtt_id)


def gtt_terms(command: Mapping[str, Any]) -> GttTerms:
    """The terms of a kernel GTT command or view. Raises LiveError for units that are not
    whole."""
    units = float(command["quantity_units"])
    if not units.is_integer():
        raise LiveError(f"kernel GTT g{command['gtt_id']} is for {units} units: whole units only")
    return GttTerms(
        str(command["side"]),
        float(command["trigger_price"]),
        float(command["price"]),
        int(units),
    )


__all__ = ["GttTakeBack", "KernelGtt", "gtt_terms"]
