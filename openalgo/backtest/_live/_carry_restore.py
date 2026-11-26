"""What a strategy carries from an earlier day, restored into today's kernel session
(20-positional-live 4.2 steps 4 and 5, section 9; the lead's ruling N5a-1), at the start's
step 11 once the plan's catch-up is done, for a strategy with nothing saved today.

- **The kernel** takes the day-close record's carried state (`restore_carried_state`): the
  positions marked at the catch-up's last close, the legs with the levels and rules saved
  with them (new parameters apply to new entries), the GTTs and GTT orders under new ids,
  the daily loss measure starting the day as a backtest's does. A state the kernel
  refuses (another capital, other instruments) stops the start with its words.
- **Expired contracts**: a carried contract the exchange settled never reaches the
  kernel (`_carry_contracts` dropped it from the carry); the strategy starts flat and a
  SESSION alert and a warning of the result name the settlement (`report_settled`).
- **The GTTs** are taken back into the strategy's GTT map by the journal's GTT the record
  binds each to (`_carry_start.BINDINGS_KEY`): one still bound at the broker is followed
  as before; one whose place was under way at the close is settled by the order worker as
  an unanswered place (`settle_place`, the lead's ruling N5a-5); one the journal ended
  today (the start's read of the GTT book found it ended: its finding, on its way to the
  loop, applies it) is taken back as bound; one it ended on an earlier day (after that
  day's close) ended outside the session, so the kernel places the leg's stop again (as
  a GTT older than 360 days the broker expires is: 20-positional-live 3.1, "expired");
  any other is unsettled: entries locked with
  an ERROR alert, never an exit (`_session_restore_gtts`). Each binding is kept with the
  strategy's saved state (`gtt_bindings`), so a restart later today takes them back too.

Quantities are signed whole units, prices rupees. Run by the main thread at the start.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .._engine import call
from .._series import ist_time
from ..errors import LiveError
from ._carry_start import Carried, gtt_seqs
from ._gtt_types import GttRequest
from ._journal_gtt_rows import GttBinding, GttState
from ._plan_catch_up import INSTRUMENT
from ._session_gtts import lock_unresolved
from ._session_trader import Trader
from ._stops_gtt import overnight
from ._types import AlertKind
from ._venue_gtts import GttMap

#: The journal's states of a carried GTT still bound at the broker.
_BOUND = frozenset({GttState.PLACED, GttState.ACTIVE, GttState.TRIGGERED})
#: The journal's states of a GTT that ended without firing.
_ENDED_UNFIRED = frozenset({GttState.ENDED, GttState.CANCELLED})


def restore_carried(session: Any, trader: Trader, carried: Carried) -> None:
    """Restore what `trader` carries (module docstring) into its kernel session; `session`
    is the `_session_loop.Session`. Raises LiveError when the kernel refuses the carried
    state."""
    record = carried.record
    try:
        ids = call(trader.engine.restore_carried_state, record.carried)
    except Exception as error:
        raise LiveError(
            f"{trader.name}: the engine refused what it carries from {record.day} ({error}): "
            "the session cannot continue it. Check its position in OpenAlgo"
        ) from error
    session.event("carried", strategy=trader.name, day=str(record.day), gtts=len(ids["gtts"]))
    seqs = gtt_seqs(record)
    requests = []
    for old, new in sorted(ids["gtts"].items(), key=lambda pair: int(pair[1])):
        seq = seqs.get(int(old))
        row = None if seq is None else session.parts.journal.gtt(seq)
        requests += _take_back(session, trader, int(new), row)
    session.hand_over(requests)


def _take_back(session: Any, trader: Trader, gtt_id: int, row: GttBinding | None) -> list[Any]:
    """Take carried kernel GTT `gtt_id` back by its journal GTT `row` (module docstring);
    the requests for the order worker."""
    gtts = trader.gtts
    view = call(trader.engine.gtt, gtt_id)
    if gtts is None or view is None:
        return []
    bound = row is not None and row.trigger_id is not None and row.state in _BOUND
    # One the start's read of the GTT book ended today: its finding, on its way to the
    # loop, applies it through the GTT map.
    ended_today = row is not None and ist_time(row.changed_ns).date() == session.parts.spec.day
    if row is not None and (bound or (row.state in _ENDED_UNFIRED and ended_today)):
        gtts.take_back(view, seq=row.seq, trigger_id=row.trigger_id or "", unsettled=False)
        return []
    if row is not None and row.state is GttState.REQUESTED:
        return settle_place(gtts, view, row)
    if row is not None and row.state in _ENDED_UNFIRED:
        gtts.take_back(view, seq=row.seq, trigger_id=row.trigger_id or "", unsettled=False)
        call(trader.engine.apply_gtt_cancelled, gtt_id)
        session.event("carried_gtt_ended", strategy=trader.name, gtt=gtt_id, seq=row.seq)
        return []
    seq = None if row is None else row.seq
    trigger_id = "" if row is None else row.trigger_id or ""
    gtts.take_back(view, seq=seq, trigger_id=trigger_id, unsettled=True)
    state = "not in the journal" if row is None else f"{row.state.value} in the journal"
    lock_unresolved(
        session,
        trader,
        f"its GTT g{gtt_id} carried from an earlier day cannot be taken back ({state})",
    )
    return []


def settle_place(gtts: GttMap, view: Mapping[str, Any], row: GttBinding) -> list[GttRequest]:
    """The request that settles a restored GTT whose place the journal shows under way and
    unbound (`row`; the lead's ruling N5a-5): its place, naming the journal's GTT, which the
    order worker settles as an unanswered place before anything is sent (`_gtt_send`). The
    GTT map takes the kernel's GTT `view` (an `Engine.gtt` dict) as a place handed over, on
    the terms of its modify pending when it has one (a replace's place after its cancel);
    its later commands wait for the answer."""
    terms = view.get("pending") or {}
    (request,) = gtts.requests({**view, **terms, "command": "place_gtt"})
    return [replace(request, seq=row.seq)]


def report_settled(session: Any, words: str) -> None:
    """Name a carried contract the exchange settled (module docstring): a warning of the
    result and a SESSION alert, in `words` (`CarryAtStart.expired`)."""
    session.warn(words)
    session.alert(AlertKind.SESSION, words)


def gtt_bindings(trader: Trader) -> list[dict[str, int]]:
    """The journal's GTT each of `trader`'s kernel GTTs is bound to, as its saved state
    keeps them under `BINDINGS_KEY` (module docstring); none for an MIS strategy."""
    if not overnight(trader) or trader.gtts is None:
        return []
    bindings = []
    for view in call(trader.engine.gtts, INSTRUMENT, trader.product):
        known = trader.gtts.gtt(int(view["gtt_id"]))
        if known is not None and known.seq is not None:
            bindings.append({"gtt": int(view["gtt_id"]), "seq": known.seq})
    return bindings


__all__ = ["gtt_bindings", "report_settled", "restore_carried", "settle_place"]
