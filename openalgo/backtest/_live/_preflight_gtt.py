"""The GTT capability check of a live session's preflight (20-positional-live 5 and 6;
decisions P2, P3 and P5; the lead's ruling N2-3 (a)), run for a session with a CNC or NRML
strategy: whether the broker OpenAlgo is logged in to holds GTT orders, read from the server
at every start, never from a list kept in the SDK (OpenAlgo adds GTT to more brokers).

- **LIVE mode**: one `gttorderbook()` read, which only reads. An answer the sandbox gave
  (`"mode": "analyze"`) says nothing of the broker: the analyzer switch changed after the
  session read the mode, and the start is refused as the funds check refuses it
  (`_preflight_answers.switch_changed`). Success means GTT orders are supported.
  OpenAlgo's answer for a broker that ships no GTT module (HTTP 501, "GTT orders
  are not supported for broker '<name>' yet", services/gtt_orderbook_service.py) means they
  are not: the start is refused, naming the broker, with the two choices (trade MIS, or
  `positional_stops="rearm_daily"`) and, for the second, what it leaves unprotected
  (`REARM_DAILY_SENTENCE`); unless the run chose `rearm_daily` already, which holds each
  carried position's stop as a day order placed at each start. Any other answer refuses
  the start: it cannot tell, and says so with the server's message
  (`_preflight_answers.answer_reason`).
- **SANDBOX mode**: no call. OpenAlgo answers every GTT call from the sandbox before it
  looks at the broker, so it would succeed for any broker; the check passes with the
  banner line `SANDBOX_GTT_LINE`.

On a broker with GTT orders a run that chose `rearm_daily` passes with a warning that GTTs
would protect its positions overnight (20-positional-live 6).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from ..errors import ConfigError
from ._answers import server_answer
from ._guard_answers import SUCCESS
from ._preflight_answers import answer_reason, switch_changed
from ._preflight_plan import WARNING, Check
from ._types import Mode, mode_of

#: How a CNC or NRML position's stop is held overnight (20-positional-live 6): a GTT at the
#: broker (the default), or a day stop placed again at each start.
GTT_STOPS = "gtt"
REARM_DAILY = "rearm_daily"
POSITIONAL_STOPS = (GTT_STOPS, REARM_DAILY)
#: The name of the check.
GTT_CHECK = "gtt"
#: OpenAlgo's answer for a broker that ships no GTT module: its HTTP status, and the words
#: its message begins with (services/gtt_orderbook_service.py and the other GTT services).
NOT_SUPPORTED_STATUS = 501
NOT_SUPPORTED_WORDS = "GTT orders are not supported for broker"
#: The start banner's line of a sandbox session with a CNC or NRML strategy
#: (20-positional-live 5).
SANDBOX_GTT_LINE = "GTT support of your live broker is checked when you start in live mode."
#: What `rearm_daily` leaves unprotected, said wherever it is offered or used
#: (20-positional-live 6, decision P5).
REARM_DAILY_SENTENCE = (
    "Positions held overnight have no stop at the broker from the close until your next "
    "start; start bt.live before the open on every trading day you hold a position."
)


class GttBookClient(Protocol):
    """What the check needs of an `openalgo.api` client: its GTT book."""

    def gttorderbook(self, **kwargs: Any) -> Mapping[str, Any]:
        """The server's answer to `/api/v1/gttorderbook`."""


def gtt_check(client: GttBookClient, *, mode: Mode, broker: str, positional_stops: str) -> Check:
    """Whether the stops of the session's CNC and NRML positions can rest at `broker` (the
    broker `ping` named) as GTT orders, read through `client` in LIVE `mode` (module
    docstring); `positional_stops` is the run's choice. Raises ConfigError for a mode that
    is not a `Mode` and for a `positional_stops` other than "gtt" and "rearm_daily"."""
    if not isinstance(mode, Mode):
        raise ConfigError(f"mode is Mode.SANDBOX or Mode.LIVE, not {mode!r}")
    if positional_stops not in POSITIONAL_STOPS:
        raise ConfigError(
            f'positional_stops takes "{GTT_STOPS}" (a GTT holds each position\'s stop '
            f'overnight) or "{REARM_DAILY}" (a day stop placed again at each start), not '
            f"{positional_stops!r}"
        )
    if mode is Mode.SANDBOX:
        return Check(GTT_CHECK, True, SANDBOX_GTT_LINE)
    answer = client.gttorderbook()
    answered = mode_of(server_answer(answer))
    if answered is not mode:
        return Check(GTT_CHECK, False, switch_changed("the GTT book", answered, mode))
    rearm = positional_stops == REARM_DAILY
    if answer.get("status") == SUCCESS:
        return _supported(broker, rearm=rearm)
    if _not_supported(answer):
        return _unsupported(broker, answer_reason(answer), rearm=rearm)
    return Check(
        GTT_CHECK,
        False,
        f"the start cannot tell whether {broker} holds GTT orders: OpenAlgo's GTT book "
        f"answered {answer_reason(answer)}. A CNC or NRML position needs a GTT to keep its "
        "stop at the broker overnight: check OpenAlgo and the broker's login, then start again",
    )


def _supported(broker: str, *, rearm: bool) -> Check:
    if rearm:
        return Check(
            GTT_CHECK,
            True,
            f"{WARNING}{broker} holds GTT orders, which would protect your positions "
            f'overnight; positional_stops="{REARM_DAILY}" places a day stop at each start '
            "instead",
        )
    return Check(
        GTT_CHECK,
        True,
        f"{broker} holds GTT orders: each CNC or NRML position's stop rests at the broker "
        "as a GTT overnight",
    )


def _unsupported(broker: str, reason: str, *, rearm: bool) -> Check:
    if rearm:
        return Check(
            GTT_CHECK,
            True,
            f"{broker} holds no GTT orders in OpenAlgo ({reason}): "
            f'positional_stops="{REARM_DAILY}" places each carried position\'s stop as a day '
            "order at each start",
        )
    return Check(
        GTT_CHECK,
        False,
        f"OpenAlgo answered {reason}: a CNC or NRML position keeps its stop at the broker "
        f"overnight as a GTT, which {broker} does not hold. Trade MIS, or pass "
        f'positional_stops="{REARM_DAILY}" to place each position\'s stop as a day order '
        f"at every start. {REARM_DAILY_SENTENCE}",
    )


def _not_supported(answer: Mapping[str, Any]) -> bool:
    """Whether an answer is OpenAlgo's for a broker without GTT orders."""
    message = str(answer.get("message") or "")
    return answer.get("code") == NOT_SUPPORTED_STATUS or NOT_SUPPORTED_WORDS in message


__all__ = [
    "GTT_CHECK",
    "GTT_STOPS",
    "NOT_SUPPORTED_STATUS",
    "NOT_SUPPORTED_WORDS",
    "POSITIONAL_STOPS",
    "REARM_DAILY",
    "REARM_DAILY_SENTENCE",
    "SANDBOX_GTT_LINE",
    "GttBookClient",
    "gtt_check",
]
