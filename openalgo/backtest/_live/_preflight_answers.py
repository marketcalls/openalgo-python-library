"""How the preflight's checks word OpenAlgo's answers (`_preflight_server`, `_preflight_gtt`):
one wording for every error answer, a quoted answer, and the refusal of an answer given in
the other mode.

The answers are read as the SDK returns them. Its older calls (openalgo/account.py,
openalgo/data.py) wrap a non-200 reply as "HTTP <code>: <the reply's text>", which hides the
server's own JSON (`server_answer` takes it back out); its newer ones (`BaseAPI._post`, the
GTT calls among them) keep the server's JSON body and add its HTTP status as `code`. A body
that is not JSON comes back wrapped either way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._answers import answered_by_server, server_answer
from ._types import Mode

#: The most characters of an answer quoted in a message: a gateway's error page in front
#: of OpenAlgo can run to kilobytes of HTML.
MAX_QUOTED_CHARS = 200


def answer_reason(answer: Mapping[str, Any]) -> str:
    """What an error answer says: the server's own message with its HTTP status ("HTTP 501:
    GTT orders are not supported ..."), the SDK's "HTTP <code>: <text>" as it is when the
    text is not the server's JSON, else the SDK's message; clipped to `MAX_QUOTED_CHARS`."""
    own = server_answer(answer)
    message = own.get("message")
    text = clip(str(message)) if message else "an error without a message"
    # `server_answer` gives a new answer only when it took the SDK's wrapper apart; the
    # newer calls keep the server's body as it is, with its status as `code`.
    unwrapped = own is not answer or answered_by_server(answer)
    code = own.get("code")
    return f"HTTP {code}: {text}" if unwrapped and code is not None else text


def quoted(answer: Mapping[str, Any]) -> str:
    """The whole answer as text, clipped to `MAX_QUOTED_CHARS`: for an answer that says
    success without what the check needs."""
    return clip(repr(dict(answer)))


def clip(text: str) -> str:
    """`text`, cut to `MAX_QUOTED_CHARS` with "..." when longer."""
    return text if len(text) <= MAX_QUOTED_CHARS else f"{text[:MAX_QUOTED_CHARS]}..."


def switch_changed(what: str, answered: Mode, mode: Mode) -> str:
    """The refusal of a start whose read of `what` ("funds", "the GTT book") OpenAlgo
    answered in the `answered` mode while the session runs in `mode`."""
    return (
        f"OpenAlgo answered {what} in {answered.name} mode, but the session runs in "
        f"{mode.name} mode: its analyzer switch changed after the session read it. Set the "
        "switch in OpenAlgo to the mode you want, then start the session again"
    )


__all__ = ["MAX_QUOTED_CHARS", "answer_reason", "clip", "quoted", "switch_changed"]
