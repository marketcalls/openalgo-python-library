"""Which futures contract a rolled strategy holds on each session (04-execution 7).

`settings.roll` names the contracts in the order they are traded (`chain`) and the number
of sessions before the near contract's expiry at which the position rolls
(`sessions_before_expiry`, default 1). From the roll session on, the strategy trades, and
on_bar sees, the next contract.
"""

from __future__ import annotations

from .timegrid import Case, sessions


def roll_session_date(case: Case) -> str | None:
    """IST date (YYYY-MM-DD) of the roll session, or None when the case does not roll.

    Raises ValueError when the near contract's expiry is not a session of the calendar or
    when the calendar has too few sessions before it.
    """
    roll = case["settings"]["roll"]
    if roll is None:
        return None
    near = roll["chain"][0]
    expiry = next(item["expiry_date"] for item in case["instruments"] if item["symbol"] == near)
    exchange = next(item["exchange"] for item in case["instruments"] if item["symbol"] == near)
    dates = [session.open.date().isoformat() for session in sessions(case, exchange)]
    if expiry not in dates:
        raise ValueError(f"{case['id']}: the expiry {expiry} of {near} is not a session")
    index = dates.index(expiry) - roll["sessions_before_expiry"]
    if index < 0:
        raise ValueError(f"{case['id']}: the calendar has no session to roll {near} on")
    return dates[index]


def contract_on(case: Case, session_date: str) -> str:
    """The contract a strategy trades on a session: its only instrument without a roll; with
    one, the near contract before the roll session and the next contract from it."""
    roll_date = roll_session_date(case)
    if roll_date is None:
        return case["instruments"][0]["symbol"]
    near, following = case["settings"]["roll"]["chain"][:2]
    return near if session_date < roll_date else following
