"""The preflight's checks of a session that carries positions from an earlier day
(20-positional-live 3.5 and 4.2; decision P5; the lead's ruling N2-3 (b) to (e)), each read
from the journal's day-close records and kept refusals (`_journal_carry`):

- **Missed days** (`missed_days`): the trading days between a record's day and today on
  which no session ran; the banner names them, and a stop that fired on one has no fill
  price the engine can read (4.3).
- **The other ledger** (`other_ledger_lines`): positions carried in the other mode's ledger
  are named in the start banner ("held in sandbox, not continued in live") and never
  restored into this one (4.2 step 1).
- **Funds on a carry** (`held_rupees`, `carry_funds_check`, `funds_with_carry`): a carried
  position has spent cash already, so the account's available cash is set against the
  capital less what the carried positions hold (4.2 step 3, review H7): the margin the
  engine blocked for each, which its carried state keeps, so the check never works it out
  a second way. That is a CNC position's cost and a bought option's premium (all of their
  value), a future's margin by the engine's futures rule on its entry day (20 percent of
  the contract value by default, or the `margin=` the run gave, `_futures_margin`), each
  cut with the units a part exit sold. The failure's words are the preflight's own funds
  check's (`_preflight_server.funds_check`).
- **Kept refusals** (`kept_refusal_checks`): a strategy whose stop GTT a broker refused for
  a reason no price explains is refused at every start with the broker's message until
  the user passes `positional_stops="rearm_daily"` or trades it in another product (3.5).

A carried position is read from a day-close record's carried state, the kernel's saved
state (its `instruments` and `positions`, rust/oa_bt/src/engine/venue_state/book.rs).
Amounts are rupees, quantities signed whole units, days IST dates.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..calendar import Calendar
from ..errors import LiveError
from ..reference import Instrument
from ..report._format import indian_digits
from ._journal_carry import DayClose, KeptRefusal
from ._preflight_gtt import GTT_STOPS, REARM_DAILY
from ._preflight_plan import Check, PreflightPlan, positive_amount, refused
from ._preflight_server import FUNDS_CHECK, funds_check
from ._types import Mode


@dataclass(frozen=True)
class CarriedPosition:
    """A position a strategy's day-close record carries: its instrument and product, its
    signed `units`, and `margin_rupees`, the cash its engine blocked for it (module
    docstring)."""

    strategy: str
    symbol: str
    exchange: str
    product: str
    units: float
    margin_rupees: float


def carried_positions(record: DayClose) -> tuple[CarriedPosition, ...]:
    """The open positions of `record`'s carried state, in its order (flat ones left out).
    Raises LiveError for a carried state that cannot be read, naming the strategy: a
    carried position is never guessed."""
    state = record.carried
    try:
        instruments = state["instruments"]
        positions = []
        for position in state["positions"]:
            units = float(position["quantity_units"])
            if units == 0:
                continue
            instrument = instruments[int(position["instrument"])]
            positions.append(
                CarriedPosition(
                    record.strategy,
                    str(instrument["symbol"]),
                    str(instrument["exchange"]),
                    str(position["product"]),
                    units,
                    float(position["margin_rupees"]),
                )
            )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise LiveError(
            f"the day-close record of {record.strategy} of {record.day} in the "
            f"{record.mode.value} ledger cannot be read ({error!r}): its carried positions "
            "are not guessed. Check them in OpenAlgo, then start under another account name "
            "(account=)"
        ) from None
    return tuple(positions)


def missed_days(
    exchange: str, closed: dt.date, today: dt.date, calendar: Calendar
) -> tuple[dt.date, ...]:
    """The trading days of `exchange` after `closed` (a day-close record's day) and before
    `today`, oldest first: none when `closed` is the previous trading day or today. Raises
    LiveError for a record of a day after today: this computer's date is wrong."""
    if closed > today:
        raise LiveError(
            f"the journal holds a day-close record of {closed}, after today ({today}): check "
            "this computer's date and time zone, then start again"
        )
    first, last = closed + dt.timedelta(days=1), today - dt.timedelta(days=1)
    if first > last:
        return ()
    return tuple(calendar.sessions(exchange, first, last)["date"])


def other_ledger_lines(records: Mapping[str, DayClose], mode: Mode) -> list[str]:
    """The start banner's lines of the positions the other mode's ledger carries
    (`records`, its latest day-close record by strategy), in a session of `mode`: one per
    position ("hold: 40 SBIN NSE (CNC) held in sandbox, not continued in live"). Raises
    LiveError for a record whose carried state cannot be read."""
    lines = []
    for record in records.values():
        for position in carried_positions(record):
            lines.append(
                f"{position.strategy}: {_units(position)} {position.symbol} "
                f"{position.exchange} ({position.product}) held in {record.mode.value}, not "
                f"continued in {mode.value}"
            )
    return lines


def held_rupees(position: CarriedPosition, instrument: Instrument) -> float:
    """Rupees of the account's cash `position` holds: the margin its engine blocked
    (module docstring). `instrument` is the position's contract. Raises LiveError for a
    sold option: none is carried overnight in this release (20-positional-live 11.5), and
    its margin is the broker's."""
    if instrument.kind == "option" and position.units < 0:
        raise LiveError(
            f"{position.strategy} carries {_units(position)} {position.symbol}, a sold "
            "option: sold options are not held overnight in this release. Close it in OpenAlgo"
        )
    return position.margin_rupees


def carry_funds_check(
    available: float,
    capital: float,
    held: Sequence[tuple[CarriedPosition, float]],
    mode: Mode,
) -> Check:
    """The funds check of a session that carries positions (module docstring): the
    `available` cash against the `capital` (the strategies' together) less the rupees each
    carried position holds (`held`: each position with its `held_rupees`), what is left
    never below 0. Its message says the figures; a failure says where to add funds."""
    carried = sum(rupees for _, rupees in held)
    needed = max(capital - carried, 0.0)
    parts = ", ".join(
        f"{position.strategy} {position.symbol} {position.product} {indian_digits(rupees)}"
        for position, rupees in held
    )
    figures = (
        f"the capital of {indian_digits(capital)} rupees less the {indian_digits(carried)} "
        f"rupees the carried positions hold ({parts}), {indian_digits(needed)} rupees"
    )
    return funds_check(available, needed, figures, mode)


def funds_with_carry(
    checks: Sequence[Check], available: float | None, plan: PreflightPlan
) -> list[Check]:
    """The server's `checks` with the funds check set against the capital less what the
    carried positions hold (`carry_funds_check`) when `plan` continues any
    (`PreflightPlan.carried`); as they are when it continues none, or when the cash or the
    capital is not known (their own checks fail then)."""
    capital = positive_amount(plan.capital)
    if not plan.carried or available is None or capital is None:
        return list(checks)
    carried = carry_funds_check(available, capital, plan.carried, plan.mode)
    return [carried if check.name == FUNDS_CHECK else check for check in checks]


def kept_refusal_checks(
    refusals: Sequence[KeptRefusal],
    strategies: Sequence[tuple[str, str, str, str]],
    positional_stops: str,
) -> list[Check]:
    """A failed check for each strategy (name, symbol, exchange, product) whose leg a
    broker's refusal kept in the journal (`refusals`) names, quoting the broker's message;
    none with `positional_stops="rearm_daily"`, which places no GTT."""
    if positional_stops == REARM_DAILY:
        return []
    kept = {(r.strategy, r.symbol, r.exchange, r.product): r for r in refusals}
    checks = []
    for leg in strategies:
        refusal = kept.get(tuple(leg))
        if refusal is not None:
            checks.append(_refused(refusal))
    return checks


def refuse_kept(
    refusals: Sequence[KeptRefusal], strategies: Sequence[tuple[str, str, str, str]]
) -> None:
    """Refuse the start of a session whose CNC or NRML strategies (name, symbol, exchange,
    product) a kept refusal names (`kept_refusal_checks`, with the GTT stops a live session
    holds; the lead's ruling N3b-3). Raises PreflightError quoting each stored message."""
    checks = kept_refusal_checks(refusals, strategies, GTT_STOPS)
    failures = tuple(check.message for check in checks)
    if failures:
        raise refused(failures)


def _units(position: CarriedPosition) -> str:
    """A carried position's signed units, whole and grouped as Indian figures are
    ("12,34,567")."""
    return indian_digits(position.units, 0)


def _refused(refusal: KeptRefusal) -> Check:
    leg = f"{refusal.exchange}:{refusal.symbol} ({refusal.product})"
    return Check(
        f"kept_refusal:{refusal.strategy}",
        False,
        f"{refusal.strategy}: on {refusal.day} the broker refused the GTT that holds its stop "
        f'on {leg}: "{refusal.message}". It is refused at every start until you pass '
        f'positional_stops="{REARM_DAILY}" or trade it in another product',
    )


__all__ = [
    "CarriedPosition",
    "carried_positions",
    "carry_funds_check",
    "funds_with_carry",
    "held_rupees",
    "kept_refusal_checks",
    "missed_days",
    "other_ledger_lines",
    "refuse_kept",
]
