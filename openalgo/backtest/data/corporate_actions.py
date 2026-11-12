"""Corporate actions a user declares for a share: splits, bonuses, and dates checked to
have none.

OpenAlgo's sources supply no corporate-action data, and the history they return may or
may not be adjusted for splits and bonuses depending on the source. The quality check
(`splits`) therefore looks for unadjusted splits and bonuses in the bars themselves, and
a declaration tells it what happened on an ex-date:

- `CorporateAction.split(ex_date, from_face_value, to_face_value)`: a face-value split,
  for example from 10 to 2 rupees, or a consolidation when the face value rises.
- `CorporateAction.bonus(ex_date, bonus, held)`: a bonus issue of `bonus` new shares for
  every `held` shares, announced as "bonus:held" (a 1:1 bonus doubles the shares).
- `CorporateAction.none(ex_date)`: the date was checked and had no corporate action, so
  a gap there is a genuine price move.

Units: `ex_date` is the IST date of the first session at the new price; face values are
rupees per share; `factor` is the number of shares held after the event for each share
held before it, so the price after the event is the price before divided by it.

A declaration does not adjust anything in this version: adjusting prices, quantities and
orders from supplied corporate actions is not supported yet.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from ..errors import ConfigError
from ._sessions import to_date

SPLIT = "split"
BONUS = "bonus"
NONE = "none"
KINDS = (SPLIT, BONUS, NONE)
# The largest number of shares before the event shown when a factor is written as whole
# shares ("3 shares for every 2"); announced ratios use small whole numbers.
_LARGEST_SHARE_COUNT = 100


def _positive_number(value: object, name: str) -> float:
    """`value` as a finite positive float. Raises ConfigError naming `name` otherwise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ConfigError(f"{name} must be a finite number above 0, got {value!r}")
    return number


def _whole_number(value: object, name: str) -> int:
    """`value` as a positive int. Raises ConfigError naming `name` otherwise."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a whole number of shares of 1 or more, got {value!r}")
    return value


def shares_text(factor: float) -> str:
    """A factor as whole shares, for messages: 2.0 is "2 shares for every 1", 1.5 is "3
    shares for every 2", 0.1 is "1 share for every 10"."""
    ratio = Fraction(factor).limit_denominator(_LARGEST_SHARE_COUNT)
    after, before = ratio.numerator, ratio.denominator
    return f"{after} share{'' if after == 1 else 's'} for every {before}"


@dataclass(frozen=True)
class CorporateAction:
    """What happened to a share on one ex-date, as the user declares it (see the module
    docstring for units).

    ex_date: the IST date of the first session at the new price; a date or an ISO string.
    kind: "split", "bonus" or "none".
    factor: shares held after the event per share held before it (5 for a split from
        face value 10 to 2, 2 for a 1:1 bonus, 0.1 for a consolidation from 1 to 10);
        1 for "none".

    Build one with `split`, `bonus` or `none`. Raises ConfigError for a date that cannot
    be read, an unknown kind, a factor that is not a finite positive number, a split of
    factor 1, a bonus of factor 1 or less, or "none" with a factor other than 1.
    """

    ex_date: dt.date
    kind: str
    factor: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ex_date", to_date(self.ex_date, "ex_date"))
        if self.kind not in KINDS:
            raise ConfigError(f"kind must be one of {KINDS}, got {self.kind!r}")
        factor = _positive_number(self.factor, "factor")
        object.__setattr__(self, "factor", factor)
        if self.kind == NONE and factor != 1.0:
            raise ConfigError(f"a declaration of no corporate action has factor 1, got {factor}")
        if self.kind == SPLIT and factor == 1.0:
            raise ConfigError("a split of factor 1 changes nothing: check the face values")
        if self.kind == BONUS and factor <= 1.0:
            raise ConfigError(f"a bonus adds shares, so its factor is above 1, got {factor}")

    @classmethod
    def split(
        cls, ex_date: dt.date | str, from_face_value: float, to_face_value: float
    ) -> CorporateAction:
        """A face-value split from `from_face_value` to `to_face_value` rupees per share
        (a consolidation when it rises): factor from / to. Raises ConfigError for a face
        value that is not a finite positive number, or two equal ones."""
        before = _positive_number(from_face_value, "from_face_value")
        after = _positive_number(to_face_value, "to_face_value")
        if before == after:
            raise ConfigError(f"a split changes the face value; both are {before}")
        return cls(ex_date, SPLIT, before / after)

    @classmethod
    def bonus(cls, ex_date: dt.date | str, bonus: int, held: int) -> CorporateAction:
        """A bonus issue of `bonus` new shares for every `held` shares: factor
        (bonus + held) / held. Raises ConfigError unless both are whole numbers of 1 or
        more."""
        new = _whole_number(bonus, "bonus")
        old = _whole_number(held, "held")
        return cls(ex_date, BONUS, (new + old) / old)

    @classmethod
    def none(cls, ex_date: dt.date | str) -> CorporateAction:
        """A date checked to have no corporate action: a gap there is a genuine price
        move. Raises ConfigError for a date that cannot be read."""
        return cls(ex_date, NONE)

    def describe(self) -> str:
        """The declaration in words, for messages: "a split (5 shares for every 1)"."""
        if self.kind == NONE:
            return "no corporate action"
        return f"a {self.kind} ({shares_text(self.factor)})"

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form: the ISO ex-date, the kind and the factor."""
        return {"ex_date": self.ex_date.isoformat(), "kind": self.kind, "factor": self.factor}


def declared(actions: Iterable[CorporateAction] | None) -> tuple[CorporateAction, ...]:
    """The declarations of one series, checked and in ex-date order.

    actions: None (nothing declared) or an iterable of CorporateAction.

    Raises ConfigError for anything else, for the same kind declared twice on one date,
    and for "none" declared on a date that also has a split or bonus. A split and a bonus
    on the same date are both kept: their factors multiply.
    """
    if actions is None:
        return ()
    if isinstance(actions, CorporateAction | str | bytes) or not isinstance(actions, Iterable):
        raise ConfigError(
            "corporate_actions takes a list of openalgo.backtest.data.CorporateAction, got "
            f"{type(actions).__name__}"
        )
    found = list(actions)
    strangers = [item for item in found if not isinstance(item, CorporateAction)]
    if strangers:
        raise ConfigError(
            "corporate_actions takes CorporateAction.split(...), .bonus(...) or .none(...) "
            f"items, got {strangers[0]!r}"
        )
    kinds_by_date: dict[dt.date, list[str]] = {}
    for action in found:
        kinds = kinds_by_date.setdefault(action.ex_date, [])
        if action.kind in kinds:
            raise ConfigError(f"a {action.kind} is declared twice on {action.ex_date}")
        kinds.append(action.kind)
    for day, kinds in kinds_by_date.items():
        if NONE in kinds and len(kinds) > 1:
            other = next(kind for kind in kinds if kind != NONE)
            raise ConfigError(f"{day} is declared both with no corporate action and with a {other}")
    return tuple(sorted(found, key=lambda action: (action.ex_date, KINDS.index(action.kind))))


__all__ = ["BONUS", "KINDS", "NONE", "SPLIT", "CorporateAction", "declared", "shares_text"]
