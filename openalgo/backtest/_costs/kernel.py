"""The native cost kernel: its functions, the tables it publishes, and its errors.

Every table here is read from the kernel once at import, so the segment list, the
component order, the note groups, the rounding modes and the trust codes have one
source of truth (the Rust crate).
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from .._native import native
from ..errors import CostError

__all__ = [
    "COMPONENTS",
    "CONTRACT_NOTE_RULE",
    "MAX_DAY",
    "MIN_DAY",
    "NOTE_GROUPS",
    "ROUNDING_MODES",
    "SEGMENTS",
    "SEGMENT_TABLE",
    "TRUST_CODES",
    "VERIFIED",
    "call",
    "native",
]

# The kernel reports a bad value as ValueError; TypeError and OverflowError come
# from the argument conversion in front of it (a string where a number is expected,
# an integer that does not fit).
_KERNEL_ERRORS = (ValueError, TypeError, OverflowError)


def call(function: Callable[..., Any], *args: Any) -> Any:
    """Call a kernel function, re-raising its errors as CostError with the same message."""
    try:
        return function(*args)
    except CostError:
        raise
    except _KERNEL_ERRORS as exc:
        raise CostError(str(exc)) from None


#: Names of the six cost components, in output order.
COMPONENTS: tuple[str, ...] = tuple(native.cost_components())

_SEGMENT_FIELDS = ("id", "code", "exchange", "brokerage_class", "note_group", "description")

#: One read-only mapping per cost segment, in id order.
SEGMENT_TABLE: tuple[Mapping[str, Any], ...] = tuple(
    MappingProxyType(dict(zip(_SEGMENT_FIELDS, row, strict=True))) for row in native.cost_segments()
)

#: Every cost segment code, in id order.
SEGMENTS: tuple[str, ...] = tuple(row["code"] for row in SEGMENT_TABLE)

#: Contract-note group codes, indexed by group id.
NOTE_GROUPS: tuple[str, ...] = tuple(native.cost_note_groups())

#: Rounding mode names, the default first.
ROUNDING_MODES: tuple[str, ...] = tuple(native.cost_rounding_codes())

#: How far a result's rates can be trusted, indexed by the kernel's trust id.
TRUST_CODES: tuple[str, ...] = tuple(native.cost_trust_codes())

#: The trust code of rates verified for the trade date.
VERIFIED: str = TRUST_CODES[0]

#: The first and last IST day number (days since 1970-01-01) the engine supports.
MIN_DAY, MAX_DAY = native.cost_day_range()

#: The contract-note rounding rule the kernel applies. It is an assumption.
CONTRACT_NOTE_RULE: str = native.cost_contract_note_rule()
