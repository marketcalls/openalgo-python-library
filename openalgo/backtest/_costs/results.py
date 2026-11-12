"""What a costing returns: the itemised breakdown, the DP charge and the warning for
rates that are not verified. Every amount comes from the kernel; nothing is
recomputed here."""

from dataclasses import dataclass

from .kernel import COMPONENTS

__all__ = ["CostBreakdown", "DpCharge", "UnverifiedScheduleWarning"]


class UnverifiedScheduleWarning(UserWarning):
    """A cost came from rates that are not verified for the trade date: a schedule row
    that was never verified, a date after the row's verified-through date, or a date
    before the first schedule version (backfill)."""


@dataclass(frozen=True)
class CostBreakdown:
    """The itemised cost of a trade or a round trip. Every amount is in rupees.

    The six components are brokerage, stt (STT, or CTT on MCX), exchange_txn (the
    exchange transaction charge including the IPFT levy), sebi_fee, stamp_duty and gst.
    `total` is their sum as the rounding mode reports it: in contract-note mode the
    components are unrounded accruals and the total their plain sum; in calculator
    mode they are whole paise, rounded the way the Zerodha calculator rounds.

    `schedules` names the schedule version or versions that priced it. `verified` is
    True only when every rate used was verified for its trade date, and `backfilled`
    is True when a trade was dated before the first schedule version.
    """

    brokerage: float
    stt: float
    exchange_txn: float
    sebi_fee: float
    stamp_duty: float
    gst: float
    total: float
    segment: str
    rounding: str
    schedules: tuple[str, ...]
    verified: bool
    backfilled: bool

    def as_dict(self) -> dict[str, float]:
        """The six components and the total as a plain dict of rupee amounts."""
        return {name: getattr(self, name) for name in (*COMPONENTS, "total")}


@dataclass(frozen=True)
class DpCharge:
    """The depository participant charge for one scrip sold from holdings, in rupees.

    `fee` is before GST, `gst` is rounded to the paisa and `total` is what is
    debited. `fee_source` says whose fee it is: "zerodha_preset", "zerodha_stand_in"
    (the preset has no fee of its own, so Zerodha's Rs 13 was used) or "supplied".
    `verified` is False when the GST rate is not verified for the date.
    """

    fee: float
    gst: float
    total: float
    fee_source: str
    schedule: str
    verified: bool
