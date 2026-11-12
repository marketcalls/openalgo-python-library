"""`CostModel`: dated schedules, a brokerage preset and a rounding mode."""

import copy
import os
import warnings
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..errors import CostError
from .inputs import (
    DateLike,
    common_length,
    day_to_date,
    floats,
    number,
    segment_code,
    segment_ids,
    side_ids,
    to_day,
    to_days,
)
from .kernel import COMPONENTS, SEGMENTS, TRUST_CODES, VERIFIED, call, native
from .presets import Brokerage, preset
from .results import CostBreakdown, DpCharge, UnverifiedScheduleWarning
from .schedules import schedule_spec, schedule_view

__all__ = ["CostModel"]

# Frames inside openalgo.backtest are skipped when a warning is attributed, so it
# points at the caller's line whichever public function the caller came through.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BACKFILLED = "backfilled"
_WHY_UNTRUSTED = MappingProxyType(
    {
        "unverified": "were never verified against their source",
        "past_verification": "are verified only up to an earlier date",
        _BACKFILLED: "belong to the first schedule version, which starts later (backfill)",
    }
)


class CostModel:
    """A cost model: dated schedules, a brokerage preset and a rounding mode.

    Parameters
    ----------
    brokerage : str or Brokerage
        "zerodha" (default), "zero_brokerage", or a preset from `costs.flat()`,
        `costs.percent()`, `costs.zero_brokerage()` or `costs.custom()`.
    rounding : str
        "contract_note" (default) returns unrounded accruals; round them per day with
        `costs.contract_note()`. "calculator" rounds each trade or round trip the way
        the Zerodha brokerage calculator does (to the paisa for equity and equity
        derivatives, within one paisa for MCX futures).
    schedules : list of dict, optional
        User schedule versions, in the shape `costs.schedules()` returns (dates may
        be date objects or ISO strings). They go through the same validation and
        lookup as the built-in versions.
    include_builtin : bool
        True (default) merges the user versions with the built-in ones: a user
        version may cover an earlier period, or start later and take over from the
        current version on its start date. False uses the user versions only.
    backfill : bool
        False (default): a trade dated before the first version raises CostError.
        True charges such trades at the first version's rates and warns.

    Raises CostError for an unknown preset or rounding mode and for a schedule that
    is malformed, has implausible rates or overlaps another version.

    Units everywhere: prices and costs in rupees; a quantity is the number of price
    units, never lots (for an MCX future, lots times the contract multiplier); rates
    are fractions; trade dates are IST calendar dates. One trade is one executed
    order: an order filled in several parts is costed once, on its total.

    Every result says which schedule version priced it and whether the rates were
    verified for the trade date. When they were not, an UnverifiedScheduleWarning is
    issued as well.
    """

    def __init__(
        self,
        brokerage: str | Brokerage = "zerodha",
        rounding: str = "contract_note",
        schedules: list[dict[str, Any]] | None = None,
        include_builtin: bool = True,
        backfill: bool = False,
    ) -> None:
        self._brokerage = preset(brokerage)
        if isinstance(schedules, dict):
            schedules = [schedules]
        specs = [schedule_spec(schedule) for schedule in schedules or []]
        if not include_builtin and not specs:
            raise CostError("include_builtin=False needs at least one schedule in schedules")
        self._kernel = call(
            native.CostModel,
            self._brokerage._kernel,
            rounding,
            specs,
            bool(include_builtin),
            bool(backfill),
        )
        self._views = tuple(schedule_view(raw) for raw in self._kernel.schedules())

    def __repr__(self) -> str:
        ids = [view["id"] for view in self._views]
        return (
            f"CostModel(brokerage={self._brokerage.name!r}, rounding={self.rounding!r}, "
            f"schedules={ids!r}, backfill={self.backfill!r})"
        )

    @property
    def brokerage(self) -> Brokerage:
        """The brokerage preset."""
        return self._brokerage

    @property
    def rounding(self) -> str:
        """The rounding mode: "contract_note" or "calculator"."""
        return self._kernel.rounding

    @property
    def backfill(self) -> bool:
        """Whether trades before the first version are charged at its rates."""
        return self._kernel.backfill

    def schedules(self) -> list[dict[str, Any]]:
        """The schedule versions of this model, oldest first.

        Each is a dict with id, valid_from, valid_to and verified_through (dates;
        valid_to is None for the version in force, verified_through None when never
        verified), gst_rate, source, notes and rates: a dict of segment code to a row
        of rates (fractions of traded value) with the row's own verified_through.
        """
        return copy.deepcopy(list(self._views))

    def _untrusted_texts(self, status: tuple[int, int], what: str, day: int) -> tuple[str, str]:
        """The short and the full text of the warning for a kernel status whose rates are
        not verified: the short one names the schedule, `what` (a segment code or "GST")
        and why, the same for every date; the full one adds the date and the notes."""
        schedule, trust = status
        view, code = self._views[schedule], TRUST_CODES[trust]
        short = (
            f"the {what} rates of cost schedule {view['id']} {_WHY_UNTRUSTED[code]}, so the "
            "charges may be off"
        )
        notes = view["rates"].get(what, {}).get("notes") or view["notes"]
        full = (
            f"the {what} rates of cost schedule {view['id']} used for {day_to_date(day)} "
            f"{_WHY_UNTRUSTED[code]}, so the charges may be off. {notes}"
        ).rstrip()
        return short, full

    def _trusted(self, status: tuple[int, int], what: str, day: int) -> tuple[str, str]:
        """The schedule id and trust code of a kernel status, warning when not verified."""
        schedule, trust = status
        view, code = self._views[schedule], TRUST_CODES[trust]
        if code != VERIFIED:
            warnings.warn(
                self._untrusted_texts(status, what, day)[1],
                UnverifiedScheduleWarning,
                skip_file_prefixes=(_PACKAGE_DIR,),
            )
        return view["id"], code

    def _unverified(self, segment: str, date: DateLike) -> tuple[str, str] | None:
        """The short and the full text of the warning `trade` would give for `segment` on
        `date` (`_untrusted_texts`), or None when the rates are verified; nothing is
        warned, so a caller that checks many dates can report each reason once. Raises
        CostError as `trade` does."""
        code = segment_code(segment)
        day = to_day(date)
        _, status = call(self._kernel.fill, code, int(side_ids("BUY", 1)[0]), 1.0, 1.0, day)
        if TRUST_CODES[status[1]] == VERIFIED:
            return None
        return self._untrusted_texts(status, code, day)

    def _breakdown(
        self, amounts: tuple, segment: str, used: list[tuple[str, str]]
    ) -> CostBreakdown:
        return CostBreakdown(
            *amounts,
            segment=segment,
            rounding=self.rounding,
            schedules=tuple(dict.fromkeys(schedule for schedule, _ in used)),
            verified=all(code == VERIFIED for _, code in used),
            backfilled=any(code == _BACKFILLED for _, code in used),
        )

    def trade(
        self, side: str | int, price: float, quantity: float, date: DateLike, segment: str
    ) -> CostBreakdown:
        """The cost of one trade (one executed order), at the rates in force on `date`.

        `side` is "BUY" or "SELL" (or +1 or -1). `price` is in rupees per price unit
        and `quantity` is the number of price units, never lots, so that price times
        quantity is the traded value: shares for an equity, premium times units for
        an option, lots times the contract multiplier for an MCX future. `segment` is
        a code from `costs.SEGMENTS` (use `costs.segment_for` to get it from an
        exchange and product).

        Raises CostError for an unknown side or segment, a price, quantity or traded
        value that is negative or not finite, or a date no schedule version covers.
        Warns with UnverifiedScheduleWarning when the rates used are not verified
        for the date.
        """
        code = segment_code(segment)
        day = to_day(date)
        amounts, status = call(
            self._kernel.fill,
            code,
            int(side_ids(side, 1)[0]),
            number("price", price),
            number("quantity", quantity),
            day,
        )
        return self._breakdown(amounts, code, [self._trusted(status, code, day)])

    def round_trip(
        self,
        buy_price: float,
        sell_price: float,
        quantity: float,
        date: DateLike,
        segment: str,
        *,
        sell_date: DateLike | None = None,
    ) -> CostBreakdown:
        """The cost of buying and selling the same `quantity` (price units), each leg
        one executed order.

        With one date both legs are costed together; in calculator mode that is the
        Zerodha calculator's result. With `sell_date` on another day the buy is
        costed on `date` and the sell on `sell_date`, each at the rates of its own
        date, and the two are added.

        Raises and warns as `trade` does.
        """
        code = segment_code(segment)
        buy_day = to_day(date)
        sell_day = buy_day if sell_date is None else to_day(sell_date, "sell_date")
        amounts, bought, sold = call(
            self._kernel.round_trip,
            code,
            number("buy_price", buy_price),
            number("sell_price", sell_price),
            number("quantity", quantity),
            buy_day,
            sell_day,
        )
        legs = {buy_day: bought, sell_day: sold}
        used = [self._trusted(status, code, day) for day, status in legs.items()]
        return self._breakdown(amounts, code, used)

    def trades(
        self,
        side: Any,
        price: Any,
        quantity: Any,
        date: Any,
        segment: Any,
        *,
        as_frame: bool = True,
    ) -> pd.DataFrame | dict[str, np.ndarray]:
        """Cost many trades at once, each row one executed order.

        Every argument may be a scalar (applied to all trades) or a one-dimensional
        array-like of one common length:

        - `side`: "BUY" or "SELL" strings, or the signed numbers +1 and -1;
        - `price` in rupees per price unit and `quantity` in price units;
        - `date`: dates, ISO date strings, datetime64 values or a DatetimeIndex
          (timezone-aware values are converted to IST dates);
        - `segment`: segment codes, or segment ids as integers.

        Returns a DataFrame with one row per trade: date, segment, the rupee amounts
        brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst and total, then
        schedule (the version that priced the row), verified and backfilled. It is
        indexed like `price` when that is a Series, and `frame.attrs["rounding"]`
        records the rounding mode. With `as_frame=False` it returns the same columns
        as a dict of NumPy arrays.

        Raises CostError naming the first bad row. Warns with
        UnverifiedScheduleWarning once per schedule version and reason when rates
        that are not verified were used.
        """
        length = common_length(side, price, quantity, date, segment)
        days, ids = to_days(date, length), segment_ids(segment, length)
        amounts, schedule, trust = call(
            self._kernel.fill_batch,
            ids,
            side_ids(side, length),
            floats(price, length, "price"),
            floats(quantity, length, "quantity"),
            days,
        )
        self._warn_for_batch(schedule, trust, ids, days)
        codes = np.array(TRUST_CODES, dtype=object)[trust]
        columns: dict[str, np.ndarray] = {
            "date": days.astype("datetime64[D]"),
            "segment": np.array(SEGMENTS, dtype=object)[ids],
            **dict(zip((*COMPONENTS, "total"), amounts, strict=True)),
            "schedule": np.array([view["id"] for view in self._views], dtype=object)[schedule],
            "verified": codes == VERIFIED,
            "backfilled": codes == _BACKFILLED,
        }
        if not as_frame:
            return columns
        index = price.index if isinstance(price, pd.Series) and len(price) == length else None
        frame = pd.DataFrame(columns, index=index)
        frame.attrs["rounding"] = self.rounding
        return frame

    def _warn_for_batch(
        self, schedule: np.ndarray, trust: np.ndarray, ids: np.ndarray, days: np.ndarray
    ) -> None:
        """Warn once for each (schedule version, reason) the kernel flagged in a batch."""
        flagged = np.flatnonzero(trust != TRUST_CODES.index(VERIFIED))
        seen = set()
        for row in flagged:
            key = (int(schedule[row]), int(trust[row]))
            if key not in seen:
                seen.add(key)
                self._trusted(key, SEGMENTS[ids[row]], int(days[row]))

    def dp_charge(self, date: DateLike) -> DpCharge:
        """The depository participant charge for selling one scrip from holdings.

        It is charged once per scrip per day on which the scrip is sold from the
        demat account, whatever the quantity or number of orders, and it is not part
        of a trade's cost or of the contract note. A CNC buy and sell of the same
        scrip on the same day is an intraday trade and has no DP charge, so the
        caller decides when to apply it.

        Returns a DpCharge: fee, gst and total in rupees (15.34 for Zerodha), whose
        fee it is (`fee_source`), and whether the GST rate is verified for `date`.
        Raises CostError when no schedule version covers `date` (GST is dated).
        Warns with UnverifiedScheduleWarning when the GST rate is not verified.
        """
        day = to_day(date)
        fee, gst, total, fee_source, *status = call(self._kernel.dp_charge, day)
        schedule, code = self._trusted(tuple(status), "GST", day)
        return DpCharge(fee, gst, total, fee_source, schedule, code == VERIFIED)

    def exercise_stt(
        self, intrinsic: float, quantity: float, date: DateLike, segment: str = "NFO_OPT"
    ) -> float:
        """STT in rupees on an option that expires in the money and is exercised.

        The buyer pays it on the intrinsic value, at the rate in force on `date` (the
        expiry date). `intrinsic` is the intrinsic value in rupees per unit at the
        settlement price (zero when out of the money) and `quantity` is in units.
        The result is unrounded in contract-note mode and rounded to the rupee in
        calculator mode.

        Raises CostError when `segment` is not an options segment, an amount is
        negative or not finite, or no schedule version covers `date`. Warns with
        UnverifiedScheduleWarning when the rate is not verified for `date`.
        """
        code = segment_code(segment)
        day = to_day(date)
        stt, status = call(
            self._kernel.exercise_stt,
            code,
            number("intrinsic", intrinsic),
            number("quantity", quantity),
            day,
        )
        self._trusted(status, code, day)
        return stt
