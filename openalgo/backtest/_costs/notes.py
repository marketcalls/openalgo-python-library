"""Contract-note rounding of per-trade accruals."""

import numpy as np
import pandas as pd

from ..errors import CostError
from .inputs import floats, segment_ids, to_days
from .kernel import COMPONENTS, CONTRACT_NOTE_RULE, NOTE_GROUPS, ROUNDING_MODES, call, native

__all__ = ["contract_note"]

# The mode whose output is unrounded accruals: the only valid input here.
_ACCRUAL_MODE = ROUNDING_MODES[0]
_NEEDED = ("date", "segment", *COMPONENTS)


def contract_note(trades: pd.DataFrame) -> pd.DataFrame:
    """Round per-trade accruals the way a contract note does.

    `trades` is the DataFrame returned by `CostModel.trades` (or `costs.trade_costs`)
    in contract-note mode, passed as it is: it carries the date, segment and
    unrounded components of each trade. A frame from calculator mode is refused,
    because its tax lines are already rounded trade by trade (a small option sale
    would show no STT at all).

    A broker issues one contract note per trade day for each of NSE cash, BSE cash,
    NSE F&O, BSE F&O and MCX. On each note every charge is summed exactly over the
    day's trades; STT or CTT and stamp duty are rounded to the nearest rupee (50
    paise up), and brokerage, exchange charge, SEBI fee and GST to the nearest paisa.
    This rule is an assumption that still has to be checked against real contract
    notes; the result says so in `attrs`.

    Returns a DataFrame with one row per (date, note): date, note, fills, the six
    charged components, total (what the note charges), accrued_total (the unrounded
    sum) and adjustment (total minus accrued_total), all in rupees. Its `attrs` hold
    `rule` (the rounding rule applied) and `rule_verified` (False).

    Raises CostError when `trades` is not such a frame, was costed in another
    rounding mode, or holds an amount that is negative or not finite.
    """
    if not isinstance(trades, pd.DataFrame) or any(name not in trades for name in _NEEDED):
        raise CostError(
            f"contract_note takes the DataFrame returned by trades(), with the columns {_NEEDED}"
        )
    mode = trades.attrs.get("rounding")
    if mode != _ACCRUAL_MODE:
        raise CostError(
            f"contract_note needs unrounded accruals, but this frame says rounding={mode!r}: "
            f"cost the trades with rounding={_ACCRUAL_MODE!r} and pass the frame as trades() "
            "returned it (operations that rebuild a frame can drop its attrs)"
        )
    length = len(trades)
    amounts = [floats(trades[name].to_numpy(), length, name) for name in COMPONENTS]
    days = to_days(trades["date"], length)
    segments = segment_ids(trades["segment"].to_numpy(), length)
    note_days, groups, fills, charged, total, accrued_total, adjustment = call(
        native.cost_contract_notes, days, segments, amounts
    )
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(np.asarray(note_days, dtype=np.int64).astype("datetime64[D]")),
            "note": [NOTE_GROUPS[group] for group in groups],
            "fills": fills,
            **dict(zip(COMPONENTS, charged, strict=True)),
            "total": total,
            "accrued_total": accrued_total,
            "adjustment": adjustment,
        }
    )
    frame.attrs["rule"] = CONTRACT_NOTE_RULE
    frame.attrs["rule_verified"] = False
    return frame
