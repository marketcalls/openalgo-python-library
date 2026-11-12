"""Data quality checks, run on every load and returned as a structured report
(PRD 06 section 5).

Severities:

- error: the data cannot be trusted (a non-positive price, a high or low that
  contradicts the open and close, a suspected unadjusted split or bonus: see
  `splits`). `load_bars` stops on these unless told otherwise.
- warning: the data is usable but incomplete or was trimmed, and the count is
  reported (duplicate timestamps, missing bars against the calendar, bars dropped
  because they were outside the regular session or built from part of a period).
- info: a conversion the user should know about (open interest of 0 turned into
  NaN where the source supplies none, pre-open bars dropped, a partial bar dropped).

Timestamp offsets and mixed conventions are not in the report: they stop
normalisation with a TimestampError. The checks of PRD 06 section 5 that this phase
does not run are listed in every report under `not_run`, so that their absence from
`issues` is never read as a pass.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._calendar_convert import format_ist, from_day, interval_nanos, to_day
from .._layout import PRICE_COLUMNS, parse_interval
from .._protocols import SessionCalendar
from ..errors import DataQualityError
from ._sessions import Sessions, ist_days
from ._severity import ERROR, INFO, SEVERITIES, WARNING
from .corporate_actions import CorporateAction, declared
from .normalise import MAX_EXAMPLES, NormaliseReport
from .splits import PriceGap, check_splits

# The checks of PRD 06 section 5 that Phase 0 does not run, with the reason.
NOT_RUN = (
    "zero_volume_runs: not checked in Phase 0",
    "outlier_returns: not checked in Phase 0",
    "survivorship_bias: a property of a universe, checked where portfolios are built",
    "adjustment_status: the sources do not say whether a series is adjusted; treat it as unknown",
)

# Rows removed during normalisation, reported with the checks of the bars themselves:
# (check name, severity, NormaliseReport count, message, NormaliseReport examples).
_NORMALISATION_ISSUES = (
    (
        "oi_zero_to_nan",
        INFO,
        "oi_zero_to_nan",
        "open interest was 0 on all {count} rows, which means the source supplies none: "
        "converted to NaN (unknown)",
        None,
    ),
    (
        "preopen_dropped",
        INFO,
        "preopen_dropped",
        "{count} pre-open bars (before the session open) were dropped",
        None,
    ),
    (
        "postclose_dropped",
        WARNING,
        "postclose_dropped",
        "{count} bars at or after the session close were dropped; check the session times "
        "of the calendar if this exchange trades later",
        None,
    ),
    (
        "closed_day_dropped",
        WARNING,
        "closed_day_dropped",
        "{count} rows on dates the calendar marks closed were dropped; add the special "
        "session to the calendar to keep them",
        "closed_day_examples",
    ),
    (
        "partial_dropped",
        INFO,
        "partial_dropped",
        "{count} bars that had not closed yet were dropped",
        None,
    ),
    (
        "incomplete_period_dropped",
        WARNING,
        "incomplete_period_dropped",
        "{count} weekly or longer bars were dropped because their period is not wholly "
        "inside the requested dates and the data obtained for them",
        None,
    ),
    (
        "late_session_start",
        WARNING,
        "late_start_days",
        "on {count} session days the first bar opens after the session open (no trade in "
        "the opening bars, or bars are missing)",
        "late_start_examples",
    ),
)


@dataclass(frozen=True)
class QualityIssue:
    """One finding of a quality check.

    check: the check's name, for example "missing_bars".
    severity: "error", "warning" or "info".
    count: rows affected (session days for "late_session_start" and
        "missing_sessions").
    message: a sentence for the user.
    examples: up to five IST timestamps or dates where it happens.
    """

    check: str
    severity: str
    count: int
    message: str
    examples: tuple[str, ...] = ()


@dataclass
class QualityReport:
    """The outcome of every quality check for one series.

    issues: what the checks found. not_run: the checks that did not run, each with
    the reason. price_gaps: the session boundaries the split check reports, with the
    prices that are its evidence (`splits.PriceGap`).
    """

    symbol: str | None = None
    exchange: str | None = None
    interval: str | None = None
    rows: int = 0
    issues: list[QualityIssue] = field(default_factory=list)
    not_run: list[str] = field(default_factory=lambda: list(NOT_RUN))
    price_gaps: list[PriceGap] = field(default_factory=list)

    def add(
        self, check: str, severity: str, count: int, message: str, examples: Iterable[str] = ()
    ) -> None:
        """Record a finding. A count of zero records nothing.

        Raises ValueError for a severity outside SEVERITIES.
        """
        if severity not in SEVERITIES:
            raise ValueError(f"unknown severity {severity!r}")
        if count:
            shown = tuple(examples)[:MAX_EXAMPLES]
            self.issues.append(QualityIssue(check, severity, int(count), message, shown))

    def of_severity(self, severity: str) -> list[QualityIssue]:
        """The issues of one severity, in the order they were found."""
        return [issue for issue in self.issues if issue.severity == severity]

    @property
    def errors(self) -> list[QualityIssue]:
        """Issues of error severity."""
        return self.of_severity(ERROR)

    @property
    def warnings(self) -> list[QualityIssue]:
        """Issues of warning severity."""
        return self.of_severity(WARNING)

    @property
    def ok(self) -> bool:
        """True when no check of error severity failed."""
        return not self.errors

    def count(self, check: str) -> int:
        """Total count reported for a check name (0 when it found nothing)."""
        return sum(issue.count for issue in self.issues if issue.check == check)

    def summary(self) -> str:
        """One line per issue, errors first, then the checks that did not run."""
        lines = [
            f"{severity}: {issue.message}"
            for severity in SEVERITIES
            for issue in self.of_severity(severity)
        ]
        lines.extend(f"not run: {text}" for text in self.not_run)
        return "\n".join(lines)

    def raise_for_errors(self) -> "QualityReport":
        """Return the report, or raise when a check of error severity failed.

        Raises DataQualityError, which carries this report as `report`.
        """
        errors = self.errors
        if not errors:
            return self
        text = "; ".join(issue.message for issue in errors)
        raise DataQualityError(
            f"{self.exchange}:{self.symbol} {self.interval}: data quality check failed: "
            f"{text}. Fix the data, or load with strict=False to receive the bars together "
            "with this report.",
            report=self,
        )


def check_bars(
    bars: pd.DataFrame,
    *,
    calendar: SessionCalendar | Sessions | None = None,
    exchange: str | None = None,
    interval: str | None = None,
    normalisation: NormaliseReport | None = None,
    now_ns: int | None = None,
    corporate_actions: Iterable[CorporateAction] | None = None,
    base_bars: pd.DataFrame | None = None,
) -> QualityReport:
    """Run the quality checks on canonical bars and return a QualityReport.

    calendar: the calendar for the missing-bars check, or `Sessions` bound to the
        exchange; None is the package default calendar.
    exchange, interval: default to `bars.attrs`.
    normalisation: the NormaliseReport of the load, so that rows removed before the
        frame was built (duplicates, out-of-session bars, a partial bar, open
        interest of 0) appear in the same report.
    now_ns: the load time in UTC nanoseconds; bars of a session that is still
        running are not counted as missing before they can have closed.
    corporate_actions: the splits, bonuses and checked dates declared for the share
        (`corporate_actions.CorporateAction`), read by the split check (`splits`).
    base_bars: the bars `bars` were resampled from, when they were; the split check
        reads them, because a split inside a weekly or longer bar cannot be seen.

    The function reports and never raises on bad data; call
    `QualityReport.raise_for_errors` to stop on error severity. Raises ConfigError
    for an interval string that is not an OpenAlgo interval, and for declarations
    that are not CorporateAction objects, conflict, or are given for a series that
    is not a share on NSE or BSE.
    """
    exchange = exchange if exchange is not None else bars.attrs.get("exchange")
    interval = interval if interval is not None else bars.attrs.get("interval")
    report = QualityReport(
        symbol=bars.attrs.get("symbol"), exchange=exchange, interval=interval, rows=len(bars)
    )
    ts_open = bars["ts_open"].to_numpy()
    _check_prices(bars, ts_open, report)
    _check_duplicates(ts_open, normalisation, report)
    if exchange is None or interval is None:
        report.not_run.append("missing_bars: the exchange or interval of the series is unknown")
    elif len(ts_open):
        _check_missing(np.sort(ts_open), calendar, exchange, interval, now_ns, report)
    check_splits(
        bars if base_bars is None else base_bars,
        report,
        exchange=exchange,
        interval=interval if base_bars is None else base_bars.attrs.get("interval"),
        declared=declared(corporate_actions),
    )
    if normalisation is not None:
        for check, severity, count_field, message, examples_field in _NORMALISATION_ISSUES:
            count = getattr(normalisation, count_field)
            examples = getattr(normalisation, examples_field) if examples_field else ()
            report.add(check, severity, count, message.format(count=count), examples)
    return report


def _examples(ts_open: np.ndarray, rows: np.ndarray) -> list[str]:
    return [format_ist(int(ts)) for ts in ts_open[rows][:MAX_EXAMPLES]]


def _check_prices(bars: pd.DataFrame, ts_open: np.ndarray, report: QualityReport) -> None:
    prices = {name: bars[name].to_numpy() for name in PRICE_COLUMNS}
    non_finite = np.zeros(len(bars), dtype=bool)
    non_positive = np.zeros(len(bars), dtype=bool)
    for values in prices.values():
        finite = np.isfinite(values)
        non_finite |= ~finite
        non_positive |= finite & (values <= 0.0)
    non_positive &= ~non_finite  # a row is reported once, for its worst price
    report.add(
        "non_finite_price",
        ERROR,
        non_finite.sum(),
        f"{int(non_finite.sum())} rows have an open, high, low or close that is NaN or infinite",
        _examples(ts_open, non_finite),
    )
    report.add(
        "non_positive_price",
        ERROR,
        non_positive.sum(),
        f"{int(non_positive.sum())} rows have a price that is zero or negative",
        _examples(ts_open, non_positive),
    )
    unusable = non_finite | non_positive
    body_high = np.maximum(prices["open"], prices["close"])
    body_low = np.minimum(prices["open"], prices["close"])
    with np.errstate(invalid="ignore"):
        contradicts = (prices["high"] < body_high) | (prices["low"] > body_low)
    contradicts &= ~unusable  # those rows are already reported above
    report.add(
        "ohlc_inconsistent",
        ERROR,
        contradicts.sum(),
        f"{int(contradicts.sum())} rows have a high below the open or close, or a low above them",
        _examples(ts_open, contradicts),
    )


def _check_duplicates(
    ts_open: np.ndarray, normalisation: NormaliseReport | None, report: QualityReport
) -> None:
    in_frame = len(ts_open) - len(np.unique(ts_open))
    removed = normalisation.duplicates_removed if normalisation is not None else 0
    conflicting = normalisation.duplicates_conflicting if normalisation is not None else 0
    parts = []
    if removed:
        parts.append(f"{removed} removed on load (the first row of each timestamp was kept)")
    if conflicting:
        parts.append(f"{conflicting} of them carried different values")
    if in_frame:
        parts.append(f"{in_frame} still present")
    total = in_frame + removed
    report.add(
        "duplicate_timestamps", WARNING, total, f"{total} duplicate timestamps: " + ", ".join(parts)
    )


def _check_missing(
    ts_open: np.ndarray,
    calendar: SessionCalendar | Sessions | None,
    exchange: str,
    interval: str,
    now_ns: int | None,
    report: QualityReport,
) -> None:
    """Count bars and sessions that are absent between the first and the last bar."""
    kind, _ = parse_interval(interval)
    if kind not in ("intraday", "D"):
        report.not_run.append(
            f"missing_bars: not checked for interval {interval!r} (only intraday and D)"
        )
        return
    days = ist_days(ts_open)
    first, last = from_day(days[0]), from_day(days[-1])
    sessions = (
        calendar if isinstance(calendar, Sessions) else Sessions(calendar, exchange, first, last)
    )
    sessions = sessions.covering(first, last)
    session_days = sessions.days(first, last)
    with_bars = set(days.tolist())
    without = [str(day) for day in session_days if to_day(day) not in with_bars]
    if kind == "intraday":
        counts = sessions.native.check_intraday(ts_open, exchange=exchange, interval=interval)
        missing = counts["missing_bars"]
        session = sessions.session(last)
        if now_ns is not None and session is not None and session[0] <= now_ns < session[1]:
            # The last session is still running: its bars that cannot have closed yet
            # are on the grid but are not missing.
            step_ns = interval_nanos(interval)
            on_grid = -(-(session[1] - session[0]) // step_ns)
            missing -= on_grid - (now_ns - session[0]) // step_ns
        report.add(
            "missing_bars",
            WARNING,
            missing,
            f"{missing} bars of the session grid are missing on the "
            f"{counts['session_days']} session days that have bars",
        )
        report.add(
            "off_lattice_bars",
            WARNING,
            counts["off_lattice_bars"],
            f"{counts['off_lattice_bars']} bars do not start a whole number of intervals after "
            "the session open",
        )
    report.add(
        "missing_sessions",
        WARNING,
        len(without),
        f"{len(without)} of {len(session_days)} session days between the first and the last "
        "bar have no bar at all",
        without,
    )


__all__ = ["QualityIssue", "QualityReport", "check_bars"]
