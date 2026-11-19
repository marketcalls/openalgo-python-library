"""The robustness report's charts as inline SVG, in the tearsheet's chart style (its
axis text, round ticks and frame, `report._dashboard.bars` and `scale`):

    fan        percentile bands of many paths (5 to 95 and 25 to 75 percent shaded, the
               median dashed), the sampled paths thin, the run's own path bold on top;
               x is the number of trades booked, or the run's sessions, y the
               cumulative return
    histogram  one figure over the runs in round-edged bins, with marked values: the
               run's own (solid), a rule's limit (dashed), a reference such as zero
               (dotted), each labelled
    sessions   a value at each session close (the run's cumulative return, or a price),
               with marked sessions shaded

Every chart reads without colour: the run's own path is the boldest line, the median is
dashed, and a key under each chart says what each mark is. Marks carry a `<title>`, the
browser's hover readout.
"""

from __future__ import annotations

import numpy as np

from ..report import _format as fmt
from ..report._dashboard import scale
from ..report._dashboard.bars import AXIS_TEXT, MUTED_TEXT, Margin, frame
from ._blocks import LIMIT, REAL, REFERENCE, FanBlock, HistogramBlock, SessionsBlock
from ._chart_text import anchor_inside
from ._paths import Fan

WIDTH, HEIGHT = 680, 300
MARGIN = Margin(22, 16, 30, 52)
#: Ticks asked of an axis, and bins asked of a histogram (round edges make about these).
AXIS_TICKS, HISTOGRAM_BINS = 6, 30
#: Points of a sampled path: thin paths are drawn at no more than this many of the
#: fan's points, to whole pixels, which keeps 100 of them light on the page.
SAMPLE_POINTS = 101
#: The class of each mark kind's line, and how the key names its line style.
MARK_CLASSES = {REAL: "mark-real", LIMIT: "mark-limit", REFERENCE: "mark-reference"}
MARK_STYLES = {REAL: "Solid", LIMIT: "Dashed", REFERENCE: "Dotted"}


def inner() -> tuple[float, float]:
    return WIDTH - MARGIN.left - MARGIN.right, HEIGHT - MARGIN.top - MARGIN.bottom


def domain(values: np.ndarray, with_zero: bool = True) -> tuple[float, float]:
    """A round axis over the finite `values`, and zero unless `with_zero` is False (a
    price), never an empty span."""
    known = values[np.isfinite(values)]
    low, high = float(known.min()), float(known.max())
    if with_zero:
        low, high = min(0.0, low), max(0.0, high)
    if low == high:
        low, high = low - 0.01, high + 0.01
    return scale.widen(low, high)


def value_axis(y: scale.Linear, low: float, high: float, percent: bool, width: float) -> str:
    decimals = scale.percent_decimals(low, high, AXIS_TICKS)
    marks = []
    for tick in scale.ticks(low, high, AXIS_TICKS):
        text = f"{tick * 100:.{decimals}f}%" if percent else fmt.number(tick, 2)
        marks.append(
            f"<line x1='0' x2='{width:.1f}' y1='{y(tick):.1f}' y2='{y(tick):.1f}' class='grid'/>"
            f"<text x='-8' y='{y(tick) + 3:.1f}' text-anchor='end' style='{AXIS_TEXT}'>"
            f"{text}</text>"
        )
    return "".join(marks)


def points(xs: np.ndarray, ys: np.ndarray, decimals: int) -> str:
    return " ".join(f"{a:.{decimals}f},{b:.{decimals}f}" for a, b in zip(xs, ys, strict=True))


def _fan_marks(fan: Fan, x: scale.Linear, y: scale.Linear) -> list[str]:
    steps = fan.steps.astype(float)
    xs = np.array([x(s) for s in steps])
    marks = []
    for low, high, css in ((0, 4, "fan-outer"), (1, 3, "fan-inner")):
        upper = points(xs, [y(v) for v in fan.bands[:, high]], 1)
        lower = points(xs[::-1], [y(v) for v in fan.bands[::-1, low]], 1)
        marks.append(f"<polygon class='{css}' points='{upper} {lower}'/>")
    shown = np.unique(np.round(np.linspace(0, len(steps) - 1, SAMPLE_POINTS)).astype(int))
    for path in fan.samples:
        ys = [y(v) for v in path[shown]]
        marks.append(f"<polyline class='fan-path' points='{points(xs[shown], ys, 0)}'/>")
    median = points(xs, [y(v) for v in fan.bands[:, 2]], 1)
    marks.append(f"<polyline class='fan-median' points='{median}'><title>median</title></polyline>")
    if fan.real is not None:
        real = points(xs, [y(v) for v in fan.real], 1)
        marks.append(
            f"<polyline class='fan-real' points='{real}'><title>the run</title></polyline>"
        )
    return marks


def fan_chart(block: FanBlock) -> str:
    """The fan of `block` with its key."""
    fan = block.fan
    width, height = inner()
    pool = np.concatenate(
        [fan.bands.ravel(), fan.samples.ravel(), *([fan.real] if fan.real is not None else [])]
    )
    low, high = domain(pool)
    last = float(fan.steps[-1]) if len(fan.steps) else 1.0
    x, y = scale.Linear(0.0, last, 0.0, width), scale.Linear(low, high, height, 0.0)
    marks = [value_axis(y, low, high, True, width)]
    marks += _fan_axis(block, x, last)
    marks += _fan_marks(fan, x, y)
    marks.append(
        f"<line x1='0' x2='{width:.1f}' y1='{y(0.0):.1f}' y2='{y(0.0):.1f}' class='zero'/>"
    )
    svg = frame(WIDTH, HEIGHT, MARGIN, fmt.text(block.title), "".join(marks))
    key = (
        f"Shaded: 5 to 95 and 25 to 75 percent of {fmt.indian_digits(fan.runs, 0)} runs; "
        f"dashed: their median; thin: {len(fan.samples)} of the runs"
    )
    if fan.real is not None:
        key += f"; bold: {fmt.text(block.real)}"
    return f"{svg}<p class='key'>{key}. Cumulative return on the start capital.</p>"


def _fan_axis(block: FanBlock, x: scale.Linear, last: float) -> list[str]:
    """The fan's x axis: whole numbers of trades, or the first and last session's dates."""
    _, height = inner()
    marks = []
    if block.dates is not None and len(block.dates):
        for at, anchor, day in ((0.0, "start", block.dates[0]), (last, "end", block.dates[-1])):
            marks.append(
                f"<text x='{x(at):.1f}' y='{height + 16}' text-anchor='{anchor}' "
                f"style='{AXIS_TEXT}'>{day:%Y-%m-%d}</text>"
            )
        label = "sessions"
    else:
        for tick in scale.ticks(0.0, last, AXIS_TICKS):
            if tick == int(tick):
                marks.append(
                    f"<text x='{x(tick):.1f}' y='{height + 16}' text-anchor='middle' "
                    f"style='{AXIS_TEXT}'>{int(tick)}</text>"
                )
        label = "trades booked"
    width, _ = inner()
    marks.append(
        f"<text x='{width:.1f}' y='{height + 28}' text-anchor='end' style='{MUTED_TEXT}'>"
        f"{label}</text>"
    )
    return marks


def _mark_lines(block: HistogramBlock, x: scale.Linear, height: float) -> tuple[list[str], str]:
    marks, words = [], []
    width, _ = inner()
    for mark in block.marks:
        text = fmt.percent(mark.value, 1) if block.percent else fmt.number(mark.value)
        at = x(mark.value)
        anchor = anchor_inside(at, mark.label, -MARGIN.left, width + MARGIN.right)
        marks.append(
            f"<line x1='{at:.1f}' x2='{at:.1f}' y1='0' y2='{height:.1f}' "
            f"class='{MARK_CLASSES[mark.kind]}'>"
            f"<title>{fmt.text(mark.label)}: {text}</title></line>"
            f"<text x='{at:.1f}' y='-6' text-anchor='{anchor}' style='{MUTED_TEXT}'>"
            f"{fmt.text(mark.label)}</text>"
        )
        words.append(f"{MARK_STYLES[mark.kind]}: {fmt.text(mark.label)} ({text})")
    return marks, "; ".join(words)


def histogram_chart(block: HistogramBlock) -> str:
    """The histogram of `block` with its marks and key."""
    values = block.values[np.isfinite(block.values)]
    if not len(values):
        return "<p class='empty'>No figure to show: every run left it undefined.</p>"
    width, height = inner()
    marked = np.array([mark.value for mark in block.marks], dtype=float)
    low, high = domain(np.concatenate([values, marked]))
    x = scale.Linear(low, high, 0.0, width)
    counted = scale.bins(values, low, high, HISTOGRAM_BINS)
    y = scale.Linear(0.0, float(max(n for _, _, n in counted) or 1), height, 0.0)
    marks = []
    for left, right, n in counted:
        span = (
            f"{fmt.percent(left, 1)} to {fmt.percent(right, 1)}"
            if block.percent
            else f"{fmt.number(left)} to {fmt.number(right)}"
        )
        marks.append(
            f"<rect class='hist-bar' x='{x(left):.1f}' y='{y(n):.1f}' "
            f"width='{max(1.0, x(right) - x(left) - 1):.1f}' height='{height - y(n):.1f}'>"
            f"<title>{span}: {n}</title></rect>"
        )
    marks.append(
        f"<line x1='0' x2='{width:.1f}' y1='{height:.1f}' y2='{height:.1f}' class='axis'/>"
    )
    decimals = scale.percent_decimals(low, high, AXIS_TICKS)
    for tick in scale.ticks(low, high, AXIS_TICKS):
        text = f"{tick * 100:.{decimals}f}%" if block.percent else fmt.number(tick)
        anchor = anchor_inside(x(tick), text, -MARGIN.left, width + MARGIN.right)
        marks.append(
            f"<text x='{x(tick):.1f}' y='{height + 16}' text-anchor='{anchor}' "
            f"style='{AXIS_TEXT}'>{text}</text>"
        )
    lines, key = _mark_lines(block, x, height)
    svg = frame(WIDTH, HEIGHT, MARGIN, fmt.text(block.title), "".join(marks + lines))
    runs = fmt.indian_digits(len(block.values), 0)
    counted = fmt.text(block.counted)
    return (
        f"{svg}<p class='key'>Bars: how many of the {runs} {counted} fall in each bin. {key}.</p>"
    )


def _spans(marked: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) positions of each run of consecutive marked sessions."""
    edges = np.diff(np.concatenate(([0], marked.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1, strict=True))


def sessions_chart(block: SessionsBlock) -> str:
    """The values of `block` with the marked sessions shaded, and its key."""
    count = len(block.values)
    if count < 2:
        return "<p class='empty'>No chart: the run has fewer than two sessions.</p>"
    width, height = inner()
    low, high = domain(block.values, with_zero=block.percent)
    x, y = scale.Linear(0.0, count - 1.0, 0.0, width), scale.Linear(low, high, height, 0.0)
    step = width / (count - 1)
    marks = [value_axis(y, low, high, block.percent, width)]
    for first, last in _spans(block.marked):
        left = max(0.0, x(first) - step / 2)
        right = min(width, x(last) + step / 2)
        marks.append(
            f"<rect class='held-out' x='{left:.1f}' y='0' width='{right - left:.1f}' "
            f"height='{height:.1f}'/>"
        )
    line = points(np.array([x(i) for i in range(count)]), [y(v) for v in block.values], 1)
    marks.append(f"<polyline class='run-line' points='{line}'/>")
    for at, date in ((0, block.dates[0]), (count - 1, block.dates[-1])):
        anchor = "start" if at == 0 else "end"
        marks.append(
            f"<text x='{x(at):.1f}' y='{height + 16}' text-anchor='{anchor}' style='{AXIS_TEXT}'>"
            f"{date:%Y-%m-%d}</text>"
        )
    svg = frame(WIDTH, HEIGHT, MARGIN, fmt.text(block.title), "".join(marks))
    held = int(block.marked.sum())
    return (
        f"{svg}<p class='key'>Line: {fmt.text(block.line)}. "
        f"Shaded: the {held} sessions {fmt.text(block.shaded)}.</p>"
    )


__all__ = [
    "AXIS_TICKS",
    "HEIGHT",
    "MARGIN",
    "WIDTH",
    "domain",
    "fan_chart",
    "histogram_chart",
    "inner",
    "points",
    "sessions_chart",
    "value_axis",
]
