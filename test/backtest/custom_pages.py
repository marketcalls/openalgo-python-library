"""Shared inputs and readers of the custom tearsheet tests (`test_report_custom_*.py`): a
small performance with a benchmark and closed trades, and readers of a page's sections,
header links and section numbers."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from openalgo.backtest import metrics

IST = "Asia/Kolkata"


def session_closes(start: str, end: str) -> pd.DatetimeIndex:
    """Weekday closes at 15:30 IST from `start` to `end`."""
    return pd.bdate_range(start, end, tz=IST) + pd.Timedelta(hours=15, minutes=30)


def small_performance() -> metrics.Performance:
    """Four months of a random walk with a benchmark and eight closed trades."""
    closes = session_closes("2025-06-02", "2025-09-30")
    rng = np.random.default_rng(11)
    equity = pd.Series(1_00_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(closes))), closes)
    bench = pd.Series(24_000 * np.cumprod(1 + rng.normal(0.0003, 0.008, len(closes))), closes)
    net = np.array([(-1) ** i * (100.0 + 10 * i) for i in range(8)])
    trades = pd.DataFrame(
        {
            "symbol": "SBIN",
            "entry_time": closes[:8],
            "exit_time": closes[1:9],
            "entry_price": 500.0,
            "exit_price": 500.0 + net / 10,
            "quantity": 10.0,
            "side": "BUY",
            "gross_pnl": net,
            "net_pnl": net,
            "costs": 0.0,
        }
    )
    return metrics.analyse(
        equity, exchanges="NSE", trades=trades, benchmark=bench, rolling_window=20
    )


def section_anchors(page: str) -> list[str]:
    """The ids of the page's numbered sections, in page order."""
    return re.findall(r"<section id='([^']+)'", page)


def header_links(page: str) -> list[str]:
    """The anchors the page's header links to, in order."""
    nav = page[page.index("<nav class='nav'>") : page.index("</nav>")]
    return re.findall(r"<a href='#([^']+)'", nav)


def section_numbers(page: str) -> list[int]:
    """The numbers of the page's sections, in page order."""
    return [int(n) for n in re.findall(r"<span class='sec-num nums'>(\d+)</span>", page)]


def section_of(page: str, anchor: str) -> str:
    """The HTML of the section with id `anchor`."""
    part = page[page.index(f"<section id='{anchor}'") :]
    return part[: part.index("</section>")]


def element_ids(page: str) -> list[str]:
    """Every id of an element of the page, in order."""
    return re.findall(r"\sid='([^']+)'", page)


__all__ = [
    "element_ids",
    "header_links",
    "section_anchors",
    "section_numbers",
    "section_of",
    "session_closes",
    "small_performance",
]
