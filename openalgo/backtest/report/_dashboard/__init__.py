"""The HTML tearsheet as a dashboard in the design of OpenStatz's dashboard: a sticky
header with section links, then numbered sections (overview, performance, risk,
monthly, distribution, metrics, trades, the strategies of a combined result, notes)
built from a `Performance`, in one self-contained page.

Modules: `page` (the page and its order of sections), `sections` (the return
sections), `sections_run` (metrics, trades, strategies, notes), `layout` (what a page
shows, sections and cards), `cards` (the stat tiles), `tables` and `trades` (tables),
`timeseries`, `bars` and `heatmaps` (charts as inline SVG and HTML), `scale` (axis
arithmetic), `data` (the series, tables and benchmark figures the sections draw),
`style` and `script` (the inline stylesheet and script).
"""

from .layout import MODE_PARTS, View
from .page import render

__all__ = ["MODE_PARTS", "View", "render"]
