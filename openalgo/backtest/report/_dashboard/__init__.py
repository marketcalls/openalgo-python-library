"""The HTML tearsheet as a dashboard in the design of OpenStatz's dashboard: a sticky
header with section links, then numbered sections (overview, performance, risk,
monthly, distribution, metrics, trades, overfitting checks, the strategies of a
combined result, notes) built from a `Performance`, in one self-contained page.

Modules: `page` (the page and its order of sections, and the page shell the robustness
report shares), `sections` (the return sections), `sections_run` (metrics, trades,
strategies, notes), `overfitting` (the quick overfitting checks), `layout` (what a page
shows, sections and cards), `cards` (the stat tiles), `tables` and `trades` (tables),
`excursions` (the trading cost block and the MAE and MFE subsection of the trades),
`timeseries`, `bars`, `scatter` and `heatmaps` (charts as inline SVG and HTML), `scale`
(axis arithmetic), `data` (the series, tables and benchmark figures the sections draw),
`style` and `script` (the inline stylesheet and script).
"""

from .layout import MODE_PARTS, View
from .page import render

__all__ = ["MODE_PARTS", "View", "render"]
