"""Built-in reports of openalgo.backtest: the HTML tearsheet in four modes, and the
metrics as a DataFrame.

    from openalgo.backtest import metrics, report

    perf = metrics.analyse(equity, exchanges="NSE", trades=trades)
    report.tearsheet(perf, "tearsheet.html")         # full report, written to a file
    html = report.tearsheet(perf, mode="basic")      # basic report, as a string
    report.tearsheet(perf, mode="metrics")           # every metric with its definition
    report.tearsheet(perf, mode="charts", log_scale=True)   # every chart
    report.metrics_table(perf)                       # the metrics as a DataFrame

The tearsheet is one self-contained HTML file: inline CSS and inline SVG charts, no
script, no font or image fetched when it is opened. It follows the reader's light or
dark setting and reads at phone width.

Modules: `tearsheet` (the entry points and page), `_pages` (the sections of each mode),
`_sections`, `_tables` and `_tables_extra` (the report sections and tables), `_charts`,
`_charts_series` and `_charts_simulation` (SVG), `_format` (numbers in Indian notation),
`_style` (the stylesheet). The metrics computation behind `openalgo.backtest.metrics`
lives here too, in `_analysis`, `_extended`, `_rolling`, `_montecarlo`, `_inputs`,
`_periods`, `_combined`, `_kernel`, `_tables_data`, `_definitions` and `_performance`.
"""

from ._pages import MODES
from .tearsheet import metrics_table, tearsheet

__all__ = ["MODES", "metrics_table", "tearsheet"]
