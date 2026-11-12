"""Built-in reports of openalgo.backtest: the HTML tearsheet in four modes, and the
metrics as a DataFrame.

    from openalgo.backtest import metrics, report

    perf = metrics.analyse(equity, exchanges="NSE", trades=trades)
    report.tearsheet(perf, "tearsheet.html")         # full report, written to a file
    html = report.tearsheet(perf, mode="basic")      # basic report, as a string
    report.tearsheet(perf, mode="metrics")           # every metric with its definition
    report.tearsheet(perf, mode="charts", log_scale=True)   # every chart
    report.metrics_table(perf)                       # the metrics as a DataFrame

The tearsheet is one self-contained HTML dashboard in the layout of OpenStatz's
dashboard: inline CSS, inline SVG charts and a small inline script for the theme
toggle, sorting, paging and the chart crosshair; no font or image is fetched when it
is opened, and it reads in full with script turned off. It follows the reader's light
or dark setting and reads at phone width.

Modules: `tearsheet` (the entry points), `_dashboard` (the page, its sections, charts
and tables), `_format` (numbers in Indian notation). The metrics computation behind
`openalgo.backtest.metrics` lives here too, in `_analysis`, `_extended`, `_rolling`,
`_montecarlo`, `_inputs`, `_periods`, `_combined`, `_kernel`, `_tables_data`,
`_definitions` and `_performance`.
"""

from .tearsheet import MODES, metrics_table, tearsheet

__all__ = ["MODES", "metrics_table", "tearsheet"]
