"""Built-in reports of openalgo.backtest: the HTML tearsheet.

    from openalgo.backtest import metrics, report

    perf = metrics.analyse(equity, exchanges="NSE", trades=trades)
    report.tearsheet(perf, "tearsheet.html")     # writes the file, returns its path
    html = report.tearsheet(perf)                # returns the HTML as a string

The tearsheet is one self-contained HTML file: inline CSS and inline SVG charts, no
script, no font or image fetched when it is opened. It follows the reader's light or
dark setting and reads at phone width.

Modules: `tearsheet` (the entry point and page), `_sections` and `_tables` (the
report sections), `_charts` (SVG), `_format` (numbers in Indian notation), `_style`
(the stylesheet). The metrics computation behind `openalgo.backtest.metrics` lives
here too, in `_analysis`, `_inputs`, `_periods`, `_combined`, `_kernel`,
`_definitions` and `_performance`.
"""

from .tearsheet import tearsheet

__all__ = ["tearsheet"]
