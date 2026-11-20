"""The machinery of custom tearsheets (`openalgo.backtest.report.custom`, design 19).

Modules: `_layout` (a layout's checks, its parts and order of sections, and the page it
renders), `_blocks` (the user blocks and the functions that make them), `_data` (what
each kind of block accepts), `_render` (a section's blocks in cards and rows; tables,
tiles, text and fragments) and `_charts` (the user charts), `_theme` (colours and the
page's second stylesheet) with `_colour_names`, `_fragment` (the escape hatch's check)
with its allow list `_allowed`, and `_text` (escaping to ASCII).

A page that asks for nothing custom is rendered by the default path, untouched.
"""
