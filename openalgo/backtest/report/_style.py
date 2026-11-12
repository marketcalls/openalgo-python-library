"""The tearsheet's stylesheet: colour tokens for light and dark, type, and layout.

Colour jobs: blue marks gains and the equity line, red marks losses and drawdowns
(the pair stays distinguishable under the common colour-vision deficiencies), grey
marks the benchmark. Text always wears an ink token, never a data colour. The monthly
grid uses four steps per arm of the blue and red ramps around a neutral grey.
Headings are set in a serif, figures and tables in the system sans with tabular
digits. Every font is a system font, so nothing is fetched.
"""

_LIGHT = """
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #6f6d68; --rule: #e1e0d9; --axis: #c3c2b7; --hover: rgba(11, 11, 11, 0.05);
  --gain: #2a78d6; --loss: #e34948; --bench: #898781;
  --gain-wash: rgba(42, 120, 214, 0.10); --loss-wash: rgba(227, 73, 72, 0.14);
  --heat-0: #f0efec; --heat-on-strong: #ffffff;
  --g1: #cde2fb; --g2: #9ec5f4; --g3: #6da7ec; --g4: #2a78d6;
  --l1: #fbdcdb; --l2: #f4b2b0; --l3: #ec8784; --l4: #e34948;
"""

_DARK = """
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7;
  --muted: #a3a199; --rule: #2c2c2a; --axis: #383835; --hover: rgba(255, 255, 255, 0.06);
  --gain: #3987e5; --loss: #e66767; --bench: #898781;
  --gain-wash: rgba(57, 135, 229, 0.16); --loss-wash: rgba(230, 103, 103, 0.20);
  --heat-0: #383835; --heat-on-strong: #0b0b0b;
  --g1: #183150; --g2: #1c4a80; --g3: #2a66b0; --g4: #3987e5;
  --l1: #47201f; --l2: #74302e; --l3: #a8403e; --l4: #e66767;
"""

_BASE = """
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 var(--sans); }
main { max-width: 60rem; margin: 0 auto; padding: 2.5rem 1rem 4rem; }
@media (min-width: 40rem) { main { padding: 3.5rem 2rem 5rem; } }
h1, h2 { font-family: var(--serif); font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: 2rem; line-height: 1.15; margin: 0; }
h2 { font-size: 1.3rem; line-height: 1.3; margin: 0 0 0.25rem; }
h3 { font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--ink-2); margin: 0 0 0.5rem; font-weight: 600; }
p { margin: 0 0 0.75rem; }
.masthead { border-bottom: 1px solid var(--rule); padding-bottom: 1.5rem; margin-bottom: 2.25rem; }
.kicker { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--ink-2); margin-bottom: 0.6rem; }
.dek { color: var(--ink-2); margin: 0.6rem 0 0; }
.dek span + span::before { content: "\\00B7"; margin: 0 0.5rem; color: var(--muted); }
section { margin: 0 0 3rem; }
.lede { color: var(--ink-2); font-size: 0.9rem; max-width: 44rem; }
.note { color: var(--ink-2); font-size: 0.8rem; }
.scroll { overflow-x: auto; }
h3.sub { margin-top: 1.5rem; }
"""

_FIGURES = """
.figures { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 6px;
  overflow: hidden; margin: 0 0 1.75rem; }
@media (min-width: 36rem) { .figures { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (min-width: 54rem) { .figures { grid-template-columns: repeat(6, minmax(0, 1fr)); } }
.figures div { background: var(--surface); padding: 0.9rem 1rem; }
.figures dt { font-size: 0.75rem; color: var(--ink-2); }
.figures dd { margin: 0.2rem 0 0; font-size: 1.45rem; font-weight: 600;
  letter-spacing: -0.01em; white-space: nowrap; }
.facts { columns: 3 15rem; column-gap: 2.25rem; }
.facts .group { break-inside: avoid; margin: 0 0 1.5rem; }
.facts dl { margin: 0; }
.facts dl div { display: flex; justify-content: space-between; gap: 1rem;
  padding: 0.3rem 0; border-bottom: 1px solid var(--rule); font-size: 0.875rem; }
.facts dt { color: var(--ink-2); }
.facts dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
"""

_CHARTS = """
.chart { display: grid; grid-template-columns: 4.5rem minmax(0, 1fr); margin: 0.5rem 0 0; }
.chart .y { position: relative; }
.chart .y span, .chart .x span { position: absolute; font-size: 0.72rem; color: var(--muted);
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.chart .y span { right: 0.6rem; transform: translateY(-50%); }
.chart .canvas { position: relative; height: var(--h); background: var(--surface);
  border-bottom: 1px solid var(--axis); }
.chart svg { display: block; width: 100%; height: 100%; }
.chart .x { grid-column: 2; position: relative; height: 1.6rem; }
.chart .x span { top: 0.35rem; transform: translateX(-50%); }
.chart .x span.first { transform: none; }
.chart .x span.last { transform: translateX(-100%); }
@media (max-width: 30rem) { .chart .x span:nth-child(even) { display: none; } }
.panel-title { font-size: 0.75rem; color: var(--ink-2); margin: 0.75rem 0 0.25rem 4.5rem; }
.grid { stroke: var(--rule); stroke-width: 1; vector-effect: non-scaling-stroke; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round;
  vector-effect: non-scaling-stroke; }
.line.equity { stroke: var(--gain); }
.line.benchmark { stroke: var(--bench); stroke-width: 1.5; }
.line.drawdown { stroke: var(--loss); stroke-width: 1.5; }
.wash { fill: var(--loss-wash); }
.hover rect { fill: transparent; }
.hover rect:hover { fill: var(--hover); }
.legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1.25rem; font-size: 0.8rem;
  color: var(--ink-2); margin: 0.25rem 0 0.5rem; }
.legend span::before { content: ""; display: inline-block; width: 1rem; height: 2px;
  vertical-align: middle; margin-right: 0.4rem; background: var(--key); }
.hist { display: flex; align-items: flex-end; gap: 2px; height: 10rem;
  border-bottom: 1px solid var(--axis); background: var(--surface); }
.hist .bin { flex: 1 1 0; height: 100%; display: flex; align-items: flex-end;
  justify-content: center; }
.hist .bin:hover { background: var(--hover); }
.hist .bin i { display: block; width: 100%; max-width: 24px; border-radius: 4px 4px 0 0;
  background: var(--gain); }
.hist .bin.loss i { background: var(--loss); }
.hist-axis { position: relative; height: 1.6rem; }
.hist-axis span { position: absolute; top: 0.35rem; transform: translateX(-50%);
  font-size: 0.72rem; color: var(--muted); white-space: nowrap; }
"""

_TABLES = """
table { font-variant-numeric: tabular-nums; }
.data { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.data th { text-align: left; font-weight: 600; color: var(--ink-2); padding: 0.45rem 0.6rem;
  border-bottom: 1px solid var(--axis); white-space: nowrap; }
.data td { padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--rule); white-space: nowrap; }
.data .num { text-align: right; }
.data tr.total td { font-weight: 600; border-bottom-color: var(--axis); }
.data tr.gap td { text-align: center; color: var(--ink-2); font-style: italic;
  background: var(--surface); }
.heat { width: 100%; border-collapse: separate; border-spacing: 2px; font-size: 0.78rem; }
.heat th { font-weight: 500; color: var(--ink-2); padding: 0.25rem; text-align: center; }
.heat td { text-align: center; padding: 0.45rem 0.25rem; border-radius: 3px;
  background: var(--heat-0); min-width: 3rem; }
.heat td.empty { background: transparent; }
.heat td.year { background: transparent; font-weight: 600; }
.heat .g1 { background: var(--g1); } .heat .g2 { background: var(--g2); }
.heat .g3 { background: var(--g3); } .heat .g4 { background: var(--g4); }
.heat .l1 { background: var(--l1); } .heat .l2 { background: var(--l2); }
.heat .l3 { background: var(--l3); } .heat .l4 { background: var(--l4); }
.heat .g4, .heat .l4 { color: var(--heat-on-strong); }
.bars td.track { width: 55%; }
.bars .halves { display: grid; grid-template-columns: 1fr 1fr; align-items: center; }
.bars .halves span { display: flex; height: 10px; }
.bars .halves span:first-child { justify-content: flex-end; border-right: 1px solid var(--axis); }
.bars .halves i { display: block; height: 100%; }
.bars .halves .down i { background: var(--loss); border-radius: 4px 0 0 4px; }
.bars .halves .up i { background: var(--gain); border-radius: 0 4px 4px 0; }
.list { padding-left: 1.2rem; margin: 0; }
.list li { margin: 0.25rem 0; }
.glossary { font-size: 0.85rem; columns: 2 18rem; column-gap: 2.25rem; }
.glossary div { break-inside: avoid; margin: 0 0 0.6rem; }
.glossary dt { font-weight: 600; }
.glossary dd { margin: 0; color: var(--ink-2); }
footer { border-top: 1px solid var(--rule); padding-top: 1rem; font-size: 0.8rem;
  color: var(--ink-2); }
@media print { body { background: #ffffff; } section { break-inside: avoid; } }
"""

_FONTS = """
  --serif: Charter, "Bitstream Charter", "Sitka Text", Cambria, Georgia, serif;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
"""

STYLESHEET = (
    ":root { color-scheme: light dark;"
    + _FONTS
    + _LIGHT
    + "}\n@media (prefers-color-scheme: dark) { :root {"
    + _DARK
    + "} }\n"
    + _BASE
    + _FIGURES
    + _CHARTS
    + _TABLES
)

__all__ = ["STYLESHEET"]
