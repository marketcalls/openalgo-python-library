"""The dashboard's stylesheet: the design tokens of OpenStatz's dashboard (a neutral
black and white theme whose only colours are gains, losses and the benchmark's blue),
in light and dark, the layout at desktop and phone widths, and the print styles.

The theme follows the reader's setting, and the page's toggle overrides it by setting
`data-theme` on the root element, so the dark tokens are written for both (one source,
`DARK`). System fonts only: nothing is fetched. The axes of the time-series charts take
their sizes from `timeseries`.
"""

from .timeseries import TIME_AXIS_PX, VALUE_AXIS_PX

#: The light theme (the default), CSS custom properties.
LIGHT = {
    "bg": "hsl(240 6% 97.5%)",
    "panel": "hsl(0 0% 100%)",
    "panel-2": "hsl(240 5% 96%)",
    "border": "hsl(240 6% 90%)",
    "border-strong": "hsl(240 5% 80%)",
    "grid": "hsl(240 6% 92%)",
    "text": "hsl(240 10% 3.9%)",
    "text-muted": "hsl(240 3.8% 46.1%)",
    "text-faint": "hsl(240 4% 64%)",
    "accent": "hsl(240 5.9% 10%)",
    "accent-ink": "hsl(0 0% 98%)",
    "label": "hsl(240 3.8% 46.1%)",
    "pnl-pos": "hsl(142 72% 33%)",
    "pnl-neg": "hsl(0 74% 50%)",
    "heat-neg": "hsl(0 74% 50%)",
    "heat-mid": "hsl(0 0% 100%)",
    "heat-pos": "hsl(142 72% 33%)",
    "benchmark": "hsl(217 91% 52%)",
}
#: The dark theme.
DARK = {
    "bg": "hsl(240 10% 3.9%)",
    "panel": "hsl(240 9% 6.5%)",
    "panel-2": "hsl(240 4% 12%)",
    "border": "hsl(240 4% 16%)",
    "border-strong": "hsl(240 4% 30%)",
    "grid": "hsl(240 4% 13%)",
    "text": "hsl(0 0% 98%)",
    "text-muted": "hsl(240 5% 65%)",
    "text-faint": "hsl(240 4% 48%)",
    "accent": "hsl(0 0% 98%)",
    "accent-ink": "hsl(240 5.9% 10%)",
    "label": "hsl(240 5% 65%)",
    "pnl-pos": "hsl(142 69% 58%)",
    "pnl-neg": "hsl(0 91% 71%)",
    "heat-neg": "hsl(0 91% 71%)",
    "heat-mid": "hsl(240 10% 7%)",
    "heat-pos": "hsl(142 69% 58%)",
    "benchmark": "hsl(217 91% 65%)",
}
FONTS = (
    "--font-sans:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',"
    "Arial,sans-serif;--font-mono:ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,"
    "'Liberation Mono',monospace;"
)


def _tokens(theme: dict[str, str]) -> str:
    return "".join(f"--{name}:{value};" for name, value in theme.items())


_BASE = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;border:0 solid}
html{line-height:1.5;-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);font-size:16px;
-webkit-font-smoothing:antialiased;font-feature-settings:"rlig" 1,"calt" 1;min-height:100vh;
transition:background-color .15s ease,color .15s ease}
h1,h2,h3{font-size:inherit;font-weight:inherit}
button,select{font:inherit;color:inherit;background:none}
button{cursor:pointer}
svg{display:block}
table{border-collapse:collapse;width:100%}
.serif{font-family:var(--font-sans);font-weight:600;letter-spacing:-.02em}
.nums{font-variant-numeric:tabular-nums;font-family:var(--font-mono);letter-spacing:-.02em}
.eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:.66rem;font-weight:600;
color:var(--label)}
.trunc{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pos{color:var(--pnl-pos)}.neg{color:var(--pnl-neg)}
.muted{color:var(--text-muted)}.faint{color:var(--text-faint)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
section[id]{scroll-margin-top:96px}
html:not(.js) .js-only{display:none!important}
"""

_HEADER = """
.top{position:sticky;top:0;z-index:20;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
background:color-mix(in oklab,var(--bg) 85%,transparent);border-bottom:1px solid var(--border)}
.top-in{max-width:1240px;margin:0 auto;padding:0 20px;height:64px;display:flex;align-items:center;
justify-content:space-between;gap:16px}
.mark{display:flex;align-items:center;gap:12px;color:var(--text)}
.mark-ring{height:36px;width:36px;display:grid;place-items:center;border-radius:9999px;
border:1px solid var(--border-strong)}
.mark-name{font-size:1.125rem;line-height:1.75rem;color:var(--text)}
.mark .eyebrow{margin-top:2px;line-height:1}
.mark>div{line-height:1}
.nav{display:none;align-items:center;gap:28px}
.nav a{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.13em;
color:var(--text-muted);text-decoration:none;transition:color .15s}
.nav a:hover,.nav a.on{color:var(--text)}
.tools{display:flex;align-items:center;gap:12px}
.btn{font-size:.75rem;line-height:1rem;font-weight:600;text-transform:uppercase;letter-spacing:.12em;
padding:8px 14px;border-radius:6px;border:1px solid var(--border);color:var(--text-muted);
transition:color .15s,border-color .15s}
.btn:hover{color:var(--text);border-color:var(--border-strong)}
.theme{height:36px;width:36px;display:grid;place-items:center;border-radius:9999px;
border:1px solid var(--border-strong);color:var(--text-muted)}
.theme:hover{color:var(--text)}
.theme .sun{display:none}
.dark-on .theme .sun{display:block}.dark-on .theme .moon{display:none}
@media (min-width:64rem){.nav{display:flex}}
"""

_MAIN = """
main{max-width:1240px;margin:0 auto;padding:28px 20px}
main>*+*{margin-top:48px}
.title{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:12px;
border-bottom:1px solid var(--border);padding-bottom:16px}
.title h1{font-size:2.25rem;line-height:1;color:var(--text);overflow-wrap:anywhere}
.title span{font-size:.75rem;line-height:1rem;color:var(--text-muted)}
.sec-head{display:flex;align-items:baseline;gap:16px;border-bottom:1px solid var(--border);
padding-bottom:12px;margin-bottom:20px}
.sec-num{color:var(--text-muted);font-size:.875rem;line-height:1.25rem;padding-top:4px}
.sec-head h2{font-size:1.7rem;line-height:1;color:var(--text)}
.sec-sub{margin-left:auto;font-size:.75rem;line-height:1rem;color:var(--text-muted);display:none}
@media (min-width:40rem){.sec-sub{display:block}}
.gap-s{margin-top:16px}.gap-l{margin-top:24px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;min-width:0}
.card-head{padding:16px 20px 12px}
.card-head h3{font-size:1.125rem;line-height:1.25;color:var(--text)}
.card-sub{font-size:.75rem;line-height:1rem;color:var(--text-muted);margin-top:4px}
.card-body{padding:0 20px 20px}
.grid-label{margin:4px 0 10px}
.grid2,.grid3{display:grid;grid-template-columns:minmax(0,1fr);gap:16px}
@media (min-width:64rem){.grid2{grid-template-columns:repeat(2,minmax(0,1fr))}
.grid3{grid-template-columns:repeat(3,minmax(0,1fr))}}
.stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--border);
border:1px solid var(--border);border-radius:8px;overflow:hidden}
@media (min-width:40rem){.stats{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media (min-width:64rem){.stats{grid-template-columns:repeat(6,minmax(0,1fr))}}
.stat{background:var(--panel);padding:14px 16px;min-width:0}
.stat-value{margin-top:8px;font-size:1.4rem;line-height:1;color:var(--text);white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
.stat-value.pos{color:var(--pnl-pos)}.stat-value.neg{color:var(--pnl-neg)}
.stat-sub{margin-top:8px;font-size:.7rem;color:var(--text-faint);white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
.empty{color:var(--text-muted);font-size:.875rem}
.foot{display:flex;align-items:center;gap:12px;padding-top:16px;border-top:1px solid var(--border);
font-size:.75rem;line-height:1rem;color:var(--text-faint)}
.foot .serif{font-size:.875rem;color:var(--text-muted)}
.foot .end{margin-left:auto}
"""

_CHARTS = """
.ts{position:relative;width:100%;margin:0}
.ts-pane{position:absolute;left:0;top:0;width:calc(100% - @AXIS@px);overflow:visible}
.ts-pane path{vector-effect:non-scaling-stroke}
.ts-pane path:not([fill]){fill:none}
.ts-pane path[fill]{stroke:none}
.ts-grid{stroke:var(--grid);stroke-width:1}
.ts-zero{stroke:var(--text-faint);stroke-width:1;stroke-dasharray:4 4}
.ts-line{stroke-linejoin:round;stroke-linecap:round}
.ts-y{position:absolute;right:0;top:0;width:@AXIS@px}
.ts-y span,.ts-x span{position:absolute;font:11px/1 var(--font-mono);color:var(--text-faint);
white-space:nowrap}
.ts-y span{left:10px;transform:translateY(-50%)}
.ts-x{position:absolute;left:0;bottom:0;height:@TIME@px;width:calc(100% - @AXIS@px)}
.ts-x span{top:10px;transform:translateX(-50%)}
.ts-x span.off{visibility:hidden}
.ts-hair{position:absolute;pointer-events:none;display:none}
.ts-v{top:0;width:0;border-left:1px dashed var(--text-faint)}
.ts-h{left:0;height:0;border-top:1px dashed var(--text-faint)}
.ts-tag{position:absolute;pointer-events:none;display:none;font:11px/1 var(--font-mono);
padding:4px 6px;border-radius:2px;background:var(--accent);color:var(--accent-ink);
white-space:nowrap}
.ts-dot{position:absolute;pointer-events:none;display:none;width:8px;height:8px;
margin:-4px 0 0 -4px;border-radius:50%;border:2px solid var(--panel)}
svg.bars .grid{stroke:var(--grid)}svg.bars .axis{stroke:var(--border)}
svg.bars .zero{stroke:var(--border-strong);stroke-dasharray:2 2}
svg.bars .whisker{stroke:var(--text-faint)}svg.bars .base{stroke:var(--border-strong)}
svg.bars .mean{stroke:var(--pnl-neg);stroke-dasharray:5 4;stroke-width:1.3}
.streak-foot{margin-top:8px;display:flex;flex-wrap:wrap;gap:8px 24px;font-size:.75rem;
color:var(--text-muted)}
.streak-foot b{font-weight:400;color:var(--text)}
.hm-scroll{width:100%;overflow-x:auto}
.hm-grid{min-width:680px;display:grid;gap:4px;
grid-template-columns:2.75rem repeat(12,minmax(0,1fr))}
.hm-month{text-align:center;font-size:.64rem;font-weight:600;text-transform:uppercase;
letter-spacing:.05em;color:var(--text-faint);padding-bottom:4px}
.hm-year{display:flex;align-items:center;justify-content:flex-end;padding-right:8px;
font-size:.75rem;color:var(--text-muted)}
.hm-cell{height:36px;border-radius:5px;border:1px solid var(--border);display:grid;
place-items:center;font-size:.72rem;color:var(--text)}
.wk{position:relative}
.wk-bar{display:flex;align-items:center;min-height:30px;margin-bottom:12px}
.wk-pick{position:absolute;right:0;top:0;background:var(--panel);border:1px solid var(--border);
border-radius:6px;padding:6px 10px;font-size:.75rem;color:var(--text)}
.wk-year+.wk-year{margin-top:20px}
.js .wk-year:not(.on){display:none}.js .wk-year+.wk-year{margin-top:0}
.js .wk-head{display:none}
.wk-grid{display:grid;gap:4px;grid-template-columns:repeat(auto-fill,minmax(64px,1fr))}
.wk-cell{height:56px;border-radius:5px;border:1px solid var(--border);display:flex;
flex-direction:column;align-items:center;justify-content:center;gap:2px;color:var(--text)}
.wk-value{font-size:.74rem}.wk-label{font-size:.56rem;opacity:.6}
.none{color:var(--text-faint)}
.under{margin-top:16px;border-top:1px solid var(--border);padding-top:8px}
"""

_TABLES = """
.scroll{overflow-x:auto}
.tbl{font-size:.875rem;line-height:1.25rem}
.tbl thead tr{border-bottom:1px solid var(--border);color:var(--text-muted)}
.tbl th{padding:8px 12px;font-weight:500;white-space:nowrap;user-select:none}
.tbl td{padding:6px 12px}
.tbl tbody tr{border-bottom:1px solid color-mix(in oklab,var(--border) 60%,transparent)}
.tbl .l{text-align:left}.tbl .r{text-align:right}
@media (max-width:39.99rem){.tbl th,.tbl td{padding-left:8px;padding-right:8px}}
.tbl.hover tbody tr{transition:background-color .15s}
.tbl.hover tbody tr:hover{background:color-mix(in oklab,var(--panel-2) 60%,transparent)}
.js .tbl th.s{cursor:pointer}
.tbl .def{color:var(--text-muted);font-size:.75rem;max-width:36rem}
.tbl tr.total td{font-weight:500}
.tbl td.best{color:var(--pnl-pos);font-weight:600;
background:color-mix(in oklab,var(--pnl-pos) 10%,transparent)}
.trades{font-size:.8125rem}
.trades td{white-space:nowrap}
.trades th,.trades td{padding-left:7px;padding-right:7px}
.kv td:first-child{color:var(--text-muted);white-space:nowrap;width:1%}
.kv td{vertical-align:top;overflow-wrap:anywhere}
.kv tr.warn td{color:var(--pnl-neg)}
.js .tbl tr.paged{display:none}
.pager{display:flex;align-items:center;justify-content:flex-end;gap:8px;margin-top:12px}
.pager-info{font-size:.75rem;color:var(--text-muted);margin-right:auto}
.pager .btn{padding:4px 10px}
.pager .btn[disabled]{opacity:.4;cursor:default}
.prose p{font-size:.8125rem;line-height:1.35rem;color:var(--text-muted)}
.prose p+p{margin-top:8px}
"""

_PRINT = """
@media print{html,body,*{-webkit-print-color-adjust:exact!important;
print-color-adjust:exact!important}
html,body{background:var(--bg)!important}.no-print{display:none!important}
.card,section[id]{break-inside:avoid}.js .wk-year:not(.on){display:none}
.js .tbl tr.paged{display:table-row}.pager{display:none!important}}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}
*{transition:none!important;animation:none!important}}
"""


def stylesheet() -> str:
    """The whole stylesheet."""
    dark = _tokens(DARK)
    themes = (
        f":root{{{FONTS}{_tokens(LIGHT)}color-scheme:light}}"
        f"@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{{dark}"
        "color-scheme:dark}}"
        f":root[data-theme=dark]{{{dark}color-scheme:dark}}"
    )
    charts = _CHARTS.replace("@AXIS@", str(VALUE_AXIS_PX)).replace("@TIME@", str(TIME_AXIS_PX))
    return themes + _BASE + _HEADER + _MAIN + charts + _TABLES + _PRINT


__all__ = ["DARK", "LIGHT", "stylesheet"]
