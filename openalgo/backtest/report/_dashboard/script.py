"""The dashboard's script, plain JavaScript inline in the page. The page reads in full
without it; the script only adds what is interactive:

    theme       the toggle switches light and dark and remembers the choice in the
                browser (`THEME_KEY`); before the first paint `HEAD` applies it
    print       the Export PDF button opens the browser's print dialog
    navigation  the section link of the section in view is marked as the reader scrolls
    tables      a click on a column heading sorts (numbers largest first, text A to Z;
                again reverses; a third time restores the first order), and the trade
                table shows one page at a time with Prev and Next
    weekly map  a year picker shows one year
    charts      a crosshair follows the pointer with the value at its height, the
                session's date, and a dot on each series; time labels that would
                overlap at the chart's real width are hidden

The script is written over several lines here for reading and joined into one line in
the page; every statement ends with a semicolon or a brace, so the join is safe.
"""

#: The browser storage key of the reader's theme choice.
THEME_KEY = "openalgo-tearsheet-theme"
#: Placeholders filled when the script is built.
_KEY_SLOT, _GAP_SLOT = "@THEME_KEY@", "@MIN_GAP@"

#: Runs in the head: marks that script runs and applies a remembered theme.
HEAD = (
    "(function(){var d=document.documentElement;d.classList.add('js');"
    f"try{{var t=localStorage.getItem('{THEME_KEY}');"
    "if(t==='light'||t==='dark')d.setAttribute('data-theme',t)}catch(e){}})();"
)

_THEME = """
var root=document.documentElement,media=window.matchMedia?
window.matchMedia('(prefers-color-scheme: dark)'):null;
function isDark(){var t=root.getAttribute('data-theme');
return t?t==='dark':!!(media&&media.matches);}
function paint(){var dark=isDark(),b=document.querySelector('.theme');
root.classList.toggle('dark-on',dark);
if(b){b.setAttribute('aria-label',dark?'Switch to light theme':'Switch to dark theme');
b.title=dark?'Light':'Dark';}}
paint();if(media&&media.addEventListener)media.addEventListener('change',paint);
var toggle=document.querySelector('.theme');
if(toggle)toggle.addEventListener('click',function(){var next=isDark()?'light':'dark';
root.setAttribute('data-theme',next);try{localStorage.setItem('@THEME_KEY@',next);}catch(e){}
paint();});
var printer=document.querySelector('.print');
if(printer)printer.addEventListener('click',function(){window.print();});
"""

_NAV = """
var links=Array.prototype.slice.call(document.querySelectorAll('.nav a'));
if(links.length&&'IntersectionObserver' in window){var inView={};
var spy=new IntersectionObserver(function(entries){entries.forEach(function(e){
inView[e.target.id]=e.isIntersecting;});
var current=links.filter(function(a){return inView[a.getAttribute('href').slice(1)];})[0];
if(!current)return;links.forEach(function(a){a.classList.toggle('on',a===current);});},
{rootMargin:'-45% 0px -50% 0px'});
links.forEach(function(a){var s=document.getElementById(a.getAttribute('href').slice(1));
if(s)spy.observe(s);});}
"""

_TABLES = """
function cellValue(row,col){var v=row.cells[col].getAttribute('data-v');
if(v===null||v==='')return null;var n=Number(v);return isNaN(n)?v:n;}
function order(a,b,col){var x=cellValue(a,col),y=cellValue(b,col);
if(x===null&&y===null)return 0;if(x===null)return -1;if(y===null)return 1;
if(typeof x==='number'&&typeof y==='number')return x-y;x=String(x);y=String(y);
return x<y?-1:x>y?1:0;}
function setupTable(table){var body=table.tBodies[0],rows=Array.prototype.slice.call(body.rows);
var heads=Array.prototype.slice.call(table.tHead.rows[0].cells);
var size=Number(table.getAttribute('data-page'))||0,page=0,col=-1,dir=0;
var pager=size?table.parentNode.nextElementSibling:null;
heads.forEach(function(th){th.setAttribute('data-label',th.textContent);});
function numeric(c){for(var i=0;i<rows.length;i++){var v=cellValue(rows[i],c);
if(v!==null)return typeof v==='number';}return true;}
function render(){var list=rows.slice();
if(dir)list.sort(function(a,b){return dir*order(a,b,col)||
(Number(a.getAttribute('data-i'))-Number(b.getAttribute('data-i')));});
list.forEach(function(r){body.appendChild(r);});
heads.forEach(function(th,c){var arrow=c===col?(dir>0?' \\u2191':' \\u2193'):'';
th.textContent=th.getAttribute('data-label')+arrow;});
if(!size)return;var pages=Math.max(1,Math.ceil(list.length/size));
page=Math.min(page,pages-1);
list.forEach(function(r,i){r.classList.toggle('paged',Math.floor(i/size)!==page);});
var from=page*size+1,to=Math.min(list.length,(page+1)*size);
pager.querySelector('.pager-info').textContent='Rows '+from+' to '+to+' of '+list.length;
pager.querySelector('.pg-prev').disabled=page===0;
pager.querySelector('.pg-next').disabled=page>=pages-1;}
heads.forEach(function(th,c){if(!th.classList.contains('s'))return;
th.addEventListener('click',function(){var first=numeric(c)?-1:1;
if(col!==c){col=c;dir=first;}else if(dir===first){dir=-first;}else{col=-1;dir=0;}
page=0;render();});});
if(pager){pager.querySelector('.pg-prev').addEventListener('click',function(){page-=1;render();});
pager.querySelector('.pg-next').addEventListener('click',function(){page+=1;render();});}
render();}
Array.prototype.forEach.call(document.querySelectorAll('table[data-sort]'),setupTable);
"""

_WEEKLY = """
Array.prototype.forEach.call(document.querySelectorAll('[data-weekly]'),function(box){
var pick=box.querySelector('.wk-pick');if(!pick)return;
pick.addEventListener('change',function(){
Array.prototype.forEach.call(box.querySelectorAll('.wk-year'),function(b){
b.classList.toggle('on',b.getAttribute('data-year')===pick.value);});});});
"""

_CHARTS = """
var MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
var dates=(document.body.getAttribute('data-dates')||'').split(',');
function shortDay(iso){if(!iso)return '';var p=iso.split('-');
return Number(p[2])+' '+MONTHS[Number(p[1])-1]+" '"+p[0].slice(2);}
function setupChart(fig){var pane=fig.querySelector('.ts-pane');
var n=Number(fig.getAttribute('data-n')),first=Number(fig.getAttribute('data-first'));
var top=Number(fig.getAttribute('data-top')),bottom=Number(fig.getAttribute('data-bottom'));
var log=fig.getAttribute('data-log')==='1',pct=fig.getAttribute('data-fmt')==='percent';
var lines=[];for(var k=0;k<Number(fig.getAttribute('data-lines'));k++){
lines.push({c:fig.getAttribute('data-c'+k),v:fig.getAttribute('data-v'+k).split(',')});}
function part(cls){var d=document.createElement('div');d.className=cls;fig.appendChild(d);
return d;}
var vline=part('ts-hair ts-v'),hline=part('ts-hair ts-h');
var ytag=part('ts-tag'),xtag=part('ts-tag');
var dots=lines.map(function(l){var d=part('ts-dot');d.style.background=l.c;return d;});
var parts=[vline,hline,ytag,xtag].concat(dots);
function show(on){parts.forEach(function(d){d.style.display=on?'block':'none';});}
function label(v){return pct?(v*100).toFixed(1)+'%':v.toFixed(2);}
fig.addEventListener('mouseleave',function(){show(false);});
fig.addEventListener('mousemove',function(e){var r=pane.getBoundingClientRect();
var x=e.clientX-r.left,y=e.clientY-r.top;
if(x<0||x>r.width||y<0||y>r.height||n<2){show(false);return;}
var i=Math.round(x/r.width*(n-1)),px=i/(n-1)*r.width;show(true);
vline.style.left=px+'px';vline.style.height=r.height+'px';
hline.style.top=y+'px';hline.style.width=r.width+'px';
var s=top-y/r.height*(top-bottom);ytag.textContent=label(log?Math.exp(s)-1:s);
ytag.style.left=(r.width+4)+'px';ytag.style.top=(y-ytag.offsetHeight/2)+'px';
xtag.textContent=shortDay(dates[first+i]);xtag.style.top=(r.height+6)+'px';
var half=xtag.offsetWidth/2;
xtag.style.left=Math.min(Math.max(px-half,0),r.width-2*half)+'px';
dots.forEach(function(d,k){var raw=lines[k].v[i];
if(raw===undefined||raw===''){d.style.display='none';return;}
var v=Number(raw),t=log?Math.log(1+v):v;d.style.left=px+'px';
d.style.top=((top-t)/(top-bottom)*r.height)+'px';});});}
Array.prototype.forEach.call(document.querySelectorAll('figure.ts'),setupChart);
function thin(){Array.prototype.forEach.call(document.querySelectorAll('.ts-x'),function(axis){
var width=axis.clientWidth,last=-1e9;
Array.prototype.forEach.call(axis.children,function(s){var at=parseFloat(s.style.left)/100*width;
var keep=at-last>=@MIN_GAP@;s.classList.toggle('off',!keep);if(keep)last=at;});});}
thin();window.addEventListener('resize',thin);
"""


def script(min_time_gap_px: int) -> str:
    """The page's script, run at the end of the body; `min_time_gap_px` is the smallest
    gap, in pixels, between two time labels."""
    body = _THEME + _NAV + _TABLES + _WEEKLY + _CHARTS
    body = body.replace(_KEY_SLOT, THEME_KEY).replace(_GAP_SLOT, str(min_time_gap_px))
    return "(function(){" + body.replace("\n", "") + "})();"


__all__ = ["HEAD", "THEME_KEY", "script"]
