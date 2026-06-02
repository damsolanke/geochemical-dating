"""Render the README architecture diagram (docs/images/architecture.png).

A hand-designed SVG (rounded cards, soft shadows, restrained palette, Geist
typography, real component logos) rendered to a high-resolution PNG. This is
deliberately not a generic Graphviz/diagrams-library output.

Dependencies:
    rsvg-convert   (macOS: brew install librsvg)   -- renders SVG -> PNG
    librsvg picks up system fonts (Geist / Helvetica Neue) via fontconfig.

Component logos are downloaded + SVG-converted at build time into a gitignored
cache (docs/images/.icons/) and embedded as base64 so the SVG is self-contained.

Run:  python scripts/generate_diagrams.py
"""

import base64
import os
import shutil
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "images")
ICON_DIR = os.path.join(OUT_DIR, ".icons")
os.makedirs(ICON_DIR, exist_ok=True)

_UA = {"User-Agent": "Mozilla/5.0 (compatible; diagram-builder)"}
ICON_SOURCES = {
    "xgboost": "https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png",
    "lightgbm": "https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/logo/LightGBM_logo_black_text.svg",
    "numpy": "https://raw.githubusercontent.com/numpy/numpy/main/branding/logo/primary/numpylogo.svg",
    "pandas": "https://raw.githubusercontent.com/pandas-dev/pandas/main/web/pandas/static/img/pandas.svg",
}
LOGO_WH = {"xgboost": (408, 141), "lightgbm": (530, 120), "numpy": (267, 120), "pandas": (297, 120)}


def get_icon(name):
    png = os.path.join(ICON_DIR, name + ".png")
    if os.path.exists(png) and os.path.getsize(png) > 1000:
        return png
    url = ICON_SOURCES[name]
    raw = os.path.join(ICON_DIR, name + (".png" if url.endswith(".png") else ".svg"))
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            open(raw, "wb").write(r.read())
    except Exception:
        return None
    if raw.endswith(".svg"):
        rsvg = shutil.which("rsvg-convert")
        if not rsvg:
            return None
        subprocess.run([rsvg, "-h", "160", raw, "-o", png], check=False)
    return png if (os.path.exists(png) and os.path.getsize(png) > 1000) else None


def logo_datauri(name):
    p = get_icon(name)
    return ("data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()) if p else None


# ---------------------------------------------------------------- palette
BG = "#FBFCFE"
INK = "#0F172A"
SUB = "#64748B"
CARD = "#FFFFFF"
CARD_STROKE = "#E2E8F0"
ENS_FILL = "#F5F7FF"
ENS_STROKE = "#E5E9F6"
KNN_FILL = "#F1FBF5"
KNN_STROKE = "#DCEFE3"
INDIGO = "#4F46E5"
INDIGO_EDGE = "#6366F1"
EMERALD = "#10B981"
PILL_INK = "#3730A3"
FONT = "Geist, 'Helvetica Neue', 'Inter', Arial, sans-serif"

W, H = 1700, 860
parts = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=14, color=INK, weight=400, anchor="middle", spacing=0):
    st = f' letter-spacing="{spacing}"' if spacing else ""
    parts.append(
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{st}>{esc(s)}</text>'
    )


def card(x, y, w, h, fill=CARD, stroke=CARD_STROKE, rx=14, sw=1.5, shadow=True, grad=None):
    flt = ' filter="url(#soft)"' if shadow else ""
    f = f'url(#{grad})' if grad else fill
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{f}" stroke="{stroke}" stroke-width="{sw}"{flt}/>'
    )


def logo(name, cx, cy, box_w, box_h):
    uri = logo_datauri(name)
    nw, nh = LOGO_WH[name]
    scale = min(box_w / nw, box_h / nh)
    w, h = nw * scale, nh * scale
    if uri:
        parts.append(
            f'<image x="{cx - w/2:.1f}" y="{cy - h/2:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'href="{uri}" preserveAspectRatio="xMidYMid meet"/>'
        )
    else:
        text(cx, cy + 5, name, 15, INK, 700)


def edge(d, color, width=2.4, dash=None, marker="arrow"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{marker})"' if marker else ""
    parts.append(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{da}{mk}/>'
    )


def elabel(x, y, s, color=SUB):
    w = 16 + len(s) * 7.2
    parts.append(f'<rect x="{x - w/2:.1f}" y="{y - 12}" width="{w:.1f}" height="22" rx="11" fill="{BG}" opacity="0.92"/>')
    text(x, y + 4, s, 12.5, color, 600)


# ---------------------------------------------------------------- defs
defs = '''<defs>
  <filter id="soft" x="-25%" y="-25%" width="150%" height="150%">
    <feDropShadow dx="0" dy="3" stdDeviation="7" flood-color="#1E293B" flood-opacity="0.13"/>
  </filter>
  <linearGradient id="fuse" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#6366F1"/><stop offset="1" stop-color="#4338CA"/>
  </linearGradient>
  <linearGradient id="ink" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#1E293B"/><stop offset="1" stop-color="#0F172A"/>
  </linearGradient>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0 1 L9 5 L0 9 z" fill="#6366F1"/></marker>
  <marker id="arrowG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0 1 L9 5 L0 9 z" fill="#10B981"/></marker>
  <marker id="arrowD" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0 1 L9 5 L0 9 z" fill="#94A3B8"/></marker>
</defs>'''

# ---------------------------------------------------------------- compose
parts.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

# header
text(60, 64, "Geochemical Dating", 34, INK, 800, anchor="start", spacing=-0.5)
text(62, 92, "Dual-branch ensemble   ·   3-class geological age   ·   Macro-F1", 16, SUB, 500, anchor="start")
# result chip
card(W - 432, 38, 372, 62, fill="url(#ink)", stroke="#0F172A", rx=14)
text(W - 412, 66, "2nd of 31", 20, "#FFFFFF", 800, anchor="start")
text(W - 412, 87, "Kaggle leaderboard", 12.5, "#94A3B8", 500, anchor="start")
text(W - 72, 64, "0.96988", 23, "#A5B4FC", 800, anchor="end")
text(W - 72, 87, "private Macro-F1", 12, "#94A3B8", 500, anchor="end")

# input
ix, iy, iw, ih = 60, 382, 168, 116
card(ix, iy, iw, ih, fill="url(#ink)", stroke="#0F172A")
text(ix + iw/2, iy + 34, "Rock samples", 15, "#FFFFFF", 700)
text(ix + iw/2, iy + 60, "2,271 train", 12.5, "#CBD5E1", 500)
text(ix + iw/2, iy + 78, "757 test", 12.5, "#CBD5E1", 500)
text(ix + iw/2, iy + 99, "33 geo features", 12, "#94A3B8", 500)

# ensemble container
ex, ey, ew, eh = 286, 128, 812, 384
card(ex, ey, ew, eh, fill=ENS_FILL, stroke=ENS_STROKE, rx=20, shadow=False)
text(ex + 26, ey + 34, "GEOCHEMISTRY ENSEMBLE", 13, "#475569", 800, anchor="start", spacing=1.2)
text(ex + 268, ey + 34, "50%  ·  generalizable", 13, INDIGO, 700, anchor="start")

# feature engineering
fx, fy, fw, fh = 312, 268, 196, 100
card(fx, fy, fw, fh)
logo("pandas", fx + fw/2, fy + 34, 122, 30)
text(fx + fw/2, fy + 68, "Feature engineering", 13, INK, 700)
text(fx + fw/2, fy + 86, "Mg#, REE slope, anomalies", 11, SUB, 500)

# model cards
mx, mw, mh = 588, 224, 78
mcy = [223, 318, 413]
models = [("xgboost", "0.52"), ("lightgbm", "0.09"), ("svm", "0.39")]
mnodes = []
for (name, wgt), cy in zip(models, mcy):
    top = cy - mh/2
    card(mx, top, mw, mh)
    if name == "svm":
        text(mx + 68, cy - 3, "SVM", 21, INK, 800)
        text(mx + 68, cy + 19, "RBF · scikit-learn", 10.5, SUB, 600)
    else:
        logo(name, mx + 68, cy, 108, 40)
    parts.append(f'<line x1="{mx+132}" y1="{top+16}" x2="{mx+132}" y2="{top+mh-16}" stroke="#EDF0FB" stroke-width="1.5"/>')
    text(mx + 180, cy - 8, "weight", 10.5, SUB, 600)
    text(mx + 180, cy + 18, wgt, 22, INDIGO, 800)
    mnodes.append((mx, mw, cy))

text(ex + ew/2, ey + eh - 22, "15-fold CV  x  5 seeds      ·      cluster-frequency sample weighting", 12, "#94A3B8", 600)

# model blend
bx, by, bw, bh = 902, 268, 168, 100
card(bx, by, bw, bh)
text(bx + bw/2, by + 42, "Weighted blend", 14, INK, 700)
text(bx + bw/2, by + 70, "0.52 / 0.09 / 0.39", 12, INDIGO, 700)
bcy = by + bh/2

# structural-prior container
kx, ky, kw, kh = 286, 548, 560, 196
card(kx, ky, kw, kh, fill=KNN_FILL, stroke=KNN_STROKE, rx=20, shadow=False)
text(kx + 26, ky + 34, "STRUCTURAL PRIOR", 13, "#475569", 800, anchor="start", spacing=1.2)
text(kx + 230, ky + 34, "50%  ·  dataset-specific", 13, "#047857", 700, anchor="start")

# id-knn card
nx, ny, nw_, nh_ = 320, 596, 496, 128
card(nx, ny, nw_, nh_)
logo("numpy", nx + 84, ny + nh_/2, 122, 40)
parts.append(f'<line x1="{nx+168}" y1="{ny+24}" x2="{nx+168}" y2="{ny+nh_-24}" stroke="#ECFDF5" stroke-width="1.5"/>')
text(nx + 192, ny + 42, "Id-KNN", 17, INK, 800, anchor="start")
text(nx + 192, ny + 66, "Id  ~  source row order   (rho = 0.999)", 12.5, SUB, 500, anchor="start")
text(nx + 192, ny + 88, "k = 3  ·  sigma = 2, exp-weighted", 12.5, SUB, 500, anchor="start")
text(nx + 192, ny + 110, "10-fold out-of-fold", 12.5, SUB, 500, anchor="start")
ncy = ny + nh_/2

# fusion
ux, uy, uw, uh = 1208, 415, 188, 152
card(ux, uy, uw, uh, grad="fuse", stroke="#3730A3", rx=16)
text(ux + uw/2, uy + 42, "Arithmetic", 16, "#FFFFFF", 800)
text(ux + uw/2, uy + 64, "blend", 16, "#FFFFFF", 800)
text(ux + uw/2, uy + 92, "0.50  ensemble", 12.5, "#E0E7FF", 600)
text(ux + uw/2, uy + 110, "+ 0.50  Id-KNN", 12.5, "#E0E7FF", 600)
text(ux + uw/2, uy + 134, "+ duplicate override", 11.5, "#C7D2FE", 500)
ucy = uy + uh/2

# output
ox, oy, ow, oh = 1456, 435, 184, 112
card(ox, oy, ow, oh, fill="url(#ink)", stroke="#0F172A")
text(ox + ow/2, oy + 38, "submission.csv", 15, "#FFFFFF", 700)
text(ox + ow/2, oy + 68, "0.96988", 24, "#A5B4FC", 800)
text(ox + ow/2, oy + 90, "private Macro-F1", 11.5, "#94A3B8", 500)

# ---------------------------------------------------------------- edges
edge(f"M {ix+iw} {iy+42} C 270 {iy+42}, 282 {fy+fh/2}, {fx} {fy+fh/2}", INDIGO_EDGE)
elabel(266, 356, "features", INDIGO)
edge(f"M {ix+iw} {iy+84} C 262 {iy+150}, 286 {ncy}, {nx} {ncy}", EMERALD, marker="arrowG")
elabel(270, 540, "Id only", "#047857")
for (mx0, mw0, cy) in mnodes:
    edge(f"M {fx+fw} {fy+fh/2} C 558 {fy+fh/2}, 558 {cy}, {mx0} {cy}", INDIGO_EDGE, 2.0)
for (mx0, mw0, cy) in mnodes:
    edge(f"M {mx0+mw0} {cy} C 882 {cy}, 882 {bcy}, {bx} {bcy}", INDIGO_EDGE, 2.0)
edge(f"M {bx+bw} {bcy} C 1150 {bcy}, 1152 {uy+uh*0.32}, {ux} {uy+uh*0.32}", INDIGO_EDGE, 2.8)
elabel(1150, 372, "0.50", INDIGO)
edge(f"M {nx+nw_} {ncy} C 1060 {ncy}, 1132 {uy+uh*0.74}, {ux} {uy+uh*0.74}", EMERALD, 2.8, marker="arrowG")
elabel(1066, 596, "0.50", "#047857")
edge(f"M {ux+uw} {ucy} L {ox} {oy+oh/2}", "#475569", 2.8, marker="arrowD")

text(W/2, H - 24, "github.com/damsolanke/geochemical-dating", 12.5, "#94A3B8", 500)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">{defs}' + "".join(parts) + "</svg>")

src = os.path.join(OUT_DIR, "architecture.svg")
out = os.path.join(OUT_DIR, "architecture.png")
open(src, "w").write(svg)
rsvg = shutil.which("rsvg-convert")
if rsvg:
    subprocess.run([rsvg, "-w", "2550", src, "-o", out], check=True)
    print("wrote", out)
else:
    print("rsvg-convert not found; wrote SVG only:", src)
