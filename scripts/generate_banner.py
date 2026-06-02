"""Render the README hero banner (docs/images/banner.png).

A dark, designed header (deep-slate gradient, soft indigo glow, accent rule,
result card) rendered to a high-resolution PNG via rsvg-convert. No emojis.

Run:  python scripts/generate_banner.py
"""

import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

FONT = "Geist, 'Helvetica Neue', 'Inter', Arial, sans-serif"
WHITE = "#F8FAFC"
SLATE = "#94A3B8"
SLATE2 = "#64748B"
ACCENT = "#818CF8"
ACCENT2 = "#A5B4FC"
W, H = 1600, 400
P = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=16, color=WHITE, weight=400, anchor="start", spacing=0):
    st = f' letter-spacing="{spacing}"' if spacing else ""
    P.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" '
             f'font-weight="{weight}" text-anchor="{anchor}"{st}>{esc(s)}</text>')


def pill(x, y, label):
    w = 26 + len(label) * 8.0
    P.append(f'<rect x="{x}" y="{y}" width="{w:.0f}" height="34" rx="17" fill="#162033" '
             f'stroke="#2A3A55" stroke-width="1"/>')
    text(x + w/2, y + 22, label, 13.5, "#CBD5E1", 600, anchor="middle")
    return x + w + 14


defs = '''<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#0B1120"/><stop offset="0.55" stop-color="#0E1628"/>
    <stop offset="1" stop-color="#141E33"/>
  </linearGradient>
  <radialGradient id="glow" cx="0.82" cy="0.18" r="0.55">
    <stop offset="0" stop-color="#4F46E5" stop-opacity="0.40"/>
    <stop offset="0.5" stop-color="#4F46E5" stop-opacity="0.10"/>
    <stop offset="1" stop-color="#4F46E5" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="glow2" cx="0.1" cy="0.95" r="0.5">
    <stop offset="0" stop-color="#0EA5E9" stop-opacity="0.18"/>
    <stop offset="1" stop-color="#0EA5E9" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="acc" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#818CF8"/><stop offset="1" stop-color="#4F46E5"/>
  </linearGradient>
</defs>'''

# background + ambient light
P.append(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
P.append(f'<rect width="{W}" height="{H}" fill="url(#glow)"/>')
P.append(f'<rect width="{W}" height="{H}" fill="url(#glow2)"/>')

# faint strata motif (rock layers) behind the result card, right side
strata = [(1090, 96, 430, 0.06), (1150, 132, 380, 0.05), (1075, 168, 300, 0.045),
          (1180, 300, 360, 0.05), (1110, 336, 300, 0.04)]
for x, y, w, op in strata:
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="10" rx="5" fill="#A5B4FC" opacity="{op}"/>')

# left accent rule
P.append('<rect x="80" y="120" width="5" height="176" rx="2.5" fill="url(#acc)"/>')

# eyebrow / title / subtitle
text(108, 138, "KAGGLE  ·  GEOCHEMICAL DATING", 15, ACCENT, 700, spacing=3)
text(104, 212, "Geochemical Dating", 66, WHITE, 800, spacing=-1.2)
text(108, 256, "3-class geological age, predicted from rock geochemistry", 22, SLATE, 500)

# stat chips
cx = 108
for label in ["Macro-F1 metric", "3 classes", "2,271 + 757 samples", "33 features"]:
    cx = pill(cx, 286, label)

# result card (right)
rx, ry, rw, rh = 1130, 92, 396, 212
P.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="18" fill="#0F1A2E" '
         f'fill-opacity="0.72" stroke="#33406A" stroke-width="1.5"/>')
text(rx + 28, ry + 42, "RUNNER-UP", 14, ACCENT, 700, spacing=3)
text(rx + 28, ry + 106, "2nd", 58, WHITE, 800)
text(rx + 174, ry + 106, "of 31 teams", 20, SLATE, 500)
P.append(f'<line x1="{rx+28}" y1="{ry+132}" x2="{rx+rw-28}" y2="{ry+132}" stroke="#2A3A55" stroke-width="1.5"/>')
text(rx + 28, ry + 186, "0.96988", 38, ACCENT2, 800)
text(rx + 206, ry + 182, "private Macro-F1", 13.5, SLATE, 500)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">{defs}' + "".join(P) + "</svg>")

src = os.path.join(OUT_DIR, "banner.svg")
out = os.path.join(OUT_DIR, "banner.png")
open(src, "w").write(svg)
rsvg = shutil.which("rsvg-convert")
if rsvg:
    subprocess.run([rsvg, "-w", "2400", src, "-o", out], check=True)
    print("wrote", out)
else:
    print("rsvg-convert not found; wrote SVG only:", src)
