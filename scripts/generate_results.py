"""Render the leaderboard shake-up slopegraph (docs/images/shakeup.png).

A rank slopegraph (public -> private) that tells the result story visually:
the robust pick held 2nd while the public leader fell to 3rd. Rendered to a
high-resolution PNG via rsvg-convert. No emojis.

Run:  python scripts/generate_results.py
"""

import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

FONT = "Geist, 'Helvetica Neue', 'Inter', Arial, sans-serif"
INK = "#0F172A"
SUB = "#64748B"
BG = "#FBFCFE"
INDIGO = "#4F46E5"
EMERALD = "#10B981"
AMBER = "#E0881E"
W, H = 1200, 560
P = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=14, color=INK, weight=400, anchor="start", spacing=0):
    st = f' letter-spacing="{spacing}"' if spacing else ""
    P.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" '
             f'font-weight="{weight}" text-anchor="{anchor}"{st}>{esc(s)}</text>')


defs = '''<defs>
  <filter id="d" x="-50%" y="-50%" width="200%" height="200%">
    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#1E293B" flood-opacity="0.18"/>
  </filter>
</defs>'''

P.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

# title
text(60, 66, "Leaderboard shake-up", 30, INK, 800, spacing=-0.4)
text(62, 96, "public -> private   ·   the robust pick held 2nd while the public leader fell to 3rd", 16, SUB, 500)

xL, xR = 392, 812
yr = {1: 214, 2: 326, 3: 438}

# highlight band for "this solution"
P.append(f'<rect x="120" y="{yr[2]-30}" width="960" height="60" rx="12" fill="#EEF2FF" opacity="0.7"/>')

# column headers
text(xL, 168, "PUBLIC LB", 14, SUB, 800, anchor="middle", spacing=2)
text(xR, 168, "PRIVATE LB", 14, SUB, 800, anchor="middle", spacing=2)

# competitor: (name, public_rank, public_score, private_rank, private_score, color, bold)
rows = [
    ("Nikita Shevyrev", 1, "0.96649", 3, "0.95728", AMBER, False),
    ("This solution", 2, "0.96292", 2, "0.96988", INDIGO, True),
    ("Raphael Stottele", 3, "0.93876", 1, "0.97092", EMERALD, False),
]

# slope lines first (under dots)
for name, pr, ps, qr, qs, color, bold in rows:
    wdt = 5 if bold else 3
    P.append(f'<path d="M {xL} {yr[pr]} L {xR} {yr[qr]}" stroke="{color}" stroke-width="{wdt}" '
             f'fill="none" stroke-linecap="round" opacity="{1 if bold else 0.85}"/>')

# dots + labels
for name, pr, ps, qr, qs, color, bold in rows:
    for x, ry in ((xL, yr[pr]), (xR, yr[qr])):
        P.append(f'<circle cx="{x}" cy="{ry}" r="{8 if bold else 6.5}" fill="{color}" '
                 f'stroke="#FFFFFF" stroke-width="2.5" filter="url(#d)"/>')
    nw = 700 if bold else 600
    nc = INK if bold else "#334155"
    # left (public)
    text(xL - 30, yr[pr] - 4, name, 15, nc, nw, anchor="end")
    text(xL - 30, yr[pr] + 16, ps, 13, SUB, 600, anchor="end")
    # right (private)
    text(xR + 30, yr[qr] - 4, name, 15, nc, nw, anchor="start")
    rank_txt = {1: "1st", 2: "2nd", 3: "3rd"}[qr]
    text(xR + 30, yr[qr] + 16, f"{qs}  ·  {rank_txt}", 13,
         (color if bold else SUB), 700 if bold else 600, anchor="start")

# margin annotation near the top of the private column
text(xR + 30, yr[2] + 36, "0.00104 behind 1st", 12, INDIGO, 600, anchor="start")

# footer
text(W/2, H - 28, "Kaggle Geochemical Dating   ·   final private leaderboard, 31 teams", 13, "#94A3B8", 500, anchor="middle")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">{defs}' + "".join(P) + "</svg>")

src = os.path.join(OUT_DIR, "shakeup.svg")
out = os.path.join(OUT_DIR, "shakeup.png")
open(src, "w").write(svg)
rsvg = shutil.which("rsvg-convert")
if rsvg:
    subprocess.run([rsvg, "-w", "1900", src, "-o", out], check=True)
    print("wrote", out)
else:
    print("rsvg-convert not found; wrote SVG only:", src)
