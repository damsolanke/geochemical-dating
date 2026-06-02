"""Render the pipeline architecture PNG for the README.

Produces docs/images/architecture.png: a dual-branch view of the 2nd-place
solution (50% geochemistry ensemble + 50% Id-KNN structural prior).

Dependencies:
    pip install diagrams            # Python 3.9+
    Graphviz  (macOS: brew install graphviz ; Linux: https://graphviz.org/download/)
    librsvg   (macOS: brew install librsvg ; provides rsvg-convert for SVG logos)

The component logos are downloaded and cached at runtime into
docs/images/.icons/ (gitignored). If a download or SVG conversion fails, the
affected node falls back to a plain labelled box, so the script still runs.

Run:  python scripts/generate_diagrams.py
"""

import os
import shutil
import subprocess
import urllib.request

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.programming.flowchart import InputOutput, PredefinedProcess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "images")
ICON_DIR = os.path.join(OUT_DIR, ".icons")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ICON_DIR, exist_ok=True)

_UA = {"User-Agent": "Mozilla/5.0 (compatible; diagram-builder)"}
ICON_SOURCES = {
    "xgboost": "https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png",
    "lightgbm": "https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/logo/LightGBM_logo_black_text.svg",
    "sklearn": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.svg",
    "numpy": "https://raw.githubusercontent.com/numpy/numpy/main/branding/logo/primary/numpylogo.svg",
    "pandas": "https://raw.githubusercontent.com/pandas-dev/pandas/main/web/pandas/static/img/pandas.svg",
}


def get_icon(name):
    """Download (and convert SVG->PNG) a component logo; return path or None."""
    png = os.path.join(ICON_DIR, name + ".png")
    if os.path.exists(png) and os.path.getsize(png) > 1000:
        return png
    url = ICON_SOURCES[name]
    raw = os.path.join(ICON_DIR, name + (".png" if url.endswith(".png") else ".svg"))
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        with open(raw, "wb") as f:
            f.write(data)
    except Exception:
        return None
    if raw.endswith(".svg"):
        rsvg = shutil.which("rsvg-convert")
        if not rsvg:
            return None
        subprocess.run([rsvg, "-h", "160", raw, "-o", png], check=False)
    if os.path.exists(png) and os.path.getsize(png) > 1000:
        return png
    return None


def comp(label, icon):
    """A Custom-logo node, or a labelled box if the logo is unavailable."""
    path = get_icon(icon)
    return Custom(label, path) if path else PredefinedProcess(label)


BLUE = "#1f6feb"   # geochemistry ensemble branch
GREEN = "#2da44e"  # Id-KNN structural-prior branch

graph_attr = {
    "fontname": "Helvetica Neue", "fontsize": "20", "bgcolor": "white",
    "dpi": "150", "pad": "0.6", "nodesep": "0.55", "ranksep": "1.1",
    "splines": "spline",
}
node_attr = {"fontname": "Helvetica Neue", "fontsize": "12", "labelloc": "b"}
edge_attr = {"fontname": "Helvetica Neue", "fontsize": "11"}

with Diagram(
    "Geochemical Dating  -  Dual-Branch Ensemble  (private Macro-F1 0.96988, 2nd of 31)",
    filename=os.path.join(OUT_DIR, "architecture"),
    show=False, direction="LR",
    graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr,
):
    data = InputOutput("Rock samples\n2,271 train / 757 test\n33 geochemical features")

    blue_cluster = {"bgcolor": "#eaf2fb", "fontname": "Helvetica Neue", "fontsize": "15",
                    "pencolor": "#9ec5f0", "style": "rounded"}
    inner_cluster = {"bgcolor": "#f6f8fa", "fontname": "Helvetica Neue", "fontsize": "13",
                     "pencolor": "#d0d7de", "style": "rounded"}
    green_cluster = {"bgcolor": "#eaf7ee", "fontname": "Helvetica Neue", "fontsize": "15",
                     "pencolor": "#9bdcae", "style": "rounded"}

    with Cluster("Geochemistry ensemble   50%   (generalizable)", graph_attr=blue_cluster):
        fe = comp("Feature engineering\nMg#, ASI, REE slope, anomalies\n+ KMeans cluster one-hots", "pandas")
        with Cluster("15-fold CV x 5 seeds, cluster-weighted", graph_attr=inner_cluster):
            xgb = comp("XGBoost   0.52", "xgboost")
            lgbm = comp("LightGBM   0.09", "lightgbm")
            svm = comp("SVM (RBF)   0.39", "sklearn")
        model_blend = PredefinedProcess("Weighted blend\n0.52 / 0.09 / 0.39")
        fe >> Edge(color=BLUE) >> [xgb, lgbm, svm] >> Edge(color=BLUE) >> model_blend

    with Cluster("Structural prior   50%   (dataset-specific)", graph_attr=green_cluster):
        knn = comp("Id-KNN\nId ~ source row order (rho 0.999)\nk=3, sigma=2 exp-weighted, 10-fold OOF", "numpy")

    fusion = PredefinedProcess("Arithmetic blend\n0.50 ensemble + 0.50 Id-KNN\n+ exact-duplicate override")
    out = InputOutput("submission.csv\nMacro-F1 0.96988")

    data >> Edge(label="engineered features", color=BLUE) >> fe
    data >> Edge(label="Id only", color=GREEN, style="bold") >> knn
    model_blend >> Edge(label="0.50", color=BLUE) >> fusion
    knn >> Edge(label="0.50", color=GREEN) >> fusion
    fusion >> Edge(color="#57606a") >> out
