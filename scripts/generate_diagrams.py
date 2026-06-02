"""Render the pipeline architecture PNG for the README.

Requires Graphviz on the system:
    macOS:  brew install graphviz
    Linux:  https://graphviz.org/download/
Then:       pip install diagrams   (Python 3.9+)
Run:        python scripts/generate_diagrams.py
Output:     docs/images/architecture.png
"""

import os

from diagrams import Cluster, Diagram, Edge
from diagrams.programming.flowchart import InputOutput, PredefinedProcess
from diagrams.programming.language import Python

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

graph_attr = {
    "fontname": "Helvetica Neue",
    "fontsize": "18",
    "bgcolor": "white",
    "dpi": "150",
    "pad": "0.5",
    "splines": "spline",
}
node_attr = {"fontname": "Helvetica Neue", "fontsize": "12"}
edge_attr = {"fontname": "Helvetica Neue", "fontsize": "11"}

with Diagram(
    "Geochemical Dating - Ensemble Pipeline",
    filename=os.path.join(OUT_DIR, "architecture"),
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    raw = InputOutput("Rock-sample\ntrain / test CSV")

    with Cluster("Feature engineering"):
        fe = PredefinedProcess("Geochem features\n+ KMeans cluster one-hots")

    with Cluster("Id-KNN branch"):
        knn = PredefinedProcess("Id-KNN\nk=3, sigma=2")

    with Cluster("Model ensemble"):
        xgb = Python("XGBoost\n0.52")
        lgbm = Python("LightGBM\n0.09")
        svm = Python("SVM (RBF)\n0.39")
        model_blend = PredefinedProcess("Weighted\nmodel blend")
        [xgb, lgbm, svm] >> model_blend

    final_blend = PredefinedProcess("50 / 50 blend\n+ duplicate override")
    submission = InputOutput("submission.csv\nprivate Macro-F1 0.96988")

    # Id-KNN reads the sample Id straight from the raw data; the model
    # ensemble reads engineered features. The two branches meet 50/50.
    raw >> Edge(color="darkgreen") >> knn
    raw >> fe >> Edge(color="darkblue") >> [xgb, lgbm, svm]
    knn >> Edge(label="0.50", color="darkgreen") >> final_blend
    model_blend >> Edge(label="0.50", color="darkblue") >> final_blend
    final_blend >> submission
