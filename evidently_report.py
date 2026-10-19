"""
OPTIONAL - richer drift report with Evidently.

  pip install evidently
  python evidently_report.py
  -> opens/creates drift_report.html (open it in a browser)

Compares the training data (reference) with a shifted copy (current).
Evidently's API changed between versions; this handles both styles.
"""
import pandas as pd
from sklearn.datasets import load_iris

from logger_config import logger

iris = load_iris(as_frame=True)
reference = iris.data.sample(frac=0.5, random_state=1)
current = iris.data.sample(frac=0.5, random_state=2).copy()
current["petal length (cm)"] *= 2.0   # simulate drift

try:  # Evidently >= 0.7
    from evidently import Report
    from evidently.presets import DataDriftPreset
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    result.save_html("drift_report.html")
except ImportError:  # older Evidently (0.4.x)
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html("drift_report.html")

logger.info("Evidently drift report saved to drift_report.html")
