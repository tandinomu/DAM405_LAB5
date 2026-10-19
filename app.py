"""
Task 2 - Prediction service with logging around /predict.

Endpoints
  POST /predict  {"features": [5.1, 3.5, 1.4, 0.2]}  -> {"prediction": 0}
  GET  /stats    monitoring summary (Task 4)
  GET  /drift    drift check on features received so far (Task 5)
  GET  /health   simple liveness check

Run:  python app.py      (then use test_client.py in a second terminal)
"""
import json
import os
import sys
import time
from collections import deque

import joblib
import numpy as np
from flask import Flask, jsonify, request

from logger_config import logger
from monitoring import drift_check, record, stats, summary

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "model.joblib")
STATS_PATH = os.path.join(BASE, "model", "train_stats.json")
SUMMARY_EVERY = 10  # log a monitoring summary every N requests

app = Flask(__name__)

# ---- load model at start-up; CRITICAL if we can't, because nothing works ----
try:
    model = joblib.load(MODEL_PATH)
    with open(STATS_PATH) as f:
        train_stats = json.load(f)
    logger.info(f"model loaded from {MODEL_PATH}")
except FileNotFoundError:
    logger.critical("model file not found - run `python train_model.py` first")
    sys.exit(1)

N_FEATURES = train_stats["n_features"]
# keep recent live inputs so we can compare them with the training data
live_inputs = deque(maxlen=500)


@app.post("/predict")
def predict():
    start = time.perf_counter()
    try:
        data = request.get_json(force=True, silent=True)
        logger.debug(f"request features: {data}")  # DEBUG: raw input, dev only

        if not data or "features" not in data:
            raise ValueError("body must be JSON with a 'features' list")
        features = data["features"]
        if len(features) != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {len(features)}")

        # unusual but handled -> WARNING
        if any(v < 0 for v in features):
            logger.warning(f"negative feature value received: {features}")

        x = np.array(features, dtype=float).reshape(1, -1)
        pred = int(model.predict(x)[0])
        live_inputs.append(features)

        ms = (time.perf_counter() - start) * 1000
        logger.info(f"prediction={pred} latency_ms={ms:.1f}")
        record(ms, ok=True)
        status = (jsonify({"prediction": pred}), 200)

    except (ValueError, TypeError) as e:
        # client sent bad input -> 400
        ms = (time.perf_counter() - start) * 1000
        logger.error(f"prediction failed (bad input): {e} latency_ms={ms:.1f}")
        record(ms, ok=False)
        status = (jsonify({"error": str(e)}), 400)

    except Exception as e:
        # anything else is our fault -> 500, with stack trace in the log
        ms = (time.perf_counter() - start) * 1000
        logger.exception(f"prediction failed (server error): {e}")
        record(ms, ok=False)
        status = (jsonify({"error": "internal error"}), 500)

    if app_stats_due():
        summary()
    return status


def app_stats_due() -> bool:
    """True every SUMMARY_EVERY requests -> periodic summary log line."""
    return stats["requests"] % SUMMARY_EVERY == 0


@app.get("/stats")
def get_stats():
    return jsonify(summary())


@app.get("/drift")
def get_drift():
    """Run the mean-based drift check for every feature."""
    if not live_inputs:
        return jsonify({"message": "no live data yet"})
    live = np.array(live_inputs)
    result = {}
    for i, name in enumerate(train_stats["feature_names"]):
        rel = drift_check(train_stats["train_means"][i], live[:, i], feature=name)
        result[name] = {"relative_change": round(rel, 3), "drift": rel > 0.25}
    return jsonify(result)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    logger.info("starting ml-service on http://127.0.0.1:5000")
    # use_reloader=False so the model isn't loaded twice
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
