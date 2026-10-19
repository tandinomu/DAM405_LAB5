"""
Trains the model used by the service (stand-in for the Practical 3 model).

Uses the Iris dataset + RandomForest. Saves:
  * model/model.joblib       - the trained model
  * model/train_stats.json   - per-feature training means (for drift check)

If you already have your Practical 3 model, you can skip this and copy
your own model.joblib into the model/ folder instead (it must accept the
same number of features you send to /predict).
"""
import json
import os

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from logger_config import logger

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def main():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"model trained accuracy={acc:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))

    # Store training statistics so the service can check for drift later
    stats = {
        "feature_names": list(data.feature_names),
        "train_means": np.mean(X_train, axis=0).round(4).tolist(),
        "n_features": int(X_train.shape[1]),
    }
    with open(os.path.join(MODEL_DIR, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"saved model and training stats to {MODEL_DIR}")


if __name__ == "__main__":
    main()
