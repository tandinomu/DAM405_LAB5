"""
Sends traffic to the running service so every expected result shows up
in logs/service.log:
  1. normal predictions        -> INFO lines with latency
  2. bad requests              -> ERROR lines
  3. unusual input             -> WARNING line
  4. monitoring summary        -> /stats
  5. drift                     -> first no drift, then shifted data -> WARNING

Start the service first (python app.py), then run:  python test_client.py
"""
import random

import requests
from sklearn.datasets import load_iris

URL = "http://127.0.0.1:5000"


def post(features):
    r = requests.post(f"{URL}/predict", json=features)
    print(r.status_code, r.json())


def main():
    X = load_iris().data
    random.seed(0)

    print("\n--- 1. normal predictions (similar to training data) ---")
    for row in random.sample(list(X), 15):
        post({"features": [float(v) for v in row]})

    print("\n--- 2. drift check on normal data (expect no drift) ---")
    print(requests.get(f"{URL}/drift").json())

    print("\n--- 3. bad requests (expect ERROR in log) ---")
    post({"features": [1, 2]})               # wrong number of features
    post({"wrong_key": [1, 2, 3, 4]})        # missing 'features'
    post({"features": ["a", "b", "c", "d"]})  # not numbers

    print("\n--- 4. unusual but valid input (expect WARNING) ---")
    post({"features": [-1.0, 3.0, 1.4, 0.2]})

    print("\n--- 5. shifted data: petal length x2 (expect drift WARNING) ---")
    for row in random.sample(list(X), 40):
        row = [float(v) for v in row]
        row[2] *= 2.0   # petal length doubled
        row[3] *= 2.0   # petal width doubled
        post({"features": row})
    print(requests.get(f"{URL}/drift").json())

    print("\n--- 6. monitoring summary ---")
    print(requests.get(f"{URL}/stats").json())
    print("\nDone - open logs/service.log to see everything.")


if __name__ == "__main__":
    main()
