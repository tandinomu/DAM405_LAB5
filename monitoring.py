"""
Task 4 - Monitoring signals, and Task 5 - basic drift check.
"""
from collections import deque

import numpy as np

from logger_config import logger

# ---------------- Task 4: counters ----------------
# latencies keeps only the last 100 values so the average shows recent behaviour
stats = {"requests": 0, "errors": 0, "latencies": deque(maxlen=100)}


def record(latency_ms: float, ok: bool = True) -> None:
    """Update counters after every request (success or failure)."""
    stats["requests"] += 1
    if not ok:
        stats["errors"] += 1
    stats["latencies"].append(latency_ms)


def summary() -> dict:
    """Log a one-line health summary and return it as a dict."""
    lat = stats["latencies"]
    avg = sum(lat) / len(lat) if lat else 0.0
    error_rate = stats["errors"] / stats["requests"] if stats["requests"] else 0.0
    logger.info(f"SUMMARY requests={stats['requests']} errors={stats['errors']} "
                f"error_rate={error_rate:.0%} avg_latency_ms={avg:.1f}")
    return {"requests": stats["requests"], "errors": stats["errors"],
            "error_rate": round(error_rate, 3), "avg_latency_ms": round(avg, 2)}


# ---------------- Task 5: drift check ----------------
def drift_check(train_mean: float, live_values, tol: float = 0.25,
                feature: str = "feature") -> float:
    """
    Compare the live mean of one feature to its training mean.
    Relative change above `tol` (default 25%) is logged as a WARNING.
    1e-9 avoids division by zero if the training mean is 0.
    """
    live_mean = float(np.mean(live_values))
    rel = abs(live_mean - train_mean) / (abs(train_mean) + 1e-9)
    if rel > tol:
        logger.warning(f"possible drift on {feature}: live={live_mean:.3f} "
                       f"train={train_mean:.3f} change={rel:.0%}")
    else:
        logger.info(f"no drift on {feature}: live={live_mean:.3f} "
                    f"train={train_mean:.3f} change={rel:.0%}")
    return rel
