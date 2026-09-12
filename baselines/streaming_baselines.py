"""
Streaming baselines (Section 7.7): Adaptive Random Forest and an
ADWIN-monitored Hoeffding Tree, both run row-by-row via `river` in the
same prequential (test-then-train) protocol as DriftFusion, aggregated
to window-level predictions for a like-for-like comparison.
"""

from __future__ import annotations

import numpy as np
from river import drift as river_drift
from river import forest as river_forest
from river import tree as river_tree


def _row_to_dict(row: np.ndarray) -> dict:
    return {str(i): float(v) for i, v in enumerate(row)}


class ARFBaseline:
    """Adaptive Random Forest: ten Hoeffding Trees, each with its own
    ADWIN drift monitor, replacing a member tree from scratch on alarm."""

    def __init__(self, n_models: int = 10, seed: int = 0):
        self.model = river_forest.ARFClassifier(n_models=n_models, seed=seed)

    def predict_and_learn_window(self, X_window: np.ndarray, y_window: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(X_window), dtype=int)
        for i, (row, label) in enumerate(zip(X_window, y_window)):
            x = _row_to_dict(row)
            pred = self.model.predict_one(x)
            preds[i] = int(pred) if pred is not None else 0
            self.model.learn_one(x, int(label))
        return preds


class AdwinHoeffdingTreeBaseline:
    """A single incrementally-grown Hoeffding Tree paired with an ADWIN
    change detector monitoring the (0/1) error stream; the tree is reset
    to a fresh instance whenever ADWIN signals a significant shift."""

    def __init__(self):
        self.model = river_tree.HoeffdingTreeClassifier()
        self.adwin = river_drift.ADWIN()
        self.n_resets = 0

    def predict_and_learn_window(self, X_window: np.ndarray, y_window: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(X_window), dtype=int)
        for i, (row, label) in enumerate(zip(X_window, y_window)):
            x = _row_to_dict(row)
            pred = self.model.predict_one(x)
            pred = int(pred) if pred is not None else 0
            preds[i] = pred
            self.model.learn_one(x, int(label))

            error = int(pred != int(label))
            self.adwin.update(error)
            if self.adwin.drift_detected:
                self.model = river_tree.HoeffdingTreeClassifier()
                self.n_resets += 1
        return preds
