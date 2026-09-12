"""
Concept Fingerprinting (Algorithm 1 / Table 7.3): reduces each window to
a 10-dimensional descriptor combining supervised performance signals,
Bayesian changepoint statistics, and distributional summaries.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .bocpd import OnlineBOCPD

FINGERPRINT_DIM = 10
FINGERPRINT_NAMES = [
    "window_error_rate", "confidence_entropy", "label_balance",
    "feature_drift_magnitude", "bocpd_changepoint_prob", "run_length_entropy",
    "normalized_modal_run_length", "error_trend_slope", "confidence_drop",
    "label_kl_divergence",
]


CONFIDENCE_BINS = 10


def _confidence_distribution_entropy(y_proba: np.ndarray,
                                      n_bins: int = CONFIDENCE_BINS) -> float:
    """Algorithm 1 Step 1: Hc = -sum p(c) log p(c) "over the distribution of
    softmax confidence values".

    The quantity is the entropy of the *distribution* the window's
    confidence values form, not the mean of each sample's own predictive
    entropy. The two differ in what they detect: the mean per-sample
    entropy tracks how unsure the model is on average, whereas this tracks
    how spread out its certainty is across the window -- a model that is
    confidently right on half the window and confidently wrong on the other
    half is the signature of a concept split, and it is invisible to the
    averaged form. Confidence for a binary head is max(p, 1-p) in [0.5, 1],
    so the histogram spans that range. Normalised by log(n_bins) to keep
    the feature in [0, 1] regardless of bin count.
    """
    conf = np.maximum(y_proba, 1.0 - y_proba).astype(np.float64)
    counts, _ = np.histogram(conf, bins=n_bins, range=(0.5, 1.0))
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    entropy = float(-np.sum(p * np.log(p)))
    return entropy / np.log(n_bins)


def _kl_bernoulli(p: float, q: float) -> float:
    p, q = np.clip(np.array([p, q], dtype=np.float64), 1e-6, 1 - 1e-6)
    return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


class FingerprintExtractor:
    def __init__(self, history_len: int = 10, hazard_lambda: float = 10.0,
                 error_slope_window: int = 5):
        # hazard_lambda (expected run length) of 10 was chosen after
        # observing that streams here span only tens to low hundreds of
        # windows: the literature-typical default of ~60 assumes a much
        # longer stream and left BOCPD's run-length posterior essentially
        # frozen at "no change has ever happened" (see run diagnostics).
        self.bocpd = OnlineBOCPD(hazard_lambda=hazard_lambda)
        self.error_history: deque[float] = deque(maxlen=error_slope_window)
        self.prev_mean_confidence: float | None = None
        self.buffer: deque[np.ndarray] = deque(maxlen=history_len)

        self.reference_mean: np.ndarray | None = None
        self.reference_std: np.ndarray | None = None
        self.reference_label_rate: float | None = None

    def set_reference(self, X_calib: np.ndarray, y_calib: np.ndarray) -> None:
        self.reference_mean = X_calib.mean(axis=0)
        self.reference_std = X_calib.std(axis=0) + 1e-8
        self.reference_label_rate = float(np.clip(y_calib.mean(), 1e-3, 1 - 1e-3))

    def compute(self, X_window: np.ndarray, y_window: np.ndarray,
                y_pred: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
        """Steps 1-4 of Algorithm 1: extract the fingerprint vector for one
        window and push it onto the rolling sequence buffer."""
        assert self.reference_mean is not None, "call set_reference() first"

        et = float(np.mean(y_pred != y_window))
        conf_entropy = _confidence_distribution_entropy(y_proba)
        bt = float(y_window.mean())
        feature_drift = float(np.mean(
            np.abs(X_window.mean(axis=0) - self.reference_mean) / self.reference_std
        ))

        bocpd_out = self.bocpd.step(et)

        self.error_history.append(et)
        if len(self.error_history) >= 2:
            xs = np.arange(len(self.error_history))
            slope = float(np.polyfit(xs, np.array(self.error_history), 1)[0])
        else:
            slope = 0.0

        mean_confidence = float(np.mean(np.maximum(y_proba, 1 - y_proba)))
        conf_drop = max(0.0, (self.prev_mean_confidence or mean_confidence) - mean_confidence)
        self.prev_mean_confidence = mean_confidence

        label_kl = _kl_bernoulli(bt, self.reference_label_rate)

        vector = np.array([
            et, conf_entropy, bt, feature_drift,
            bocpd_out["p_changepoint"], bocpd_out["run_length_entropy"],
            bocpd_out["normalized_modal_run_length"], slope, conf_drop, label_kl,
        ], dtype=np.float32)

        self.buffer.append(vector)
        return vector

    def sequence_ready(self) -> bool:
        return len(self.buffer) == self.buffer.maxlen

    def sequence(self) -> np.ndarray:
        """Returns the (history_len, FINGERPRINT_DIM) buffer, left-padded by
        repeating the oldest entry if history hasn't filled yet."""
        if len(self.buffer) == 0:
            return np.zeros((self.buffer.maxlen, FINGERPRINT_DIM), dtype=np.float32)
        arr = np.stack(self.buffer)
        if len(arr) < self.buffer.maxlen:
            pad = np.repeat(arr[:1], self.buffer.maxlen - len(arr), axis=0)
            arr = np.concatenate([pad, arr], axis=0)
        return arr
