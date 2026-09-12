"""
Prequential (interleaved test-then-train) evaluation harness (Figure 7.4):
segments a stream into fixed-size windows and, for each window, predicts
before revealing labels, then updates the model -- run identically for
DriftFusion (+ ablations) and the streaming baselines.

An initial calibration slice is reserved before the timed prequential
run (not scored) to: (a) set the fingerprint reference statistics
(Section 7.4), and (b) meta-train/warm-start each model, since Module 2's
fallback head is designed to "rely solely on parameters learned during
meta-training" rather than be learned online from a random start under
the deliberately conservative 'stable' update budget (Table 7.4). The
same calibration rows are given to the baselines via plain online
learning, so the comparison stays apples-to-apples.
"""

from __future__ import annotations

import time

import numpy as np

from ..data.preprocessing import RunningNormalizer
from ..models.driftfusion import DriftFusionModel
from ..models.drift_classifier import DriftTypeClassifier
from ..baselines.streaming_baselines import _row_to_dict
from .metrics import MetricsTracker


def make_windows(X: np.ndarray, y: np.ndarray, window_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(X) // window_size
    return [(X[i * window_size:(i + 1) * window_size], y[i * window_size:(i + 1) * window_size])
            for i in range(n)]


def _calibration_split(windows: list, calib_frac: float, min_calib_windows: int):
    n_calib = max(min_calib_windows, int(len(windows) * calib_frac))
    n_calib = min(n_calib, len(windows) - 1)
    return windows[:n_calib], windows[n_calib:]


def run_driftfusion(X: np.ndarray, y: np.ndarray, window_size: int,
                     drift_classifier: DriftTypeClassifier,
                     use_fingerprinting: bool = True, use_context_memory: bool = True,
                     use_ewc: bool = True, label_delay_k: int = 0,
                     true_drift_types: list[str] | None = None,
                     calib_frac: float = 0.2, min_calib_windows: int = 2,
                     seed: int = 0, strategy_table: dict | None = None,
                     external_onsets: list[int] | None = None,
                     return_tracker: bool = False) -> tuple[dict, list[str]]:
    windows = make_windows(X, y, window_size)
    calib_windows, eval_windows = _calibration_split(windows, calib_frac, min_calib_windows)

    normalizer = RunningNormalizer(n_features=X.shape[1])
    model = DriftFusionModel(input_dim=X.shape[1], drift_classifier=drift_classifier,
                              use_fingerprinting=use_fingerprinting,
                              use_context_memory=use_context_memory, use_ewc=use_ewc,
                              label_delay_k=label_delay_k, seed=seed, strategy_table=strategy_table)

    X_calib = np.concatenate([w[0] for w in calib_windows])
    y_calib = np.concatenate([w[1] for w in calib_windows])
    for Xw_raw, _ in calib_windows:
        normalizer.update(Xw_raw)
    model.calibrate(normalizer.transform(X_calib), y_calib)

    tracker = MetricsTracker()
    predicted_types = []
    n_calib_windows = len(calib_windows)

    for t, (Xw_raw, yw) in enumerate(eval_windows):
        Xw = normalizer.transform(Xw_raw)
        normalizer.update(Xw_raw)

        # Figure 7.4's loop: Predict -> Evaluate -> Reveal Labels -> Adapt.
        # Latency covers predict+adapt, i.e. everything the deployment must
        # do per window before the next one arrives.
        t_start = time.perf_counter()
        y_pred, y_proba, concept_id = model.predict(Xw)
        tracker.log_window(t, yw, y_pred, concept_id=concept_id)

        meta = model.observe_and_adapt(Xw, yw, y_pred, y_proba)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        tracker.log_cost(latency_ms=elapsed_ms, steps=meta["steps_taken"],
                          memory_size=sum(len(p) for p in model.predictor.memory.partitions.values()))

        predicted_types.append(meta["drift_type"])
        if true_drift_types is not None:
            tracker.log_drift_type(true_drift_types[n_calib_windows + t], meta["drift_type"])

    summary = tracker.summary(external_onsets=external_onsets)
    if return_tracker:
        return summary, predicted_types, tracker
    return summary, predicted_types


def run_baseline(baseline, X: np.ndarray, y: np.ndarray, window_size: int,
                  external_onsets: list[int] | None = None,
                  calib_frac: float = 0.2, min_calib_windows: int = 2,
                  return_tracker: bool = False) -> dict:
    windows = make_windows(X, y, window_size)
    calib_windows, eval_windows = _calibration_split(windows, calib_frac, min_calib_windows)

    normalizer = RunningNormalizer(n_features=X.shape[1])
    for Xw_raw, yw in calib_windows:
        Xw = normalizer.transform(Xw_raw)
        normalizer.update(Xw_raw)
        for row, label in zip(Xw, yw):
            baseline.model.learn_one(_row_to_dict(row), int(label))

    tracker = MetricsTracker()
    for t, (Xw_raw, yw) in enumerate(eval_windows):
        Xw = normalizer.transform(Xw_raw)
        normalizer.update(Xw_raw)
        t_start = time.perf_counter()
        y_pred = baseline.predict_and_learn_window(Xw, yw)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        tracker.log_window(t, yw, y_pred)
        # Baselines learn incrementally per row rather than by gradient
        # steps, so 'cost' is left unrecorded; latency is directly
        # comparable and is the meaningful efficiency axis here.
        tracker.log_cost(latency_ms=elapsed_ms)

    summary = tracker.summary(external_onsets=external_onsets)
    if return_tracker:
        return summary, tracker
    return summary
