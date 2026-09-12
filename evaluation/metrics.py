"""
Evaluation metrics (Table 7.5): standard prequential classification
metrics plus the proposal's security-aware measures -- Vulnerability
Window, Forgetting, Stability-Plasticity Index, Backward Transfer, and
Recovery Speed.

All metrics are computed from a per-window log collected during the
prequential run (see evaluation/prequential.py), so the same tracker
works for DriftFusion, its ablations, and the streaming baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score


def aggregate_runs(summaries: list[dict]) -> dict:
    """Mean +/- std across repeated runs (e.g. multiple random seeds) of
    the same configuration, for statistically reliable reporting."""
    keys = [k for k in summaries[0] if isinstance(summaries[0][k], (int, float))
            or summaries[0][k] is None]
    out = {}
    for k in keys:
        vals = [s[k] for s in summaries if s[k] is not None]
        if not vals:
            out[k] = (None, None)
        else:
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def paired_significance(method_runs: list[dict], baseline_runs: list[dict],
                        metric: str) -> dict:
    """Paired t-test of one metric between a method and a baseline, pairing
    on seed.

    Seeds are the unit of replication: each supplies one matched
    observation of both methods, so the pairing removes the seed-to-seed
    variation that otherwise swamps the small margins involved here. A
    difference in means is not reported as a result unless this test
    supports it.
    """
    from scipy import stats

    a = [r[metric] for r in method_runs if r.get(metric) is not None]
    b = [r[metric] for r in baseline_runs if r.get(metric) is not None]
    if len(a) != len(b) or len(a) < 2:
        return {"n": len(a), "t": None, "p": None, "mean_diff": None, "wins": None}
    t, p = stats.ttest_rel(a, b)
    return {
        "n": len(a),
        "t": float(t),
        "p": float(p),
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "wins": int(sum(x > y for x, y in zip(a, b))),
    }


@dataclass
class WindowRecord:
    window_idx: int
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray
    concept_id: str = "stable"


class MetricsTracker:
    def __init__(self, vw_threshold: float = 0.85, recovery_frac: float = 0.95):
        self.records: list[WindowRecord] = []
        self.drift_type_true: list[str] = []
        self.drift_type_pred: list[str] = []
        self.vw_threshold = vw_threshold
        self.recovery_frac = recovery_frac
        # Operational cost counters (Figure 7.4: Latency / Cost / Memory).
        self.window_latencies_ms: list[float] = []
        self.update_steps: list[int] = []
        self.memory_sizes: list[int] = []

    def log_cost(self, latency_ms: float | None = None, steps: int | None = None,
                 memory_size: int | None = None) -> None:
        if latency_ms is not None:
            self.window_latencies_ms.append(latency_ms)
        if steps is not None:
            self.update_steps.append(steps)
        if memory_size is not None:
            self.memory_sizes.append(memory_size)

    def log_window(self, window_idx: int, y_true: np.ndarray, y_pred: np.ndarray,
                   concept_id: str = "stable") -> None:
        acc = float(np.mean(y_true == y_pred))
        self.records.append(WindowRecord(window_idx, acc, y_true, y_pred, concept_id))

    def log_drift_type(self, true_type: str | None, pred_type: str) -> None:
        if true_type is not None:
            self.drift_type_true.append(true_type)
            self.drift_type_pred.append(pred_type)

    # -- Table 7.5 metrics -------------------------------------------------

    def prequential_accuracy(self) -> float:
        correct = sum(np.sum(r.y_true == r.y_pred) for r in self.records)
        total = sum(len(r.y_true) for r in self.records)
        return correct / total if total else float("nan")

    def macro_f1(self) -> float:
        y_true = np.concatenate([r.y_true for r in self.records])
        y_pred = np.concatenate([r.y_pred for r in self.records])
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

    def forgetting(self) -> float:
        best_by_concept: dict[str, float] = {}
        declines = []
        for r in self.records:
            if r.concept_id in best_by_concept:
                declines.append(max(0.0, best_by_concept[r.concept_id] - r.accuracy))
            best_by_concept[r.concept_id] = max(best_by_concept.get(r.concept_id, 0.0), r.accuracy)
        return float(np.mean(declines)) if declines else 0.0

    def _drift_onsets(self) -> list[int]:
        """Fallback onset detection: windows where the active concept
        identifier changes. Used only when no external onset list is
        supplied; cross-method comparisons should always pass
        `external_onsets` so every method is scored on the same episodes.
        """
        onsets = []
        for i in range(1, len(self.records)):
            if self.records[i].concept_id != self.records[i - 1].concept_id:
                onsets.append(i)
        return onsets

    def vulnerability_window(self, external_onsets: list[int] | None = None) -> float:
        accs = [r.accuracy for r in self.records]
        onsets = external_onsets if external_onsets is not None else self._drift_onsets()
        if not onsets:
            return 0.0
        widths = []
        for onset in onsets:
            w = 0
            for j in range(onset, len(accs)):
                if accs[j] < self.vw_threshold:
                    w += 1
                else:
                    break
            widths.append(w)
        return float(np.mean(widths))

    def recovery_speed(self, external_onsets: list[int] | None = None) -> float:
        accs = [r.accuracy for r in self.records]
        onsets = external_onsets if external_onsets is not None else self._drift_onsets()
        if not onsets:
            return 0.0
        speeds = []
        for onset in onsets:
            pre_drift_acc = np.mean(accs[max(0, onset - 3):onset]) if onset > 0 else accs[onset]
            target = self.recovery_frac * pre_drift_acc
            n = next((j - onset for j in range(onset, len(accs)) if accs[j] >= target),
                     len(accs) - onset)
            speeds.append(n)
        return float(np.mean(speeds))

    def stability_plasticity_index(self) -> float:
        accs = [r.accuracy for r in self.records]
        drops, gains = [], []
        for i in range(1, len(accs)):
            delta = accs[i] - accs[i - 1]
            drops.append(max(0.0, -delta))
            gains.append(max(0.0, delta))
        stability = 1 - np.mean(drops) if drops else 1.0
        plasticity = np.mean(gains) if gains else 0.0
        if stability + plasticity == 0:
            return 0.0
        return float(2 * stability * plasticity / (stability + plasticity))

    def backward_transfer(self) -> float:
        occurrences: dict[str, list[int]] = {}
        for i, r in enumerate(self.records):
            occurrences.setdefault(r.concept_id, []).append(i)
        bwts = []
        for concept, idxs in occurrences.items():
            if len(idxs) < 2:
                continue
            first_acc = self.records[idxs[0]].accuracy
            last_acc = self.records[idxs[-1]].accuracy
            bwts.append(last_acc - first_acc)
        return float(np.mean(bwts)) if bwts else 0.0

    def false_positive_rate(self) -> float:
        """FPR = FP / (FP + TN) (Figure 7.4). In the IDS setting this is the
        benign-traffic false-alarm rate, the metric an analyst actually
        feels: at CIC's 250k-flow scale a one-point FPR difference is
        thousands of spurious alerts, which accuracy alone hides."""
        y_true = np.concatenate([r.y_true for r in self.records])
        y_pred = np.concatenate([r.y_pred for r in self.records])
        negatives = (y_true == 0)
        n_neg = int(negatives.sum())
        if n_neg == 0:
            return float("nan")
        return float((y_pred[negatives] == 1).sum() / n_neg)

    def mean_latency_ms(self) -> float | None:
        """Mean wall-clock time to process one window (predict + adapt)."""
        return float(np.mean(self.window_latencies_ms)) if self.window_latencies_ms else None

    def adaptation_cost(self) -> float | None:
        """Mean gradient update steps per window -- the compute a deployment
        must budget for, and the axis Table 7.4's per-drift-type step counts
        directly control."""
        return float(np.mean(self.update_steps)) if self.update_steps else None

    def peak_memory_exemplars(self) -> float | None:
        """Peak exemplar count held in the context memory bank."""
        return float(np.max(self.memory_sizes)) if self.memory_sizes else None

    def drift_type_f1(self) -> float | None:
        if not self.drift_type_true:
            return None
        return f1_score(self.drift_type_true, self.drift_type_pred, average="macro", zero_division=0)

    def summary(self, external_onsets: list[int] | None = None) -> dict:
        onsets = external_onsets if external_onsets is not None else self._drift_onsets()
        return {
            # Table 7.5
            "accuracy": self.prequential_accuracy(),
            "macro_f1": self.macro_f1(),
            "forgetting": self.forgetting(),
            "vulnerability_window": self.vulnerability_window(external_onsets),
            "recovery_speed": self.recovery_speed(external_onsets),
            "spi": self.stability_plasticity_index(),
            "bwt": self.backward_transfer(),
            "drift_type_f1": self.drift_type_f1(),
            # Figure 7.4 operational metrics
            "fpr": self.false_positive_rate(),
            "latency_ms": self.mean_latency_ms(),
            "cost_steps": self.adaptation_cost(),
            "memory_exemplars": self.peak_memory_exemplars(),
            "n_windows": len(self.records),
            "n_drift_episodes": len(onsets),
        }
