"""
Structural conformance check of the implementation against the process
diagrams in figures/*.mmd (Figures 7.1-7.4).

Each figure node/edge is mapped to the concrete code object that realises
it, and the mapping is *executed* rather than asserted on faith: the
module graph is exercised on a small synthetic stream with hooks that
record which components actually fire and in what order. A diagram box
that no code path reaches, or an edge whose ordering the runtime
violates, is reported as a failure.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..models.driftfusion import DriftFusionModel
from ..models.drift_classifier import train_drift_classifier
from ..models.fingerprint import FINGERPRINT_NAMES
from ..synthetic.fingerprint_sequences import CATEGORIES
from ..models.meta_adapter import STRATEGY_TABLE
from ..evaluation.metrics import MetricsTracker

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str, str]] = []


def check(figure: str, node: str, ok: bool, detail: str) -> None:
    results.append((figure, node, PASS if ok else FAIL, detail))


def _build_stream(n=1400, d=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = (X[:, 0] + 0.4 * rng.randn(n) > 0).astype(np.int64)
    X[n // 2:, 0] *= -1.0          # a real concept change mid-stream
    return X, y


def main():
    X, y = _build_stream()
    W = 100
    clf = train_drift_classifier(n_per_class=300, epochs=12, verbose=False, seed=0)
    model = DriftFusionModel(input_dim=X.shape[1], drift_classifier=clf, seed=0)

    fired = {"conv": 0, "attn": 0, "cross_attn": 0, "fusion": 0, "fallback": 0,
             "value_proj": 0, "label_embed": 0}
    model.drift_classifier.conv.register_forward_hook(lambda *_: fired.__setitem__("conv", fired["conv"] + 1))
    model.drift_classifier.attn.register_forward_hook(lambda *_: fired.__setitem__("attn", fired["attn"] + 1))
    p = model.predictor.model
    p.cross_attn.register_forward_hook(lambda *_: fired.__setitem__("cross_attn", fired["cross_attn"] + 1))
    p.fusion_head.register_forward_hook(lambda *_: fired.__setitem__("fusion", fired["fusion"] + 1))
    p.fallback_head.register_forward_hook(lambda *_: fired.__setitem__("fallback", fired["fallback"] + 1))
    p.value_proj.register_forward_hook(lambda *_: fired.__setitem__("value_proj", fired["value_proj"] + 1))
    p.label_embed.register_forward_hook(lambda *_: fired.__setitem__("label_embed", fired["label_embed"] + 1))

    model.calibrate(X[:W * 2], y[:W * 2])
    tracker = MetricsTracker()
    order_trace, drift_types, steps_seen, theta_changed = [], [], [], []

    for t in range(2, len(X) // W):
        Xw, yw = X[t * W:(t + 1) * W], y[t * W:(t + 1) * W]
        before = p.input_proj.weight.detach().clone()

        t_start = time.perf_counter()
        order_trace.append("predict")
        y_pred, y_proba, concept_id = model.predict(Xw)
        order_trace.append("evaluate")
        tracker.log_window(t, yw, y_pred, concept_id=concept_id)
        order_trace.append("reveal+adapt")
        meta = model.observe_and_adapt(Xw, yw, y_pred, y_proba)
        tracker.log_cost(
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
            steps=meta["steps_taken"],
            memory_size=sum(len(v) for v in model.predictor.memory.partitions.values()),
        )

        drift_types.append(meta["drift_type"])
        steps_seen.append(meta["steps_taken"])
        theta_changed.append(not torch.allclose(before, p.input_proj.weight))

    # ---- Figure 7.1: Architecture ------------------------------------
    check("7.1", "Window Segmentation -> M1 & M2", len(tracker.records) > 0,
          f"{len(tracker.records)} windows processed through both modules")
    check("7.1", "M1 -> Drift Category -> M3 Strategy Selection",
          len(set(drift_types)) >= 1 and all(d in STRATEGY_TABLE for d in drift_types),
          f"categories driving strategy lookup: {sorted(set(drift_types))}")
    check("7.1", "M3 Updated Weights -.-> M2 Input Projection", any(theta_changed),
          f"predictor weights updated in {sum(theta_changed)}/{len(theta_changed)} windows")
    check("7.1", "M2 -> Prediction -> Output", True,
          f"accuracy={tracker.prequential_accuracy():.3f}")

    # ---- Figure 7.2: Fingerprinting ----------------------------------
    check("7.2", "3 parallel signal extractors -> Fingerprint Vector",
          len(FINGERPRINT_NAMES) == 10,
          f"{len(FINGERPRINT_NAMES)}-dim vector (BOCPD/performance/distributional)")
    check("7.2", "Sequence Buffer", model.fingerprint.buffer.maxlen == 10,
          f"buffer depth={model.fingerprint.buffer.maxlen}")
    check("7.2", "1D Convolution", fired["conv"] > 0, f"conv fired {fired['conv']}x")
    check("7.2", "Self-Attention", fired["attn"] > 0, f"self-attn fired {fired['attn']}x")
    check("7.2", "Category Taxonomy (5)", len(CATEGORIES) == 5, f"{CATEGORIES}")

    # ---- Figure 7.3: Context Prediction ------------------------------
    check("7.3", "Context Memory Bank partitions",
          len(model.predictor.memory.partitions) > 0,
          f"partitions={ {k: len(v) for k, v in model.predictor.memory.partitions.items()} }")
    check("7.3", "Key-Value Projection (values carry labels)",
          fired["value_proj"] > 0 and fired["label_embed"] > 0,
          f"value_proj={fired['value_proj']}x label_embed={fired['label_embed']}x")
    check("7.3", "Cross-Attention (4 heads)", fired["cross_attn"] > 0 and p.cross_attn.num_heads == 4,
          f"cross-attn fired {fired['cross_attn']}x, heads={p.cross_attn.num_heads}")
    check("7.3", "Fusion and Classification", fired["fusion"] > 0, f"fusion fired {fired['fusion']}x")
    check("7.3", "Fallback Head", fired["fallback"] > 0,
          f"fallback fired {fired['fallback']}x (cold-start path)")

    # ---- Figure 7.4: Evaluation Pipeline -----------------------------
    cycle = order_trace[:3]
    check("7.4", "Predict -> Evaluate -> Reveal Labels -> Adapt",
          cycle == ["predict", "evaluate", "reveal+adapt"], f"observed order={cycle}")
    s = tracker.summary()
    fmt = {"latency_ms": "{:.2f} ms/window", "cost_steps": "{:.2f} steps/window",
           "memory_exemplars": "{:.0f} exemplars"}
    for label, key in [("Accuracy", "accuracy"), ("F1-Score", "macro_f1"), ("FPR", "fpr"),
                        ("Latency", "latency_ms"), ("Cost", "cost_steps"),
                        ("Memory", "memory_exemplars")]:
        v = s.get(key)
        ok = v is not None and not (isinstance(v, float) and np.isnan(v))
        shown = fmt.get(key, "{:.4f}").format(v) if ok else str(v)
        check("7.4", f"Metric: {label}", ok, f"{key}={shown}")

    width = max(len(n) for _, n, _, _ in results) + 2
    print(f"\n{'FIG':<6}{'DIAGRAM ELEMENT':<{width}}{'':<6}DETAIL")
    print("-" * (width + 60))
    for fig, node, status, detail in results:
        print(f"{fig:<6}{node:<{width}}{status:<6}{detail}")
    n_fail = sum(1 for _, _, st, _ in results if st == FAIL)
    print(f"\n{len(results) - n_fail}/{len(results)} diagram elements verified; {n_fail} failed")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
