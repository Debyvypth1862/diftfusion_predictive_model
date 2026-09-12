"""
DriftFusion: orchestrates Module 1 (fingerprinting + drift-type
classification), Module 2 (in-context zero-shot prediction), and
Module 3 (meta-drift adaptation with EWC) into a single per-window
streaming pipeline (Figure 7.1).

Ablation variants (Section 7.7) are implemented as constructor flags
rather than separate classes, so the exact same driver loop exercises
the full system and each degraded variant.
"""

from __future__ import annotations

import numpy as np
import torch

from .fingerprint import FingerprintExtractor
from .drift_classifier import DriftTypeClassifier, train_drift_classifier
from .context_predictor import InContextZeroShotPredictor, ConceptTracker
from .meta_adapter import MetaDriftAdapter


class DriftFusionModel:
    def __init__(self, input_dim: int, drift_classifier: DriftTypeClassifier | None = None,
                 use_fingerprinting: bool = True, use_context_memory: bool = True,
                 use_ewc: bool = True, label_delay_k: int = 0, seed: int = 0,
                 strategy_table: dict | None = None,
                 drift_confidence_threshold: float = 0.6):
        self.use_fingerprinting = use_fingerprinting
        self.use_context_memory = use_context_memory
        self.drift_confidence_threshold = drift_confidence_threshold

        torch.manual_seed(seed)
        self.fingerprint = FingerprintExtractor()
        self.drift_classifier = drift_classifier or train_drift_classifier(verbose=False, seed=seed)
        self.predictor = InContextZeroShotPredictor(input_dim, label_delay_k=label_delay_k)
        self.adapter = MetaDriftAdapter(self.predictor.model, strategy_table=strategy_table,
                                         use_ewc=use_ewc)

        self.concepts = ConceptTracker()
        self.drift_history: list[str] = []
        self.concept_history: list[str] = []
        self.window_count = 0

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray) -> None:
        self.fingerprint.set_reference(X_calib, y_calib)
        self.predictor.pretrain(X_calib, y_calib)
        # The EWC reference snapshot must anchor to the *pretrained* weights,
        # not the random initialisation captured when MetaDriftAdapter was
        # constructed -- otherwise EWC would penalise the model for having
        # left its untrained starting point.
        self.adapter.theta_star = {n: p.detach().clone()
                                    for n, p in self.predictor.model.named_parameters()}

    def _forward_fn(self, concept_id: str, use_fallback: bool):
        def fn(X: torch.Tensor) -> torch.Tensor:
            h = self.predictor.model.embed(X)
            if use_fallback or not self.use_context_memory:
                return self.predictor.model.forward_fallback(h)
            mem = self.predictor.memory.get(concept_id)
            mem_x = torch.from_numpy(np.stack([m[0] for m in mem])).float()
            mem_y = torch.tensor([m[1] for m in mem], dtype=torch.long)
            mem_h = self.predictor.model.embed(mem_x)
            return self.predictor.model.forward_with_context(h, mem_h, mem_y)
        return fn

    def predict(self, X_window: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Test phase: predict this window using the *current* memory/weights
        (no peeking at this window's labels)."""
        concept_id = self.concepts.current_id
        y_pred, y_proba, _ = self.predictor.predict(X_window, concept_id)
        return y_pred, y_proba, concept_id

    def observe_and_adapt(self, X_window: np.ndarray, y_window: np.ndarray,
                           y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
        """Train phase: update the fingerprint/drift-type state, then let
        Module 3 adapt Module 2's parameters accordingly."""
        self.window_count += 1
        fp_vector = self.fingerprint.compute(X_window, y_window, y_pred, y_proba)

        if self.use_fingerprinting and self.fingerprint.sequence_ready():
            drift_type, drift_probs = self.drift_classifier.predict(self.fingerprint.sequence())
            # Act on a drift category only when the classifier is actually
            # confident in it. The classifier is meta-trained on synthetic
            # fingerprint sequences, so its transfer to real streams is
            # imperfect and seed-dependent; taking argmax unconditionally
            # let a low-confidence 'sudden' call trigger the most aggressive
            # entry in Table 7.4 (lr=0.012 over 12 steps). Falling back to
            # 'stable' below threshold makes an uncertain detector behave
            # conservatively rather than destructively -- the asymmetry is
            # deliberate, since a missed adaptation costs far less than an
            # unwarranted large parameter update.
            if float(np.max(drift_probs)) < self.drift_confidence_threshold:
                drift_type = "stable"
        else:
            drift_type, drift_probs = "stable", None
        self.drift_history.append(drift_type)

        # Algorithm 3 Step 7 - Context Memory Notification: resolve which
        # concept partition this window belongs to (new one on sudden,
        # historical recall on recurring) before memory is read or written.
        prev_concept = self.concept_history[-1] if self.concept_history else None
        concept_id = self.concepts.update(drift_type, fp_vector)
        self.concept_history.append(concept_id)
        concept_changed = (prev_concept is None) or (concept_id != prev_concept)

        confidence_drop = float(fp_vector[8])  # index 8: confidence_drop (Table 7.3)

        X_t = torch.from_numpy(X_window).float()
        y_t = torch.from_numpy(y_window).long()
        use_fallback = (not self.use_context_memory) or (not self.predictor.memory.ready(concept_id))
        forward_fn = self._forward_fn(concept_id, use_fallback)

        meta = self.adapter.adapt(forward_fn, X_t, y_t, drift_type, confidence_drop,
                                   concept_changed=concept_changed)

        if self.use_context_memory:
            self.predictor.observe(X_window, y_window, concept_id)

        meta["drift_type"] = drift_type
        meta["concept_id"] = concept_id
        meta["drift_probs"] = drift_probs
        meta["fingerprint"] = fp_vector
        return meta
