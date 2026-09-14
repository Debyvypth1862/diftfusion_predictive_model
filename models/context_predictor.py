"""
In-Context Zero-Shot Predictor (Algorithm 2 / Module 2): predictions are
conditioned on a per-concept exemplar memory via cross-attention, so that
responding to a recognised concept change requires no gradient step --
only a change of which memory partition is queried. A fallback direct
head handles concepts with too few exemplars.

Partitions are keyed by *concept identity*, tracked by ConceptTracker
below, not by drift-type category. An earlier version keyed them by
category; see the ConceptTracker docstring for why that was corrected.

Meta-training uses first-order MAML: see pretrain_context_predictor.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .meta_adapter import focal_loss

D_MODEL = 128
N_HEADS = 4
MIN_EXEMPLARS = 5
MEMORY_CAPACITY = 200

# First-order MAML meta-training (Finn, Abbeel and Levine, 2017). The inner
# loop adapts on an episode's support set with the same focal objective and
# the same plain-SGD form the online adapter uses at deployment; the outer
# objective is then evaluated at those *adapted* parameters, so what is
# optimised is an initialisation from which a short update generalises,
# rather than one that merely fits the calibration data.
#
# The meta-gradient is first-order: the query gradient taken at the adapted
# parameters is applied to the pre-adaptation parameters, without
# differentiating through the inner step. The second-order term would
# require retaining the inner graph across every episode, and Finn et al.
# report first-order MAML performing comparably at a fraction of the cost.
FOMAML_INNER_LR = 0.01
FOMAML_INNER_STEPS = 1


class ConceptTracker:
    """Concept identity management (Algorithm 3 Step 7 / Module 2).

    The proposal partitions context memory "for each identified concept",
    not for each drift *category*: a sudden drift must "initialise a new
    partition" and a recurring drift must "retrieve historical memory
    partition". Keying partitions by drift category instead collapses every
    sudden episode in the stream into one shared partition, so exemplars
    from unrelated attack regimes are averaged together and the recurring
    branch can never recall anything specific.

    Concepts are identified by a running centroid of their fingerprint
    vectors; a recurring drift resolves to whichever historical concept has
    the nearest centroid, which is what makes recall meaningful.
    """

    def __init__(self, match_threshold: float = 2.0):
        self._next_id = 0
        self.current_id = self._new_id()
        self.centroids: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.match_threshold = match_threshold

    def _new_id(self) -> str:
        cid = f"concept_{self._next_id}"
        self._next_id += 1
        return cid

    def _nearest_historical(self, fingerprint: np.ndarray) -> str | None:
        candidates = {c: v for c, v in self.centroids.items() if c != self.current_id}
        if not candidates:
            return None
        best = min(candidates, key=lambda c: float(np.linalg.norm(candidates[c] - fingerprint)))
        if float(np.linalg.norm(candidates[best] - fingerprint)) > self.match_threshold:
            return None
        return best

    def update(self, drift_type: str, fingerprint: np.ndarray) -> str:
        """Resolve the active concept for this window, then fold the
        fingerprint into that concept's centroid."""
        if drift_type == "sudden":
            self.current_id = self._new_id()
        elif drift_type == "recurring":
            recalled = self._nearest_historical(fingerprint)
            self.current_id = recalled if recalled is not None else self._new_id()

        cid = self.current_id
        n = self.counts.get(cid, 0)
        if n == 0:
            self.centroids[cid] = fingerprint.astype(np.float64).copy()
        else:
            self.centroids[cid] += (fingerprint - self.centroids[cid]) / (n + 1)
        self.counts[cid] = n + 1
        return cid


class ContextMemoryBank:
    def __init__(self, capacity: int = MEMORY_CAPACITY, min_exemplars: int = MIN_EXEMPLARS):
        self.capacity = capacity
        self.min_exemplars = min_exemplars
        self.partitions: dict[str, deque] = {}

    def ready(self, concept_id: str) -> bool:
        return len(self.partitions.get(concept_id, [])) >= self.min_exemplars

    def get(self, concept_id: str) -> list[tuple[np.ndarray, int]]:
        return list(self.partitions.get(concept_id, []))

    def insert(self, concept_id: str, X: np.ndarray, y: np.ndarray) -> None:
        if concept_id not in self.partitions:
            self.partitions[concept_id] = deque(maxlen=self.capacity)
        for xi, yi in zip(X, y):
            self.partitions[concept_id].append((xi, int(yi)))


class ContextPredictor(nn.Module):
    def __init__(self, input_dim: int, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_classes: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Algorithm 2 Step 3 draws memory from Mc = {(x_j, y_j)} pairs. The
        # keys are built from h(x_j) exactly as written; the *values* also
        # carry an embedding of y_j. Building values from h(x_j) alone --
        # the literal reading of the K/V formula -- discards every label in
        # the memory bank, so the retrieved context conveys only where the
        # exemplars sit in feature space and nothing about their classes.
        # A context pathway that cannot transmit label information cannot
        # produce the "immediate prediction adjustments without gradient-
        # based retraining" the module is specified to deliver, and
        # measurably degraded accuracy relative to disabling it entirely.
        self.label_embed = nn.Embedding(n_classes, d_model)
        self.value_proj = nn.Linear(2 * d_model, d_model)
        self.fusion_head = nn.Sequential(
            nn.Linear(2 * d_model, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )
        self.fallback_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )
        self.d_model = d_model

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_proj(x)

    def forward_with_context(self, h: torch.Tensor, mem_h: torch.Tensor,
                              mem_y: torch.Tensor) -> torch.Tensor:
        """h: (batch, d_model) query embeddings. mem_h: (n_mem, d_model)
        exemplar embeddings. mem_y: (n_mem,) exemplar labels."""
        q = h.unsqueeze(1)                                   # (batch, 1, d_model)
        keys = mem_h.unsqueeze(0).expand(h.size(0), -1, -1)  # (batch, n_mem, d_model)
        values = self.value_proj(torch.cat([mem_h, self.label_embed(mem_y)], dim=-1))
        values = values.unsqueeze(0).expand(h.size(0), -1, -1)
        ctx, _ = self.cross_attn(q, keys, values)
        ctx = ctx.squeeze(1)                                  # (batch, d_model)
        fused = torch.cat([h, ctx], dim=-1)
        return self.fusion_head(fused)

    def forward_fallback(self, h: torch.Tensor) -> torch.Tensor:
        return self.fallback_head(h)


def pretrain_context_predictor(model: "ContextPredictor", X_calib: np.ndarray, y_calib: np.ndarray,
                                epochs: int = 300, lr: float = 1e-3, seed: int = 0,
                                support_size: int = MEMORY_CAPACITY,
                                query_size: int = 128, val_frac: float = 0.2,
                                eval_every: int = 10,
                                inner_lr: float = FOMAML_INNER_LR,
                                inner_steps: int = FOMAML_INNER_STEPS) -> None:
    """Meta-training bootstrap for Module 2 (Section 7.4: the fallback head
    "relies solely on the model parameters learned during meta-training").

    Both prediction paths are meta-trained. Training only the fallback head
    leaves cross_attn, value_proj, label_embed and fusion_head at their
    random initialisation, so the moment a memory partition reaches
    m_min exemplars the system switches from a trained head to an untrained
    one and discards everything it had learned. Table 7.4's per-window
    update budget (one step at lr=0.0005 under 'stable') cannot recover
    that gap online, which is precisely the regime the context path is
    supposed to serve.

    Each epoch is one first-order MAML episode. A support set (standing in
    for a memory partition) and a disjoint query set are sampled; the inner
    loop adapts on the support set with the same focal objective and plain
    SGD the online adapter uses, and the outer objective is evaluated at
    those *adapted* parameters before its gradient is applied to the
    pre-adaptation weights. What is optimised is therefore an initialisation
    from which the short per-window update generalises, not one that merely
    fits the calibration rows -- the distinction that matters here, because
    at deployment the model only ever gets a handful of steps per window.

    The outer objective keeps both paths: the direct head on the query rows,
    and the context path classifying those same rows given the labelled
    support -- the same conditional prediction performed at inference. The
    disjoint split is what forces the context path to read the support
    labels rather than memorise the calibration data.

    Training is validated and the best-scoring weights retained. Without
    this the context path was observed to reach ~94% episodic accuracy and
    then diverge into constant majority-class output, from which it never
    recovered; because that collapse suppresses the attack class entirely
    it is invisible to accuracy on an imbalanced stream and shows up only
    in macro-F1 and the predicted positive rate. Selecting on validation
    macro-F1 rather than loss makes the criterion sensitive to exactly that
    failure.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_all = len(X_calib)
    perm0 = rng.permutation(n_all)
    n_val = max(MIN_EXEMPLARS * 2, int(n_all * val_frac))
    val_idx, tr_idx = perm0[:n_val], perm0[n_val:]

    x_tr = torch.from_numpy(X_calib[tr_idx]).float()
    y_tr = torch.from_numpy(y_calib[tr_idx]).long()
    x_va = torch.from_numpy(X_calib[val_idx]).float()
    y_va = torch.from_numpy(y_calib[val_idx]).long()
    n = len(x_tr)

    best_score, best_state = -1.0, None

    def _macro_f1(pred: torch.Tensor, true: torch.Tensor) -> float:
        f1s = []
        for c in (0, 1):
            tp = ((pred == c) & (true == c)).sum().item()
            fp = ((pred == c) & (true != c)).sum().item()
            fn = ((pred != c) & (true == c)).sum().item()
            denom = 2 * tp + fp + fn
            f1s.append((2 * tp / denom) if denom else 0.0)
        return sum(f1s) / len(f1s)

    for ep in range(epochs):
        model.train()

        # Sample one task: a support set standing in for a memory partition
        # and a disjoint query set, so the context path must read the
        # support labels rather than memorise the calibration data.
        sup_idx = qry_idx = None
        if n >= MIN_EXEMPLARS * 2:
            perm = rng.permutation(n)
            n_sup = min(support_size, n // 2)
            sup_idx = perm[:n_sup]
            qry_idx = perm[n_sup:n_sup + query_size]

        if sup_idx is None or len(qry_idx) == 0:
            # Not enough calibration rows to form an episode; fall back to a
            # plain supervised step on the direct head.
            opt.zero_grad()
            F.cross_entropy(model.forward_fallback(model.embed(x_tr)), y_tr).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            x_s, y_s = x_tr[sup_idx], y_tr[sup_idx]
            x_q, y_q = x_tr[qry_idx], y_tr[qry_idx]

            theta = [p.detach().clone() for p in model.parameters()]

            # --- inner loop: adapt on the support set, exactly as the
            # online adapter would on a newly encountered concept.
            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
            for _ in range(inner_steps):
                inner_opt.zero_grad()
                focal_loss(model.forward_fallback(model.embed(x_s)), y_s).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                inner_opt.step()

            # --- outer objective, evaluated at the *adapted* parameters:
            # the direct head on the query rows, plus the context path
            # classifying those same rows given the labelled support.
            loss = F.cross_entropy(model.forward_fallback(model.embed(x_q)), y_q)
            loss = loss + F.cross_entropy(
                model.forward_with_context(model.embed(x_q), model.embed(x_s), y_s), y_q)
            meta_grads = torch.autograd.grad(loss, list(model.parameters()),
                                             allow_unused=True)

            # --- restore the pre-adaptation parameters and apply the
            # first-order meta-gradient to them.
            with torch.no_grad():
                for p, t in zip(model.parameters(), theta):
                    p.copy_(t)
            opt.zero_grad()
            for p, g in zip(model.parameters(), meta_grads):
                p.grad = None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if ep % eval_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_fb = model.forward_fallback(model.embed(x_va)).argmax(-1)
                score = _macro_f1(pred_fb, y_va)
                if sup_idx is not None and len(sup_idx) >= MIN_EXEMPLARS:
                    mem_h = model.embed(x_tr[sup_idx])
                    pred_ctx = model.forward_with_context(
                        model.embed(x_va), mem_h, y_tr[sup_idx]).argmax(-1)
                    score = 0.5 * (score + _macro_f1(pred_ctx, y_va))
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)


class InContextZeroShotPredictor:
    """Wraps ContextPredictor + ContextMemoryBank, implementing Algorithm 2's
    predict/insert cycle including the k-window label-delay fallback."""

    def __init__(self, input_dim: int, label_delay_k: int = 0):
        self.model = ContextPredictor(input_dim)
        self.memory = ContextMemoryBank()
        self.label_delay_k = label_delay_k
        self._pending: deque = deque()  # (window_features, window_labels, insert_at_step)
        self._step = 0

    def pretrain(self, X_calib: np.ndarray, y_calib: np.ndarray, **kwargs) -> None:
        pretrain_context_predictor(self.model, X_calib, y_calib, **kwargs)

    @torch.no_grad()
    def predict(self, X_window: np.ndarray, concept_id: str) -> tuple[np.ndarray, np.ndarray, bool]:
        self.model.eval()
        x = torch.from_numpy(X_window).float()
        h = self.model.embed(x)
        use_fallback = not self.memory.ready(concept_id)
        if use_fallback:
            logits = self.model.forward_fallback(h)
        else:
            mem = self.memory.get(concept_id)
            mem_x = torch.from_numpy(np.stack([m[0] for m in mem])).float()
            mem_y = torch.tensor([m[1] for m in mem], dtype=torch.long)
            mem_h = self.model.embed(mem_x)
            logits = self.model.forward_with_context(h, mem_h, mem_y)
        probs = F.softmax(logits, dim=-1).numpy()
        y_pred = probs.argmax(axis=-1)
        y_proba_pos = probs[:, 1]
        return y_pred, y_proba_pos, use_fallback

    def observe(self, X_window: np.ndarray, y_window: np.ndarray, concept_id: str) -> None:
        """Step 8 of Algorithm 2: insert labels into memory, respecting the
        configured label-delay k (k=0 => immediate prequential reveal)."""
        self._step += 1
        if self.label_delay_k <= 0:
            self.memory.insert(concept_id, X_window, y_window)
            return
        self._pending.append((X_window, y_window, concept_id, self._step + self.label_delay_k))
        while self._pending and self._pending[0][3] <= self._step:
            X_d, y_d, cid_d, _ = self._pending.popleft()
            self.memory.insert(cid_d, X_d, y_d)
