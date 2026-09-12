"""
Synthetic fingerprint-sequence generator for meta-training the Drift-Type
Classifier (Section 7.4 / Module 1). Real-world datasets have no ground-truth
drift-type annotations, so the classifier is trained entirely on procedurally
generated 10-step fingerprint sequences that embody each category's
characteristic temporal signature, with known labels.

Categories (fixed order used throughout the codebase):
    0 stable | 1 sudden_attack | 2 gradual_evasion | 3 incremental_shift | 4 recurring_campaign
"""

from __future__ import annotations

import numpy as np

from ..models.fingerprint import FINGERPRINT_DIM

CATEGORIES = ["stable", "sudden", "gradual", "incremental", "recurring"]
SEQ_LEN = 10

# Plausible per-dimension ranges for a fingerprint vector, used to sample
# random "regime" base vectors (order matches FINGERPRINT_NAMES).
DIM_RANGES = np.array([
    [0.02, 0.45],   # window_error_rate
    [0.0, 1.0],     # confidence_entropy (normalised by log(n_bins), so in [0,1])
    [0.05, 0.90],   # label_balance
    [0.05, 2.50],   # feature_drift_magnitude
    [0.02, 0.30],   # bocpd_changepoint_prob (baseline/stable level)
    [0.0, 1.0],     # run_length_entropy (normalised by log|R|, so in [0,1])
    [0.0, 1.0],     # normalized_modal_run_length
    [-0.05, 0.05],  # error_trend_slope
    [0.0, 0.15],    # confidence_drop
    [0.0, 0.6],     # label_kl_divergence
], dtype=np.float32)

NOISE_STD = 0.03


def _random_regime(rng: np.random.RandomState) -> np.ndarray:
    lo, hi = DIM_RANGES[:, 0], DIM_RANGES[:, 1]
    return rng.uniform(lo, hi).astype(np.float32)


def _add_noise(vec: np.ndarray, rng: np.random.RandomState, scale: float = NOISE_STD) -> np.ndarray:
    span = DIM_RANGES[:, 1] - DIM_RANGES[:, 0]
    return np.clip(vec + rng.normal(0, scale, size=vec.shape) * span,
                    DIM_RANGES[:, 0], DIM_RANGES[:, 1]).astype(np.float32)


def _changepoint_burst(step_offset: int, seq_len: int = SEQ_LEN) -> np.ndarray:
    """A one-hot-ish burst in dims [4,5,6] (BOCPD signals) at the moment a
    changepoint occurs, decaying over the following steps -- mirrors what
    OnlineBOCPD actually emits around a real changepoint."""
    burst = np.zeros((seq_len, FINGERPRINT_DIM), dtype=np.float32)
    for t in range(seq_len):
        d = t - step_offset
        if d < 0:
            continue
        decay = np.exp(-d / 3.0)
        burst[t, 4] += 0.6 * decay if d == 0 else 0.15 * decay
        burst[t, 5] -= 0.25 * decay   # entropy collapses at a changepoint ([0,1] scale)
        burst[t, 6] = min(burst[t, 6], d / max(seq_len, 1))
    return burst


def gen_stable(rng: np.random.RandomState) -> np.ndarray:
    base = _random_regime(rng)
    base[4] = rng.uniform(0.05, 0.30)   # low-to-moderate changepoint "noise floor"
    base[7] = rng.uniform(-0.01, 0.01)  # near-zero error slope
    seq = np.stack([_add_noise(base, rng, scale=0.015) for _ in range(SEQ_LEN)])
    # Modal run-length keeps growing steadily -- nothing resets.
    seq[:, 6] = np.clip(np.linspace(base[6], min(base[6] + 0.3, 1.0), SEQ_LEN)
                         + rng.normal(0, 0.02, SEQ_LEN), 0, 1)
    return seq


def gen_sudden(rng: np.random.RandomState) -> np.ndarray:
    t0 = rng.randint(3, 8)
    regime_a, regime_b = _random_regime(rng), _random_regime(rng)
    regime_b[0] = np.clip(regime_a[0] + rng.uniform(0.15, 0.3), *DIM_RANGES[0])  # error spikes up
    seq = np.zeros((SEQ_LEN, FINGERPRINT_DIM), dtype=np.float32)
    for t in range(SEQ_LEN):
        base = regime_b if t >= t0 else regime_a
        seq[t] = _add_noise(base, rng, scale=0.02)
    seq += _changepoint_burst(t0)
    # Modal run-length resets sharply at t0, then regrows.
    for t in range(SEQ_LEN):
        seq[t, 6] = max(0.0, (t - t0) / SEQ_LEN) if t >= t0 else seq[t, 6]
    seq[:, :] = np.clip(seq, DIM_RANGES[:, 0], DIM_RANGES[:, 1])
    return seq


def gen_gradual(rng: np.random.RandomState) -> np.ndarray:
    t0 = rng.randint(1, 4)
    transition = rng.randint(4, 8)
    regime_a, regime_b = _random_regime(rng), _random_regime(rng)
    regime_b[0] = np.clip(regime_a[0] + rng.uniform(0.1, 0.25), *DIM_RANGES[0])
    seq = np.zeros((SEQ_LEN, FINGERPRINT_DIM), dtype=np.float32)
    for t in range(SEQ_LEN):
        if t < t0:
            w = 0.0
        elif t >= t0 + transition:
            w = 1.0
        else:
            w = (t - t0) / transition
        base = (1 - w) * regime_a + w * regime_b
        seq[t] = _add_noise(base, rng, scale=0.02)
        # Rising, moderate changepoint mass across the whole transition
        # (not a single spike) -- the hallmark that separates gradual from sudden.
        if t0 <= t < t0 + transition:
            seq[t, 4] = np.clip(0.15 + 0.25 * w + rng.normal(0, 0.03), 0.02, 0.6)
            seq[t, 6] = np.clip(0.3 + 0.2 * (1 - w), 0, 1)
    return np.clip(seq, DIM_RANGES[:, 0], DIM_RANGES[:, 1])


def gen_incremental(rng: np.random.RandomState) -> np.ndarray:
    base = _random_regime(rng)
    drift_vec = rng.normal(0, 1, FINGERPRINT_DIM) * (DIM_RANGES[:, 1] - DIM_RANGES[:, 0]) * 0.015
    seq = np.zeros((SEQ_LEN, FINGERPRINT_DIM), dtype=np.float32)
    for t in range(SEQ_LEN):
        cum = base + drift_vec * t
        seq[t] = _add_noise(cum, rng, scale=0.02)
        seq[t, 4] = np.clip(0.06 + 0.01 * t + rng.normal(0, 0.02), 0.02, 0.35)  # slowly rising
        seq[t, 6] = np.clip(0.5 + 0.03 * t, 0, 1)  # keeps growing, no hard reset
    return np.clip(seq, DIM_RANGES[:, 0], DIM_RANGES[:, 1])


def gen_recurring(rng: np.random.RandomState) -> np.ndarray:
    regime_a, regime_b = _random_regime(rng), _random_regime(rng)
    boundaries = sorted(rng.choice(range(2, SEQ_LEN), size=2, replace=False))
    seq = np.zeros((SEQ_LEN, FINGERPRINT_DIM), dtype=np.float32)
    current = regime_a
    for t in range(SEQ_LEN):
        if t in boundaries:
            current = regime_b if np.array_equal(current, regime_a) else regime_a
        seq[t] = _add_noise(current, rng, scale=0.02)
        if t in boundaries:
            seq[t, 4] = np.clip(0.5 + rng.normal(0, 0.08), 0.2, 0.9)
            seq[t, 6] = rng.uniform(0.0, 0.1)
    return np.clip(seq, DIM_RANGES[:, 0], DIM_RANGES[:, 1])


GENERATORS = {
    "stable": gen_stable, "sudden": gen_sudden, "gradual": gen_gradual,
    "incremental": gen_incremental, "recurring": gen_recurring,
}


def build_synthetic_dataset(n_per_class: int = 2000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X, y = [], []
    for label_idx, name in enumerate(CATEGORIES):
        gen = GENERATORS[name]
        for _ in range(n_per_class):
            X.append(gen(rng))
            y.append(label_idx)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


if __name__ == "__main__":
    X, y = build_synthetic_dataset(n_per_class=200)
    print("Synthetic fingerprint dataset:", X.shape, y.shape)
    for i, name in enumerate(CATEGORIES):
        print(f"  {name}: {np.sum(y == i)} sequences")
