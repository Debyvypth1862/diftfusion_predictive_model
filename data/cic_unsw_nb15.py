"""
CIC-UNSW-NB15 (Canadian Institute for Cybersecurity, 2024) with controlled
drift injection.

This is the dataset the proposal's Table 7.2 is describing: a CIC 2024
release of CICFlowMeter flow features with binary benign/attack labels and
fine-grained attack-family sub-labels. It supersedes the CSE-CIC-IDS-2018
substitution used earlier.

Its decisive advantage is that it carries no timestamp, so the stream must
be *constructed* -- which is exactly what Section 7.3 prescribes: "the
detailed sub-labels will be retained as metadata for drift injection
purposes - different attack families can be selectively introduced or
withdrawn to create specific drift profiles". Building the stream from an
explicit schedule means the drift type of every window is known by
construction rather than inferred, so the Drift-Type Classification F1 of
Table 7.5 is measured against genuine ground truth instead of the coarse
attack-family proxy the 2018 data forced.

Files (as distributed):
    Data.csv   - 447,915 rows x 76 CICFlowMeter features
    Label.csv  - matching multi-class family labels, 0 = Benign
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "CIC-IDS-2024"

FAMILY_NAMES = {
    0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
    5: "Fuzzers", 6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms",
}

# Families with enough flows to sustain a multi-window regime.
USABLE_ATTACKS = ["Exploits", "Fuzzers", "Reconnaissance", "Generic", "DoS", "Shellcode"]


@dataclass
class Phase:
    """One segment of the constructed stream.

    kind      - 'stable' | 'sudden' | 'gradual' | 'incremental' | 'recurring'
    windows   - length of the phase in windows
    family    - attack family active during the phase (None => benign only)
    from_family - for 'gradual', the family being mixed out of
    attack_rate - proportion of attack flows per window
    """
    kind: str
    windows: int
    family: str | None = None
    from_family: str | None = None
    attack_rate: float = 0.3


# The schedule realises all four drift profiles named in Section 7.3:
# (a) abrupt insertion, (b) linear mixing, (c) cumulative perturbation,
# (d) reintroduction after a gap of multiple windows.
DEFAULT_SCHEDULE = [
    Phase("stable",      12, "Exploits",        attack_rate=0.30),
    Phase("sudden",       8, "DoS",             attack_rate=0.45),
    Phase("stable",       8, "DoS",             attack_rate=0.45),
    Phase("gradual",     10, "Fuzzers",         from_family="DoS", attack_rate=0.35),
    Phase("stable",       8, "Fuzzers",         attack_rate=0.35),
    Phase("incremental", 10, "Fuzzers",         attack_rate=0.35),
    Phase("stable",       8, "Reconnaissance",  attack_rate=0.30),
    Phase("recurring",    8, "DoS",             attack_rate=0.45),
    Phase("stable",       8, "DoS",             attack_rate=0.45),
    Phase("sudden",       8, "Generic",         attack_rate=0.40),
    Phase("stable",      10, "Generic",         attack_rate=0.40),
]

# A detected drift persists in the fingerprint buffer for several windows,
# so the onset label is held for this many windows before reverting to
# 'stable'. Matches the sequence length the drift classifier consumes.
ONSET_PERSISTENCE = 4

INCREMENTAL_MAGNITUDE = 0.02


def load_pools() -> tuple[dict[str, np.ndarray], list[str]]:
    """Load the dataset and split it into per-family feature pools."""
    data_path, label_path = RAW_DIR / "Data.csv", RAW_DIR / "Label.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected {data_path}")

    X = pd.read_csv(data_path, low_memory=False)
    y = pd.read_csv(label_path)["Label"].to_numpy()
    X.columns = [c.strip() for c in X.columns]

    feature_cols = list(X.columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    pools: dict[str, np.ndarray] = {}
    for code, name in FAMILY_NAMES.items():
        mask = (y == code)
        if mask.sum() > 0:
            pools[name] = X.to_numpy(dtype=np.float32)[mask]
    return pools, feature_cols


def _draw(pool: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample n rows with replacement (pools are finite and reused across
    phases, notably for the recurring profile)."""
    idx = rng.randint(0, len(pool), size=n)
    return pool[idx]


def build_injected_stream(schedule: list[Phase] | None = None, window_size: int = 500,
                           seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Construct the stream from the schedule.

    Returns (X, y, drift_types, families, phase_kinds) where drift_types
    holds the ground-truth drift category of each window.
    """
    schedule = schedule or DEFAULT_SCHEDULE
    pools, _ = load_pools()
    rng = np.random.RandomState(seed)
    benign = pools["Benign"]

    Xs, ys, drift_types, families = [], [], [], []
    incremental_step = 0

    for phase in schedule:
        for w in range(phase.windows):
            n_attack = int(window_size * phase.attack_rate)
            n_benign = window_size - n_attack

            if phase.kind == "gradual" and phase.from_family:
                # (b) linear mixing: the new family displaces the old one
                # progressively across the transition.
                frac_new = (w + 1) / phase.windows
                n_new = int(n_attack * frac_new)
                atk = np.concatenate([
                    _draw(pools[phase.family], n_new, rng),
                    _draw(pools[phase.from_family], n_attack - n_new, rng),
                ]) if n_attack else np.empty((0, benign.shape[1]), dtype=np.float32)
            else:
                atk = (_draw(pools[phase.family], n_attack, rng)
                       if phase.family else np.empty((0, benign.shape[1]), dtype=np.float32))

            ben = _draw(benign, n_benign, rng)
            Xw = np.concatenate([ben, atk]).astype(np.float32)
            yw = np.concatenate([np.zeros(len(ben), dtype=np.int64),
                                  np.ones(len(atk), dtype=np.int64)])

            if phase.kind == "incremental":
                # (c) small cumulative perturbation of the feature
                # distribution, accumulating window over window.
                incremental_step += 1
                scale = np.abs(Xw).mean(axis=0) + 1e-8
                drift_vec = rng.normal(0, 1, Xw.shape[1])
                Xw = Xw + (drift_vec * scale * INCREMENTAL_MAGNITUDE * incremental_step)
            else:
                incremental_step = 0

            order = rng.permutation(len(Xw))
            Xs.append(Xw[order]); ys.append(yw[order])

            # Ground-truth drift label: the onset category is held for
            # ONSET_PERSISTENCE windows, matching how long the change
            # remains visible in the fingerprint buffer.
            if phase.kind == "stable":
                drift_types.append("stable")
            elif phase.kind in ("sudden", "recurring"):
                drift_types.append(phase.kind if w < ONSET_PERSISTENCE else "stable")
            else:  # gradual, incremental persist for the whole phase
                drift_types.append(phase.kind)
            families.append(phase.family or "Benign")

    X = np.concatenate(Xs); y = np.concatenate(ys)
    phase_kinds = [p.kind for p in schedule for _ in range(p.windows)]
    return X, y, drift_types, families, phase_kinds


if __name__ == "__main__":
    X, y, dt, fam, _ = build_injected_stream()
    from collections import Counter
    print(f"CIC-UNSW-NB15 injected stream: {X.shape[0]} flows, {X.shape[1]} features")
    print(f"attack rate: {y.mean():.3f}   windows: {len(dt)}")
    print(f"drift-type ground truth: {dict(Counter(dt))}")
    print(f"families: {dict(Counter(fam))}")
