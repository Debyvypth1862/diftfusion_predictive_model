"""
Shared pre-processing pipeline (proposal Section 7.3): mRMR feature
selection, streaming-safe running normalisation, and the controlled
drift-injection protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# ---------------------------------------------------------------------------
# Feature cleaning (Section 7.3, steps i-iii)
# ---------------------------------------------------------------------------

def clean_features(X: pd.DataFrame, calibration_frac: float = 0.2) -> pd.DataFrame:
    """Constant/quasi-constant removal, missing-value imputation, and
    infinite-value treatment, using an initial calibration slice to
    decide which columns to drop (mirrors the proposal's steps i-iii)."""
    n_calib = max(50, int(len(X) * calibration_frac))
    calib = X.iloc[:n_calib]

    keep = calib.columns[calib.var(numeric_only=True) > 1e-6]
    X = X[keep].copy()

    missing_frac = X.isna().mean()
    X = X.drop(columns=missing_frac[missing_frac > 0.5].index)

    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().any():
            non_inf_99 = X[col].dropna().quantile(0.99)
            X[col] = X[col].fillna(non_inf_99)
        running_median = X[col].expanding(min_periods=1).median()
        X[col] = X[col].fillna(running_median)
    return X


# ---------------------------------------------------------------------------
# mRMR feature selection (Section 7.3, step iv)
# ---------------------------------------------------------------------------

def mrmr_select(X: pd.DataFrame, y: np.ndarray, k: int = 40,
                 calibration_rows: int = 5000, random_state: int = 0) -> list[str]:
    """Minimum-Redundancy-Maximum-Relevance forward selection (Peng, Long
    and Ding, 2005), computed on a calibration subsample for tractability:
    at each step pick argmax[ I(Xj;Y) - mean_i I(Xj;Xi) ] over already
    selected S, giving the O(K*D) cost the proposal specifies."""
    if len(X) > calibration_rows:
        rng = np.random.RandomState(random_state)
        idx = np.sort(rng.choice(len(X), calibration_rows, replace=False))
        Xc, yc = X.iloc[idx].to_numpy(), y[idx]
    else:
        Xc, yc = X.to_numpy(), y

    cols = list(X.columns)
    relevance = mutual_info_classif(Xc, yc, random_state=random_state)
    relevance = dict(zip(cols, relevance))

    selected: list[str] = []
    redundancy_sum = {c: 0.0 for c in cols}
    remaining = set(cols)

    k = min(k, len(cols))
    for _ in range(k):
        if not selected:
            best = max(remaining, key=lambda c: relevance[c])
        else:
            last = selected[-1]
            last_col = Xc[:, cols.index(last)].reshape(-1, 1)
            for c in list(remaining):
                mi = mutual_info_regression(last_col, Xc[:, cols.index(c)],
                                             random_state=random_state)[0]
                redundancy_sum[c] += mi
            best = max(remaining, key=lambda c: relevance[c] - redundancy_sum[c] / len(selected))
        selected.append(best)
        remaining.remove(best)
    return selected


class MRMRStabilityMonitor:
    """Step (v) of Section 7.3: "mRMR rankings are recomputed every 50
    windows during streaming operation; if the Kendall tau correlation
    between the current and original ranking drops below 0.7, the feature
    set is refreshed by re-running mRMR on the recent calibration buffer."

    Feature selection made once on a calibration slice silently assumes the
    informative features never change -- the exact assumption a drifting
    stream violates, and the one this check exists to catch.
    """

    def __init__(self, original_ranking: list[str], recompute_every: int = 50,
                 tau_threshold: float = 0.7, buffer_windows: int = 10,
                 k: int = 40, random_state: int = 0):
        self.original_ranking = list(original_ranking)
        self.recompute_every = recompute_every
        self.tau_threshold = tau_threshold
        self.buffer_windows = buffer_windows
        self.k = k
        self.random_state = random_state
        self.current_ranking = list(original_ranking)
        self._buffer_X: list[np.ndarray] = []
        self._buffer_y: list[np.ndarray] = []
        self.window_count = 0
        self.checks: list[dict] = []

    def observe(self, X_window: np.ndarray, y_window: np.ndarray) -> dict | None:
        """Feed one window; returns a report on windows where the check runs."""
        from scipy.stats import kendalltau

        self.window_count += 1
        self._buffer_X.append(X_window)
        self._buffer_y.append(y_window)
        if len(self._buffer_X) > self.buffer_windows:
            self._buffer_X.pop(0)
            self._buffer_y.pop(0)

        if self.window_count % self.recompute_every != 0:
            return None

        X_buf = pd.DataFrame(np.concatenate(self._buffer_X), columns=self.original_ranking)
        y_buf = np.concatenate(self._buffer_y)
        if len(np.unique(y_buf)) < 2:
            return None

        new_ranking = mrmr_select(X_buf, y_buf, k=self.k, random_state=self.random_state)

        pos_cur = {f: i for i, f in enumerate(self.current_ranking)}
        pos_new = {f: i for i, f in enumerate(new_ranking)}
        shared = [f for f in self.current_ranking if f in pos_new]
        if len(shared) < 2:
            tau = 0.0
        else:
            tau = float(kendalltau([pos_cur[f] for f in shared],
                                    [pos_new[f] for f in shared]).statistic)

        refreshed = tau < self.tau_threshold
        if refreshed:
            self.current_ranking = new_ranking

        report = {"window": self.window_count, "kendall_tau": tau, "refreshed": refreshed,
                  "n_shared": len(shared)}
        self.checks.append(report)
        return report


# ---------------------------------------------------------------------------
# Streaming-safe running z-score normalisation
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """Welford-style running mean/std, updated window-by-window. Window t
    is normalised using statistics accumulated over windows [0, t-1] only,
    preventing the leakage a whole-dataset scaler would introduce."""

    def __init__(self, n_features: int):
        self.n = 0
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)

    def stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n < 2:
            return self.mean, np.ones_like(self.mean)
        var = self.M2 / (self.n - 1)
        return self.mean, np.sqrt(np.maximum(var, 1e-12))

    def transform(self, window: np.ndarray) -> np.ndarray:
        if self.n == 0:
            mu, sigma = window.mean(axis=0), window.std(axis=0) + 1e-8
        else:
            mu, sigma = self.stats()
        return (window - mu) / sigma

    def update(self, window: np.ndarray) -> None:
        for row in window:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.M2 += delta * delta2


# ---------------------------------------------------------------------------
# Drift injection protocol (Section 7.3)
# ---------------------------------------------------------------------------

@dataclass
class DriftEvent:
    kind: str            # 'sudden' | 'gradual' | 'incremental' | 'recurring'
    start_row: int
    end_row: int
    description: str = ""


def inject_sudden_drift(X: pd.DataFrame, y: np.ndarray, at_row: int,
                         new_family_mask: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Abruptly insert samples from `new_family_mask` at `at_row`, no transition."""
    new_rows = X[new_family_mask]
    new_y = y[new_family_mask]
    X_out = pd.concat([X.iloc[:at_row], new_rows, X.iloc[at_row:]], ignore_index=True)
    y_out = np.concatenate([y[:at_row], new_y, y[at_row:]])
    return X_out, y_out


def inject_gradual_drift(old_mask: np.ndarray, new_mask: np.ndarray,
                          transition_len: int, rng: np.random.RandomState) -> np.ndarray:
    """Return a boolean array of length `transition_len` selecting, at each
    step, the old (False) or new (True) regime with linearly increasing
    probability of the new regime -- used to interleave two label pools."""
    probs = np.linspace(0.0, 1.0, transition_len)
    return rng.random(transition_len) < probs


def inject_incremental_drift(X: np.ndarray, magnitude: float = 0.01,
                              rng: np.random.RandomState | None = None) -> np.ndarray:
    """Small cumulative additive perturbation to feature distributions."""
    rng = rng or np.random.RandomState(0)
    drift_vector = rng.normal(0, magnitude, size=X.shape[1])
    cumulative = np.outer(np.arange(len(X)), drift_vector)
    return X + cumulative


def build_recurring_schedule(family_order: list[str], gap_windows: int,
                              repeats: int = 2) -> list[str]:
    """Reintroduce earlier families after a quiescent gap: A, B, ..., A."""
    schedule = list(family_order)
    for _ in range(repeats - 1):
        schedule += ["stable"] * 0 + family_order[:1]
    return schedule
