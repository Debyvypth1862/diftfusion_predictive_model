"""
Experiment 2 (secondary, domain-specific benchmark): CSE-CIC-IDS-2018,
five days each dominated by a distinct attack family, giving natural
(not synthetically injected) drift across day boundaries. Window size
500, matching the proposal's CIC default.

The dataset carries no official "drift type" annotation (nothing does --
that's exactly why Module 1 is meta-trained on synthetic sequences). For
the Drift-Type Classification F1 metric we need *some* ground truth, so
each day's dominant attack family is mapped to the closest of the five
proposal categories by its real-world behavioural profile:
    DDoS          -> sudden       (abrupt, high-volume flood)
    Brute-force   -> gradual      (prolonged credential guessing that
                                    tries to stay under detection thresholds
                                    -- the proposal's own definition of
                                    "gradual evasion")
    Web attack    -> incremental  (sparse, scattered SQLi/XSS/brute-force
                                    attempts embedded in mostly-benign traffic)
    Infiltration  -> incremental  (stealthy, low-volume lateral movement)
    Botnet        -> recurring    (periodic C2 beaconing)
This is a coarse, documented proxy, not ground truth handed to us by CIC.

Run across multiple seeds and reported as mean +/- std (data loading and
mRMR selection happen once and are shared across seeds -- they aren't
stochastic in any seed-sensitive way that matters here).
"""

from __future__ import annotations

import time

import numpy as np

from ..data.cic_ids2018 import build_cic_stream, DAY_ATTACK_FAMILY, DAY_FILES
from ..data.preprocessing import clean_features, mrmr_select, MRMRStabilityMonitor
from ..models.drift_classifier import train_drift_classifier
from ..baselines.streaming_baselines import ARFBaseline, AdwinHoeffdingTreeBaseline
from ..evaluation.prequential import run_driftfusion, run_baseline, make_windows, _calibration_split
from ..evaluation.metrics import aggregate_runs

WINDOW_SIZE = 500
MRMR_K = 40
SEEDS = (0, 1, 2)

FAMILY_TO_DRIFT_TYPE = {
    "ddos": "sudden",
    "brute_force": "gradual",
    "web_attack": "incremental",
    "infiltration": "incremental",
    "botnet": "recurring",
}


def build_true_drift_types(df, window_size: int) -> list[str]:
    n_windows = len(df) // window_size
    types = []
    for i in range(n_windows):
        chunk = df["attack_family_source_day"].iloc[i * window_size:(i + 1) * window_size]
        dominant_family = chunk.mode().iloc[0]
        types.append(FAMILY_TO_DRIFT_TYPE[dominant_family])
    return types


def main():
    t0 = time.time()
    df, raw_feature_cols = build_cic_stream()
    print(f"CIC-IDS-2018 stream: {len(df)} rows, {len(raw_feature_cols)} raw features, "
          f"attack rate={df['label'].mean():.3f}")

    X_raw = clean_features(df[raw_feature_cols])
    y = df["label"].to_numpy(dtype=np.int64)

    print(f"Selecting top {MRMR_K} features via mRMR...")
    selected = mrmr_select(X_raw, y, k=MRMR_K)
    X = X_raw[selected].to_numpy(dtype=np.float32)
    print(f"Selected features: {selected}")

    true_drift_types = build_true_drift_types(df, WINDOW_SIZE)
    print(f"n_windows={len(X)//WINDOW_SIZE}, seeds={SEEDS}")

    # Section 7.3 step (v): mRMR stability verification during streaming.
    monitor = MRMRStabilityMonitor(original_ranking=selected, recompute_every=50,
                                    tau_threshold=0.7, k=MRMR_K)
    X_sel_df = X_raw[selected]
    for i in range(len(X) // WINDOW_SIZE):
        sl = slice(i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE)
        monitor.observe(X_sel_df.iloc[sl].to_numpy(dtype=np.float32), y[sl])
    if monitor.checks:
        print("mRMR stability checks (step v):")
        for c in monitor.checks:
            print(f"  window {c['window']:4d}  kendall_tau={c['kendall_tau']:.3f}  "
                  f"refresh_triggered={c['refreshed']}")
    else:
        print("mRMR stability: no checkpoints reached")

    windows = make_windows(X, y, WINDOW_SIZE)
    calib_windows, _ = _calibration_split(windows, calib_frac=0.2, min_calib_windows=2)
    n_calib = len(calib_windows)
    external_onsets = [i for i, t in enumerate(true_drift_types[n_calib:])
                        if i == 0 or true_drift_types[n_calib + i] != true_drift_types[n_calib + i - 1]]

    raw_results = {name: [] for name in
                   ["DriftFusion (full)", "DriftFusion (label-delay k=3)",
                    "Ablation -Fingerprint", "Ablation -Context",
                    "Ablation -EWC", "ARF", "ADWIN+HT"]}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        drift_clf = train_drift_classifier(verbose=False, seed=seed)

        r, pred_types = run_driftfusion(X, y, WINDOW_SIZE, drift_clf,
                                         true_drift_types=true_drift_types, seed=seed)
        raw_results["DriftFusion (full)"].append(r)
        from collections import Counter
        print(f"  drift types detected: {dict(Counter(pred_types))}")

        # Proposal Section 7.4: 3-tier label-delay strategy, k=3 default.
        r, _ = run_driftfusion(X, y, WINDOW_SIZE, drift_clf, label_delay_k=3,
                                true_drift_types=true_drift_types, seed=seed)
        raw_results["DriftFusion (label-delay k=3)"].append(r)

        r, _ = run_driftfusion(X, y, WINDOW_SIZE, drift_clf, use_fingerprinting=False,
                                true_drift_types=true_drift_types, seed=seed)
        raw_results["Ablation -Fingerprint"].append(r)
        r, _ = run_driftfusion(X, y, WINDOW_SIZE, drift_clf, use_context_memory=False,
                                true_drift_types=true_drift_types, seed=seed)
        raw_results["Ablation -Context"].append(r)
        r, _ = run_driftfusion(X, y, WINDOW_SIZE, drift_clf, use_ewc=False,
                                true_drift_types=true_drift_types, seed=seed)
        raw_results["Ablation -EWC"].append(r)

        raw_results["ARF"].append(run_baseline(ARFBaseline(seed=seed), X, y, WINDOW_SIZE,
                                                external_onsets=external_onsets))
        raw_results["ADWIN+HT"].append(run_baseline(AdwinHoeffdingTreeBaseline(), X, y, WINDOW_SIZE,
                                                     external_onsets=external_onsets))
        print(f"  done seed {seed} ({time.time()-t0:.1f}s elapsed)")

    results = {name: aggregate_runs(runs) for name, runs in raw_results.items()}

    def fmt(r, k):
        m, s = r[k]
        return f"{m:.3f}+/-{s:.3f}" if m is not None else "n/a"

    print("\n--- Table 7.5 metrics ---")
    print(f"{'Method':<30}{'Acc':>14}{'F1':>14}{'VW':>14}{'RecSpd':>14}{'Fgt':>14}{'SPI':>14}{'BWT':>14}{'DriftF1':>14}")
    for name, r in results.items():
        print(f"{name:<30}{fmt(r,'accuracy'):>14}{fmt(r,'macro_f1'):>14}{fmt(r,'vulnerability_window'):>14}"
              f"{fmt(r,'recovery_speed'):>14}{fmt(r,'forgetting'):>14}{fmt(r,'spi'):>14}"
              f"{fmt(r,'bwt'):>14}{fmt(r,'drift_type_f1'):>14}")

    print("\n--- Figure 7.4 operational metrics ---")
    print(f"{'Method':<30}{'FPR':>14}{'Latency(ms)':>16}{'Cost(steps)':>14}{'Memory(ex)':>14}")
    for name, r in results.items():
        print(f"{name:<30}{fmt(r,'fpr'):>14}{fmt(r,'latency_ms'):>16}"
              f"{fmt(r,'cost_steps'):>14}{fmt(r,'memory_exemplars'):>14}")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
