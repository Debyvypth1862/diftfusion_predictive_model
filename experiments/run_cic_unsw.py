"""
Experiment 3: CIC-UNSW-NB15 (CIC-IDS-2024) with controlled drift injection.

This is the closest available realisation of the proposal's intended
secondary benchmark. Two properties make it stronger evidence than the
CSE-CIC-IDS-2018 run:

1. The drift schedule is constructed, so every window's drift category is
   known by construction. Drift-Type Classification F1 (Table 7.5) is
   therefore measured against real ground truth rather than the
   attack-family proxy the 2018 data required, giving RQ1 a sound test.
2. Drift onsets are likewise known exactly, so Vulnerability Window and
   Recovery Speed are scored over the true injected episodes for every
   method -- removing the last place where onset definition could favour
   one method over another.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from ..data.cic_unsw_nb15 import build_injected_stream, DEFAULT_SCHEDULE
from ..data.preprocessing import clean_features, mrmr_select, MRMRStabilityMonitor
from ..models.drift_classifier import train_drift_classifier
from ..baselines.streaming_baselines import ARFBaseline, AdwinHoeffdingTreeBaseline
from ..evaluation.prequential import (run_driftfusion, run_baseline,
                                       make_windows, _calibration_split)
from ..evaluation.metrics import aggregate_runs, paired_significance

import pandas as pd

WINDOW_SIZE = 500
MRMR_K = 40
SEEDS = (0, 1, 2, 3, 4)


def main():
    t0 = time.time()
    X_raw_arr, y, drift_types, families, _ = build_injected_stream(window_size=WINDOW_SIZE, seed=0)
    print(f"CIC-UNSW-NB15 (CIC-IDS-2024) injected stream: {len(X_raw_arr)} flows, "
          f"{X_raw_arr.shape[1]} raw features, attack rate={y.mean():.3f}")
    print(f"ground-truth drift schedule: {dict(Counter(drift_types))}")

    X_df = pd.DataFrame(X_raw_arr, columns=[f"f{i}" for i in range(X_raw_arr.shape[1])])
    X_clean = clean_features(X_df)
    print(f"after cleaning steps (i)-(iii): {X_clean.shape[1]} features")

    selected = mrmr_select(X_clean, y, k=MRMR_K)
    X = X_clean[selected].to_numpy(dtype=np.float32)
    print(f"mRMR selected {len(selected)} features")

    n_windows = len(X) // WINDOW_SIZE
    windows = make_windows(X, y, WINDOW_SIZE)
    calib_windows, _ = _calibration_split(windows, 0.2, 2)
    n_calib = len(calib_windows)

    # True injected onsets: windows where the drift category changes away
    # from 'stable'. Shared by every method.
    eval_dt = drift_types[n_calib:]
    onsets = [i for i in range(1, len(eval_dt))
              if eval_dt[i] != "stable" and eval_dt[i - 1] == "stable"]
    print(f"n_windows={n_windows}, calibration={n_calib}, "
          f"true injected drift episodes in eval slice={len(onsets)}")

    monitor = MRMRStabilityMonitor(original_ranking=selected, recompute_every=25,
                                    tau_threshold=0.7, k=MRMR_K)
    X_sel = X_clean[selected]
    for i in range(n_windows):
        sl = slice(i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE)
        monitor.observe(X_sel.iloc[sl].to_numpy(dtype=np.float32), y[sl])
    if monitor.checks:
        print("mRMR stability checks (Section 7.3 step v):")
        for c in monitor.checks:
            print(f"  window {c['window']:4d}  kendall_tau={c['kendall_tau']:+.3f}  "
                  f"refresh_triggered={c['refreshed']}")

    raw_results = {n: [] for n in
                   ["DriftFusion (full)", "DriftFusion (label-delay k=3)",
                    "Ablation -Fingerprint", "Ablation -Context", "Ablation -EWC",
                    "ARF", "ADWIN+HT"]}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        clf = train_drift_classifier(verbose=False, seed=seed)

        r, pred = run_driftfusion(X, y, WINDOW_SIZE, clf, true_drift_types=drift_types,
                                   seed=seed, external_onsets=onsets)
        raw_results["DriftFusion (full)"].append(r)
        print(f"  predicted: {dict(Counter(pred))}")
        print(f"  true     : {dict(Counter(eval_dt))}")
        print(f"  drift-type F1={r['drift_type_f1']:.3f}  acc={r['accuracy']:.3f}")

        r, _ = run_driftfusion(X, y, WINDOW_SIZE, clf, true_drift_types=drift_types,
                                label_delay_k=3, seed=seed, external_onsets=onsets)
        raw_results["DriftFusion (label-delay k=3)"].append(r)
        r, _ = run_driftfusion(X, y, WINDOW_SIZE, clf, true_drift_types=drift_types,
                                use_fingerprinting=False, seed=seed, external_onsets=onsets)
        raw_results["Ablation -Fingerprint"].append(r)
        r, _ = run_driftfusion(X, y, WINDOW_SIZE, clf, true_drift_types=drift_types,
                                use_context_memory=False, seed=seed, external_onsets=onsets)
        raw_results["Ablation -Context"].append(r)
        r, _ = run_driftfusion(X, y, WINDOW_SIZE, clf, true_drift_types=drift_types,
                                use_ewc=False, seed=seed, external_onsets=onsets)
        raw_results["Ablation -EWC"].append(r)

        raw_results["ARF"].append(run_baseline(ARFBaseline(seed=seed), X, y, WINDOW_SIZE,
                                                external_onsets=onsets))
        raw_results["ADWIN+HT"].append(run_baseline(AdwinHoeffdingTreeBaseline(), X, y,
                                                     WINDOW_SIZE, external_onsets=onsets))
        print(f"  done ({time.time()-t0:.0f}s elapsed)")

    results = {k: aggregate_runs(v) for k, v in raw_results.items()}

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

    # Significance testing against both required baselines. Margins here are
    # small relative to seed variance, so a difference in means is not
    # reported as a result unless a paired test over seeds supports it.
    print("\n--- Paired significance tests vs baselines (n={} seeds) ---".format(len(SEEDS)))
    print(f"{'Comparison':<44}{'metric':>10}{'mean diff':>12}{'t':>8}{'p':>9}{'wins':>7}{'':>4}")
    for method in ["DriftFusion (full)", "DriftFusion (label-delay k=3)", "Ablation -Context"]:
        for base in ["ARF", "ADWIN+HT"]:
            for metric in ["accuracy", "macro_f1"]:
                s = paired_significance(raw_results[method], raw_results[base], metric)
                if s["p"] is None:
                    continue
                mark = "SIG" if s["p"] < 0.05 and s["mean_diff"] > 0 else (
                       "sig-neg" if s["p"] < 0.05 else "ns")
                print(f"{method+' vs '+base:<44}{metric:>10}{s['mean_diff']:>+12.4f}"
                      f"{s['t']:>8.3f}{s['p']:>9.4f}{s['wins']:>4}/{s['n']:<3}{mark:>4}")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
