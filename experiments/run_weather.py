"""
Experiment 1 (primary benchmark): NOAA weather stream at the proposal's
specified window size of 250 (Section 7.3).

The stream is the full multi-decade NOAA daily record for the station
rather than the 2025-only slice of Table 7.1 -- see
data/noaa_weather.build_weather_stream_full() for why the two parts of
the proposal cannot both be satisfied by a 365-row slice, and why the
full record is the configuration that satisfies the window size, the
"standard benchmark" claim, and the stated drift characteristics.

Run across multiple seeds and reported as mean +/- std, per the
proposal's own Phase 3 commitment to "multiple random seeds for
statistical reliability".
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from ..data.noaa_weather import build_weather_stream_full
from ..data.preprocessing import clean_features
from ..models.drift_classifier import train_drift_classifier
from ..baselines.streaming_baselines import ARFBaseline, AdwinHoeffdingTreeBaseline
from ..evaluation.prequential import (run_driftfusion, run_baseline,
                                       make_windows, _calibration_split)
from ..evaluation.metrics import aggregate_runs, paired_significance

WINDOW_SIZE = 250
SEEDS = (0, 1, 2, 3, 4)

# Meteorological seasons, used to define drift onsets independently of any
# model's own detections (see seasonal_drift_onsets).
_SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                    6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def seasonal_drift_onsets(df, window_size: int, n_calib_windows: int) -> list[int]:
    """Drift onsets defined by meteorological season change.

    Vulnerability Window and Recovery Speed must be measured over the same
    episodes for every method, otherwise they are not comparable: scored
    against self-detected onsets, a method that never reports drift records
    VW=0 by construction, which reads as perfect resilience when it
    actually means the detector was silent. Seasons give a model-
    independent onset definition, and the proposal itself identifies
    seasonal transitions as the Weather stream's principal drift mechanism.
    Indices are returned relative to the start of the evaluation slice.
    """
    n_windows = len(df) // window_size
    seasons = []
    for i in range(n_windows):
        months = df["date"].iloc[i * window_size:(i + 1) * window_size].dt.month
        seasons.append(_SEASON_OF_MONTH[int(months.mode().iloc[0])])
    eval_seasons = seasons[n_calib_windows:]
    return [i for i in range(1, len(eval_seasons))
            if eval_seasons[i] != eval_seasons[i - 1]]


def main():
    t0 = time.time()
    df = build_weather_stream_full()
    feature_cols = [c for c in df.columns if c not in ("window_index", "date", "rain")]
    # Section 7.3: "For the Weather stream (8 features), steps (i)-(iii)
    # below are applied but mRMR selection is bypassed as all features are
    # retained." Steps (i)-(iii) are constant/quasi-constant removal,
    # missing-value filtering and infinite-value treatment.
    X_clean = clean_features(df[feature_cols])
    dropped = set(feature_cols) - set(X_clean.columns)
    if dropped:
        print(f"  preprocessing (i)-(iii) dropped: {sorted(dropped)}")
    X = X_clean.to_numpy(dtype=np.float32)
    y = df["rain"].to_numpy(dtype=np.int64)
    print(f"Weather stream: {len(df)} rows, {X.shape[1]} features, rain rate={y.mean():.3f}, "
          f"n_windows={len(X)//WINDOW_SIZE}, seeds={SEEDS}")

    windows = make_windows(X, y, WINDOW_SIZE)
    calib_windows, _ = _calibration_split(windows, calib_frac=0.2, min_calib_windows=2)
    onsets = seasonal_drift_onsets(df, WINDOW_SIZE, len(calib_windows))
    print(f"seasonal drift onsets (shared by all methods): {len(onsets)} episodes")

    raw_results = {name: [] for name in
                   ["DriftFusion (full)", "DriftFusion (label-delay k=3)",
                    "Ablation -Fingerprint", "Ablation -Context",
                    "Ablation -EWC", "ARF", "ADWIN+HT"]}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        drift_clf = train_drift_classifier(verbose=False, seed=seed)

        r, pred = run_driftfusion(X, y, WINDOW_SIZE, drift_clf, seed=seed,
                                   external_onsets=onsets)
        raw_results["DriftFusion (full)"].append(r)
        print(f"  drift types detected: {dict(Counter(pred))}")

        # Proposal Section 7.4 specifies a 3-tier label-delay strategy with
        # k=3 windows as the default operational assumption.
        r, _ = run_driftfusion(X, y, WINDOW_SIZE, drift_clf, label_delay_k=3, seed=seed,
                                external_onsets=onsets)
        raw_results["DriftFusion (label-delay k=3)"].append(r)

        r, _ = run_driftfusion(X, y, WINDOW_SIZE, drift_clf, use_fingerprinting=False,
                                seed=seed, external_onsets=onsets)
        raw_results["Ablation -Fingerprint"].append(r)
        r, _ = run_driftfusion(X, y, WINDOW_SIZE, drift_clf, use_context_memory=False,
                                seed=seed, external_onsets=onsets)
        raw_results["Ablation -Context"].append(r)
        r, _ = run_driftfusion(X, y, WINDOW_SIZE, drift_clf, use_ewc=False,
                                seed=seed, external_onsets=onsets)
        raw_results["Ablation -EWC"].append(r)
        raw_results["ARF"].append(run_baseline(ARFBaseline(seed=seed), X, y, WINDOW_SIZE,
                                                external_onsets=onsets))
        raw_results["ADWIN+HT"].append(run_baseline(AdwinHoeffdingTreeBaseline(), X, y,
                                                     WINDOW_SIZE, external_onsets=onsets))
        print(f"  done seed {seed} ({time.time()-t0:.1f}s elapsed)")

    results = {name: aggregate_runs(runs) for name, runs in raw_results.items()}

    def fmt(r, k):
        m, s = r[k]
        return f"{m:.3f}+/-{s:.3f}" if m is not None else "n/a"

    print("\n--- Table 7.5 metrics ---")
    print(f"{'Method':<30}{'Acc':>14}{'F1':>14}{'VW':>14}{'RecSpd':>14}{'Fgt':>14}{'SPI':>14}{'BWT':>14}")
    for name, r in results.items():
        print(f"{name:<30}{fmt(r,'accuracy'):>14}{fmt(r,'macro_f1'):>14}{fmt(r,'vulnerability_window'):>14}"
              f"{fmt(r,'recovery_speed'):>14}{fmt(r,'forgetting'):>14}{fmt(r,'spi'):>14}{fmt(r,'bwt'):>14}")

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
