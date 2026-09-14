"""Single-pass regeneration of every number and figure in Chapter 5.

Runs the same 5-seed, 7-configuration protocol as make_figures.py -- reusing
its loaders, VARIANTS list and figure functions, so the figures and the
tables cannot drift apart -- and additionally prints every aggregated
metric, including recovery speed and the stability-plasticity index that
Section 3.10 promises.

Emits, for each stream:
  * performance measures        -> Tables 5.1 / 5.5
  * operational measures        -> Tables 5.2 / 5.6
  * paired t-tests vs baselines -> Tables 5.3 / 5.7
plus the ceiling analysis (Table 5.4) and the frozen-vs-adapted comparison
(Section 5.4.1), and rewrites every PNG under figures/results/.
"""

import numpy as np

from .make_figures import (load_weather, load_cic_unsw, collect, VARIANTS,
                           W_WIN, C_WIN, OUT, fig_ceiling, draw_ceiling,
                           fig_baseline_comparison, fig_trace, fig_ablation,
                           fig_adaptation, fig_drift_confusion)
from ..evaluation.metrics import aggregate_runs, paired_significance

METHODS = [n for n, _ in VARIANTS] + ["ARF", "ADWIN+HT"]


def fmt(agg, key, dp=3):
    m, s = agg[key]
    return f"{m:.{dp}f} ({s:.{dp}f})" if m is not None else "n/a"


def report(name, raw):
    aggs = {m: aggregate_runs(raw[m]) for m in METHODS}

    print(f"\n=== {name}: PERFORMANCE (Tables 5.1 / 5.5) ===")
    print(f"{'Method':<22}{'Acc':>15}{'MacroF1':>15}{'VW':>15}{'RecSpd':>15}"
          f"{'Fgt':>15}{'SPI':>15}{'BWT':>15}{'DriftF1':>15}")
    for m in METHODS:
        a = aggs[m]
        print(f"{m:<22}{fmt(a,'accuracy'):>15}{fmt(a,'macro_f1'):>15}"
              f"{fmt(a,'vulnerability_window',2):>15}{fmt(a,'recovery_speed',2):>15}"
              f"{fmt(a,'forgetting'):>15}{fmt(a,'spi'):>15}{fmt(a,'bwt'):>15}"
              f"{fmt(a,'drift_type_f1'):>15}")

    print(f"\n=== {name}: OPERATIONAL (Tables 5.2 / 5.6) ===")
    print(f"{'Method':<22}{'FPR':>15}{'Latency ms':>17}{'Steps/win':>15}{'PeakMem':>15}")
    for m in METHODS:
        a = aggs[m]
        print(f"{m:<22}{fmt(a,'fpr'):>15}{fmt(a,'latency_ms',1):>17}"
              f"{fmt(a,'cost_steps',2):>15}{fmt(a,'memory_exemplars',0):>15}")

    print(f"\n=== {name}: PAIRED TESTS (Tables 5.3 / 5.7) ===")
    print(f"{'Comparison':<40}{'Measure':<10}{'MeanDiff':>12}{'t':>10}{'p':>10}{'Won':>6}")
    pairs = [("Framework against ensemble", "DriftFusion", "ARF"),
             ("Framework against tree", "DriftFusion", "ADWIN+HT"),
             ("Without context against ensemble", "-Context", "ARF"),
             ("Without context against tree", "-Context", "ADWIN+HT")]
    for label, a_name, b_name in pairs:
        for metric in ("accuracy", "macro_f1"):
            s = paired_significance(raw[a_name], raw[b_name], metric)
            if s["p"] is None:
                continue
            print(f"{label:<40}{metric:<10}{s['mean_diff']:>+12.4f}"
                  f"{s['t']:>10.3f}{s['p']:>10.4f}{s['wins']:>4} of 5")


def main():
    print("Weather: 5-seed collection...", flush=True)
    Xw, yw, ons_w = load_weather()
    raw_w, traces_w, _ = collect(Xw, yw, W_WIN, ons_w)
    report("NOAA Weather stream", raw_w)

    print("\nCIC-UNSW-NB15 (CIC-IDS-2024): 5-seed collection...", flush=True)
    Xc, yc, ons_c, dt_c, nc_c = load_cic_unsw()
    raw_c, traces_c, pred_c = collect(Xc, yc, C_WIN, ons_c, true_dt=dt_c)
    report("CIC-UNSW-NB15 (CIC-IDS-2024) stream", raw_c)

    print("\n=== CEILING ANALYSIS (Table 5.4) ===", flush=True)
    for stale in OUT.glob("*.png"):
        stale.unlink()
    lin, rf, mlp, maj = fig_ceiling(Xw, yw)
    df_acc = aggregate_runs(raw_w["DriftFusion"])["accuracy"][0]
    arf_acc = aggregate_runs(raw_w["ARF"])["accuracy"][0]
    draw_ceiling(maj, lin, arf_acc, rf, df_acc, mlp)
    for label, v in [("Majority class predictor", maj),
                     ("Logistic regression refitted each window", lin),
                     ("Adaptive random forest, streaming", arf_acc),
                     ("Random forest refitted each window on all history", rf),
                     ("Proposed framework, streaming", df_acc),
                     ("Prediction model architecture refitted on all history", mlp)]:
        print(f"  {label:<58}{v:.4f}")

    fig_baseline_comparison(raw_w, raw_c)
    fig_trace("weather", traces_w, ons_w,
              "NOAA Weather: per-window accuracy, seed 0 (grey = seasonal drift onsets)")
    fig_trace("cic_unsw", traces_c, ons_c,
              "CIC-UNSW-NB15 (CIC-IDS-2024): per-window accuracy, seed 0 "
              "(grey = injected drift onsets)")
    fig_ablation("weather", raw_w, "Ablation study — NOAA Weather stream (five seeds)")
    fig_ablation("cic_unsw", raw_c, "Ablation study — CIC-UNSW-NB15 (CIC-IDS-2024), five seeds")
    if pred_c is not None:
        fig_drift_confusion(dt_c, pred_c, nc_c)

    print("\n=== ADAPTATION EFFECT (Section 5.4.1) ===", flush=True)
    fig_adaptation(Xw, yw, ons_w)

    print(f"\nFigures written to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
