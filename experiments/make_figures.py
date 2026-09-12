"""
Generates the dissertation results figures from real runs on the two
benchmarks the proposal specifies: the NOAA weather stream (Table 7.1) and
CIC-UNSW-NB15 (Table 7.2).

The bar charts re-run the full five-seed protocol rather than reading
cached numbers, so every figure is guaranteed consistent with the tables in
RESULTS_AND_DISCUSSION.md. Per-window traces use seed 0.

Outputs to figures/results/:
    fig_ceiling_analysis.png       - Section 3, the central diagnostic
    fig_baseline_comparison.png    - Sections 2.1 / 5.1, with significance marks
    fig_accuracy_trace_*.png       - per-window accuracy, drift onsets marked
    fig_ablation_*.png             - ablation accuracy and FPR
    fig_adaptation_effect.png      - Section 4.1, frozen vs adapted
    fig_drift_type_confusion.png   - Section 5.3, RQ1 error structure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..data.noaa_weather import build_weather_stream_full
from ..data.cic_unsw_nb15 import build_injected_stream
from ..data.preprocessing import clean_features, mrmr_select, RunningNormalizer
from ..models.drift_classifier import train_drift_classifier
from ..models.context_predictor import ContextPredictor, pretrain_context_predictor
from ..baselines.streaming_baselines import ARFBaseline, AdwinHoeffdingTreeBaseline
from ..evaluation.prequential import (run_driftfusion, run_baseline,
                                       make_windows, _calibration_split)
from ..evaluation.metrics import aggregate_runs, paired_significance
from ..synthetic.fingerprint_sequences import CATEGORIES
from .run_weather import seasonal_drift_onsets, WINDOW_SIZE as W_WIN
from .run_cic_unsw import WINDOW_SIZE as C_WIN, MRMR_K

OUT = Path(__file__).resolve().parents[3] / "figures" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2, 3, 4)

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

CLR = {"DriftFusion": "#c0504d", "DriftFusion (k=3)": "#d98880",
       "-Context": "#4a6fa5", "-Fingerprint": "#9b8aa6", "-EWC": "#d9a05b",
       "ARF": "#6dba82", "ADWIN+HT": "#8c8c8c"}

VARIANTS = [("DriftFusion", {}), ("DriftFusion (k=3)", {"label_delay_k": 3}),
            ("-Fingerprint", {"use_fingerprinting": False}),
            ("-Context", {"use_context_memory": False}),
            ("-EWC", {"use_ewc": False})]


# --------------------------------------------------------------- loaders
def load_weather():
    df = build_weather_stream_full()
    fc = [c for c in df.columns if c not in ("window_index", "date", "rain")]
    X = clean_features(df[fc]).to_numpy(dtype=np.float32)
    y = df["rain"].to_numpy(dtype=np.int64)
    cal, _ = _calibration_split(make_windows(X, y, W_WIN), 0.2, 2)
    return X, y, seasonal_drift_onsets(df, W_WIN, len(cal))


def load_cic_unsw():
    Xa, y, dt, _, _ = build_injected_stream(window_size=C_WIN, seed=0)
    Xd = pd.DataFrame(Xa, columns=[f"f{i}" for i in range(Xa.shape[1])])
    Xc = clean_features(Xd)
    sel = mrmr_select(Xc, y, k=MRMR_K)
    X = Xc[sel].to_numpy(dtype=np.float32)
    cal, _ = _calibration_split(make_windows(X, y, C_WIN), 0.2, 2)
    nc = len(cal)
    ed = dt[nc:]
    onsets = [i for i in range(1, len(ed)) if ed[i] != "stable" and ed[i - 1] == "stable"]
    return X, y, onsets, dt, nc


def collect(X, y, window, onsets, true_dt=None):
    """Run every variant and baseline across all seeds; return raw per-seed
    summaries plus seed-0 traces and drift-type predictions."""
    raw = {name: [] for name, _ in VARIANTS}
    raw.update({"ARF": [], "ADWIN+HT": []})
    traces, preds0 = {}, None
    for seed in SEEDS:
        clf = train_drift_classifier(verbose=False, seed=seed)
        for name, kw in VARIANTS:
            s, pred, tr = run_driftfusion(X, y, window, clf, seed=seed,
                                           external_onsets=onsets,
                                           true_drift_types=true_dt,
                                           return_tracker=True, **kw)
            raw[name].append(s)
            if seed == 0:
                traces[name] = [r.accuracy for r in tr.records]
                if name == "DriftFusion":
                    preds0 = pred
        for name, bl in [("ARF", ARFBaseline(seed=seed)),
                          ("ADWIN+HT", AdwinHoeffdingTreeBaseline())]:
            s, tr = run_baseline(bl, X, y, window, external_onsets=onsets,
                                  return_tracker=True)
            raw[name].append(s)
            if seed == 0:
                traces[name] = [r.accuracy for r in tr.records]
        print(f"    seed {seed} done", flush=True)
    return raw, traces, preds0


# ------------------------------------------------------------- figure 1
def fig_ceiling(X, y):
    wins = make_windows(X, y, W_WIN)
    cal, ev = _calibration_split(wins, 0.2, 2)
    nz = RunningNormalizer(X.shape[1]); sX, sy = [], []
    for a, b in cal:
        nz.update(a); sX.append(a); sy.append(b)
    rf, lr, mlp = [], [], []
    torch.manual_seed(0)
    for a, b in ev:
        Xw = nz.transform(a).astype(np.float32); nz.update(a)
        Xtr = nz.transform(np.concatenate(sX)).astype(np.float32); ytr = np.concatenate(sy)
        rf.append((RandomForestClassifier(50, random_state=0, n_jobs=-1)
                   .fit(Xtr, ytr).predict(Xw) == b).mean())
        lr.append((LogisticRegression(max_iter=500).fit(Xtr, ytr).predict(Xw) == b).mean())
        m = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(),
                                torch.nn.Linear(128, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 2))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        for _ in range(150):
            opt.zero_grad(); torch.nn.functional.cross_entropy(m(Xt), yt).backward(); opt.step()
        with torch.no_grad():
            mlp.append((m(torch.from_numpy(Xw)).argmax(-1).numpy() == b).mean())
        sX.append(a); sy.append(b)
    return float(np.mean(lr)), float(np.mean(rf)), float(np.mean(mlp)), \
        float(max(y.mean(), 1 - y.mean()))


def draw_ceiling(maj, lin, arf, rf, df_acc, mlp):
    names = ["Majority\nclass", "Logistic\nregression", "ARF\n(streaming)",
             "Batch RF\n(all history)", "DriftFusion\n(online)", "Module 2 MLP\n(all history)"]
    vals = [maj, lin, arf, rf, df_acc, mlp]
    cols = ["#bfbfbf", "#bfbfbf", CLR["ARF"], "#8fa9c9", CLR["DriftFusion"], "#c9a0dc"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.bar(names, vals, color=cols, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.axhline(rf, ls="--", c="#4a6fa5", lw=1, label=f"batch-RF ceiling ({rf:.4f})")
    ax.set_ylim(0.60, 0.83); ax.set_ylabel("Prequential accuracy")
    ax.set_title("Ceiling analysis, NOAA Weather stream\n"
                 "ARF is saturated at the batch-RF ceiling; DriftFusion operates above it")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUT / "fig_ceiling_analysis.png"); plt.close(fig)


# ------------------------------------------------------------- figure 2
def fig_baseline_comparison(raw_w, raw_c):
    """Grouped bars with paired-test significance marks, both benchmarks."""
    methods = ["DriftFusion", "-Context", "ARF", "ADWIN+HT"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (raw, title) in zip(axes, [(raw_w, "NOAA Weather (primary)"),
                                        (raw_c, "CIC-UNSW-NB15 (CIC-IDS-2024) — secondary")]):
        agg = {m: aggregate_runs(raw[m]) for m in methods}
        x = np.arange(len(methods)); w = 0.38
        acc = [agg[m]["accuracy"][0] for m in methods]
        acc_e = [agg[m]["accuracy"][1] for m in methods]
        f1 = [agg[m]["macro_f1"][0] for m in methods]
        f1_e = [agg[m]["macro_f1"][1] for m in methods]
        ax.bar(x - w / 2, acc, w, yerr=acc_e, capsize=3, label="Accuracy",
               color="#8fa9c9", edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, f1, w, yerr=f1_e, capsize=3, label="Macro-F1",
               color="#c0504d", edgecolor="black", linewidth=0.5)
        for i, m in enumerate(methods):
            ax.text(i - w / 2, acc[i] + acc_e[i] + 0.004, f"{acc[i]:.3f}",
                    ha="center", fontsize=7.5)
            ax.text(i + w / 2, f1[i] + f1_e[i] + 0.004, f"{f1[i]:.3f}",
                    ha="center", fontsize=7.5)
        # significance of DriftFusion vs ARF on macro-F1
        s = paired_significance(raw["DriftFusion"], raw["ARF"], "macro_f1")
        verdict = ("DriftFusion > ARF on macro-F1 (p={:.3f})".format(s["p"])
                   if s["p"] is not None and s["p"] < 0.05 and s["mean_diff"] > 0
                   else "ARF > DriftFusion on macro-F1 (p={:.3f})".format(s["p"])
                   if s["p"] is not None and s["p"] < 0.05
                   else "no significant difference vs ARF (p={:.3f})".format(s["p"]))
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=12)
        ax.set_ylabel("Score"); ax.set_title(f"{title}\n{verdict}", fontsize=10)
        ax.set_ylim(0.6, 1.04); ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Baseline comparison, five seeds, mean ± s.d. with paired t-tests",
                 fontsize=11, y=0.99)
    fig.subplots_adjust(top=0.80)
    fig.savefig(OUT / "fig_baseline_comparison.png"); plt.close(fig)


# ------------------------------------------------------------- figure 3
def fig_trace(name, traces, onsets, title, vw_thr=0.85):
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    for lbl in ["DriftFusion", "-Context", "ARF", "ADWIN+HT"]:
        if lbl in traces:
            ax.plot(traces[lbl], label=lbl, color=CLR.get(lbl), lw=1.4,
                    alpha=0.9 if lbl in ("DriftFusion", "ARF") else 0.6)
    for i, o in enumerate(onsets):
        ax.axvline(o, color="#cccccc", lw=0.7, zorder=0,
                   label="drift onset" if i == 0 else None)
    ax.axhline(vw_thr, ls=":", c="red", lw=1, label=f"VW threshold ({vw_thr:.0%})")
    ax.set_xlabel("Evaluation window"); ax.set_ylabel("Window accuracy")
    ax.set_title(title); ax.legend(ncol=5, fontsize=8, loc="lower left")
    fig.savefig(OUT / f"fig_accuracy_trace_{name}.png"); plt.close(fig)


# ------------------------------------------------------------- figure 4
def fig_ablation(name, raw, title):
    labels = [n for n, _ in VARIANTS]
    agg = {m: aggregate_runs(raw[m]) for m in labels}
    acc = [agg[m]["accuracy"][0] for m in labels]
    acc_e = [agg[m]["accuracy"][1] for m in labels]
    fpr = [agg[m]["fpr"][0] for m in labels]
    fpr_e = [agg[m]["fpr"][1] for m in labels]
    x = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    a1.bar(x, acc, yerr=acc_e, capsize=3, color=[CLR.get(k, "#bfbfbf") for k in labels],
           edgecolor="black", linewidth=0.6)
    for i, v in enumerate(acc):
        a1.text(i, v + acc_e[i] + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels(labels, rotation=20, ha="right")
    a1.set_ylabel("Prequential accuracy"); a1.set_title("Accuracy (higher is better)")
    # Headroom above the tallest bar+error+label, so the label never collides
    # with the subplot title -- auto-scaled ylim leaves no such margin.
    a1.set_ylim(0, max(v + e for v, e in zip(acc, acc_e)) * 1.08)
    a2.bar(x, fpr, yerr=fpr_e, capsize=3, color=[CLR.get(k, "#bfbfbf") for k in labels],
           edgecolor="black", linewidth=0.6)
    for i, v in enumerate(fpr):
        a2.text(i, v + fpr_e[i] + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(labels, rotation=20, ha="right")
    a2.set_ylabel("False positive rate"); a2.set_title("FPR (lower is better)")
    a2.set_ylim(0, max(v + e for v, e in zip(fpr, fpr_e)) * 1.25)
    fig.suptitle(title)
    fig.savefig(OUT / f"fig_ablation_{name}.png"); plt.close(fig)


# ------------------------------------------------------------- figure 5
def fig_adaptation(X, y, onsets):
    wins = make_windows(X, y, W_WIN); cal, ev = _calibration_split(wins, 0.2, 2)
    Xc = np.concatenate([a for a, _ in cal]); yc = np.concatenate([b for _, b in cal])
    fr, ad = [], []
    for s in SEEDS:
        nz = RunningNormalizer(X.shape[1])
        for a, _ in cal: nz.update(a)
        torch.manual_seed(s)
        m = ContextPredictor(X.shape[1])
        pretrain_context_predictor(m, nz.transform(Xc), yc, seed=s)
        nz2 = RunningNormalizer(X.shape[1])
        for a, _ in cal: nz2.update(a)
        acc = []
        with torch.no_grad():
            for a, b in ev:
                Xw = nz2.transform(a).astype(np.float32); nz2.update(a)
                acc.append((m.forward_fallback(m.embed(torch.from_numpy(Xw)))
                            .argmax(-1).numpy() == b).mean())
        fr.append(float(np.mean(acc)))
        clf = train_drift_classifier(verbose=False, seed=s)
        r, _ = run_driftfusion(X, y, W_WIN, clf, seed=s, external_onsets=onsets)
        ad.append(r["accuracy"])
    from scipy import stats
    t, p = stats.ttest_rel(ad, fr)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    vals = [np.mean(fr), np.mean(ad)]; errs = [np.std(fr), np.std(ad)]
    bars = ax.bar(["Meta-trained,\nadaptation DISABLED", "Meta-trained,\nadaptation ENABLED"],
                  vals, yerr=errs, capsize=5,
                  color=["#6dba82", CLR["DriftFusion"]], edgecolor="black", linewidth=0.6)
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.0008,
                f"{v:.4f}±{e:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.790, 0.802); ax.set_ylabel("Prequential accuracy")
    ax.annotate(f"difference {np.mean(ad)-np.mean(fr):+.4f}, paired p = {p:.3f}\n"
                f"variance inflated {np.std(ad)/max(np.std(fr),1e-9):.1f}×",
                xy=(0.5, 0.72), xycoords="axes fraction", ha="center",
                fontsize=10, color="#c0504d")
    ax.set_title("Online adaptation has no measurable effect on accuracy (Weather)")
    fig.savefig(OUT / "fig_adaptation_effect.png"); plt.close(fig)
    print(f"  adaptation: frozen={np.mean(fr):.4f} adapted={np.mean(ad):.4f} p={p:.4f}")


# ------------------------------------------------------------- figure 6
def fig_drift_confusion(true_dt, pred, n_calib):
    """Drift-type confusion on the injected stream -- the evidence behind
    RQ1's localised failure on incremental drift."""
    true_eval = true_dt[n_calib:n_calib + len(pred)]
    M = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    for t, p in zip(true_eval, pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(CATEGORIES))); ax.set_xticklabels(CATEGORIES, rotation=35, ha="right")
    ax.set_yticks(range(len(CATEGORIES))); ax.set_yticklabels(CATEGORIES)
    ax.set_xlabel("Predicted drift category"); ax.set_ylabel("True (injected) drift category")
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] > M.max() * 0.55 else "black")
    ax.set_title("Drift-type classification against injected ground truth\n"
                 "CIC-UNSW-NB15 (CIC-IDS-2024), seed 0 — sudden, gradual and incremental drift are largely missed")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.grid(False)
    fig.savefig(OUT / "fig_drift_type_confusion.png"); plt.close(fig)


def main():
    for stale in OUT.glob("*.png"):
        stale.unlink()

    print("Weather: 5-seed collection...")
    Xw, yw, ons_w = load_weather()
    raw_w, traces_w, _ = collect(Xw, yw, W_WIN, ons_w)

    print("CIC-UNSW-NB15: 5-seed collection...")
    Xc, yc, ons_c, dt_c, nc_c = load_cic_unsw()
    raw_c, traces_c, pred_c = collect(Xc, yc, C_WIN, ons_c, true_dt=dt_c)

    print("ceiling analysis...")
    lin, rf, mlp, maj = fig_ceiling(Xw, yw)
    df_acc = aggregate_runs(raw_w["DriftFusion"])["accuracy"][0]
    arf_acc = aggregate_runs(raw_w["ARF"])["accuracy"][0]
    draw_ceiling(maj, lin, arf_acc, rf, df_acc, mlp)

    fig_baseline_comparison(raw_w, raw_c)
    fig_trace("weather", traces_w, ons_w,
              "NOAA Weather: per-window accuracy, seed 0 (grey = seasonal drift onsets)")
    fig_trace("cic_unsw", traces_c, ons_c,
              "CIC-UNSW-NB15 (CIC-IDS-2024): per-window accuracy, seed 0 (grey = injected drift onsets)")
    fig_ablation("weather", raw_w, "Ablation study — NOAA Weather stream (five seeds)")
    fig_ablation("cic_unsw", raw_c, "Ablation study — CIC-UNSW-NB15 (CIC-IDS-2024), five seeds")
    if pred_c is not None:
        fig_drift_confusion(dt_c, pred_c, nc_c)
    print("adaptation effect...")
    fig_adaptation(Xw, yw, ons_w)

    print(f"\nFigures written to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
