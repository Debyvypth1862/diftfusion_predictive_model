# DriftFusion

A streaming intrusion detection framework that identifies the *category* of
concept drift affecting a data stream and adapts to it accordingly, rather
than applying a single uniform response to every change. Developed as part
of an MSc Data Science thesis at Liverpool John Moores University.

## Overview

DriftFusion combines three modules into a single per-window pipeline:

1. **Concept fingerprinting and drift-type classification** — condenses each
   window into a ten-dimensional descriptor built from Bayesian online
   changepoint statistics, prediction-confidence signals and distributional
   features, then classifies the drift into one of five categories (stable,
   sudden, gradual, incremental, recurring) using a single-layer LSTM
   trained on synthetic fingerprint sequences.
2. **In-context zero-shot prediction** — produces the benign/attack decision
   from a model conditioned on a bounded store of labelled examples drawn
   from the active operating context, via cross-attention. Meta-trained with
   first-order MAML: each episode adapts on a support set using the same
   objective the online adapter uses, then evaluates on a disjoint query set
   at the *adapted* parameters, so what is optimised is an initialisation
   from which a short update generalises.
3. **Meta drift adapter** — selects a learning rate, a number of update
   steps, and the weight of an Elastic Weight Consolidation (EWC) retention
   penalty according to the drift category identified by module 1, updating
   the prediction model with Nesterov momentum. This is the inner loop of
   the meta-learning scheme, with its step count and learning rate
   conditioned on drift type rather than held fixed.

## Repository structure

```
driftfusion/
├── models/
│   ├── driftfusion.py           # Orchestrates all three modules per window
│   ├── drift_classifier.py      # LSTM drift-type classifier
│   ├── fingerprint.py           # Concept fingerprint extractor
│   ├── bocpd.py                 # Bayesian online changepoint detection
│   ├── context_predictor.py     # In-context predictor + first-order MAML meta-training
│   └── meta_adapter.py          # Drift-conditioned adaptation with EWC
├── data/
│   ├── noaa_weather.py          # NOAA GSOD weather stream loader
│   ├── cic_unsw_nb15.py         # CIC-UNSW-NB15 stream loader with drift injection
│   └── preprocessing.py         # Cleaning, mRMR feature selection, running normalisation
├── baselines/
│   └── streaming_baselines.py   # Adaptive Random Forest, ADWIN+Hoeffding Tree
├── evaluation/
│   ├── prequential.py           # Test-then-train evaluation loop
│   └── metrics.py               # Vulnerability Window, recovery speed, SPI, forgetting,
│                                #   backward transfer, significance tests
├── experiments/
│   ├── run_weather.py           # Full experiment on the NOAA weather stream
│   ├── run_cic_unsw.py          # Full experiment on the CIC-UNSW-NB15 stream
│   ├── make_figures.py          # Regenerates all results figures from real runs
│   ├── collect_full_metrics.py  # Single pass: every reported metric, table and figure
│   ├── drift_per_seed.py        # Per-seed drift-type behaviour
│   ├── validate_figures.py      # Structural conformance of code against the design diagrams
│   └── sweep_strategy.py        # Adaptation hyperparameter sweep
└── synthetic/
    └── fingerprint_sequences.py # Synthetic sequences for training the drift classifier
```

## Requirements

Python 3.13, PyTorch 2.13, NumPy 2.4, pandas 3.0, scikit-learn 1.8,
SciPy 1.17, river 0.25, Matplotlib 3.10. No GPU required — all reported
results were produced on a single CPU workstation.

## Usage

```bash
python -m driftfusion.experiments.collect_full_metrics  # Every reported number + all figures
python -m driftfusion.experiments.run_weather           # NOAA weather stream alone
python -m driftfusion.experiments.run_cic_unsw          # CIC-UNSW-NB15 stream alone
python -m driftfusion.experiments.validate_figures      # 21 structural checks against the diagrams
```

`collect_full_metrics` is the single source of every published figure and
table: it runs the complete framework, four ablated variants (without drift
classification, without context memory, without EWC, and with labels delayed
by three windows) and both baselines, across five seeds under a prequential
protocol, and reports means with standard deviations and paired significance
tests. Expect roughly 45 minutes on a CPU workstation.

## Datasets

Both streams used are public. Raw data files are not included in this
repository (see `.gitignore`).

- **NOAA weather stream**: National Centers for Environmental Information
  (NCEI), Global Summary of the Day (GSOD), station 72530094846 (Chicago
  O'Hare International Airport, IL).
  https://www.ncei.noaa.gov/access/search/data-search/global-summary-of-the-day
- **CIC-UNSW-NB15 stream**: Canadian Institute for Cybersecurity (CIC),
  University of New Brunswick.
  https://www.unb.ca/cic/datasets/cic-unsw-nb15.html

## Results summary

Evaluated against an Adaptive Random Forest (ARF) and an ADWIN-augmented
Hoeffding Tree, across five seeds with paired statistical testing:

| Stream | Accuracy | Macro F1 | vs ARF | vs ADWIN+HT |
|---|---|---|---|---|
| NOAA weather | 0.797 | 0.774 | significantly better on macro-F1 (p=0.019); accuracy p=0.051, not claimed | significantly better |
| CIC-UNSW-NB15 | 0.971 | 0.969 | significantly worse by 1.1 points (p=0.015) | significantly better by 9.0 points |

On the CIC-UNSW-NB15 stream the Vulnerability Window (mean windows below 85%
accuracy following a drift onset) is zero for every configuration of the
framework and for ARF, against 6.50 for the lighter baseline; recovery speed
shows the same contrast at 0.00 against 13.00 windows.

Component ablation is reported in full, including where it is unfavourable:

- **Online adaptation contributes.** Frozen 0.7938 vs adapting 0.7969 on the
  weather stream, paired p=0.018. The frozen model is level with ARF, so the
  margin over that baseline comes from the adaptation procedure rather than
  from the meta-training that precedes it.
- **Drift-typed adaptation reduces forgetting** on both streams relative to a
  uniform response (0.069 → 0.059 on weather, 0.029 → 0.019 on CIC).
- **The context-conditioned prediction module does not earn its cost.**
  Removing it leaves accuracy and macro-F1 unchanged on the weather stream,
  improves both by roughly a point on CIC-UNSW-NB15, improves forgetting on
  both, and cuts per-window processing time by more than an order of
  magnitude.
- **The EWC retention penalty exerts no force.** With and without it the
  results are numerically identical to every decimal place: the penalty term
  is seven or more orders of magnitude below the task loss and is absorbed by
  single-precision rounding.
- **The drift classifier does not beat a trivial baseline.** It reaches macro
  F1 0.144 against known categories, while a configuration that labels every
  window `stable` scores 0.157 — because `stable` accounts for 51 of 79
  evaluation windows. The classifier collapses onto stable and recurring
  calls, missing sudden and gradual drift entirely on three of five seeds.

Full analysis, including why each negative result arises, is in the
accompanying thesis.

## License

Provided for academic and research purposes.
