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
   meta-trained on synthetic fingerprint sequences.
2. **In-context zero-shot prediction** — produces the benign/attack decision
   from a model conditioned on a bounded store of labelled examples drawn
   from the active operating context, via cross-attention.
3. **Meta drift adapter** — selects a learning rate, a number of update
   steps, and the weight of an Elastic Weight Consolidation (EWC) retention
   penalty according to the drift category identified by module 1, updating
   the prediction model with Nesterov momentum.

## Repository structure

```
driftfusion/
├── models/
│   ├── driftfusion.py          # Orchestrates all three modules per window
│   ├── drift_classifier.py     # LSTM drift-type classifier
│   ├── fingerprint.py           # Concept fingerprint extractor
│   ├── bocpd.py                 # Bayesian online changepoint detection
│   ├── context_predictor.py    # In-context zero-shot predictor
│   └── meta_adapter.py          # Drift-conditioned adaptation with EWC
├── data/
│   ├── noaa_weather.py          # NOAA GSOD weather stream loader
│   ├── cic_unsw_nb15.py         # CIC-UNSW-NB15 stream loader with drift injection
│   └── preprocessing.py         # Cleaning, mRMR feature selection, running normalisation
├── baselines/
│   └── streaming_baselines.py   # Adaptive Random Forest, ADWIN+Hoeffding Tree
├── evaluation/
│   ├── prequential.py            # Test-then-train evaluation loop
│   └── metrics.py                # Vulnerability Window, forgetting, significance tests
├── experiments/
│   ├── run_weather.py            # Full experiment on the NOAA weather stream
│   ├── run_cic_unsw.py           # Full experiment on the CIC-UNSW-NB15 stream
│   └── make_figures.py           # Regenerates all results figures from real runs
└── synthetic/
    └── fingerprint_sequences.py  # Synthetic sequences for meta-training the classifier
```

## Requirements

Python 3.13, PyTorch, NumPy, pandas, scikit-learn, SciPy, river, Matplotlib.

## Usage

```bash
python -m driftfusion.experiments.run_weather      # NOAA weather stream, 5 seeds, 7 configurations
python -m driftfusion.experiments.run_cic_unsw      # CIC-UNSW-NB15 stream, 5 seeds, 7 configurations
python -m driftfusion.experiments.make_figures      # Regenerates all results figures
```

Each run script evaluates the complete framework, four ablated variants
(without drift classification, without context memory, without EWC, and with
delayed labels), and both baselines under a prequential protocol, reporting
means and standard deviations over five random seeds with paired significance
testing.

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
| NOAA weather | 0.797 | 0.773 | not significant | significantly better |
| CIC-UNSW-NB15 | 0.969 | 0.967 | significantly worse | significantly better |

The Vulnerability Window (mean windows below 85% accuracy following a drift
onset) fell from 6.50 to 0.05 against the lighter baseline. Component
ablation found that removing the context-conditioned prediction module
improved every measure on both streams, and that the EWC retention penalty
exerted no measurable force once tested in isolation — both reported as
negative results rather than omitted. Full analysis is in the accompanying
thesis.

## License

Provided for academic and research purposes.
