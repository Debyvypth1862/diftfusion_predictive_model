"""
Ad-hoc sweep over Table 7.4's adaptation strategy hyperparameters on the
full CIC-IDS-2018 stream, used to diagnose/fix the "sudden"/"recurring"
learning rates being too aggressive (observed: the no-fingerprinting
ablation, which forces every window through the ultra-conservative
'stable' strategy, outperformed the full model).
"""

from __future__ import annotations

import time

import numpy as np

from ..data.cic_ids2018 import build_cic_stream
from ..data.preprocessing import clean_features, mrmr_select
from ..models.drift_classifier import train_drift_classifier
from ..models.meta_adapter import STRATEGY_TABLE
from ..evaluation.prequential import run_driftfusion
from .run_cic import build_true_drift_types, WINDOW_SIZE, MRMR_K

CANDIDATES = {
    "proposal_default": STRATEGY_TABLE,
    "gentler_sudden_recurring": {
        **STRATEGY_TABLE,
        "sudden":    dict(lr=0.003, steps=6, ewc_lambda=0.3),
        "recurring": dict(lr=0.002, steps=4, ewc_lambda=1.0),
    },
    "much_gentler_all_nonstable": {
        "stable":      dict(lr=0.0005, steps=1, ewc_lambda=5.0),
        "sudden":      dict(lr=0.001,  steps=3, ewc_lambda=1.0),
        "gradual":     dict(lr=0.001,  steps=2, ewc_lambda=2.0),
        "incremental": dict(lr=0.0008, steps=1, ewc_lambda=3.0),
        "recurring":   dict(lr=0.001,  steps=2, ewc_lambda=1.0),
    },
    "moderate_all_nonstable": {
        "stable":      dict(lr=0.0005, steps=1, ewc_lambda=5.0),
        "sudden":      dict(lr=0.002,  steps=4, ewc_lambda=1.0),
        "gradual":     dict(lr=0.0015, steps=3, ewc_lambda=2.0),
        "incremental": dict(lr=0.001,  steps=2, ewc_lambda=3.0),
        "recurring":   dict(lr=0.0015, steps=3, ewc_lambda=1.0),
    },
}


def main():
    df, raw_cols = build_cic_stream()
    X_raw = clean_features(df[raw_cols])
    y = df["label"].to_numpy(dtype=np.int64)
    selected = mrmr_select(X_raw, y, k=MRMR_K)
    X = X_raw[selected].to_numpy(dtype=np.float32)
    true_types = build_true_drift_types(df, WINDOW_SIZE)

    for seed in (0, 1):
        drift_clf = train_drift_classifier(verbose=False, seed=seed)
        for name, table in CANDIDATES.items():
            t0 = time.time()
            r, pred = run_driftfusion(X, y, WINDOW_SIZE, drift_clf, true_drift_types=true_types,
                                       seed=seed, strategy_table=table)
            print(f"seed={seed} {name:28s} acc={r['accuracy']:.3f} f1={r['macro_f1']:.3f} "
                  f"vw={r['vulnerability_window']:.1f} fgt={r['forgetting']:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
