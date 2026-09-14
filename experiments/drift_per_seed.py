"""Per-seed drift-type behaviour on the injected stream, and a redraw of the
adaptation-effect figure with its corrected title.

Section 5.6 makes per-seed claims -- how many seeds never predict a given
category, and what the best seed reaches -- that an aggregate collection
cannot support. This recovers them directly.
"""

import collections

import numpy as np

from .make_figures import load_weather, load_cic_unsw, fig_adaptation, C_WIN, SEEDS
from ..models.drift_classifier import train_drift_classifier
from ..evaluation.prequential import run_driftfusion


def main():
    Xc, yc, ons_c, dt_c, nc_c = load_cic_unsw()
    true_eval = dt_c[nc_c:]

    print("Per-seed drift-type classification, CIC-UNSW-NB15 (CIC-IDS-2024)\n")
    f1s = []
    for seed in SEEDS:
        clf = train_drift_classifier(verbose=False, seed=seed)
        s, pred = run_driftfusion(Xc, yc, C_WIN, clf, seed=seed,
                                  external_onsets=ons_c, true_drift_types=dt_c)
        f1s.append(s["drift_type_f1"])
        counts = collections.Counter(pred)
        print(f"seed {seed}: drift_type_f1={s['drift_type_f1']:.3f}  "
              f"predicted={dict(sorted(counts.items()))}")

    print(f"\nmean={np.mean(f1s):.3f}  sd={np.std(f1s):.3f}  "
          f"best={max(f1s):.3f}  worst={min(f1s):.3f}")

    n = len(true_eval)
    print(f"\nTrue categories over {n} evaluation windows: "
          f"{dict(sorted(collections.Counter(true_eval).items()))}")

    print("\nRedrawing the adaptation-effect figure with the corrected title...")
    Xw, yw, ons_w = load_weather()
    fig_adaptation(Xw, yw, ons_w)


if __name__ == "__main__":
    main()
