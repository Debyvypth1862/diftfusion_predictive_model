"""
Bayesian Online Changepoint Detection (Adams & MacKay, 2007) with a
Normal-Inverse-Gamma conjugate prior, per Section 7.4 / Algorithm 1.

Operates on a univariate signal derived from the stream -- here, the
per-window classification error rate `e_t`, so that a changepoint
indicates a shift in how well the current model explains the data
(the same quantity later used to drive drift-type classification).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


class OnlineBOCPD:
    def __init__(self, hazard_lambda: float = 10.0, mu0: float = 0.0,
                 kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0,
                 prune_threshold: float = 1e-6):
        self.hazard = 1.0 / hazard_lambda
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.prune_threshold = prune_threshold

        # Run-length posterior and per-hypothesis NIG parameters, indexed
        # by run length r = 0, 1, 2, ... (hypothesis 0 = "just changed").
        self.log_R = np.array([0.0])  # log P(r_0 = 0) = 1
        self.mu = np.array([mu0])
        self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])
        self.t = 0

        # Running standardisation of the raw input: an NIG prior with
        # beta0=1 implicitly assumes the signal has roughly unit variance.
        # Error rates (and most other candidate signals) live on a much
        # smaller scale, which would otherwise make the prior dominate
        # the posterior and mask real changepoints. Feeding a running
        # z-score into the NIG updates instead keeps the prior well
        # calibrated regardless of the input signal's native scale.
        self._raw_n = 0
        self._raw_mean = 0.0
        self._raw_M2 = 0.0

    def _standardize(self, x: float) -> float:
        if self._raw_n < 2:
            z = x - self._raw_mean if self._raw_n else x
        else:
            var = self._raw_M2 / (self._raw_n - 1)
            z = (x - self._raw_mean) / np.sqrt(max(var, 1e-8))
        self._raw_n += 1
        delta = x - self._raw_mean
        self._raw_mean += delta / self._raw_n
        self._raw_M2 += delta * (x - self._raw_mean)
        return float(z)

    def _log_student_t_pred(self, x: float) -> np.ndarray:
        """log predictive probability of x under each run-length hypothesis's
        Student-t posterior predictive (standard NIG-Normal conjugacy)."""
        df = 2 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        z = (x - self.mu) / scale
        log_norm = (gammaln((df + 1) / 2) - gammaln(df / 2)
                    - 0.5 * np.log(df * np.pi) - np.log(scale))
        log_kernel = -((df + 1) / 2) * np.log1p(z ** 2 / df)
        return log_norm + log_kernel

    def step(self, x: float) -> dict:
        """Ingest one observation (a window's error rate) and return the
        three BOCPD-derived fingerprint components."""
        self.t += 1
        z = self._standardize(x)
        log_pred = self._log_student_t_pred(z)

        # Growth probabilities: hypothesis r survives and grows to r+1.
        log_growth = self.log_R + log_pred + np.log1p(-self.hazard)
        # Changepoint probability: mass collapses to r = 0.
        log_cp = _logsumexp(self.log_R + log_pred + np.log(self.hazard))

        new_log_R = np.concatenate([[log_cp], log_growth])
        new_log_R -= _logsumexp(new_log_R)

        # Each hypothesis's sufficient statistics are updated with z: the
        # r=0 (changepoint) lineage updates from the raw prior, and every
        # growth lineage updates from its own previous-step statistics.
        all_mu = np.concatenate([[self.mu0], self.mu])
        all_kappa = np.concatenate([[self.kappa0], self.kappa])
        all_alpha = np.concatenate([[self.alpha0], self.alpha])
        all_beta = np.concatenate([[self.beta0], self.beta])

        new_kappa = all_kappa + 1
        new_mu = (all_kappa * all_mu + z) / new_kappa
        new_alpha = all_alpha + 0.5
        new_beta = all_beta + all_kappa * (z - all_mu) ** 2 / (2 * new_kappa)

        keep = new_log_R > np.log(self.prune_threshold)
        if keep.sum() == 0:
            keep[np.argmax(new_log_R)] = True
        self.log_R = new_log_R[keep]
        self.log_R -= _logsumexp(self.log_R)
        self.mu, self.kappa, self.alpha, self.beta = (
            new_mu[keep], new_kappa[keep], new_alpha[keep], new_beta[keep]
        )

        R = np.exp(self.log_R)
        # Under a constant hazard, P(r_t = 0 | x_1:t) collapses to exactly
        # the hazard rate for any data whatsoever -- a direct consequence
        # of the recursion (the r=0 branch and every growth branch draw
        # from the same evidence pool, split only by the fixed H:(1-H)
        # ratio). It carries no signal. The informative, data-dependent
        # quantity is how much mass sits at a *small* run length shortly
        # after ingesting a surprising observation, so "changepoint
        # probability" is reported as P(r_t <= 1) = R[0] + R[1].
        p_changepoint = float(R[0] + (R[1] if len(R) > 1 else 0.0))
        # Normalised to [0, 1] by its own maximum, log|R|. Raw Shannon
        # entropy grows with the number of surviving run-length hypotheses,
        # which itself grows with stream position, so an absolute value is
        # not comparable across windows and drifts steadily out of any
        # fixed range the drift-type classifier was meta-trained on.
        raw_entropy = float(-np.sum(R * np.log(R + 1e-12)))
        max_entropy = float(np.log(len(R))) if len(R) > 1 else 1.0
        entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
        modal_run_length = int(np.argmax(R))
        normalized_modal_run_length = modal_run_length / max(self.t, 1)

        return {
            "p_changepoint": p_changepoint,
            "run_length_entropy": entropy,
            "normalized_modal_run_length": normalized_modal_run_length,
        }


def _logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))
