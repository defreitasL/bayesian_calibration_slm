from __future__ import annotations

"""Pure Mixture-of-Experts ensemble.

This workflow combines multiple independent experts (physics, ML, statistical)
into an ensemble prediction using a *gating* model.

The gating model is a multinomial (softmax) regression trained on calibration
observations to predict which expert performs best given forcing features.

Design goals
------------
* Expert handlers live in `slmcal.models`.
* The MoE engine does not assume which experts you use.
* Optional dependencies (pymc-bart, statsmodels) remain isolated.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from slmcal.data import TimeSeriesDataset
from slmcal.features import build_feature_matrix

from slmcal.models.expert import Expert, ExpertPrediction


@dataclass
class GateSoftmaxConfig:
    """Softmax-gate configuration."""

    forcing_names: Sequence[str] = ("E", "hs", "tp")
    include_doy: bool = True
    include_time: bool = False
    standardize: bool = True

    # Regularization strength (L2) on coefficients.
    l2: float = 1.0
    maxiter: int = 500


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _fit_softmax_regression(X: np.ndarray, y: np.ndarray, *, l2: float = 1.0, maxiter: int = 500) -> np.ndarray:
    """Fit multinomial logistic regression with L2 using scipy.optimize.

    Returns
    -------
    theta
        Parameter matrix with shape (K, p+1) including intercept, with the
        last class fixed to zeros for identifiability.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n, p = X.shape
    K = int(np.max(y)) + 1

    # add intercept column
    X1 = np.column_stack([np.ones((n, 1)), X])
    p1 = p + 1

    # optimize parameters for first K-1 classes; last class is zeros
    d = (K - 1) * p1

    def unpack(w: np.ndarray) -> np.ndarray:
        W = w.reshape((K - 1, p1))
        theta = np.zeros((K, p1), dtype=float)
        theta[: K - 1, :] = W
        return theta

    def nll_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        theta = unpack(w)
        logits = X1 @ theta.T  # (n, K)
        P = _softmax(logits)

        # negative log likelihood
        eps = 1e-12
        ll = np.sum(np.log(P[np.arange(n), y] + eps))

        # L2 penalty (exclude intercept)
        pen = 0.5 * float(l2) * np.sum(theta[: K - 1, 1:] ** 2)
        loss = -ll + pen

        # gradient
        Y = np.zeros_like(P)
        Y[np.arange(n), y] = 1.0
        G = (P - Y).T @ X1  # (K, p1)
        # apply penalty gradient
        G[: K - 1, 1:] += float(l2) * theta[: K - 1, 1:]
        grad = G[: K - 1, :].reshape(-1)
        return loss, grad

    w0 = np.zeros((d,), dtype=float)
    res = minimize(lambda w: nll_and_grad(w)[0], w0, jac=lambda w: nll_and_grad(w)[1], method="L-BFGS-B", options={"maxiter": int(maxiter)})
    return unpack(res.x)


def _predict_gate_weights(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X1 = np.column_stack([np.ones((X.shape[0], 1)), X])
    logits = X1 @ theta.T
    return _softmax(logits)


@dataclass
class MoEEnsembleResult:
    time: np.ndarray
    weights: np.ndarray  # (n_time, K)
    expert_predictions: list[ExpertPrediction]
    ensemble: ExpertPrediction


class MoEEnsembleWorkflow:
    """Fit experts and build a softmax-gated ensemble."""

    def __init__(
        self,
        experts: Sequence[Expert],
        gate_cfg: GateSoftmaxConfig = GateSoftmaxConfig(),
        *,
        name: str = "moe",
    ):
        self.experts = list(experts)
        self.gate_cfg = gate_cfg
        self.name = name
        self._theta: np.ndarray | None = None
        self._ref_stats: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, dataset_cal: TimeSeriesDataset, *, n_gate_draws: int = 300, random_seed: int = 42) -> "MoEEnsembleWorkflow":
        # Fit experts
        for ex in self.experts:
            ex.fit(dataset_cal)

        # Expert medians at obs times
        med = []
        for ex in self.experts:
            pred = ex.predict_draws(dataset_cal, n_draws=int(n_gate_draws), random_seed=int(random_seed))
            m = pred.median()
            idx = np.asarray(dataset_cal.idx_obs, dtype=int)
            med.append(m[idx])
        med = np.column_stack(med)  # (n_obs, K)

        y_obs = np.asarray(dataset_cal.obs, dtype=float)
        abs_err = np.abs(med - y_obs[:, None])
        winner = np.argmin(abs_err, axis=1).astype(int)

        # Gate features (obs)
        X_obs, _, ref_stats = build_feature_matrix(
            dataset_cal,
            self.gate_cfg.forcing_names,
            include_doy=self.gate_cfg.include_doy,
            include_time=self.gate_cfg.include_time,
            at="obs",
            standardize=self.gate_cfg.standardize,
        )
        self._ref_stats = ref_stats

        self._theta = _fit_softmax_regression(X_obs, winner, l2=self.gate_cfg.l2, maxiter=self.gate_cfg.maxiter)
        return self

    def predict(self, dataset_full: TimeSeriesDataset, *, n_draws: int = 1000, random_seed: int = 42) -> MoEEnsembleResult:
        if self._theta is None:
            raise RuntimeError("Call fit() before predict()")

        rng = np.random.default_rng(int(random_seed))

        # Gate weights on full time grid
        X_full, _, _ = build_feature_matrix(
            dataset_full,
            self.gate_cfg.forcing_names,
            include_doy=self.gate_cfg.include_doy,
            include_time=self.gate_cfg.include_time,
            at="full",
            standardize=self.gate_cfg.standardize,
            ref_stats=self._ref_stats,
        )
        weights = _predict_gate_weights(self._theta, X_full)  # (n_time, K)

        # Expert draws
        preds: list[ExpertPrediction] = []
        for ex in self.experts:
            preds.append(ex.predict_draws(dataset_full, n_draws=int(n_draws), random_seed=int(random_seed)))

        # Blend draws (soft mixture)
        K = len(preds)
        draws = np.zeros((int(n_draws), dataset_full.time.shape[0]), dtype=float)
        for k in range(K):
            wk = weights[:, k][None, :]  # (1, n_time)
            draws += wk * np.asarray(preds[k].draws, dtype=float)

        ens = ExpertPrediction(name=self.name, time=np.asarray(dataset_full.time), draws=draws)
        return MoEEnsembleResult(time=np.asarray(dataset_full.time), weights=weights, expert_predictions=preds, ensemble=ens)
