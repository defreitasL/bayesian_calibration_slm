from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slmcal.data import TimeSeriesDataset
from slmcal.models.base import ShorelineModel
from slmcal.optimization import (
    NSGA2Config,
    NSGA2Result,
    run_nsga2,
    FastOptNSGA2Config,
    FastOptNSGA2Result,
    run_nsga2_fastopt,
)
from slmcal.bayes.pymc import (
    Prior,
    PriorMVN,
    PriorKDE,
    PriorCopulaKDE,
    fit_mvn_prior_from_nsga2,
    fit_kde_prior_from_nsga2,
    fit_copula_kde_prior_from_nsga2,
    bayesian_calibrate,
)


@dataclass
class CalibrationWorkflow:
    """End-to-end calibration workflow.

    Parameters
    ----------
    model
        Shoreline model adapter.
    dataset
        Data container with forcings + observations.
    """

    model: ShorelineModel
    dataset: TimeSeriesDataset

    nsga2_result: NSGA2Result | FastOptNSGA2Result | None = None
    prior: Prior | None = None

    def precalibrate_nsga2(
        self,
        metrics: tuple[str, ...] = ("kge", "spbias", "spearman"),
        cfg: NSGA2Config = NSGA2Config(),
        fast_cfg: FastOptNSGA2Config | None = None,
        backend: str = "fast_optimization",
        progress: bool = True,
    ) -> NSGA2Result | FastOptNSGA2Result:
        if backend == "fast_optimization":
            self.nsga2_result = run_nsga2_fastopt(
                model=self.model,
                dataset=self.dataset,
                metrics=metrics,
                cfg=fast_cfg or FastOptNSGA2Config(),
            )
        elif backend == "internal":
            self.nsga2_result = run_nsga2(
                model=self.model,
                dataset=self.dataset,
                metrics=metrics,
                cfg=cfg,
                progress=progress,
            )
        else:
            raise ValueError("backend must be 'fast_optimization' or 'internal'")
        return self.nsga2_result

    def build_prior_from_nsga2(
        self,
        kind: str = "mvn",
        cov_scale: float = 30.0,
        bw_method: str | float | None = "scott",
        jitter: float = 1e-9,
        copula_method: str = "factor",
        copula_grid_size: int = 512,
        copula_rank: int | None = None,
        copula_explained_var: float = 0.99,
        copula_shrinkage: float = 0.05,
        use_valid_only: bool = True,
        validity_thresholds: dict | None = None,
    ) -> Prior:
        if self.nsga2_result is None:
            raise RuntimeError("Run precalibrate_nsga2() first")

        # Optional filtering of "valid" solutions is implemented by the internal backend.
        # The fast_optimization backend returns a generic Pareto set without built-in validity helpers.
        if use_valid_only and hasattr(self.nsga2_result, "select_valid"):
            th = validity_thresholds or {}
            ind, _ = self.nsga2_result.select_valid(**th)
        else:
            ind = self.nsga2_result.individuals_raw

        if ind.shape[0] < 5:
            raise RuntimeError(
                "Too few individuals for a stable empirical prior. "
                "Try relaxing validity thresholds or increasing NSGA-II restarts."
            )


        kind_l = kind.lower().strip()

        if kind_l in ("mvn", "gaussian", "normal"):
            self.prior = fit_mvn_prior_from_nsga2(ind, cov_scale=cov_scale)
        elif kind_l in ("kde", "empirical"):
            from slmcal.models.base import bounds_matrix

            bnd = bounds_matrix(self.model.parameters)
            self.prior = fit_kde_prior_from_nsga2(
                ind,
                bw_method=bw_method,
                jitter=jitter,
                bounds_lower=bnd[:, 0],
                bounds_upper=bnd[:, 1],
            )
        elif kind_l in ("copula_kde", "copula", "copula-kde", "gaussian_copula_kde"):
            from slmcal.models.base import bounds_matrix

            bnd = bounds_matrix(self.model.parameters)
            self.prior = fit_copula_kde_prior_from_nsga2(
                ind,
                bw_method=bw_method,
                grid_size=copula_grid_size,
                shrinkage=copula_shrinkage,
                method=copula_method,
                rank=copula_rank,
                explained_var=copula_explained_var,
                bounds_lower=bnd[:, 0],
                bounds_upper=bnd[:, 1],
            )
        else:
            raise ValueError("Unknown prior kind. Use 'mvn', 'kde', or 'copula_kde'.")

        return self.prior

    def bayesian_calibrate(
        self,
        prior_from: str = "nsga2",
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int = 42,
        likelihood: str = "normal",
        sigma: float | None = None,
        estimate_sigma: bool = False,
        include_bias: bool = True,
        cores: int | None = None,
    ):
        if prior_from == "nsga2":
            if self.prior is None:
                self.build_prior_from_nsga2()
            prior = self.prior
        else:
            raise ValueError("Only prior_from='nsga2' is implemented in v0.1")

        trace, ppc = bayesian_calibrate(
            model=self.model,
            dataset=self.dataset,
            prior=prior,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            likelihood=likelihood,
            sigma=sigma,
            estimate_sigma=estimate_sigma,
            include_bias=include_bias,
            cores=cores,
        )
        return trace, ppc

    
    def prior_summary(self) -> dict:
        if self.prior is None:
            raise RuntimeError("Prior not built")

        if isinstance(self.prior, PriorMVN):
            return {
                "kind": "mvn",
                "mean": self.prior.mean,
                "std": np.sqrt(np.diag(self.prior.cov)),
            }

        if isinstance(self.prior, PriorKDE):
            s = np.asarray(self.prior.samples, dtype=float)
            return {
                "kind": "kde",
                "n_samples": int(s.shape[0]),
                "mean": np.mean(s, axis=0),
                "std": np.std(s, axis=0),
                "p05": np.quantile(s, 0.05, axis=0),
                "p50": np.quantile(s, 0.50, axis=0),
                "p95": np.quantile(s, 0.95, axis=0),
            }

        if isinstance(self.prior, PriorCopulaKDE):
            # Summarize using the *samples* implied by the NSGA-II set is not stored
            # for copula KDE (we store tabulated marginals). We provide a lightweight summary.
            return {
                "kind": "copula_kde",
                "method": self.prior.method,
                "n_params": int(self.prior.grid.shape[0]),
                "grid_size": int(self.prior.grid.shape[1]),
            }

        raise TypeError(f"Unsupported prior type: {type(self.prior)}")
