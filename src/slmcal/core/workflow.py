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
    PriorProduct,
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
    trace: object | None = None
    ppc: object | None = None

    def precalibrate_nsga2(
        self,
        metrics: tuple[str, ...] = ("kge", "pbias", "spearman"),
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
        top_k: int | None = None,
        cov_inflation: float | None = None,
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
            ind, obj = self.nsga2_result.select_valid(**th)
        else:
            ind = self.nsga2_result.individuals_raw
            obj = getattr(self.nsga2_result, "objectives", None)

        if ind.shape[0] < 5:
            raise RuntimeError(
                "Too few individuals for a stable empirical prior. "
                "Try relaxing validity thresholds or increasing NSGA-II restarts."
            )


        # Optionally keep only the top-K solutions (scalarized in normalized objective space).
        if top_k is not None and int(top_k) > 0 and ind.shape[0] > int(top_k):
            if obj is None:
                # No objective info available: fall back to random subset.
                rng = np.random.default_rng(42)
                idx = rng.choice(ind.shape[0], size=int(top_k), replace=False)
                ind = ind[idx]
            else:
                obj = np.asarray(obj, dtype=float)
                # Normalize each objective to [0,1] and scalarize by mean.
                lo = np.nanmin(obj, axis=0)
                hi = np.nanmax(obj, axis=0)
                span = hi - lo
                span[span == 0.0] = 1.0
                z = (obj - lo) / span
                score = np.nanmean(z, axis=1)
                idx = np.argsort(score)[: int(top_k)]
                ind = ind[idx]
                # keep matching objectives for potential future use
                obj = obj[idx]

        kind_l = kind.lower().strip()

        if kind_l in ("mvn", "gaussian", "normal"):
            infl = float(cov_inflation) if cov_inflation is not None else float(cov_scale)
            # Use selected individuals also as an init pool for chain initialisation.
            self.prior = fit_mvn_prior_from_nsga2(ind, cov_scale=infl, init_pool=ind)
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
        save_outputs: bool = False,
        out_dir: str | None = None,
        dataset_predict: TimeSeriesDataset | None = None,
        n_output_draws: int = 1000,
        output_max_bytes: int = 500_000_000,
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
        self.trace = trace
        self.ppc = ppc

        if save_outputs:
            if out_dir is None:
                raise ValueError("out_dir must be provided when save_outputs=True")
            self.save_bayesian_outputs(
                out_dir=out_dir,
                trace=trace,
                ppc=ppc,
                dataset_predict=dataset_predict,
                n_draws=n_output_draws,
                random_seed=random_seed,
                sigma_fixed=(sigma if not estimate_sigma else None),
                include_bias=include_bias,
                max_bytes=output_max_bytes,
            )

        return trace, ppc


    def save_bayesian_outputs(
        self,
        *,
        out_dir: str,
        trace=None,
        ppc=None,
        dataset_predict: TimeSeriesDataset | None = None,
        n_draws: int = 1000,
        random_seed: int = 42,
        sigma_fixed: float | None = None,
        include_bias: bool = True,
        include_noise: bool = True,
        max_bytes: int = 500_000_000,
    ) -> dict:
        """Save posterior sampler and decomposed uncertainty propagation files.

        This method is intentionally thin: it delegates to
        :func:`slmcal.bayes.outputs.save_bayesian_artifacts` so scripts and
        notebooks can use the same artifact format without instantiating the
        full workflow object.
        """
        if self.prior is None:
            raise RuntimeError("Prior not built")
        tr = trace if trace is not None else self.trace
        if tr is None:
            raise RuntimeError("No trace available. Run bayesian_calibrate() first or pass trace=...")
        pp = ppc if ppc is not None else self.ppc
        ds_pred = dataset_predict if dataset_predict is not None else self.dataset

        from slmcal.bayes.outputs import save_bayesian_artifacts

        return save_bayesian_artifacts(
            out_dir=out_dir,
            trace=tr,
            ppc=pp,
            prior=self.prior,
            model=self.model,
            dataset_predict=ds_pred,
            n_draws=n_draws,
            seed=random_seed,
            sigma_fixed=sigma_fixed,
            include_bias=include_bias,
            include_noise=include_noise,
            max_bytes=max_bytes,
            metadata={"workflow": self.__class__.__name__},
        )

    
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



        if isinstance(self.prior, PriorProduct):
            out = {"kind": "product", "n_blocks": len(self.prior.priors), "blocks": []}
            for pr, d in zip(self.prior.priors, self.prior.dims):
                out["blocks"].append({
                    "dim": int(d),
                    "mean": pr.mean,
                    "std": np.sqrt(np.diag(pr.cov)),
                })
            return out
        raise TypeError(f"Unsupported prior type: {type(self.prior)}")
