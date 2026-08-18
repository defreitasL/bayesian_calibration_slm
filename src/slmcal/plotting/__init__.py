"""Plotting helpers.

This subpackage provides lightweight functions to visualize common outputs:

- Prior vs posterior marginal distributions
- Posterior predictive envelopes vs observations
- Posterior predictive checks at observation points
- Likelihood distribution diagnostics (observations vs posterior predictive)
"""

from .io import open_precalibration_results, open_calibration_results, extract_prior_and_posterior_raw
from .moose_plots import plot_transect_timeseries_bands, plot_rotation_bands
from .plots import (
    make_default_calibration_plots,
    plot_prior_posterior,
    plot_posterior_predictive_split,
    plot_posterior_predictive_with_likelihood,
    plot_ppc_timeseries,
    plot_likelihood_distribution_kde,
    plot_residual_diagnostics
)

__all__ = [
    "open_precalibration_results",
    "open_calibration_results",
    "extract_prior_and_posterior_raw",
    "plot_prior_posterior",
    "plot_posterior_predictive_split",
    "plot_posterior_predictive_with_likelihood",
    "plot_ppc_timeseries",
    "plot_likelihood_distribution_kde",
    # "plot_posterior_predictive",
    "plot_residual_diagnostics",
    "make_default_calibration_plots",
    "plot_transect_timeseries_bands",
    "plot_rotation_bands",
    "set_paper_style",
    "load_site_datasets",
    "plot_site_timeseries",
    "plot_monthly_shoreline_distribution",
    "plot_wave_rose_by_hs",
    "plot_uncertainty_component_bands",
    "collect_synthetic_metrics",
    "plot_emv_heatmaps",
    "plot_pca_scatter_grid",
]


from .paper_figures import (
    set_paper_style,
    load_site_datasets,
    plot_site_timeseries,
    plot_monthly_shoreline_distribution,
    plot_wave_rose_by_hs,
    plot_uncertainty_component_bands,
    collect_synthetic_metrics,
    plot_emv_heatmaps,
    plot_pca_scatter_grid,
)
