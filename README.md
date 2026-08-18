# slmcal

**slmcal** provides a reproducible workflow to perform a Bayesian calibration of shoreline evolution models:

1) **NSGA-II pre-calibration** (multi-objective) to build informative, site-specific parameter priors.
2) **Bayesian calibration** (PyMC) using those priors with a black-box forward model wrapper.


## Install

```bash
pip install -e .
# optional extras
pip install -e ".[bayes,plot,numba]"
# fast NSGA-II + metric suite from fast_optimization
pip install -e ".[fastopt]"
```

To install the exact versions used to produce the published results:

```bash
pip install -e ".[paper]"
```

## Reproducing the published results

The two study sites (Angourie Back Beach and La Jolla Shores) are reproduced by
running the following **in order**. Input data is in `data/`; all results are
written to `outputs/`, which is not tracked in this repository.

```bash
python examples/00_prepare_data.py
python examples/vitousek21_yates_Angourie.py
python examples/vitousek21_yates_LaJolla.py
jupyter notebook notebooks/regenerate_paper_figures.ipynb
```

`00_prepare_data.py` builds the single-series `*_default.nc` files that the two
calibration scripts read, from the raw multi-transect files in `data/`. It must
be run first.

### Run configuration

Both calibration scripts contain the settings used for the published results.
The NSGA-II pre-calibration is identical for both sites (50 generations,
population 500, 8 restarts, `kge`/`pbias`/`spearman` objectives, seed 42),
followed by a copula-KDE prior and a gradient-free `DEMetropolisZ` sampler:

| | Angourie | La Jolla |
|---|---|---|
| Chains | 6 | 4 |
| Draws | 15 000 | 10 000 |
| Tuning steps | 15 000 | 10 000 |
| Calibration / validation split | 2010-01-01 | 2010-01-01 |

Approximate cost on 6 cores: NSGA-II takes ~22 min for Angourie and ~7 min for
La Jolla; MCMC sampling takes ~6 min and ~1.5 min respectively, plus posterior
predictive sampling over the full record.

## Core concepts

- `TimeSeriesDataset`: holds time, observation series, and any forcing time series needed by the model.
- `ShorelineModel` protocol: your shoreline model adapter (simulate + parameter transforms + bounds).
- `CalibrationWorkflow`: one object that runs NSGA-II and then Bayesian calibration.

## Using your own shoreline model

Create a model adapter by implementing:

- `param_names`: list of parameter names
- `bounds_raw`: array of shape `(n_params, 2)` in the *raw* parameter space used by NSGA-II + PyMC
- `transform(raw_params) -> physical_params`
- `simulate(physical_params, dataset) -> shoreline_pred_on_dataset_time`

Then:

```python
from slmcal.core.workflow import CalibrationWorkflow
from slmcal.optimization import NSGA2Config

workflow = CalibrationWorkflow(model=my_model, dataset=my_dataset)

nsga = workflow.precalibrate_nsga2(
    metrics=("kge", "pbias", "spearman"),
    cfg=NSGA2Config(n_generations=60, pop_size=200, n_restarts=10),
    backend="internal",
)

trace = workflow.bayesian_calibrate(
    prior_from="nsga2",
    draws=2000,
    tune=1000,
    chains=4,
    likelihood="normal",  # or "studentt"
)
```

> The two calibration scripts under `examples/` do not use `CalibrationWorkflow`;
> they call the underlying functions directly. Both routes are supported.

## Repository layout

| Path | Contents |
|---|---|
| `src/slmcal/` | The package. Shoreline models, NSGA-II pre-calibration, Bayesian calibration, plotting. |
| `examples/` | Runnable scripts, including the two that produce the published results. |
| `notebooks/` | `regenerate_paper_figures.ipynb`, which rebuilds every figure from `outputs/`. |
| `data/` | Input NetCDF datasets for both sites. |
| `outputs/` | Written at run time. Not tracked here — see the Zenodo archive. |

Not every module is exercised by the published results: the package also carries
mixture-of-experts, BART, SARIMAX and 2D/multi-transect (IH-MOOSE) code used in
related work.

## Citation

Please cite the associated paper (in preparation). Software metadata for citing
this release directly is in [`CITATION.cff`](CITATION.cff).

## NSGA-II backend and metric selection

By default, `CalibrationWorkflow.precalibrate_nsga2()` uses the **fast_optimization** metric system (fast_optimization).
You choose objectives by **metric name**:

```python
from slmcal.core.workflow import CalibrationWorkflow
from slmcal.optimization import FastOptNSGA2Config

wf = CalibrationWorkflow(model, dataset)

nsga = wf.precalibrate_nsga2(
    metrics=("kge", "pbias", "spearman"),
    fast_cfg=FastOptNSGA2Config(population_size=2000, num_generations=150, n_restarts=30),
    backend="fast_optimization",
)
wf.build_prior_from_nsga2()
```

### Prior options (from NSGA-II samples)

By default the workflow fits a multivariate normal (MVN) prior in **raw parameter space**:

```python
wf.build_prior_from_nsga2(kind="mvn", cov_scale=30.0)
```

If you want to use the **empirical distribution directly**, fit a **multivariate KDE prior** instead
(requires SciPy and uses gradient-free sampling):

```python
wf.build_prior_from_nsga2(kind="kde", bw_method="scott")
```

For **high-dimensional** parameter vectors, a full multivariate KDE can become
slow and unstable. In that case, use the **Gaussian-copula KDE prior**:

```python
wf.build_prior_from_nsga2(
    kind="copula_kde",
    copula_method="factor",      # 'factor' (recommended) or 'full'
    copula_grid_size=512,        # marginal grid resolution
    copula_explained_var=0.99,   # choose factor rank automatically
    copula_shrinkage=0.05,       # regularization toward independence
)
```


If you don't have `fast_optimization` installed, set `backend="internal"` to use the built-in NSGA-II implementation.

To list the metric names supported by installed `fast_optimization`, you can run:

```python
from fast_optimization.metrics import backtot
names, _ = backtot()
print(names)
```

## Author ✍️

- **Lucas de Freitas** – 👨‍💻 [GitHub 🌐](https://github.com/defreitasL) 🌊

If you use this package in a paper or report, please consider citing the associated work and/or acknowledging the use of *slmcal* in your methodology section. 🙏