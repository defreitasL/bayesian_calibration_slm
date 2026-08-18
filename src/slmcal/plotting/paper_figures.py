from __future__ import annotations

"""
Clean figure helpers for the shoreline uncertainty paper.

"""

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


PAPER_RC = {
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def set_paper_style() -> None:
    plt.rcParams.update(PAPER_RC)


def _save(fig, out_path: str | Path | None, **kwargs) -> None:
    if out_path is None:
        return
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", **kwargs)


def _np_time(x) -> np.ndarray:
    return np.asarray(x)


def _get_var(ds, names: Sequence[str]) -> np.ndarray:
    for n in names:
        if n in ds:
            return np.asarray(ds[n].values if hasattr(ds[n], "values") else ds[n], dtype=float)
    raise KeyError(f"None of the variables {names!r} were found.")


def _get_time(ds, names: Sequence[str] = ("time",)) -> np.ndarray:
    for n in names:
        if n in ds:
            return np.asarray(ds[n].values if hasattr(ds[n], "values") else ds[n])
    if hasattr(ds, "coords"):
        for n in names:
            if n in ds.coords:
                return np.asarray(ds.coords[n].values)
    raise KeyError(f"None of the time variables {names!r} were found.")


def load_site_datasets(paths: Mapping[str, str | Path], source: str | None = None, trs: List[str] | None = None) -> Mapping[str, object]:
    """Open site NetCDF files as xarray datasets keyed by site label."""
    import xarray as xr

    if source is None:
        return {label: xr.open_dataset(path) for label, path in paths.items()}

    if source == "IH-SET":
        new_paths = default_from_ihset(paths, trs=trs)
        return {label: xr.open_dataset(path) for label, path in new_paths.items()}


def default_from_ihset(paths: Mapping[str, str | Path], trs: List[str] | None = None) -> Mapping[str, Path]:
    """Transform IH-SET files into the default format."""

    import xarray as xr
    from scipy.stats import circmean

    new_paths = []
    i = 0
    
    for _, path in paths.items():
        ds = xr.open_dataset(path)
        
        trs_ = trs[i] if trs is not None else None

        time_obs = ds["time_obs"].values

        if trs_ == "average" or trs_ is None:
            obs = np.asarray(ds["average_obs"].values , dtype=float)
            hs = np.asarray(np.mean(ds["hs"].values, axis=1) , dtype=float)
            tp = np.asarray(np.mean(ds["tp"].values, axis=1) , dtype=float)
            dire = np.asarray(circmean(ds["dir"].values, high=360, low=0, axis=1) , dtype=float)


        else:
            trs_ = int(float(trs_))
            obs = np.asarray(ds["obs"].values[:, trs_], dtype=float)
            hs = np.asarray(ds["hs"].values[:, trs_], dtype=float)
            tp = np.asarray(ds["tp"].values[:, trs_], dtype=float)
            dire = np.asarray(ds["dir"].values[:, trs_], dtype=float)

        nan_mask = np.isnan(obs)
        time_obs = time_obs[~nan_mask]
        obs = obs[~nan_mask]

        # detrend obs

        mean_obs = np.nanmean(obs)
        linear_trend = np.polyfit(np.arange(obs.size), obs, 1)

        detrended_obs = obs - (linear_trend[0] * np.arange(obs.size) + linear_trend[1]) + mean_obs

        new_ds = xr.Dataset(
                    coords={"time": ds["time"],
                             "time_obs": time_obs},)

        new_ds["obs"] = (("time_obs",), detrended_obs)
        new_ds["hs"] = (("time",), hs)
        new_ds["tp"] = (("time",), tp)
        new_ds["dir"] = (("time",), dire)

        out_file = Path(path).with_name(f"{Path(path).stem}_default.nc")

        encoding = {v: {"zlib": True, "complevel": 9} for v in new_ds.data_vars}
        new_ds.to_netcdf(out_file, mode="w", format="NETCDF4", encoding=encoding)

        new_paths.append(out_file)

    paths_ = {label: out_file for label, out_file in zip(paths.keys(), new_paths)}

    return paths_


def plot_site_timeseries(
    sites: Mapping[str, object],
    *,
    obs_names: Sequence[str] = ("obs", "Obs", "shoreline"),
    obs_time_names: Sequence[str] = ("time_obs", "obs_time", "time"),
    hs_names: Sequence[str] = ("hs", "Hs", "H"),
    energy_shading: bool = True,
    shading_step: int = 30,
    out_path: str | Path | None = None,
):
    """Plot one shoreline-observation time series per site with optional wave-energy shading."""
    set_paper_style()
    n = len(sites)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(2.2, 2.0 * n)), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, (label, ds) in zip(axes, sites.items()):
        obs = _get_var(ds, obs_names)
        obs_time = _get_time(ds, obs_time_names)

        if energy_shading:
            try:
                hs = _get_var(ds, hs_names)
                time = _get_time(ds, ("time",))
                step = max(1, int(shading_step))
                n_blocks = min(time.size, hs.size) // step
                if n_blocks > 0:
                    tt = time[: n_blocks * step : step]
                    ee = (hs[: n_blocks * step] ** 2).reshape(n_blocks, step).mean(axis=1)
                    ee = (ee - np.nanmin(ee)) / max(np.nanmax(ee) - np.nanmin(ee), 1e-12)
                    cmap = plt.get_cmap("Reds")
                    for i in range(tt.size - 1):
                        ax.axvspan(tt[i], tt[i + 1], color=cmap(float(ee[i])), alpha=0.35, lw=0)
            except Exception:
                pass

        ax.plot(obs_time, obs, color="black", lw=0.8, marker="s", ms=1.5, label="observed shoreline")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylabel("Shoreline position [m]")
        ax.grid(alpha=0.25)
        m = np.isfinite(obs)
        if m.any():
            txt = f"N={m.sum()} | mean={np.nanmean(obs):.1f} m | sd={np.nanstd(obs):.1f} m"
            ax.text(0.01, 0.04, txt, transform=ax.transAxes, ha="left", va="bottom")
        if np.issubdtype(np.asarray(obs_time).dtype, np.datetime64):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    _save(fig, out_path)
    return fig, axes


def plot_monthly_shoreline_distribution(
    sites: Mapping[str, object],
    *,
    obs_names: Sequence[str] = ("obs", "Obs", "shoreline"),
    obs_time_names: Sequence[str] = ("time_obs", "obs_time", "time"),
    hs_names: Sequence[str] = ("hs", "Hs", "H"),
    out_path: str | Path | None = None,
):
    """Plot monthly shoreline distributions and monthly mean wave energy by site."""
    set_paper_style()
    n = len(sites)
    fig, axes = plt.subplots(n, 1, figsize=(8, max(2.4, 2.1 * n)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (label, ds) in zip(axes, sites.items()):
        obs = _get_var(ds, obs_names)
        obs_time = pd.to_datetime(_get_time(ds, obs_time_names))
        df = pd.DataFrame({"month": obs_time.month, "obs": obs})
        data = [df.loc[df.month == m, "obs"].dropna().values for m in range(1, 13)]
        ax.boxplot(data, positions=np.arange(1, 13), widths=0.65, showfliers=False)
        ax.plot(np.arange(1, 13), [np.nanmean(v) if len(v) else np.nan for v in data], "k--", lw=1, marker="o", ms=2)
        ax.set_ylabel("Shoreline [m]")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)

        try:
            hs = _get_var(ds, hs_names)
            time = pd.to_datetime(_get_time(ds, ("time",)))
            e = pd.DataFrame({"month": time.month, "E": hs ** 2}).groupby("month")["E"].mean()
            ax2 = ax.twinx()
            ax2.plot(e.index.values, e.values, color="forestgreen", lw=1, marker="s", ms=2)
            ax2.set_ylabel("Mean $H_s^2$ [m$^2$]", color="forestgreen")
            ax2.tick_params(axis="y", labelcolor="forestgreen")
        except Exception:
            pass

    axes[-1].set_xticks(np.arange(1, 13))
    axes[-1].set_xticklabels(list("JFMAMJJASOND"))
    axes[-1].set_xlabel("Month")
    fig.tight_layout()
    _save(fig, out_path)
    return fig, axes


def plot_wave_rose_by_hs(
    ds,
    *,
    ax=None,
    dir_names: Sequence[str] = ("dir", "Dir", "wave_dir"),
    hs_names: Sequence[str] = ("hs", "Hs", "H"),
    dir_bins: int = 36,
    hs_bins: Sequence[float] | int = 6,
    cmap: str = "turbo",
    out_path: str | Path | None = None,
):
    """Draw a polar wave rose stacked by Hs classes."""
    set_paper_style()
    directions_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    theta = np.deg2rad(_get_var(ds, dir_names) % 360.0)
    hs = _get_var(ds, hs_names)
    m = np.isfinite(theta) & np.isfinite(hs)
    theta = theta[m]
    hs = hs[m]

    if ax is None:
        fig = plt.figure(figsize=(4.2, 4.2))
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.figure

    edges = np.linspace(0, 2 * np.pi, int(dir_bins) + 1)
    width = edges[1] - edges[0]
    if isinstance(hs_bins, int):
        q = np.linspace(0, 1, int(hs_bins) + 1)
        hs_edges = np.unique(np.nanquantile(hs, q))
    else:
        hs_edges = np.asarray(hs_bins, dtype=float)
    if hs_edges.size < 2:
        hs_edges = np.linspace(np.nanmin(hs), np.nanmax(hs), 4)

    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.9, hs_edges.size - 1))
    bottom = np.zeros(int(dir_bins), dtype=float)
    for k in range(hs_edges.size - 1):
        mm = (hs >= hs_edges[k]) & (hs < hs_edges[k + 1] if k < hs_edges.size - 2 else hs <= hs_edges[k + 1])
        counts, _ = np.histogram(theta[mm], bins=edges)
        frac = counts / max(1, theta.size)
        ax.bar(edges[:-1], frac, width=width, bottom=bottom, align="edge", color=colors[k], edgecolor="white", lw=0.2,
               label=f"{hs_edges[k]:.1f}–{hs_edges[k+1]:.1f} m")
        bottom += frac

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(directions_labels)

    ylim = ax.get_ylim()
    yticks = np.linspace(ylim[0], ylim[1], 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.0%}" for y in yticks])
    ax.set_title("Wave rose by $H_s$")
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.0), frameon=False)
    _save(fig, out_path)
    return fig, ax

def plot_forcing_timeseries(ds, *, start_date: pd.Timestamp | None = None, end_date: pd.Timestamp | None = None, ax=None, out_path: str | Path | None = None):
    """Plot forcing timeseries."""
    set_paper_style()
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(8, 4), sharex=True, constrained_layout=True)
    else:
        axes = np.atleast_1d(ax)
        if axes.size != 3:
            raise ValueError(f"Expected three axes, got {axes.size}.")
        ax = axes
        fig = ax[0].figure

    time = _get_time(ds, ("time",))
    hs = _get_var(ds, ("hs", "Hs", "H"))
    tp = _get_var(ds, ("tp", "Tp", "T"))
    dire = _get_var(ds, ("dir", "Dir", "wave_dir"))

    series = (
        (ax[0], hs, r"$H_s$ [m]", "blue", "Hs"),
        (ax[1], tp, r"$T_p$ [s]", "red", "Tp"),
        (ax[2], dire, r"$\theta_{dir}$ [$^\circ$N]", "black", "Dir"),
    )

    for a, values, ylabel, color, label in series:
        if label == "Dir":
            a.scatter(time, values, label=label, color=color, s=2, alpha=0.7, linewidths=0)
        else:
            a.plot(time, values, label=label, color=color, lw=1.0)
        a.set_ylabel(ylabel)
        a.grid(alpha=0.25)
        # a.legend(frameon=False, loc="upper right")

        # set x-axis limits if start_date or end_date are provided
        if start_date is not None or end_date is not None:
            
            a.set_xlim(start_date, end_date)

            # set one tick every 2 years
            a.xaxis.set_major_locator(mdates.YearLocator(4))

    # if np.issubdtype(np.asarray(time).dtype, np.datetime64):
    #     locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    #     formatter = mdates.ConciseDateFormatter(locator)
    #     for a in ax:
    #         a.xaxis.set_major_locator(locator)
    #         a.xaxis.set_major_formatter(formatter)

    # ax[-1].set_xlabel("Time")

    _save(fig, out_path)
    return fig, ax

def load_npz_artifact(path_or_dir: str | Path, filename: str = "uncertainty_propagator.npz"):
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / filename
    return np.load(p, allow_pickle=True)


def reconstruct_uncertainty_component(
    prop,
    component: str,
    *,
    time_indices: np.ndarray | Sequence[int] | None = None,
    dtype=None,
) -> np.ndarray:
    """
    Return an absolute or effect ensemble from compact or legacy artifacts.

    The optional ``time_indices`` argument reconstructs only selected time
    columns. This is substantially more memory-efficient for coverage analyses
    because the complete hourly posterior ensemble is never materialised.

    Supported compact formats
    -------------------------
    Version 4:
        deterministic_reference
        parameter_effect
        reference_position_effect
        residual_noise

    Version 3:
        deterministic
        parameter_effect
        initial_condition_effect
        nuisance_error
    """
    requested = str(component).strip().lower()

    if hasattr(prop, "files"):
        available = set(prop.files)
    else:
        available = set(prop.keys())

    def _cast(arr):
        arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    # --------------------------------------------------------------
    # Deterministic reference and requested time selection
    # --------------------------------------------------------------
    if "deterministic_reference" in available:
        deterministic = np.asarray(
            prop["deterministic_reference"]
        ).squeeze()
    elif "deterministic" in available:
        deterministic = np.asarray(
            prop["deterministic"]
        ).squeeze()
    else:
        deterministic = None

    if deterministic is not None and deterministic.ndim != 1:
        raise ValueError(
            "The deterministic reference must be one-dimensional, "
            f"got shape {deterministic.shape}."
        )

    if deterministic is not None:
        n_time = int(deterministic.size)
    elif "time" in available:
        n_time = int(np.asarray(prop["time"]).size)
    else:
        n_time = None

    if time_indices is None:
        selected = None
    else:
        selected = np.asarray(time_indices, dtype=int).ravel()
        if selected.size == 0:
            raise ValueError("time_indices cannot be empty.")
        if n_time is not None and (
            np.any(selected < 0) or np.any(selected >= n_time)
        ):
            raise IndexError(
                "time_indices contains values outside the available "
                f"time range [0, {n_time - 1}]."
            )

    def _orient_and_slice(arr, key: str):
        """
        Orient an ensemble as [draw, time] and immediately retain only the
        requested columns. Arrays remain in their stored dtype unless ``dtype``
        is explicitly requested.
        """
        arr = np.asarray(arr)

        if arr.ndim == 1:
            if n_time is not None and arr.size != n_time:
                raise ValueError(
                    f"Component '{key}' has length {arr.size}, "
                    f"but the time coordinate has length {n_time}."
                )
            out = arr[None, :]
        elif arr.ndim == 2:
            if n_time is None:
                out = arr
            elif arr.shape[1] == n_time:
                out = arr
            elif arr.shape[0] == n_time:
                out = arr.T
            else:
                raise ValueError(
                    f"Could not identify the time dimension for '{key}'. "
                    f"Component shape={arr.shape}; time length={n_time}."
                )
        else:
            raise ValueError(
                f"Component '{key}' must be one- or two-dimensional, "
                f"got shape {arr.shape}."
            )

        if selected is not None:
            out = out[:, selected]

        return _cast(out)

    # Explicitly materialised ensembles take precedence.
    if requested in available and requested not in {
        "deterministic",
        "deterministic_reference",
    }:
        return _orient_and_slice(prop[requested], requested)

    if deterministic is None:
        raise KeyError(
            "The uncertainty artifact does not contain "
            "'deterministic_reference' or 'deterministic'. "
            f"Available keys: {sorted(available)}"
        )

    deterministic_selected = (
        deterministic
        if selected is None
        else deterministic[selected]
    )
    deterministic_selected = _cast(deterministic_selected)

    def _load_effect(
        preferred_key: str,
        legacy_key: str | None = None,
    ) -> np.ndarray | None:
        key = None
        if preferred_key in available:
            key = preferred_key
        elif legacy_key is not None and legacy_key in available:
            key = legacy_key

        if key is None:
            return None

        # Slice immediately after decompression so only the selected columns
        # are retained when this function is used for coverage calculations.
        return _orient_and_slice(prop[key], key)

    parameter_effect = _load_effect("parameter_effect")
    reference_effect = _load_effect(
        "reference_position_effect",
        "initial_condition_effect",
    )
    residual_effect = _load_effect(
        "residual_noise",
        "nuisance_error",
    )

    sample_counts = [
        arr.shape[0]
        for arr in (
            parameter_effect,
            reference_effect,
            residual_effect,
        )
        if arr is not None
    ]

    if sample_counts and len(set(sample_counts)) != 1:
        raise ValueError(
            "The propagated uncertainty components do not contain the "
            f"same number of posterior draws: {sample_counts}."
        )

    n_samples = sample_counts[0] if sample_counts else 1
    n_selected = int(deterministic_selected.size)

    base = np.broadcast_to(
        deterministic_selected[None, :],
        (n_samples, n_selected),
    )

    zeros = np.zeros(
        (n_samples, n_selected),
        dtype=(
            dtype
            if dtype is not None
            else deterministic_selected.dtype
        ),
    )

    if parameter_effect is None:
        parameter_effect = zeros
    if reference_effect is None:
        reference_effect = zeros
    if residual_effect is None:
        residual_effect = zeros

    effect_mapping = {
        "parameter_effect": parameter_effect,
        "physical_parameter_effect": parameter_effect,
        "reference_position_effect": reference_effect,
        "reference_effect": reference_effect,
        "initial_condition_effect": reference_effect,
        "residual_noise": residual_effect,
        "residual_effect": residual_effect,
        "nuisance_error": residual_effect,
    }

    if requested in effect_mapping:
        return _cast(effect_mapping[requested])

    if requested in {
        "reference",
        "deterministic",
        "deterministic_reference",
    }:
        return _cast(base)

    effect_sets = {
        "parameter_only": (parameter_effect,),
        "physical_parameters_only": (parameter_effect,),
        "physical_parameter_only": (parameter_effect,),

        "reference_position_only": (reference_effect,),
        "reference_only": (reference_effect,),

        "residual_only": (residual_effect,),
        "noise_only": (residual_effect,),
        "uncertainty_only": (residual_effect,),
        "nuisance_only": (residual_effect,),

        "latent": (parameter_effect, reference_effect),
        "latent_only": (parameter_effect, reference_effect),
        "latent_shoreline": (parameter_effect, reference_effect),
        "joint_model_only": (parameter_effect, reference_effect),
        "model_only": (parameter_effect, reference_effect),

        "full": (
            parameter_effect,
            reference_effect,
            residual_effect,
        ),
        "posterior_predictive": (
            parameter_effect,
            reference_effect,
            residual_effect,
        ),
        "predictive": (
            parameter_effect,
            reference_effect,
            residual_effect,
        ),
    }

    if requested not in effect_sets:
        raise KeyError(
            f"Unknown propagated component {component!r}. "
            f"Available artifact keys: {sorted(available)}. "
            "Supported reconstructed components include "
            "'parameter_only', 'reference_position_only', "
            "'residual_only', 'latent' and 'full'."
        )

    # One output matrix is allocated and each effect is added in-place.
    # This avoids the several full-size temporary arrays created by chained
    # NumPy additions.
    out = np.array(base, copy=True)
    for effect in effect_sets[requested]:
        out += effect

    return _cast(out)


def plot_uncertainty_component_bands(
    artifact_dir: str | Path,
    *,
    label: str = "posterior propagation",
    out_path: str | Path | None = None,
):
    """Plot physical-parameter, y0, nuisance and full posterior bands."""
    set_paper_style()
    prop = load_npz_artifact(
        artifact_dir,
        "uncertainty_propagator.npz",
    )
    time = prop["time"]

    comps = [
        "parameter_only",
        "initial_condition_only",
        "uncertainty_only",
        "full",
    ]
    titles = [
        "Physical parameters only",
        "Initial condition $Y_0$ only",
        r"Bias + residual $\sigma$ only",
        "Full posterior predictive uncertainty",
    ]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 8.8),
        sharex=True,
        sharey=True,
    )

    for ax, comp, title in zip(axes, comps, titles):
        d = reconstruct_uncertainty_component(prop, comp)
        p05, p50, p95 = np.nanpercentile(
            d,
            [5, 50, 95],
            axis=0,
        )
        p01, p99 = np.nanpercentile(
            d,
            [1, 99],
            axis=0,
        )
        ax.fill_between(
            time,
            p01,
            p99,
            alpha=0.16,
            label="1–99%",
        )
        ax.fill_between(
            time,
            p05,
            p95,
            alpha=0.32,
            label="5–95%",
        )
        ax.plot(
            time,
            p50,
            lw=1.0,
            color="black",
            label="median",
        )
        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
        )
        ax.set_ylabel("Shoreline [m]")
        ax.grid(alpha=0.25)

        if np.issubdtype(
            np.asarray(time).dtype,
            np.datetime64,
        ):
            ax.xaxis.set_major_locator(
                mdates.AutoDateLocator(
                    minticks=3,
                    maxticks=8,
                )
            )
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(
                    ax.xaxis.get_major_locator()
                )
            )

    axes[0].legend(
        ncol=3,
        frameon=False,
        loc="best",
    )
    axes[0].set_title(label, loc="right")
    axes[-1].set_xlabel("Time")

    fig.tight_layout()
    _save(fig, out_path)
    return fig, axes


def plot_uncertainty_component_bands_full(
    artifact_dir: str | Path,
    *,
    label: str = "posterior propagation",
    out_path: str | Path | None = None,
):
    """Plot full, parameter-only and uncertainty-only posterior bands from saved artifacts."""
    set_paper_style()
    prop = load_npz_artifact(artifact_dir, "uncertainty_propagator.npz")
    time = prop["time"]
    comp = "full"

    fig, ax = plt.subplots(1, 1, figsize=(11, 3), sharey=True)
    d = reconstruct_uncertainty_component(prop, comp)
    p10, p50, p90 = np.nanpercentile(d, [10, 50, 90], axis=0)
    p01, p99 = np.nanpercentile(d, [1, 99], axis=0)
    ax.fill_between(time, p01, p10, alpha=0.60, label="1–99%", color="red", linewidth=0)
    ax.fill_between(time, p90, p99, alpha=0.60, label="", color="red", linewidth=0)
    ax.fill_between(time, p10, p90, alpha=0.45, label="10–90%", color="green", linewidth=0)
    ax.plot(time, p50, lw=0.5, color="black", label="median")
    ax.set_ylabel("Shoreline position [m]")
    ax.grid(alpha=0.25)
    if np.issubdtype(np.asarray(time).dtype, np.datetime64):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        # set one tick every two years
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    ax.set_xlim(time[0], time[-1])
    ax.legend(ncol=3, frameon=False, loc="best")
    ax.set_title(label, loc="right")
    ax.set_xlabel("Time")
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax

def collect_synthetic_metrics(
    results_dir: str | Path,
    *,
    record_lengths: Sequence[int],
    noise_magnitudes: Sequence[float],
    sampling_windows: Sequence[int],
    file_template: str = "results_synthetic_rl{rl}_wn{wn}_dsw{dsw}.nc",
    posterior_var: str = "posterior",
    lines_var: str = "new_lines",
    cache_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect EMV/variance metrics and 2D PCA posterior coordinates efficiently.

    The function opens one NetCDF file at a time and closes it immediately,
    avoiding the memory build-up of the exploratory notebook.
    """
    import xarray as xr
    from sklearn.decomposition import PCA

    results_dir = Path(results_dir)
    rows = []
    pc_rows = []

    for rl in record_lengths:
        for wn in noise_magnitudes:
            for dsw in sampling_windows:
                path = results_dir / file_template.format(rl=rl, wn=wn, dsw=dsw)
                if not path.exists():
                    continue
                with xr.open_dataset(path) as ds:
                    if lines_var in ds:
                        lines = np.asarray(ds[lines_var].values, dtype=float)
                        emv = np.nanmax(lines, axis=1) - np.nanmin(lines, axis=1)
                        var = np.nanvar(lines, axis=1)
                        std = np.nanstd(lines, axis=1)
                        rows.append({
                            "RL": int(rl), "RL_years": int(round(rl / 365)), "WNM": float(wn), "DSW": int(dsw),
                            "EMV_mean": float(np.nanmean(emv)), "EMV_std": float(np.nanstd(emv)), "EMV_max": float(np.nanmax(emv)),
                            "var_mean": float(np.nanmean(var)), "var_max": float(np.nanmax(var)),
                            "std_mean": float(np.nanmean(std)), "std_max": float(np.nanmax(std)),
                        })
                    if posterior_var in ds:
                        par = np.asarray(ds[posterior_var].values, dtype=float).reshape(-1, ds[posterior_var].shape[-1])
                        par = par[np.all(np.isfinite(par), axis=1)]
                        if par.shape[0] >= 3:
                            pc = PCA(n_components=2).fit_transform(par)
                            # Store a bounded subset for plotting large experiments.
                            n_keep = min(pc.shape[0], 5000)
                            idx = np.linspace(0, pc.shape[0] - 1, n_keep).astype(int)
                            tmp = pd.DataFrame(pc[idx], columns=["PC1", "PC2"])
                            tmp["RL"] = int(rl)
                            tmp["RL_years"] = int(round(rl / 365))
                            tmp["WNM"] = float(wn)
                            tmp["DSW"] = int(dsw)
                            pc_rows.append(tmp)

    df_metrics = pd.DataFrame(rows)
    df_pc = pd.concat(pc_rows, ignore_index=True) if pc_rows else pd.DataFrame()
    if cache_csv is not None:
        cache = Path(cache_csv)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(cache, index=False)
        if not df_pc.empty:
            df_pc.to_csv(cache.with_name(cache.stem + "_pc.csv"), index=False)
    return df_metrics, df_pc


def plot_emv_heatmaps(
    df_metrics: pd.DataFrame,
    *,
    value: str = "EMV_mean",
    fixed: tuple[str, float | int] | None = None,
    out_path: str | Path | None = None,
):
    """Plot a compact set of heatmaps for synthetic uncertainty metrics."""
    set_paper_style()
    df = df_metrics.copy()
    if fixed is not None:
        key, val = fixed
        df = df[df[key] == val]

    panels = []
    if "RL_years" in df and "WNM" in df and "DSW" in df:
        # Default: one panel per record length.
        for rl in sorted(df["RL_years"].dropna().unique()):
            panels.append((f"RL={rl:g} yr", df[df["RL_years"] == rl], "DSW", "WNM"))

    n = max(1, len(panels))
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.0), squeeze=False)
    axes = axes.ravel()
    mappable = None
    for ax, (title, sub, x, y) in zip(axes, panels):
        tab = sub.pivot_table(index=y, columns=x, values=value, aggfunc="mean").sort_index(ascending=True)
        im = ax.imshow(tab.values, origin="lower", aspect="auto")
        mappable = im
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_xticks(np.arange(tab.shape[1]))
        ax.set_xticklabels([str(v) for v in tab.columns])
        ax.set_yticks(np.arange(tab.shape[0]))
        ax.set_yticklabels([str(v) for v in tab.index])
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                if np.isfinite(tab.values[i, j]):
                    ax.text(j, i, f"{tab.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes.tolist(), shrink=0.8, label=value)
    fig.tight_layout()
    _save(fig, out_path)
    return fig, axes


def plot_pca_scatter_grid(
    df_pc: pd.DataFrame,
    *,
    color_by: str = "DSW",
    fixed: Mapping[str, float | int] | None = None,
    out_path: str | Path | None = None,
):
    """Plot posterior PCA coordinates for quick prior/posterior contraction checks."""
    set_paper_style()
    df = df_pc.copy()
    if fixed:
        for k, v in fixed.items():
            df = df[df[k] == v]
    if df.empty:
        raise ValueError("No PCA rows available after filtering.")

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    vals = pd.Categorical(df[color_by])
    sc = ax.scatter(df["PC1"], df["PC2"], c=vals.codes, s=4, alpha=0.25)
    handles = []
    for code, cat in enumerate(vals.categories):
        handles.append(ax.scatter([], [], c=[sc.cmap(sc.norm(code))], s=20, label=f"{color_by}={cat}"))
    ax.legend(handles=handles, frameon=False, markerscale=1.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, out_path)
    return fig, ax


def plot_bayesian_marginal_distributions(
    artifact_dirs: Mapping[str, str | Path],
    *,
    model_param_names: Sequence[str] | None = None,
    split_timescales: bool = False,
    include_bias: bool = True,
    include_sigma: bool = True,
    show_prior_for_model_params: bool = True,
    show_prior_for_bias_sigma: bool = True,
    bias_prior_mean: float = 0.0,
    bias_prior_sd: float = 10.0,
    sigma_prior_scale: float = 30.0,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    max_points: int = 12000,
    n_grid: int = 350,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (3.5, 2.75),
    title: str | None = (
        "Vitousek21-Yates prior/posterior marginal distributions"
    ),
    normalize_density: str | None = None,
    show_prior_fill: bool = True,
    prior_fill_alpha: float = 0.10,
    posterior_fill_alpha: float = 0.00,
    out_path: str | Path | None = None,
    summary_csv: str | Path | None = None,
):
    """
    Plot one-dimensional prior and posterior marginal distributions.

    Parameters
    ----------
    artifact_dirs
        Mapping between site labels and output directories. Each path may point
        to:

            site_output/
            site_output/bayesian_artifacts/
            bayesian_parameter_sampler.npz

    model_param_names
        Names of the physical model parameters. When omitted, names are inferred
        for the Vitousek21-Yates formulation.

    normalize_density
        Controls optional vertical normalization:

        None
            Plot the true probability density.

        "panel_max"
            Divide every curve in one panel by the largest density found in that
            panel. The relative peak magnitude among curves is retained.

        "curve_max"
            Divide each curve by its own maximum. Every curve has a peak of one.
            This is useful for comparing distribution location and shape.

    show_prior_for_bias_sigma
        Plot the analytical priors:

            bias  ~ Normal(bias_prior_mean, bias_prior_sd)
            sigma ~ HalfNormal(sigma_prior_scale)

    Returns
    -------
    fig, axes, summary_df
    """
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde, norm, halfnorm
    import warnings

    set_paper_style()

    valid_normalizations = {None, "panel_max", "curve_max"}
    if normalize_density not in valid_normalizations:
        raise ValueError(
            "normalize_density must be None, 'panel_max', or 'curve_max'."
        )

    # ------------------------------------------------------------------
    # Display labels
    # ------------------------------------------------------------------
    default_labels = {
        "DeltaY": r"$\Delta Y$ [m]",
        "DeltaT": r"$\Delta T$ [days]",
        "DeltaT_acc": r"$\Delta T_{\mathrm{acc}}$ [days]",
        "DeltaT_ero": r"$\Delta T_{\mathrm{ero}}$ [days]",
        "Hhat": r"$\hat{H}$ [m]",
        "bias": r"Bias [m]",
        "sigma": r"$\sigma$ [m]",
        "delta_y": r"$\Delta Y$ [m]",
        "delta_t": r"$\Delta T$ [days]",
        "delta_t_acc": r"$\Delta T_{\mathrm{acc}}$ [days]",
        "delta_t_ero": r"$\Delta T_{\mathrm{ero}}$ [days]",
        "hhat": r"$\hat{H}$ [m]",
    }

    if labels is not None:
        default_labels.update(dict(labels))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _artifact_file(path_or_dir: str | Path) -> Path:
        p = Path(path_or_dir)

        if p.name == "bayesian_parameter_sampler.npz":
            return p

        if p.name == "bayesian_artifacts":
            return p / "bayesian_parameter_sampler.npz"

        direct = p / "bayesian_parameter_sampler.npz"
        if direct.exists():
            return direct

        return p / "bayesian_artifacts" / "bayesian_parameter_sampler.npz"

    def _available_key(z, candidates: Sequence[str]) -> str | None:
        for key in candidates:
            if key in z.files:
                return key
        return None

    def _as_1d_finite(x) -> np.ndarray:
        arr = np.asarray(x, dtype=float).ravel()
        return arr[np.isfinite(arr)]

    def _subsample(
        x: np.ndarray,
        *,
        seed: int = 42,
    ) -> np.ndarray:
        x = _as_1d_finite(x)

        if x.size > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(
                x.size,
                size=max_points,
                replace=False,
            )
            x = x[idx]

        return x

    def _is_constant(x: np.ndarray) -> bool:
        x = _as_1d_finite(x)

        if x.size <= 1:
            return True

        spread = np.nanmax(x) - np.nanmin(x)

        return bool(
            np.nanstd(x) <= 1e-12
            or spread <= 1e-12
        )

    def _clean_name(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _infer_param_names(
        npar: int,
        stored_names=None,
    ) -> list[str]:
        if model_param_names is not None:
            names = list(model_param_names)

            if len(names) != npar:
                raise ValueError(
                    f"model_param_names has length {len(names)}, "
                    f"but posterior_physical has {npar} columns."
                )

            return names

        if stored_names is not None and len(stored_names) == npar:
            return [_clean_name(value) for value in stored_names]

        if split_timescales and npar == 4:
            return [
                "DeltaY",
                "DeltaT_acc",
                "DeltaT_ero",
                "Hhat",
            ]

        if not split_timescales and npar == 3:
            return [
                "DeltaY",
                "DeltaT",
                "Hhat",
            ]

        return [
            f"param_{i + 1}"
            for i in range(npar)
        ]

    def _summary_row(
        site: str,
        source: str,
        variable: str,
        samples: np.ndarray,
    ) -> dict:
        x = _as_1d_finite(samples)

        if x.size == 0:
            return {
                "site": site,
                "source": source,
                "variable": variable,
                "n": 0,
                "mean": np.nan,
                "sd": np.nan,
                "q02_5": np.nan,
                "q05": np.nan,
                "q50": np.nan,
                "q95": np.nan,
                "q97_5": np.nan,
                "is_constant": True,
            }

        return {
            "site": site,
            "source": source,
            "variable": variable,
            "n": int(x.size),
            "mean": float(np.nanmean(x)),
            "sd": float(np.nanstd(x)),
            "q02_5": float(np.nanquantile(x, 0.025)),
            "q05": float(np.nanquantile(x, 0.05)),
            "q50": float(np.nanquantile(x, 0.50)),
            "q95": float(np.nanquantile(x, 0.95)),
            "q97_5": float(np.nanquantile(x, 0.975)),
            "is_constant": bool(_is_constant(x)),
        }

    def _analytic_prior_summary(variable: str) -> dict:
        if variable == "bias":
            dist = norm(
                loc=bias_prior_mean,
                scale=bias_prior_sd,
            )

        elif variable == "sigma":
            dist = halfnorm(
                scale=sigma_prior_scale,
            )

        else:
            raise ValueError(variable)

        return {
            "site": "all",
            "source": "prior_analytic",
            "variable": variable,
            "n": np.nan,
            "mean": float(dist.mean()),
            "sd": float(dist.std()),
            "q02_5": float(dist.ppf(0.025)),
            "q05": float(dist.ppf(0.05)),
            "q50": float(dist.ppf(0.50)),
            "q95": float(dist.ppf(0.95)),
            "q97_5": float(dist.ppf(0.975)),
            "is_constant": False,
        }

    def _sample_plot_range(
        arrays: Sequence[np.ndarray],
        *,
        lower_quantile: float = 0.005,
        upper_quantile: float = 0.995,
        padding: float = 0.10,
        lower_bound: float | None = None,
    ) -> tuple[float, float]:
        finite_arrays = []

        for arr in arrays:
            values = _as_1d_finite(arr)

            if values.size:
                finite_arrays.append(values)

        if not finite_arrays:
            raise ValueError(
                "No finite values were available to define the plotting range."
            )

        combined = np.concatenate(finite_arrays)

        lo, hi = np.nanquantile(
            combined,
            [lower_quantile, upper_quantile],
        )

        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.nanmin(combined))
            hi = float(np.nanmax(combined))

        if np.isclose(lo, hi):
            centre = float(np.nanmean(combined))
            half_width = max(
                abs(centre) * 0.05,
                1e-3,
            )
            lo = centre - half_width
            hi = centre + half_width

        pad = padding * (hi - lo)

        lo -= pad
        hi += pad

        if lower_bound is not None:
            lo = max(float(lower_bound), lo)

        return float(lo), float(hi)

    def _kde_on_grid(
        samples: np.ndarray,
        x_grid: np.ndarray,
    ) -> np.ndarray | None:
        x = _subsample(samples)

        if x.size == 0 or _is_constant(x):
            return None

        try:
            kde = gaussian_kde(x)
            y = np.asarray(kde(x_grid), dtype=float)

            if not np.any(np.isfinite(y)):
                return None

            return y

        except Exception:
            return None

    def _normalise_curves(curves: list[dict]) -> list[dict]:
        density_curves = [
            curve
            for curve in curves
            if curve.get("y") is not None
        ]

        if normalize_density is None:
            return curves

        if normalize_density == "panel_max":
            maxima = [
                np.nanmax(curve["y"])
                for curve in density_curves
                if np.any(np.isfinite(curve["y"]))
            ]

            panel_max = (
                float(np.nanmax(maxima))
                if maxima
                else np.nan
            )

            if np.isfinite(panel_max) and panel_max > 0:
                for curve in density_curves:
                    curve["y"] = curve["y"] / panel_max

        elif normalize_density == "curve_max":
            for curve in density_curves:
                curve_max = np.nanmax(curve["y"])

                if np.isfinite(curve_max) and curve_max > 0:
                    curve["y"] = curve["y"] / curve_max

        return curves

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    loaded: dict[str, dict] = {}
    summary: list[dict] = []

    for site, artifact_dir in artifact_dirs.items():
        sampler_file = _artifact_file(artifact_dir)

        if not sampler_file.exists():
            warnings.warn(
                f"Missing Bayesian sampler file for {site}: "
                f"{sampler_file}"
            )
            continue

        with np.load(
            sampler_file,
            allow_pickle=True,
        ) as z:

            if "posterior_physical" not in z.files:
                raise KeyError(
                    f"{sampler_file} does not contain "
                    "'posterior_physical'. "
                    f"Available keys: {z.files}"
                )

            posterior = np.asarray(
                z["posterior_physical"],
                dtype=float,
            )

            if posterior.ndim != 2:
                posterior = posterior.reshape(
                    -1,
                    posterior.shape[-1],
                )

            stored_names = None

            if "param_names" in z.files:
                stored_names = np.asarray(
                    z["param_names"]
                ).ravel()

            names = _infer_param_names(
                posterior.shape[1],
                stored_names,
            )

            site_data = {
                "posterior": {},
                "prior": {},
                "keys": list(z.files),
            }

            for i, name in enumerate(names):
                samples = posterior[:, i]

                site_data["posterior"][name] = samples

                summary.append(
                    _summary_row(
                        site,
                        "posterior",
                        name,
                        samples,
                    )
                )

            if (
                show_prior_for_model_params
                and "prior_physical" in z.files
            ):
                prior = np.asarray(
                    z["prior_physical"],
                    dtype=float,
                )

                if prior.ndim != 2:
                    prior = prior.reshape(
                        -1,
                        prior.shape[-1],
                    )

                if prior.shape[1] == len(names):
                    for i, name in enumerate(names):
                        samples = prior[:, i]

                        site_data["prior"][name] = samples

                        summary.append(
                            _summary_row(
                                site,
                                "prior",
                                name,
                                samples,
                            )
                        )

                else:
                    warnings.warn(
                        f"prior_physical for {site} has "
                        f"{prior.shape[1]} columns, but "
                        f"{len(names)} were expected. "
                        "The model-parameter prior will be skipped."
                    )

            bias_key = _available_key(
                z,
                [
                    "bias",
                    "posterior_bias",
                    "bias_samples",
                    "posterior_bias_samples",
                ],
            )

            if include_bias and bias_key is not None:
                bias = _as_1d_finite(z[bias_key])

                site_data["posterior"]["bias"] = bias

                summary.append(
                    _summary_row(
                        site,
                        "posterior",
                        "bias",
                        bias,
                    )
                )

            elif include_bias:
                warnings.warn(
                    f"No bias posterior was found for {site}. "
                    f"Available keys: {z.files}"
                )

            sigma_key = _available_key(
                z,
                [
                    "sigma",
                    "posterior_sigma",
                    "sigma_samples",
                    "posterior_sigma_samples",
                ],
            )

            if include_sigma and sigma_key is not None:
                sigma = _as_1d_finite(z[sigma_key])

                site_data["posterior"]["sigma"] = sigma

                summary.append(
                    _summary_row(
                        site,
                        "posterior",
                        "sigma",
                        sigma,
                    )
                )

            elif include_sigma:
                warnings.warn(
                    f"No sigma posterior was found for {site}. "
                    f"Available keys: {z.files}"
                )

            loaded[site] = site_data

    if not loaded:
        raise FileNotFoundError(
            "No valid bayesian_parameter_sampler.npz files were found."
        )

    if show_prior_for_bias_sigma:
        if include_bias:
            summary.append(
                _analytic_prior_summary("bias")
            )

        if include_sigma:
            summary.append(
                _analytic_prior_summary("sigma")
            )

    # ------------------------------------------------------------------
    # Establish panel order
    # ------------------------------------------------------------------
    plot_vars: list[str] = []

    first_site = next(iter(loaded))

    for variable in loaded[first_site]["posterior"]:
        if variable not in plot_vars:
            plot_vars.append(variable)

    for site_data in loaded.values():
        for variable in site_data["posterior"]:
            if variable not in plot_vars:
                plot_vars.append(variable)

    nvars = len(plot_vars)
    ncols = max(
        1,
        min(int(ncols), nvars),
    )
    nrows = int(np.ceil(nvars / ncols))

    # ------------------------------------------------------------------
    # Colours
    # ------------------------------------------------------------------
    if colors is None:
        cmap = plt.get_cmap("tab10")

        colors = {
            site: cmap(i % 10)
            for i, site in enumerate(loaded)
        }

    else:
        colors = dict(colors)
        cmap = plt.get_cmap("tab10")

        for i, site in enumerate(loaded):
            colors.setdefault(
                site,
                cmap(i % 10),
            )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig_width = figsize_per_panel[0] * ncols
    fig_height = figsize_per_panel[1] * nrows + 0.75

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=False,
    )

    axes_flat = axes.ravel()

    # Reserve space for title and one clean global legend.
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.10,
        top=0.79,
        wspace=0.38,
        hspace=0.42,
    )

    # ------------------------------------------------------------------
    # Plot panels
    # ------------------------------------------------------------------
    for panel_index, (ax, variable) in enumerate(
        zip(axes_flat, plot_vars)
    ):
        sample_arrays: list[np.ndarray] = []

        # Parameters use prior + posterior ranges.
        if variable not in {"bias", "sigma"}:
            for site_data in loaded.values():
                if variable in site_data["prior"]:
                    sample_arrays.append(
                        site_data["prior"][variable]
                    )

                if variable in site_data["posterior"]:
                    sample_arrays.append(
                        site_data["posterior"][variable]
                    )

            x_lo, x_hi = _sample_plot_range(
                sample_arrays,
                padding=0.08,
            )

        # Bias and sigma use the posterior range. The broad analytical
        # prior is evaluated over this range to preserve readability.
        else:
            for site_data in loaded.values():
                if variable in site_data["posterior"]:
                    sample_arrays.append(
                        site_data["posterior"][variable]
                    )

            lower_bound = 0.0 if variable == "sigma" else None

            x_lo, x_hi = _sample_plot_range(
                sample_arrays,
                padding=0.15,
                lower_bound=lower_bound,
            )

        x_grid = np.linspace(
            x_lo,
            x_hi,
            n_grid,
        )

        curves: list[dict] = []

        # --------------------------------------------------------------
        # Analytical prior for bias and sigma
        # --------------------------------------------------------------
        if (
            show_prior_for_bias_sigma
            and variable == "bias"
        ):
            y_prior = norm(
                loc=bias_prior_mean,
                scale=bias_prior_sd,
            ).pdf(x_grid)

            curves.append(
                {
                    "kind": "density",
                    "role": "prior",
                    "site": None,
                    "color": "0.35",
                    "x": x_grid,
                    "y": y_prior,
                    "analytic": True,
                }
            )

        elif (
            show_prior_for_bias_sigma
            and variable == "sigma"
        ):
            y_prior = halfnorm(
                scale=sigma_prior_scale,
            ).pdf(x_grid)

            curves.append(
                {
                    "kind": "density",
                    "role": "prior",
                    "site": None,
                    "color": "0.35",
                    "x": x_grid,
                    "y": y_prior,
                    "analytic": True,
                }
            )

        # --------------------------------------------------------------
        # Site priors and posteriors
        # --------------------------------------------------------------
        for site, site_data in loaded.items():
            site_color = colors[site]

            if (
                show_prior_for_model_params
                and variable in site_data["prior"]
            ):
                prior_samples = site_data["prior"][variable]

                if _is_constant(prior_samples):
                    curves.append(
                        {
                            "kind": "constant",
                            "role": "prior",
                            "site": site,
                            "color": site_color,
                            "value": float(
                                np.nanmean(prior_samples)
                            ),
                            "analytic": False,
                        }
                    )

                else:
                    prior_density = _kde_on_grid(
                        prior_samples,
                        x_grid,
                    )

                    if prior_density is not None:
                        curves.append(
                            {
                                "kind": "density",
                                "role": "prior",
                                "site": site,
                                "color": site_color,
                                "x": x_grid,
                                "y": prior_density,
                                "analytic": False,
                            }
                        )

            if variable in site_data["posterior"]:
                posterior_samples = site_data["posterior"][variable]

                if _is_constant(posterior_samples):
                    curves.append(
                        {
                            "kind": "constant",
                            "role": "posterior",
                            "site": site,
                            "color": site_color,
                            "value": float(
                                np.nanmean(posterior_samples)
                            ),
                            "analytic": False,
                        }
                    )

                else:
                    posterior_density = _kde_on_grid(
                        posterior_samples,
                        x_grid,
                    )

                    if posterior_density is not None:
                        curves.append(
                            {
                                "kind": "density",
                                "role": "posterior",
                                "site": site,
                                "color": site_color,
                                "x": x_grid,
                                "y": posterior_density,
                                "analytic": False,
                            }
                        )

        curves = _normalise_curves(curves)

        # --------------------------------------------------------------
        # Draw curves
        # --------------------------------------------------------------
        for curve in curves:
            color = curve["color"]
            role = curve["role"]

            if curve["kind"] == "constant":
                ax.axvline(
                    curve["value"],
                    color=color,
                    linestyle=(
                        "--"
                        if role == "posterior"
                        else "-"
                    ),
                    linewidth=(
                        2.0
                        if role == "posterior"
                        else 1.3
                    ),
                    alpha=0.95,
                    zorder=4,
                )
                continue

            x_values = curve["x"]
            y_values = curve["y"]

            if role == "prior":
                ax.plot(
                    x_values,
                    y_values,
                    color=color,
                    linestyle="-",
                    linewidth=1.35,
                    alpha=0.95,
                    zorder=2,
                )

                if show_prior_fill:
                    ax.fill_between(
                        x_values,
                        0.0,
                        y_values,
                        color=color,
                        alpha=prior_fill_alpha,
                        linewidth=0,
                        zorder=1,
                    )

            else:
                ax.plot(
                    x_values,
                    y_values,
                    color=color,
                    linestyle="--",
                    linewidth=2.1,
                    alpha=1.0,
                    zorder=4,
                )

                if posterior_fill_alpha > 0:
                    ax.fill_between(
                        x_values,
                        0.0,
                        y_values,
                        color=color,
                        alpha=posterior_fill_alpha,
                        linewidth=0,
                        zorder=3,
                    )

        # --------------------------------------------------------------
        # Panel formatting
        # --------------------------------------------------------------
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(
            default_labels.get(
                variable,
                variable,
            )
        )

        if panel_index % ncols == 0:
            if normalize_density is None:
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("Relative density")
        else:
            ax.set_ylabel("")

        if normalize_density is None:
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_ylim(0.0, 1.03)
            ax.set_yticks(
                [0.0, 0.5, 1.0]
            )

        ax.grid(
            alpha=0.18,
            linestyle="-",
            linewidth=0.6,
        )

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
        )

    # Hide unused axes.
    for ax in axes_flat[nvars:]:
        ax.axis("off")

    # ------------------------------------------------------------------
    # Clean global legend
    # ------------------------------------------------------------------
    site_handles = [
        Line2D(
            [0],
            [0],
            color=colors[site],
            linewidth=2.5,
            linestyle="-",
            label=site,
        )
        for site in loaded
    ]

    style_handles = []

    if (
        show_prior_for_model_params
        or show_prior_for_bias_sigma
    ):
        style_handles.append(
            Line2D(
                [0],
                [0],
                color="0.30",
                linewidth=1.5,
                linestyle="-",
                label="Prior",
            )
        )

    style_handles.append(
        Line2D(
            [0],
            [0],
            color="0.30",
            linewidth=2.1,
            linestyle="--",
            label="Posterior",
        )
    )

    legend_handles = site_handles + style_handles

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=len(legend_handles),
        frameon=False,
        columnspacing=1.6,
        handlelength=2.3,
        handletextpad=0.55,
        borderaxespad=0.0,
    )

    # Compact note describing the analytical residual priors.
    if show_prior_for_bias_sigma and (
        include_bias or include_sigma
    ):
        prior_notes = []

        if include_bias:
            prior_notes.append(
                rf"Bias prior: $\mathcal{{N}}"
                rf"({bias_prior_mean:g},\,{bias_prior_sd:g})$"
            )

        if include_sigma:
            prior_notes.append(
                rf"$\sigma$ prior: HalfNormal"
                rf"$({sigma_prior_scale:g})$"
            )

        fig.text(
            0.5,
            0.845,
            "   |   ".join(prior_notes),
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
        )

    if title:
        fig.suptitle(
            title,
            fontweight="bold",
            y=0.985,
        )

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary)

    if summary_csv is not None:
        summary_path = Path(summary_csv)
        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        summary_df.to_csv(
            summary_path,
            index=False,
        )

    _save(
        fig,
        out_path,
    )

    return fig, axes, summary_df


def plot_full_uncertainty_bands_with_coverage_panel(
    artifact_dir: str | Path,
    *,
    obs=None,
    obs_time=None,
    obs_dataset=None,
    obs_names: Sequence[str] = ("obs", "Obs", "shoreline"),
    obs_time_names: Sequence[str] = ("time_obs", "obs_time", "time"),
    split_date: str | pd.Timestamp | np.datetime64 | None = None,
    start_date: str | pd.Timestamp | np.datetime64 | None = None,
    end_date: str | pd.Timestamp | np.datetime64 | None = None,
    label: str = "posterior propagation",
    comp: str = "full",
    coverage_levels: Sequence[float] | None = None,
    match_tolerance: str | pd.Timedelta | None = None,
    figsize: tuple[float, float] = (12.5, 3.2),
    out_path: str | Path | None = None,
    max_plot_points: int | None = 12000,
    coverage_seed: int = 42,
    antithetic_residuals: bool = True,
):
    """
    Efficiently plot posterior uncertainty bands and coverage diagnostics.

    The visual style is the original one:

    - outer 1–99% interval in red;
    - central 10–90% interval in green;
    - posterior median in blue;
    - observations in black;
    - compact coverage panel on the right;
    - one centred legend above the time-series panel.

    Efficiency
    ----------
    The time-series bands are read from
    ``uncertainty_propagator_summary.npz`` whenever available. Coverage is
    evaluated only at matched observation times. For version-4 ``full``
    artifacts, residual draws are generated from the saved posterior ``sigma``
    values at those observation times, so the enormous hourly ``residual_noise``
    matrix is not loaded.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import warnings

    set_paper_style()

    if coverage_levels is None:
        coverage_levels = np.array(
            [0.50, 0.67, 0.80, 0.90, 0.95, 0.975, 0.99, 1.00]
        )
    else:
        coverage_levels = np.asarray(coverage_levels, dtype=float)

    if np.any(~np.isfinite(coverage_levels)):
        raise ValueError("coverage_levels must contain finite values.")
    if np.any((coverage_levels <= 0.0) | (coverage_levels > 1.0)):
        raise ValueError("coverage_levels must lie in the interval (0, 1].")

    def _artifact_file(
        path_or_dir: str | Path,
        filename: str,
    ) -> Path:
        p = Path(path_or_dir)

        if p.is_file():
            if p.name == filename:
                return p
            return p.parent / filename

        direct = p / filename
        if direct.exists():
            return direct

        nested = p / "bayesian_artifacts" / filename
        if nested.exists():
            return nested

        return direct

    def _as_datetime_index(x) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(x)))
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        return idx

    def _as_timestamp_or_none(x):
        if x is None:
            return None
        ts = pd.Timestamp(x)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts

    def _load_observations(prop):
        nonlocal obs, obs_time

        if obs is None or obs_time is None:
            possible_obs = (
                "obs",
                "Obs",
                "shoreline",
                "y_obs",
                "observations",
            )
            possible_time = (
                "time_obs",
                "obs_time",
                "time_observed",
                "t_obs",
            )

            obs_key = next(
                (key for key in possible_obs if key in prop.files),
                None,
            )
            time_key = next(
                (key for key in possible_time if key in prop.files),
                None,
            )

            if obs is None and obs_key is not None:
                obs = np.asarray(prop[obs_key], dtype=float)
            if obs_time is None and time_key is not None:
                obs_time = np.asarray(prop[time_key])

        if obs is None or obs_time is None:
            if obs_dataset is None:
                raise ValueError(
                    "Observations were not found in the artifact. "
                    "Pass obs and obs_time, or provide obs_dataset."
                )

            close_after = False
            if isinstance(obs_dataset, (str, Path)):
                import xarray as xr
                ds = xr.open_dataset(obs_dataset)
                close_after = True
            else:
                ds = obs_dataset

            try:
                if obs is None:
                    obs = np.asarray(
                        _get_var(ds, obs_names),
                        dtype=float,
                    )
                if obs_time is None:
                    obs_time = np.asarray(
                        _get_time(ds, obs_time_names)
                    )
            finally:
                if close_after:
                    ds.close()

        obs_array = np.asarray(obs, dtype=float).ravel()
        obs_time_array = np.asarray(obs_time).ravel()

        if obs_array.size != obs_time_array.size:
            raise ValueError(
                "obs and obs_time must have the same length. "
                f"Received {obs_array.size} and {obs_time_array.size}."
            )

        valid = np.isfinite(obs_array)
        return obs_array[valid], obs_time_array[valid]

    def _summary_component_candidates(component: str) -> list[str]:
        name = str(component).strip().lower()
        mapping = {
            "full": ["full"],
            "posterior_predictive": ["full"],
            "predictive": ["full"],

            "parameter_only": [
                "physical_parameter_only",
                "parameter_only",
            ],
            "physical_parameters_only": [
                "physical_parameter_only",
                "parameter_only",
            ],
            "physical_parameter_only": [
                "physical_parameter_only",
                "parameter_only",
            ],

            "reference_position_only": [
                "reference_position_only",
            ],
            "reference_only": [
                "reference_position_only",
            ],

            "residual_only": [
                "residual_only",
                "uncertainty_only",
            ],
            "uncertainty_only": [
                "residual_only",
                "uncertainty_only",
            ],
            "nuisance_only": [
                "residual_only",
                "uncertainty_only",
            ],

            "latent": [
                "latent_shoreline",
                "joint_model_only",
                "model_only",
            ],
            "latent_shoreline": [
                "latent_shoreline",
                "joint_model_only",
                "model_only",
            ],
            "joint_model_only": [
                "latent_shoreline",
                "joint_model_only",
                "model_only",
            ],
            "model_only": [
                "latent_shoreline",
                "joint_model_only",
                "model_only",
            ],
        }
        return mapping.get(name, [name])

    def _load_plot_quantiles(
        time: np.ndarray,
        component: str,
    ):
        """
        Load the five display quantiles from the compact summary. This avoids
        reconstructing the complete posterior ensemble over the hourly record.
        """
        summary_path = _artifact_file(
            artifact_dir,
            "uncertainty_propagator_summary.npz",
        )

        if not summary_path.exists():
            return None

        with np.load(summary_path, allow_pickle=True) as summary:
            summary_time = (
                np.asarray(summary["time"])
                if "time" in summary.files
                else time
            )

            if summary_time.size != time.size:
                return None

            candidate = None
            for name in _summary_component_candidates(component):
                required = [
                    f"{name}_p05",
                    f"{name}_p2.5",
                    f"{name}_p50",
                    f"{name}_p97.5",
                    f"{name}_p95",
                ]
                if all(key in summary.files for key in required):
                    candidate = name
                    break

            if candidate is None:
                return None

            return tuple(
                np.asarray(summary[f"{candidate}_{key}"], dtype=float)
                for key in ("p05", "p2.5", "p50", "p97.5", "p95")
            )

    def _match_observations_to_model(
        model_time,
        observation_time,
        observation_values,
        tolerance=None,
    ) -> pd.DataFrame:
        model_df = pd.DataFrame(
            {
                "time": _as_datetime_index(model_time),
                "model_index": np.arange(
                    len(model_time),
                    dtype=np.int64,
                ),
            }
        ).sort_values("time")

        obs_df = pd.DataFrame(
            {
                "obs_time": _as_datetime_index(observation_time),
                "obs": np.asarray(observation_values, dtype=float),
            }
        ).sort_values("obs_time")

        tol = None if tolerance is None else pd.Timedelta(tolerance)

        matched = pd.merge_asof(
            obs_df,
            model_df,
            left_on="obs_time",
            right_on="time",
            direction="nearest",
            tolerance=tol,
        )

        matched = matched.dropna(
            subset=["model_index", "time", "obs"]
        ).copy()

        matched["model_index"] = matched["model_index"].astype(np.int64)
        return matched

    def _load_aligned_sigma(n_draws: int) -> np.ndarray | None:
        sampler_path = _artifact_file(
            artifact_dir,
            "bayesian_parameter_sampler.npz",
        )

        if not sampler_path.exists():
            return None

        with np.load(sampler_path, allow_pickle=True) as sampler:
            if "sigma" not in sampler.files:
                return None

            sigma_values = np.asarray(
                sampler["sigma"],
                dtype=np.float32,
            ).ravel()

            if sigma_values.size != n_draws:
                warnings.warn(
                    "The saved sigma sample is not aligned with the "
                    "propagation draws. Falling back to the saved full "
                    "component for coverage.",
                    RuntimeWarning,
                )
                return None

            return np.maximum(sigma_values, 0.0)

    def _predictive_at_indices(
        prop,
        unique_indices: np.ndarray,
        component: str,
    ) -> np.ndarray:
        """
        Reconstruct only the columns required for coverage. Version-4 ``full``
        coverage is sampled from the latent shoreline and posterior sigma,
        avoiding the large saved residual-noise matrix.
        """
        requested = str(component).strip().lower()
        available = set(prop.files)

        is_v4_full = (
            requested in {"full", "posterior_predictive", "predictive"}
            and "parameter_effect" in available
            and "reference_position_effect" in available
            and "deterministic_reference" in available
        )

        if is_v4_full:
            latent = reconstruct_uncertainty_component(
                prop,
                "latent",
                time_indices=unique_indices,
                dtype=np.float32,
            )

            sigma_values = _load_aligned_sigma(latent.shape[0])

            if sigma_values is not None:
                rng = np.random.default_rng(int(coverage_seed))
                z = rng.standard_normal(
                    latent.shape,
                    dtype=np.float32,
                )
                scaled_noise = sigma_values[:, None] * z

                if antithetic_residuals:
                    return np.concatenate(
                        (
                            latent + scaled_noise,
                            latent - scaled_noise,
                        ),
                        axis=0,
                    )

                return latent + scaled_noise

        # Generic fallback for legacy formats or non-full components.
        return reconstruct_uncertainty_component(
            prop,
            component,
            time_indices=unique_indices,
            dtype=np.float32,
        )

    def _coverage_table(
        matched: pd.DataFrame,
        predictive_unique: np.ndarray,
        unique_inverse: np.ndarray,
    ) -> pd.DataFrame:
        required_percentiles = []
        for nominal in coverage_levels:
            lower_q = 50.0 * (1.0 - float(nominal))
            upper_q = 100.0 - lower_q
            required_percentiles.extend((lower_q, upper_q))

        required_percentiles = np.unique(
            np.asarray(required_percentiles, dtype=float)
        )

        q_unique = np.nanpercentile(
            predictive_unique,
            required_percentiles,
            axis=0,
        )

        q_lookup = {
            float(q): q_unique[i, unique_inverse]
            for i, q in enumerate(required_percentiles)
        }

        split_ts = _as_timestamp_or_none(split_date)

        if split_ts is None:
            masks = {
                "Calibration": np.ones(len(matched), dtype=bool),
                "Validation": np.zeros(len(matched), dtype=bool),
            }
        else:
            obs_dates = matched["obs_time"].to_numpy()
            masks = {
                "Calibration": obs_dates <= np.datetime64(split_ts),
                "Validation": obs_dates > np.datetime64(split_ts),
            }

        obs_values = matched["obs"].to_numpy(dtype=float)
        rows = []

        for split_name, split_mask in masks.items():
            for nominal in coverage_levels:
                nominal = float(nominal)
                lower_q = float(50.0 * (1.0 - nominal))
                upper_q = float(100.0 - lower_q)

                lower = q_lookup[lower_q]
                upper = q_lookup[upper_q]

                valid = (
                    split_mask
                    & np.isfinite(obs_values)
                    & np.isfinite(lower)
                    & np.isfinite(upper)
                )

                if np.any(valid):
                    inside = (
                        (obs_values[valid] >= lower[valid])
                        & (obs_values[valid] <= upper[valid])
                    )
                    empirical = float(np.mean(inside))
                    n_valid = int(np.sum(valid))
                else:
                    empirical = np.nan
                    n_valid = 0

                rows.append(
                    {
                        "nominal": nominal,
                        "empirical": empirical,
                        "n": n_valid,
                        "split": split_name,
                    }
                )

        return pd.DataFrame(rows)

    def _last_finite_xy(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            return None, None
        return float(x[valid][-1]), float(y[valid][-1])

    # --------------------------------------------------------------
    # Read compact propagation and observations
    # --------------------------------------------------------------
    prop_path = _artifact_file(
        artifact_dir,
        "uncertainty_propagator.npz",
    )

    if not prop_path.exists():
        raise FileNotFoundError(prop_path)

    with np.load(prop_path, allow_pickle=True) as prop:
        time = np.asarray(prop["time"])
        time_dt = _as_datetime_index(time)

        obs_values, obs_times = _load_observations(prop)
        obs_time_dt = _as_datetime_index(obs_times)

        # Fast path: use the already-saved full-record quantiles.
        display_quantiles = _load_plot_quantiles(time, comp)

        if display_quantiles is None:
            warnings.warn(
                "The uncertainty summary was not found or did not contain "
                f"quantiles for {comp!r}. Reconstructing the full ensemble "
                "as a compatibility fallback; regenerating the Bayesian "
                "artifacts with the current outputs.py will make this plot "
                "substantially faster.",
                RuntimeWarning,
            )
            complete = reconstruct_uncertainty_component(
                prop,
                comp,
                dtype=np.float32,
            )
            display_quantiles = tuple(
                np.nanpercentile(
                    complete,
                    [5, 2.5, 50, 97.5, 95],
                    axis=0,
                )
            )
            del complete

        p05, p2_5, p50, p97_5, p95 = display_quantiles

        # Match observations before loading any propagated effect matrices.
        matched = _match_observations_to_model(
            time,
            obs_times,
            obs_values,
            tolerance=match_tolerance,
        )

        if matched.empty:
            raise ValueError(
                "No observations could be matched to posterior time. "
                "Increase match_tolerance or check the time coordinates."
            )

        model_indices = matched["model_index"].to_numpy(dtype=np.int64)
        unique_indices, unique_inverse = np.unique(
            model_indices,
            return_inverse=True,
        )

        predictive_unique = _predictive_at_indices(
            prop,
            unique_indices,
            comp,
        )

        coverage_df = _coverage_table(
            matched,
            predictive_unique,
            unique_inverse,
        )

        del predictive_unique

    # --------------------------------------------------------------
    # Restrict and downsample only the displayed time series
    # --------------------------------------------------------------
    start_ts = _as_timestamp_or_none(start_date)
    end_ts = _as_timestamp_or_none(end_date)

    visible = np.ones(time_dt.size, dtype=bool)
    if start_ts is not None:
        visible &= time_dt >= start_ts
    if end_ts is not None:
        visible &= time_dt <= end_ts

    visible_indices = np.flatnonzero(visible)
    if visible_indices.size == 0:
        raise ValueError(
            "The selected start_date/end_date window contains no model times."
        )

    if (
        max_plot_points is not None
        and int(max_plot_points) > 1
        and visible_indices.size > int(max_plot_points)
    ):
        stride = int(
            np.ceil(visible_indices.size / int(max_plot_points))
        )
        plot_indices = visible_indices[::stride]
        if plot_indices[-1] != visible_indices[-1]:
            plot_indices = np.append(
                plot_indices,
                visible_indices[-1],
            )
    else:
        plot_indices = visible_indices

    plot_time = time_dt[plot_indices]
    plot_p05 = np.asarray(p05)[plot_indices]
    plot_p2_5 = np.asarray(p2_5)[plot_indices]
    plot_p50 = np.asarray(p50)[plot_indices]
    plot_p97_5 = np.asarray(p97_5)[plot_indices]
    plot_p95 = np.asarray(p95)[plot_indices]

    split_ts = _as_timestamp_or_none(split_date)

    # --------------------------------------------------------------
    # Original figure layout and style
    # --------------------------------------------------------------
    fig = plt.figure(
        figsize=figsize,
        constrained_layout=False,
    )

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[4.9, 1.25],
        left=0.075,
        right=0.975,
        bottom=0.18,
        top=0.80,
        wspace=0.26,
    )

    ax_ts = fig.add_subplot(gs[0, 0])
    ax_cov = fig.add_subplot(gs[0, 1])

    ax_ts.fill_between(
        plot_time,
        plot_p05,
        plot_p2_5,
        color="red",
        alpha=0.60,
        linewidth=0,
        zorder=1,
    )
    ax_ts.fill_between(
        plot_time,
        plot_p97_5,
        plot_p95,
        color="red",
        alpha=0.60,
        linewidth=0,
        zorder=1,
    )
    ax_ts.fill_between(
        plot_time,
        plot_p05,
        plot_p95,
        color="green",
        alpha=0.45,
        linewidth=0,
        zorder=2,
    )
    ax_ts.plot(
        plot_time,
        plot_p50,
        color="blue",
        lw=0.55,
        zorder=4,
    )

    if split_ts is None:
        ax_ts.scatter(
            obs_time_dt,
            obs_values,
            s=5,
            color="black",
            alpha=0.80,
            linewidths=0,
            zorder=5,
        )
    else:
        cal_obs_mask = obs_time_dt <= split_ts
        val_obs_mask = obs_time_dt > split_ts

        ax_ts.scatter(
            obs_time_dt[cal_obs_mask],
            obs_values[cal_obs_mask],
            s=5,
            color="black",
            alpha=0.85,
            linewidths=0,
            zorder=5,
        )
        ax_ts.scatter(
            obs_time_dt[val_obs_mask],
            obs_values[val_obs_mask],
            s=5,
            color="black",
            alpha=0.85,
            linewidths=0,
            zorder=6,
        )
        ax_ts.axvline(
            split_ts,
            color="0.25",
            linestyle="--",
            lw=1.1,
            zorder=7,
        )

    ax_ts.set_ylabel("Shoreline position [m]")

    if start_ts is not None or end_ts is not None:
        ax_ts.set_xlim(start_ts, end_ts)
    else:
        ax_ts.set_xlim(time_dt[0], time_dt[-1])

    ax_ts.grid(
        alpha=0.25,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )

    if label:
        ax_ts.set_title(
            label,
            loc="left",
            fontweight="bold",
            fontsize=11,
            pad=4,
        )

    if np.issubdtype(np.asarray(time).dtype, np.datetime64):
        locator = mdates.YearLocator(4)
        ax_ts.xaxis.set_major_locator(locator)
        ax_ts.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(locator)
        )

    main_handles = [
        Patch(
            facecolor="red",
            alpha=0.60,
            edgecolor="none",
            label="2.5–97.5%",
        ),
        Patch(
            facecolor="green",
            alpha=0.45,
            edgecolor="none",
            label="5–95%",
        ),
        Line2D(
            [0],
            [0],
            color="blue",
            lw=0.8,
            label="median",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=3.0,
            linestyle="none",
            label="observations",
        ),
    ]

    if split_ts is not None:
        main_handles.append(
            Line2D(
                [0],
                [0],
                color="0.25",
                lw=1.1,
                linestyle="--",
                label="cal/val split",
            )
        )

    ts_box = ax_ts.get_position()
    ts_legend_x = 0.5 * (ts_box.x0 + ts_box.x1)

    fig.legend(
        handles=main_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(ts_legend_x, 0.975),
        ncol=len(main_handles),
        columnspacing=1.10,
        handlelength=1.45,
        handletextpad=0.42,
        borderaxespad=0.0,
        fontsize=10,
    )

    # --------------------------------------------------------------
    # Coverage reliability panel
    # --------------------------------------------------------------
    ax_cov.plot(
        [0.5, 1.0],
        [0.5, 1.0],
        "--",
        color="0.45",
        lw=1.6,
        zorder=1,
    )

    cov_cal = coverage_df[
        coverage_df["split"] == "Calibration"
    ]
    cov_val = coverage_df[
        coverage_df["split"] == "Validation"
    ]

    if not cov_cal.empty and np.isfinite(cov_cal["empirical"]).any():
        ax_cov.plot(
            cov_cal["nominal"],
            cov_cal["empirical"],
            marker=None,
            markersize=4.2,
            lw=1.6,
            color="teal",
            zorder=3,
        )

    if not cov_val.empty and np.isfinite(cov_val["empirical"]).any():
        ax_cov.plot(
            cov_val["nominal"],
            cov_val["empirical"],
            marker=None,
            markersize=4.2,
            lw=1.6,
            color="coral",
            zorder=3,
        )

    ax_cov.set_xlim(0.50, 1.0)
    ax_cov.set_ylim(0.50, 1.00)
    ax_cov.set_xlabel("Nominal coverage")
    ax_cov.set_ylabel("Empirical coverage")
    ax_cov.set_title(
        "Coverage",
        fontweight="bold",
        fontsize=11,
        pad=4,
    )
    ax_cov.grid(
        alpha=0.25,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )

    coverage_labels = [
        {"text": "Ideal", "y": 0.72, "color": "0.35"}
    ]

    if not cov_cal.empty:
        _, y_last = _last_finite_xy(
            cov_cal["nominal"],
            cov_cal["empirical"],
        )
        # if y_last is not None:
        coverage_labels.append(
            {"text": "Cal.", "y": 0.67, "color": "teal"}
        )

    if not cov_val.empty:
        _, y_last = _last_finite_xy(
            cov_val["nominal"],
            cov_val["empirical"],
        )
        if y_last is not None:
            coverage_labels.append(
                {"text": "Val.", "y": 0.62, "color": "coral"}
            )

    coverage_labels = sorted(
        coverage_labels,
        key=lambda item: item["y"],
        reverse=True,
    )

    previous_y = None
    for item in coverage_labels:
        # if previous_y is None:
        y_position = item["y"]
        # if previous_y is not None:
        #     y_position = previous_y -0.05
            
        # previous_y = y_position

        ax_cov.text(
            0.985,
            y_position,
            item["text"],
            color=item["color"],
            fontsize=11,
            fontweight="bold",
            ha="right",
            va="center",
            clip_on=True,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 0.35,
            },
            zorder=10,
        )

    _save(fig, out_path, dpi=350)

    return fig, (ax_ts, ax_cov), coverage_df
