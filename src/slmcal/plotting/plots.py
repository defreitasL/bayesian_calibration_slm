"""slmcal.plotting

Default plots + diagnostics for calibration runs.

This file is intentionally dependency-light and tolerant to missing optional
libraries. If SciPy/ArviZ are not available, the corresponding plots are skipped.

What gets produced ("Option B")
--------------------------------
In addition to the original default figures (prior vs posterior, posterior
predictive envelope, likelihood PPC time-series), this version also saves:

1) **Calibration metrics** at observation times (CSV + PNG table)
   - RMSE, MAE, bias, Pearson r, NSE
   - 90% interval coverage (obs within [p5,p95]) and mean interval width

2) **Sampler diagnostics** (CSV + PNG table)
   - r_hat, ess_bulk, ess_tail for all (non-huge) posterior variables

3) **Residual diagnostics** (PNG)
   - Residual time series, histogram, and QQ plot (based on posterior median)

4) **Likelihood distribution check** (PNG)
   - KDE of observed values vs PPC KDE bands (median, 5–95%, min–max)

All outputs are written into the provided ``out_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde
from scipy.stats import norm as _norm
from scipy.stats import t as _student_t
from scipy.special import gammaln as _gammaln

import numpy as np
import re


# ---------------------------
# Helpers
# ---------------------------


def _as_float_time(t: np.ndarray) -> np.ndarray:
    """Convert a time vector to float64 (days since epoch) for interpolation."""
    t = np.asarray(t)
    if t.size == 0:
        return t.astype("float64")
    if np.issubdtype(t.dtype, np.datetime64):
        # days since epoch
        return t.astype("datetime64[ns]").astype("int64") / (1e9 * 86400.0)
    return t.astype("float64")


def _interp_to_obs(
    time: np.ndarray,
    series: np.ndarray,
    obs_time: np.ndarray,
) -> np.ndarray:
    """Interpolate a series defined on `time` onto `obs_time` (1D)."""
    x = _as_float_time(np.asarray(time))
    xo = _as_float_time(np.asarray(obs_time))

    y = np.asarray(series, dtype="float64")
    if y.ndim != 1:
        raise ValueError("series must be 1D")

    # Ensure increasing x for np.interp
    if x.size != y.size:
        raise ValueError("time and series must have same length")
    if x.size < 2:
        return np.full_like(xo, y[0] if y.size else np.nan, dtype="float64")
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
    # IMPORTANT: avoid extrapolation outside the model time window.
    # Using NaN outside prevents misleading endpoint repetition in metrics.
    return np.interp(xo, x, y, left=np.nan, right=np.nan)


def _interp_to_obs_2d(
    time: np.ndarray,
    series_2d: np.ndarray,
    obs_time: np.ndarray,
) -> np.ndarray:
    """Interpolate a 2D series (time, transect) onto obs_time.

    Returns array of shape (n_obs, n_transects).
    """
    time = np.asarray(time)
    obs_time = np.asarray(obs_time)
    Y = np.asarray(series_2d, dtype="float64")
    if Y.ndim != 2:
        raise ValueError("series_2d must be 2D (time, transect)")
    if time.size != Y.shape[0]:
        raise ValueError("time length must match series_2d.shape[0]")
    n_obs = int(obs_time.size)
    n_tr = int(Y.shape[1])
    out = np.full((n_obs, n_tr), np.nan, dtype="float64")
    for j in range(n_tr):
        out[:, j] = _interp_to_obs(time, Y[:, j], obs_time)
    return out



def _safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a = a[m]
    b = b[m]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 3:
        return float("nan")
    o = obs[m]
    s = sim[m]
    denom = float(np.sum((o - o.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((o - s) ** 2) / denom)

def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 3:
        return float("nan")
    o = obs[m]
    s = sim[m]

    r_num = float(np.sum((o - o.mean()) * (s - s.mean())))
    r_denom = float(np.sqrt(np.sum((o - o.mean()) ** 2) * np.sum((s - s.mean()) ** 2)))
    if r_denom <= 0:
        return float("nan")
    r = r_num / r_denom

    alpha = s.std() / (o.std() + 1e-12)
    beta = s.mean() / (o.mean() + 1e-12)

    kge_val = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return float(kge_val)


def _rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((obs[m] - sim[m]) ** 2)))


def _mae(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(obs[m] - sim[m])))


def _bias(sim: np.ndarray, obs: np.ndarray) -> float:
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(sim[m] - obs[m]))


def _coverage(obs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    m = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((obs[m] >= lo[m]) & (obs[m] <= hi[m])))


def _mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    m = np.isfinite(lo) & np.isfinite(hi)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(hi[m] - lo[m]))



_THEME = {
    "navy": "#234E70",
    "teal": "#2A9D8F",
    "sky": "#4C78A8",
    "coral": "#E76F51",
    "gold": "#E9C46A",
    "purple": "#7B6DAB",
    "slate": "#5B6675",
    "grid": "#D7DEE8",
    "header": "#EEF3F8",
    "stripe": "#F8FAFC",
    "ink": "#23313F",
}


def _apply_axes_style(ax, *, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_THEME["grid"])
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=_THEME["ink"], labelsize=9)
    ax.xaxis.label.set_color(_THEME["ink"])
    ax.yaxis.label.set_color(_THEME["ink"])
    ax.title.set_color(_THEME["ink"])
    ax.grid(True, axis=grid_axis, color=_THEME["grid"], alpha=0.7, linewidth=0.7)


def _interval_bounds(mu: np.ndarray, sig: np.ndarray, level: float, use_studentt: bool, nu: float | None) -> tuple[np.ndarray, np.ndarray]:
    a = 0.5 * (1.0 - float(level))
    lo_p = a
    hi_p = 1.0 - a
    if use_studentt:
        qlo = np.asarray(mu, dtype=float) + np.asarray(sig, dtype=float) * _student_t.ppf(lo_p, float(nu))
        qhi = np.asarray(mu, dtype=float) + np.asarray(sig, dtype=float) * _student_t.ppf(hi_p, float(nu))
    else:
        qlo = np.asarray(mu, dtype=float) + np.asarray(sig, dtype=float) * _norm.ppf(lo_p)
        qhi = np.asarray(mu, dtype=float) + np.asarray(sig, dtype=float) * _norm.ppf(hi_p)
    return qlo, qhi


def _dist_cdf_at_threshold(threshold: float | np.ndarray, mu: np.ndarray, sig: np.ndarray, use_studentt: bool, nu: float | None) -> np.ndarray:
    thr = np.asarray(threshold, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-9)
    z = (thr - mu) / sig
    if use_studentt:
        return _student_t.cdf(z, float(nu))
    return _norm.cdf(z)


def _ignorance_bits(logpdf: np.ndarray) -> np.ndarray:
    return -np.asarray(logpdf, dtype=float) / np.log(2.0)


def _brier_score(prob: np.ndarray, event: np.ndarray) -> float:
    p = np.asarray(prob, dtype=float)
    e = np.asarray(event, dtype=float)
    m = np.isfinite(p) & np.isfinite(e)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((p[m] - e[m]) ** 2))


def _save_table_png(
    rows: Sequence[Sequence[object]],
    col_labels: Sequence[str],
    out_png: Path,
    title: Optional[str] = None,
    font_size: int = 9,
    max_rows_per_page: int = 35,
) -> list[Path]:
    """Save one or more PNG pages for a long table with improved styling."""
    rows = list(rows)
    if not rows:
        rows = [["", "", ""]]

    n_pages = int(np.ceil(len(rows) / max_rows_per_page))
    written = []

    for i in range(n_pages):
        chunk = rows[i * max_rows_per_page:(i + 1) * max_rows_per_page]
        n_cols = max(1, len(col_labels))
        fig_h = min(14.0, 1.35 + 0.34 * (len(chunk) + 2))
        fig_w = max(7.5, 1.5 * n_cols + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        if title:
            ax.set_title(title + ("" if n_pages == 1 else f" (page {i+1}/{n_pages})"), pad=14, color=_THEME["ink"], fontsize=12, fontweight="semibold")
        tbl = ax.table(
            cellText=[[str(c) for c in r] for r in chunk],
            colLabels=list(col_labels),
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(font_size)
        tbl.scale(1.0, 1.25)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(_THEME["grid"])
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor(_THEME["navy"])
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor(_THEME["stripe"] if (r % 2) == 0 else "white")
                if c == 0:
                    cell.get_text().set_fontweight("semibold")
                    cell.get_text().set_color(_THEME["ink"])

        fig.tight_layout()
        out_i = out_png if n_pages == 1 else out_png.with_name(out_png.stem + f"_part{i+1:02d}" + out_png.suffix)
        fig.savefig(out_i, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(out_i)
    return written


    n_pages = int(np.ceil(len(rows) / max_rows_per_page))
    for i in range(n_pages):
        chunk = rows[i * max_rows_per_page : (i + 1) * max_rows_per_page]
        fig_h = 0.35 * (len(chunk) + 2)
        fig, ax = plt.subplots(figsize=(12, max(3.5, fig_h)))
        ax.axis("off")
        if title:
            ax.set_title(title + ("" if n_pages == 1 else f" (page {i+1}/{n_pages})"), pad=12)
        tbl = ax.table(
            cellText=[[str(c) for c in r] for r in chunk],
            colLabels=list(col_labels),
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(font_size)
        tbl.scale(1.0, 1.2)
        fig.tight_layout()

        out_i = out_png
        if n_pages > 1:
            out_i = out_png.with_name(out_png.stem + f"_part{i+1:02d}" + out_png.suffix)
        fig.savefig(out_i, dpi=200)
        plt.close(fig)
        written.append(out_i)
    return written


def _try_import_scipy_kde():
    try:
        from scipy.stats import gaussian_kde

        return gaussian_kde
    except Exception:
        return None


# ---------------------------
# Plots
# ---------------------------


def plot_prior_posterior(
    prior_raw: np.ndarray,
    posterior_raw: np.ndarray,
    param_names: Sequence[str],
    label: str,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    prior_raw = np.asarray(prior_raw, dtype=float)
    posterior_raw = np.asarray(posterior_raw, dtype=float)
    if prior_raw.ndim != 2 or posterior_raw.ndim != 2:
        return

    n_params = prior_raw.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(8.2, max(2.5, 2.2 * n_params)), sharex=False)
    if n_params == 1:
        axes = [axes]

    kde_cls = _try_import_scipy_kde()
    for i, ax in enumerate(axes):
        pr = prior_raw[:, i]
        po = posterior_raw[:, i]
        pr = pr[np.isfinite(pr)]
        po = po[np.isfinite(po)]
        if pr.size == 0 or po.size == 0:
            ax.set_axis_off()
            continue

        lo = float(np.nanmin([np.nanmin(pr), np.nanmin(po)]))
        hi = float(np.nanmax([np.nanmax(pr), np.nanmax(po)]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.08 * (hi - lo)
        xs = np.linspace(lo - pad, hi + pad, 400)

        if kde_cls is not None and pr.size > 2 and po.size > 2:
            try:
                kde_pr = kde_cls(pr)
                kde_po = kde_cls(po)
                y_pr = kde_pr(xs)
                y_po = kde_po(xs)
                ax.fill_between(xs, 0.0, y_pr, color=_THEME["slate"], alpha=0.22, label="Prior")
                ax.plot(xs, y_pr, color=_THEME["slate"], lw=1.6)
                ax.fill_between(xs, 0.0, y_po, color=_THEME["teal"], alpha=0.28, label="Posterior")
                ax.plot(xs, y_po, color=_THEME["teal"], lw=1.8)
            except Exception:
                ax.hist(pr, bins=40, density=True, alpha=0.35, color=_THEME["slate"], label="Prior")
                ax.hist(po, bins=40, density=True, alpha=0.45, color=_THEME["teal"], label="Posterior")
        else:
            ax.hist(pr, bins=40, density=True, alpha=0.35, color=_THEME["slate"], label="Prior")
            ax.hist(po, bins=40, density=True, alpha=0.45, color=_THEME["teal"], label="Posterior")

        ax.axvline(np.nanmedian(pr), color=_THEME["slate"], lw=1.2, ls="--", alpha=0.9)
        ax.axvline(np.nanmedian(po), color=_THEME["teal"], lw=1.2, ls="--", alpha=0.95)
        ax.set_ylabel(param_names[i] if i < len(param_names) else f"p{i}")
        _apply_axes_style(ax, grid_axis="y")
        if i == 0:
            ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")

    axes[-1].set_xlabel("Parameter value")
    fig.suptitle(f"Prior vs posterior — {label}", y=1.01, fontsize=13, fontweight="semibold", color=_THEME["ink"])
    fig.tight_layout()
    fig.savefig(out_dir / f"prior_posterior_{label}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def plot_posterior_predictive_split(
    *,
    time: np.ndarray,
    per5: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per95: np.ndarray,
    per1: np.ndarray,
    per99: np.ndarray,
    mini: np.ndarray,
    maxi: np.ndarray,
    draws: Optional[np.ndarray] = None,
    obs_time: np.ndarray,
    obs: np.ndarray,
    label: str,
    out_dir: Path,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Posterior envelope over full model time, highlighting calibration vs validation obs."""
    import matplotlib.pyplot as plt

    time = np.asarray(time)
    per5 = np.asarray(per5, dtype=float)
    per10 = np.asarray(per10, dtype=float)
    per50 = np.asarray(per50, dtype=float)
    per90 = np.asarray(per90, dtype=float)
    per95 = np.asarray(per95, dtype=float)
    mini = np.asarray(mini, dtype=float)
    maxi = np.asarray(maxi, dtype=float)
    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)

    if obs_mask_cal is None:
        obs_mask_cal = np.isfinite(obs)
    else:
        obs_mask_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)

    if obs_mask_val is None:
        obs_mask_val = np.zeros_like(obs_mask_cal, dtype=bool)
    else:
        obs_mask_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    fig, ax = plt.subplots(figsize=(12, 4))
    # ax.fill_between(time, mini, maxi, alpha=0.15, color='red', label="Posterior min–max", linewidth=0.0)
    # ax.fill_between(time, per1, per99, alpha=0.25, color='orange', label="Posterior 1–99%", linewidth=0.0)
    # ax.fill_between(time, per5, per95, alpha=0.45, color='green', label="Posterior 5–95%", linewidth=0.0)
    ax.fill_between(time, per1, per10, alpha=0.30, color='red', label="Posterior 1–99%", linewidth=0.0, zorder=1)
    ax.fill_between(time, per90, per99, alpha=0.30, color='red', label="", linewidth=0.0, zorder=1)
    ax.fill_between(time, per10, per90, alpha=0.45, color='green', label="Posterior 10–90%", linewidth=0.0, zorder=2)
    ax.plot(time, per50, lw=1.0, color='blue', label="Posterior median", zorder=3)

    # obs points
    if obs_mask_cal.any():
        ax.scatter(obs_time[obs_mask_cal], obs[obs_mask_cal], s=4, c="k", label="Obs (cal)", zorder=4)
    if obs_mask_val.any():
        ax.scatter(obs_time[obs_mask_val], obs[obs_mask_val], s=4, c="r", label="Obs (val)", zorder=4, alpha=0.9)

    # split marker and validation shading
    if split_date is not None:
        try:
            sd = np.datetime64(split_date)
            ax.axvline(sd, lw=1.2, ls="--", color='black', label="Split date", zorder=5)
            # shade validation region up to end of model time
            ax.axvspan(sd, time.max(), alpha=0.15, color='gray', zorder=0)
        except Exception:
            pass

    ax.set_title(f"Posterior predictive envelope — {label}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Shoreline position")
    ax.set_xlim(time.min(), time.max())
    ax.legend(loc="best", ncols=6)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / f"posterior_predictive_{label}.png", dpi=200)
    plt.close(fig)


def plot_posterior_predictive_with_likelihood(
    *,
    time: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per1: np.ndarray,
    per99: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    label: str,
    out_dir: Path,
    trace=None,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Posterior predictive bands + log predictive density at observation times.

    This figure complements `plot_posterior_predictive_split` by adding a bottom
    panel showing the *log predictive density* (LPD) at observation times.

    The predictive distribution is approximated from the provided quantiles.
    The family (Normal vs Student-t) is inferred automatically:

    - If `trace` contains a posterior variable named `nu`, Student-t is used.
    - Otherwise, Normal is used unless the 99–1 / 90–10 spread ratio indicates
      heavy tails, in which case Student-t is used with an estimated df.
    """
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    time = np.asarray(time)
    per10 = np.asarray(per10, dtype=float)
    per50 = np.asarray(per50, dtype=float)
    per90 = np.asarray(per90, dtype=float)
    per1 = np.asarray(per1, dtype=float)
    per99 = np.asarray(per99, dtype=float)
    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)

    # masks
    if obs_mask_cal is None:
        obs_mask_cal = np.isfinite(obs)
    else:
        obs_mask_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
    if obs_mask_val is None:
        obs_mask_val = np.zeros_like(obs_mask_cal, dtype=bool)
    else:
        obs_mask_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    # interpolate quantiles to observation times
    q10 = _interp_to_obs(time, per10, obs_time)
    q50 = _interp_to_obs(time, per50, obs_time)
    q90 = _interp_to_obs(time, per90, obs_time)
    q01 = _interp_to_obs(time, per1, obs_time)
    q99 = _interp_to_obs(time, per99, obs_time)

    m = np.isfinite(obs) & np.isfinite(q10) & np.isfinite(q50) & np.isfinite(q90)
    if not np.any(m):
        return

    # -------------------------
    # Infer distribution family
    # -------------------------
    use_studentt = False
    nu_val: float | None = None

    # 1) From trace (preferred)
    try:
        if trace is not None and hasattr(trace, "posterior"):
            post = trace.posterior
            if hasattr(post, "data_vars") and ("nu" in post.data_vars):
                nu_arr = np.asarray(post["nu"].values).reshape(-1)
                nu_arr = nu_arr[np.isfinite(nu_arr)]
                if nu_arr.size:
                    use_studentt = True
                    nu_val = float(np.nanmedian(nu_arr))
    except Exception:
        pass

    # 2) Heuristic from band spread ratio (fallback)
    # For Normal: ratio ≈ z0.99 / z0.90 ≈ 1.815
    if not use_studentt:
        try:
            spread_90 = (q90 - q10)
            spread_99 = (q99 - q01)
            rr = spread_99[m] / np.maximum(spread_90[m], 1e-12)
            rr = rr[np.isfinite(rr) & (rr > 0)]
            if rr.size:
                r_med = float(np.nanmedian(rr))
                if r_med > 1.95:  # heavier than Normal
                    use_studentt = True
                    # estimate nu by matching ratio = tppf(0.99,nu)/tppf(0.90,nu)
                    target = r_med
                    lo, hi = 2.1, 200.0
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        r_mid = float(_student_t.ppf(0.99, mid) / _student_t.ppf(0.90, mid))
                        if r_mid > target:
                            lo = mid
                        else:
                            hi = mid
                    nu_val = float(0.5 * (lo + hi))
        except Exception:
            use_studentt = False
            nu_val = None

    if use_studentt and (nu_val is None or not np.isfinite(nu_val)):
        use_studentt = False

    # -------------------------
    # Estimate scale from quantiles
    # -------------------------
    z90 = float(_norm.ppf(0.90))
    width = (q90 - q10)
    width = np.maximum(width, 1e-9)

    if use_studentt:
        t90 = float(_student_t.ppf(0.90, nu_val))
        sig = width / np.maximum(2.0 * t90, 1e-9)
    else:
        sig = width / np.maximum(2.0 * z90, 1e-9)

    sig = np.maximum(sig, 1e-9)

    # -------------------------
    # Log predictive density
    # -------------------------
    y = obs
    mu = q50
    z = (y - mu) / sig

    if use_studentt:
        nu = float(nu_val)
        logp = (
            _gammaln((nu + 1.0) / 2.0)
            - _gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
            - np.log(sig)
            - ((nu + 1.0) / 2.0) * np.log1p((z * z) / nu)
        )
        like_name = f"studentt (nu≈{nu:.2g})"
    else:
        logp = -0.5 * (z * z) - np.log(sig) - 0.5 * np.log(2.0 * np.pi)
        like_name = "normal"

    # -------------------------
    # Plot
    # -------------------------
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(12, 6.2),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex=True,
    )

    ax0.fill_between(time, per1, per99, alpha=0.18, color="red", label="Posterior 1–99%", linewidth=0.0)
    ax0.fill_between(time, per10, per90, alpha=0.30, color="green", label="Posterior 10–90%", linewidth=0.0)
    ax0.plot(time, per50, lw=1.1, color="blue", label="Posterior median")

    if obs_mask_cal.any():
        ax0.scatter(obs_time[obs_mask_cal], obs[obs_mask_cal], s=6, c="k", label="Obs (cal)", zorder=3)
    if obs_mask_val.any():
        ax0.scatter(obs_time[obs_mask_val], obs[obs_mask_val], s=6, c="r", label="Obs (val)", zorder=3, alpha=0.9)

    if split_date is not None:
        try:
            sd = np.datetime64(split_date)
            ax0.axvline(sd, lw=1.2, ls="--", color="black", zorder=4)
            ax0.axvspan(sd, time.max(), alpha=0.12, color="gray", zorder=0)
            ax1.axvline(sd, lw=1.2, ls="--", color="black", zorder=4)
            ax1.axvspan(sd, time.max(), alpha=0.12, color="gray", zorder=0)
        except Exception:
            pass

    ax0.set_title(f"Predictive bands + obs — {label}")
    ax0.set_ylabel("Shoreline")
    ax0.legend(loc="best", ncols=4, fontsize=9)
    fig.autofmt_xdate()

    lpd = np.asarray(logp, dtype=float)
    mm = np.isfinite(lpd) & np.isfinite(obs)
    if mm.any():
        ax1.scatter(obs_time[mm & obs_mask_cal], lpd[mm & obs_mask_cal], s=10, c="k", alpha=0.85)
        if obs_mask_val.any():
            ax1.scatter(obs_time[mm & obs_mask_val], lpd[mm & obs_mask_val], s=10, c="r", alpha=0.85)
        med = float(np.nanmedian(lpd[mm]))
        ax1.axhline(med, lw=1.0, alpha=0.6)

    ax1.set_title(f"Log predictive density at obs ({like_name})")
    ax1.set_ylabel("log p(y_obs)")
    ax1.set_xlabel("Time")

    if np.issubdtype(time.dtype, np.datetime64):
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    fig.tight_layout()
    fig.savefig(out_dir / f"posterior_predictive_likelihood_{label}.png", dpi=200)
    plt.close(fig)


# def plot_posterior_predictive(
#     time: np.ndarray,
#     per5: np.ndarray,
#     per50: np.ndarray,
#     per95: np.ndarray,
#     obs_time: np.ndarray,
#     obs: np.ndarray,
#     label: str,
#     out_dir: Path,
#     mini: Optional[np.ndarray] = None,
#     maxi: Optional[np.ndarray] = None,
# ) -> None:
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.plot(time, per50, label="Posterior median")
#     ax.fill_between(time, per5, per95, alpha=0.3, label="Posterior 5–95%")
#     if mini is not None and maxi is not None:
#         ax.fill_between(time, mini, maxi, alpha=0.15, label="Posterior min–max")
#     ax.scatter(obs_time, obs, s=10, c="k", label="Observations", zorder=3)
#     ax.set_title(f"Posterior predictive: {label}")
#     ax.set_xlabel("time")
#     ax.set_ylabel("shoreline position")
#     ax.legend(loc="best")
#     fig.tight_layout()
#     fig.savefig(out_dir / f"posterior_predictive_{label}.png", dpi=200)
#     plt.close(fig)


def plot_ppc_timeseries(
    ppc,
    obs_time: np.ndarray,
    obs: np.ndarray,
    label: str,
    out_dir: Path,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Posterior predictive time-series envelope.

    Note
    ----
    `ppc` is typically generated on the *calibration* observation vector.
    If `obs_mask_cal` is provided (defined on the full obs arrays), we use it to
    select the calibration points and (optionally) overlay validation observations.
    """
    if ppc is None or (not hasattr(ppc, "posterior_predictive")):
        return

    pp = ppc.posterior_predictive
    if pp is None or len(getattr(pp, "data_vars", {})) == 0:
        return

    var_name = "likelihood" if "likelihood" in pp else list(pp.data_vars)[0]
    arr = np.asarray(pp[var_name].values)
    if arr.ndim == 2:
        # (draw, obs)
        flat = arr
    else:
        # (chain, draw, obs)
        flat = arr.reshape(-1, arr.shape[-1])

    if flat.size == 0:
        return

    per5 = np.percentile(flat, 5, axis=0)
    per50 = np.percentile(flat, 50, axis=0)
    per95 = np.percentile(flat, 95, axis=0)
    mini = np.min(flat, axis=0)
    maxi = np.max(flat, axis=0)

    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)

    # If masks were provided, use them to pick calibration vs validation points.
    if obs_mask_cal is None:
        m_cal = np.isfinite(obs)
    else:
        m_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
    if obs_mask_val is None:
        m_val = np.zeros_like(m_cal, dtype=bool)
    else:
        m_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    obs_time_cal = obs_time[m_cal]
    obs_cal = obs[m_cal]
    obs_time_val = obs_time[m_val]
    obs_val = obs[m_val]

    n_ppc = int(per50.size)
    n_cal = int(obs_cal.size)

    note = ""
    if n_cal != n_ppc:
        n_use = int(min(n_cal, n_ppc))
        if n_use <= 0:
            return
        # trim both to common length (keeps time ordering)
        obs_cal = obs_cal[:n_use]
        obs_time_cal = obs_time_cal[:n_use]
        per5 = per5[:n_use]
        per50 = per50[:n_use]
        per95 = per95[:n_use]
        mini = mini[:n_use]
        maxi = maxi[:n_use]
        note = f"Length mismatch: cal obs n={n_cal}, ppc n={n_ppc} -> using n={n_use}"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(obs_time_cal, mini, maxi, alpha=0.15, color='red', label="PPC min–max")
    ax.fill_between(obs_time_cal, per5, per95, alpha=0.45, color='green', label="PPC 5–95%")
    ax.plot(obs_time_cal, per50, lw=1.0, color='blue', label="PPC median")
    ax.scatter(obs_time_cal, obs_cal, s=4, c="k", label="Obs (cal)", zorder=3, marker='s')
    if obs_time_val.size:
        ax.scatter(obs_time_val, obs_val, s=4, c="r", label="Obs (val)", zorder=3, alpha=0.9)

    if split_date is not None:
        try:
            sd = np.datetime64(split_date)
            ax.axvline(sd, lw=1.2)
            ax.axvspan(sd, obs_time.max(), alpha=0.06, color='gray')
        except Exception:
            pass

    ax.set_title(f"Posterior Predictive Check (timeseries) — {label}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Shoreline")
    ax.set_xlim(obs_time.min(), obs_time.max())
    ax.legend(loc="best", ncols=4)
    fig.autofmt_xdate()

    if note:
        ax.text(
            0.01,
            0.01,
            note,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            alpha=0.8,
        )

    fig.tight_layout()
    fig.savefig(out_dir / f"ppc_timeseries_{label}.png", dpi=200)
    plt.close(fig)



def plot_likelihood_distribution_kde(
    ppc,
    obs: np.ndarray,
    label: str,
    out_dir: Path,
    obs_mask_cal: Optional[np.ndarray] = None,
) -> None:
    """Compare observed distribution vs PPC distribution (KDE).

    If `obs_mask_cal` is provided, the observed KDE is computed on the
    calibration subset only (recommended, since `ppc` is generated on
    calibration points).
    """
    if ppc is None or (not hasattr(ppc, "posterior_predictive")):
        return

    pp = ppc.posterior_predictive
    if pp is None or len(getattr(pp, "data_vars", {})) == 0:
        return

    var_name = "likelihood" if "likelihood" in pp else list(pp.data_vars)[0]
    arr = np.asarray(pp[var_name].values)
    if arr.ndim == 2:
        flat = arr
    else:
        flat = arr.reshape(-1, arr.shape[-1])

    if flat.size == 0:
        return

    per50 = np.percentile(flat, 50, axis=0)

    obs = np.asarray(obs, dtype=float)
    if obs_mask_cal is not None:
        m = np.asarray(obs_mask_cal, dtype=bool)
        if m.shape[0] == obs.shape[0]:
            obs = obs[m]

    n_ppc = int(per50.size)
    n_obs = int(obs.size)
    note = "" if n_obs == n_ppc else f"Obs n={n_obs}, ppc n={n_ppc}"

    obs = obs[np.isfinite(obs)]
    flat = flat[:, np.isfinite(np.nanmedian(flat, axis=0))]

    if obs.size == 0 or flat.size == 0:
        return

    vmin = float(min(np.nanmin(flat), np.nanmin(obs)))
    vmax = float(max(np.nanmax(flat), np.nanmax(obs)))

    xs = np.linspace(vmin, vmax, 200)

    fig, ax = plt.subplots(figsize=(8, 4))

    # observed KDE
    if obs.size > 1:
        kde_obs = gaussian_kde(obs)
        ax.plot(xs, kde_obs(xs), label="Obs KDE", color='black', lw=2)

    # PPC KDE (draw-level average)
    if flat.shape[0] > 1 and flat.shape[1] > 1:
        # sample up to 200 draws to keep it fast
        n_draw = min(flat.shape[0], 200)
        idx = np.random.default_rng(0).choice(flat.shape[0], size=n_draw, replace=False)
        ys = []
        for i in idx:
            x_i = flat[i]
            x_i = x_i[np.isfinite(x_i)]
            if x_i.size > 1:
                ys.append(gaussian_kde(x_i)(xs))
        if ys:
            ax.plot(xs, np.mean(ys, axis=0), label="PPC mean KDE", color='red', lw=1)

    ax.set_title(f"Likelihood distribution (KDE) — {label}")
    ax.set_xlabel("Shoreline")
    ax.set_ylabel("Density")
    ax.legend(loc="best")

    if note:
        ax.text(
            0.01,
            0.01,
            note,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            alpha=0.8,
        )

    fig.tight_layout()
    fig.savefig(out_dir / f"ppc_kde_{label}.png", dpi=200)
    plt.close(fig)

def plot_density_kde_bands_cal_val(
    *,
    draws: np.ndarray,                 # (S, T) on model grid
    time: np.ndarray,                  # (T,)
    obs_time: np.ndarray,              # (Nobs,)
    obs: np.ndarray,                   # (Nobs,)
    out_dir: Path,
    label: str,
    split_date: Optional[np.datetime64] = None,
    # speed knobs
    max_draws: int = 600,              # subsample posterior draws for density bands
    n_bins: int = 220,
    smooth_bw_frac: float = 0.03,      # bandwidth as fraction of data range
    # bands
    q_lo_hi_1: tuple[float, float] = (1, 99),
    q_lo_hi_2: tuple[float, float] = (10, 90),
) -> None:
    """
    Two panels (cal/val): KDE density of obs vs KDE-like density bands of modelled shoreline positions.
    Obs density uses true gaussian_kde (fast; only 1 KDE per panel).
    Model density bands use fast histogram + Gaussian smoothing per draw (fast).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    D = np.asarray(draws, dtype=float)
    t = np.asarray(time)
    ot = np.asarray(obs_time)
    oy = np.asarray(obs, dtype=float)

    if D.ndim != 2 or t.ndim != 1:
        return
    S, T = D.shape
    if T != t.size:
        raise ValueError("draws shape (S,T) must match time length")

    # ---- map obs -> nearest model index ----
    if np.issubdtype(t.dtype, np.datetime64) and np.issubdtype(ot.dtype, np.datetime64):
        tt = t.astype("datetime64[ns]")
        oo = ot.astype("datetime64[ns]")
        idx = np.searchsorted(tt, oo)
        idx = np.clip(idx, 1, len(tt) - 1)
        left = idx - 1
        right = idx
        choose_right = (oo - tt[left]) > (tt[right] - oo)
        idx = np.where(choose_right, right, left).astype(int)
    else:
        idx = np.asarray(ot, dtype=int)
        idx = np.clip(idx, 0, T - 1)

    # ---- cal/val mask in obs space ----
    if split_date is not None and np.issubdtype(ot.dtype, np.datetime64):
        sd = np.datetime64(split_date)
        m_cal = ot < sd
        m_val = ot >= sd
    else:
        m_cal = np.ones_like(oy, dtype=bool)
        m_val = np.zeros_like(oy, dtype=bool)

    # Extract model draws at obs indices: (S, Nobs)
    D_obs = D[:, idx]

    # Subsample draws for speed
    rng = np.random.default_rng(123)
    if S > max_draws:
        sel = rng.choice(S, size=max_draws, replace=False)
        D_obs = D_obs[sel, :]
        S_use = max_draws
    else:
        S_use = S

    # Smoother for the model densities
    try:
        from scipy.ndimage import gaussian_filter1d
        _has_scipy = True
    except Exception:
        gaussian_filter1d = None
        _has_scipy = False

    def _density_band(x_obs: np.ndarray, X_draws: np.ndarray):
        """
        x_obs: (N,)
        X_draws: (S_use, N)
        returns centers and:
          obs_kde, model_median, band1_low/high, band2_low/high
        """
        x_obs = np.asarray(x_obs, dtype=float)
        x_obs = x_obs[np.isfinite(x_obs)]
        if x_obs.size < 2:
            raise ValueError("Not enough finite observations for KDE")

        all_vals = np.concatenate([x_obs, X_draws.reshape(-1)])
        lo = np.nanpercentile(all_vals, 0.5)
        hi = np.nanpercentile(all_vals, 99.5)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = np.nanmin(all_vals)
            hi = np.nanmax(all_vals)
        pad = 0.05 * (hi - lo)
        lo -= pad
        hi += pad

        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bw = max(1e-12, smooth_bw_frac * (hi - lo))  # bandwidth in data units
        bin_w = (hi - lo) / n_bins
        sigma_bins = max(1e-6, bw / bin_w)

        # --- OBS: true KDE (cheap: one KDE per panel) ---
        try:
            from scipy.stats import gaussian_kde
            sdx = float(np.std(x_obs))
            if sdx > 0:
                bw_factor = bw / sdx  # gaussian_kde bandwidth factor
                kde = gaussian_kde(x_obs, bw_method=bw_factor)
            else:
                kde = gaussian_kde(x_obs)  # degenerate, fallback
            f_obs = kde(centers)
        except Exception:
            # fallback: histogram density + smoothing
            h_obs, _ = np.histogram(x_obs, bins=edges)
            f_obs = h_obs.astype(float) / max(1, np.sum(h_obs)) / bin_w
            if _has_scipy:
                f_obs = gaussian_filter1d(f_obs, sigma=sigma_bins, mode="nearest")

        # --- MODEL: fast density per draw (hist+smooth) ---
        idxb = np.searchsorted(edges, X_draws, side="right") - 1
        valid = (idxb >= 0) & (idxb < n_bins)
        idxb = np.where(valid, idxb, -1)

        F = np.zeros((S_use, n_bins), dtype=float)
        for s in range(S_use):
            ii = idxb[s]
            ii = ii[ii >= 0]
            if ii.size == 0:
                continue
            c = np.bincount(ii, minlength=n_bins).astype(float)
            F[s] = c / max(1, c.sum()) / bin_w

        if _has_scipy:
            F_s = gaussian_filter1d(F, sigma=sigma_bins, axis=1, mode="nearest")
        else:
            # fallback convolution kernel
            rad = int(max(3, np.ceil(3 * sigma_bins)))
            xk = np.arange(-rad, rad + 1)
            ker = np.exp(-0.5 * (xk / sigma_bins) ** 2)
            ker /= ker.sum()
            F_s = np.vstack([np.convolve(F[s], ker, mode="same") for s in range(S_use)])

        med = np.nanpercentile(F_s, 50, axis=0)
        lo1, hi1 = np.nanpercentile(F_s, [q_lo_hi_1[0], q_lo_hi_1[1]], axis=0)
        lo2, hi2 = np.nanpercentile(F_s, [q_lo_hi_2[0], q_lo_hi_2[1]], axis=0)

        return centers, f_obs, med, lo1, hi1, lo2, hi2

    # Build densities for cal and val
    panels = [("calibration", m_cal), ("validation", m_val)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)

    for ax, (title, mask) in zip(axes, panels):
        if not np.any(mask):
            ax.set_axis_off()
            ax.set_title(f"{title} (no data)")
            continue

        x_obs = oy[mask]
        Xd = D_obs[:, mask]

        centers, f_obs, med, lo1, hi1, lo2, hi2 = _density_band(x_obs, Xd)

        ax.fill_between(centers, lo1, hi1, alpha=0.18, label=f"{q_lo_hi_1[0]}–{q_lo_hi_1[1]}% band")
        ax.fill_between(centers, lo2, hi2, alpha=0.28, label=f"{q_lo_hi_2[0]}–{q_lo_hi_2[1]}% band")
        ax.plot(centers, med, lw=1.6, label="model median density")
        ax.plot(centers, f_obs, lw=1.6, label="obs KDE")

        ax.set_title(title)
        ax.set_xlabel("shoreline position")
        ax.grid(alpha=0.25, lw=0.6)

    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=9, frameon=False, ncol=1)
    fig.suptitle(f"Obs KDE vs model density bands: {label}", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"density_kde_bands_{label}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residual_diagnostics(
    obs_time: np.ndarray,
    obs: np.ndarray,
    pred_med_obs: np.ndarray,
    label: str,
    out_dir: Path,
) -> None:
    """Residual time series + histogram + QQ plot (based on posterior median)."""
    import matplotlib.pyplot as plt

    res = np.asarray(obs, dtype=float) - np.asarray(pred_med_obs, dtype=float)
    m = np.isfinite(res)
    res = res[m]
    t = np.asarray(obs_time)[m]
    if res.size < 5:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), gridspec_kw={"width_ratios": [2.0, 1.0, 1.0]})

    axes[0].plot(t, res, color=_THEME["sky"], lw=1.2, alpha=0.95)
    axes[0].scatter(t, res, s=12, color=_THEME["sky"], alpha=0.65)
    axes[0].axhline(0.0, lw=1.1, color=_THEME["slate"], ls="--")
    axes[0].set_title("Residuals vs time")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("obs − median")
    _apply_axes_style(axes[0], grid_axis="both")
    if np.issubdtype(np.asarray(obs_time).dtype, np.datetime64):
        axes[0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[0].xaxis.get_major_locator()))

    axes[1].hist(res, bins=min(32, max(12, int(np.sqrt(res.size)))), density=True, color=_THEME["teal"], alpha=0.6, edgecolor="white", linewidth=0.8)
    try:
        kde = gaussian_kde(res)
        xs = np.linspace(np.nanpercentile(res, 1), np.nanpercentile(res, 99), 300)
        axes[1].plot(xs, kde(xs), color=_THEME["navy"], lw=1.7)
    except Exception:
        pass
    axes[1].axvline(np.nanmean(res), color=_THEME["coral"], lw=1.1, ls="--")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Density")
    _apply_axes_style(axes[1], grid_axis="y")

    try:
        from scipy.stats import norm
        z = (res - np.mean(res)) / (np.std(res) + 1e-12)
        z = np.sort(z)
        q = norm.ppf((np.arange(1, z.size + 1) - 0.5) / z.size)
        axes[2].scatter(q, z, s=14, color=_THEME["purple"], alpha=0.75)
        lim = float(max(np.max(np.abs(q)), np.max(np.abs(z))))
        axes[2].plot([-lim, lim], [-lim, lim], lw=1.1, color=_THEME["slate"], ls="--")
        axes[2].set_xlim(-lim, lim)
        axes[2].set_ylim(-lim, lim)
        axes[2].set_title("QQ plot vs Normal")
        axes[2].set_xlabel("Theoretical quantiles")
        axes[2].set_ylabel("Empirical quantiles")
        _apply_axes_style(axes[2], grid_axis="both")
    except Exception:
        axes[2].axis("off")

    fig.suptitle(f"Residual diagnostics — {label}", y=1.02, fontsize=13, fontweight="semibold", color=_THEME["ink"])
    fig.tight_layout()
    fig.savefig(out_dir / f"residual_diagnostics_{label}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)



# ---------------------------
# Diagnostics + metrics
# ---------------------------


def save_calibration_metrics(
    *,
    time: np.ndarray,
    per5: np.ndarray,
    per50: np.ndarray,
    per95: np.ndarray,
    mini: np.ndarray,
    maxi: np.ndarray,
    draws: Optional[np.ndarray] = None,
    obs_time: np.ndarray,
    obs: np.ndarray,
    label: str,
    out_dir: Path,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Compute + save comparable metrics for calibration vs validation.

    If `obs_mask_cal/obs_mask_val` are provided, metrics are computed separately
    on those subsets (masks are defined on the full obs arrays).

    Validation convention:
        calibration: [start_date, end_date)
        validation:  [end_date, end_of_record]
    """
    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)

    if obs_mask_cal is None:
        obs_mask_cal = np.isfinite(obs)
    else:
        obs_mask_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
    if obs_mask_val is None:
        obs_mask_val = np.zeros_like(obs_mask_cal, dtype=bool)
    else:
        obs_mask_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    # Map envelopes to observation times (no extrapolation: NaN outside window)
    per50_o = _interp_to_obs(time, per50, obs_time)
    per5_o = _interp_to_obs(time, per5, obs_time)
    per95_o = _interp_to_obs(time, per95, obs_time)
    mini_o = _interp_to_obs(time, mini, obs_time)
    maxi_o = _interp_to_obs(time, maxi, obs_time)

    base = np.isfinite(obs) & np.isfinite(per50_o) & np.isfinite(per5_o) & np.isfinite(per95_o)
    m_cal = base & obs_mask_cal
    m_val = base & obs_mask_val

    def _compute(mask: np.ndarray) -> dict:
        if mask.sum() < 3:
            return {
                "rmse_median": float("nan"),
                "mae_median": float("nan"),
                "bias_median": float("nan"),
                "pearson_r": float("nan"),
                "nse_median": float("nan"),
                "kge_median": float("nan"),
                "coverage_95": float("nan"),
                "coverage_min_max": float("nan"),
                "mean_width_90": float("nan"),
                "n_used": int(mask.sum()),
            }
        o = obs[mask]
        p50 = per50_o[mask]
        p5 = per5_o[mask]
        p95 = per95_o[mask]
        mi = mini_o[mask]
        ma = maxi_o[mask]
        return {
            "rmse_median": _rmse(o, p50),
            "mae_median": _mae(o, p50),
            "bias_median": _bias(p50, o),
            "pearson_r": _pearsonr(o, p50),
            "nse_median": _nse(o, p50),
            "kge_median": _kge(o, p50),
            "coverage_p5_p95": _coverage(o, p5, p95),
            "coverage_min_max": _coverage(o, mi, ma),
            "mean_width_90": _mean_width(p5, p95),
            "n_used": int(mask.sum()),
        }

    met_cal = _compute(m_cal)
    met_val = _compute(m_val)

    # Shared counts
    n_obs_total = int(np.isfinite(obs).sum())
    n_cal_total = int(obs_mask_cal.sum())
    n_val_total = int(obs_mask_val.sum())

    metric_order = [
        "rmse_median",
        "mae_median",
        "bias_median",
        "pearson_r",
        "nse_median",
        "kge_median",
        "coverage_p5_p95",
        "coverage_min_max",
        "mean_width_90",
        "n_obs_total",
        "n_cal_total",
        "n_val_total",
        "n_used_cal",
        "n_used_val",
    ]

    rows = []
    def _fmt(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if v is None:
            return ""
        try:
            vf = float(v)
            return "" if not np.isfinite(vf) else f"{vf:.6g}"
        except Exception:
            return str(v)

    # Build a flat dict for CSV convenience
    cal_col = {
        **{k: met_cal.get(k, float("nan")) for k in met_cal},
        "n_obs_total": n_obs_total,
        "n_cal_total": n_cal_total,
        "n_val_total": n_val_total,
        "n_used_cal": met_cal.get("n_used", 0),
        "n_used_val": met_val.get("n_used", 0),
    }
    val_col = {
        **{k: met_val.get(k, float("nan")) for k in met_val},
        "n_obs_total": n_obs_total,
        "n_cal_total": n_cal_total,
        "n_val_total": n_val_total,
        "n_used_cal": met_cal.get("n_used", 0),
        "n_used_val": met_val.get("n_used", 0),
    }

    for k in metric_order:
        rows.append([k, _fmt(cal_col.get(k)), _fmt(val_col.get(k))])

    # Save CSV
    out_csv = out_dir / f"calibration_metrics_{label}.csv"
    try:
        import pandas as pd

        df = pd.DataFrame(rows, columns=["metric", "calibration", "validation"])
        # optionally embed split date
        if split_date is not None:
            df.attrs["split_date"] = str(np.datetime64(split_date))
        df.to_csv(out_csv, index=False)
    except Exception:
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("metric,calibration,validation\n")
            for r in rows:
                f.write(",".join(r) + "\n")

    title = f"Metrics (calibration vs validation): {label}"
    if split_date is not None:
        try:
            title += f"\nSplit @ {np.datetime64(split_date)}"
        except Exception:
            pass

    _save_table_png(
        rows=rows,
        col_labels=["metric", "calibration", "validation"],
        out_png=out_dir / f"calibration_metrics_{label}.png",
        title=title,
        font_size=10,
        max_rows_per_page=60,
    )




# ---------------------------
# Probabilistic diagnostics (from draws or quantile-based approximation)
# ---------------------------

_DEFAULT_COVERAGE_LEVELS = (0.50, 0.67, 0.80, 0.90, 0.95, 0.99)


def _nearest_time_index(time: np.ndarray, obs_time: np.ndarray) -> np.ndarray:
    """Map obs_time to nearest index on time grid."""
    t = np.asarray(time)
    ot = np.asarray(obs_time)
    if t.size == 0 or ot.size == 0:
        return np.zeros((ot.size,), dtype=int)

    if np.issubdtype(t.dtype, np.datetime64) and np.issubdtype(ot.dtype, np.datetime64):
        tt = t.astype("datetime64[ns]")
        oo = ot.astype("datetime64[ns]")
        idx = np.searchsorted(tt, oo)
        idx = np.clip(idx, 1, len(tt) - 1)
        left = idx - 1
        right = idx
        choose_right = (oo - tt[left]) > (tt[right] - oo)
        return np.where(choose_right, right, left).astype(int)

    # numeric case
    tnum = np.asarray(t, dtype=float)
    onum = np.asarray(ot, dtype=float)
    idx = np.searchsorted(tnum, onum)
    idx = np.clip(idx, 1, len(tnum) - 1)
    left = idx - 1
    right = idx
    choose_right = (onum - tnum[left]) > (tnum[right] - onum)
    return np.where(choose_right, right, left).astype(int)


def _infer_studentt_nu_from_trace_or_spread(
    *,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    trace=None,
) -> tuple[bool, float | None]:
    """Infer whether to use Student-t and (if so) df nu.

    Priority:
      1) If trace contains posterior 'nu' -> Student-t with median nu.
      2) Else heuristic from spread ratio (99-1)/(90-10).
    """
    use_studentt = False
    nu_val: float | None = None

    # 1) From trace
    try:
        if trace is not None and hasattr(trace, "posterior"):
            post = trace.posterior
            if hasattr(post, "data_vars") and ("nu" in post.data_vars):
                nu_arr = np.asarray(post["nu"].values).reshape(-1)
                nu_arr = nu_arr[np.isfinite(nu_arr)]
                if nu_arr.size:
                    use_studentt = True
                    nu_val = float(np.nanmedian(nu_arr))
    except Exception:
        pass

    # 2) Heuristic
    if not use_studentt:
        try:
            spread_90 = (q90 - q10)
            spread_99 = (q99 - q01)
            m = np.isfinite(spread_90) & np.isfinite(spread_99) & (spread_90 > 0) & (spread_99 > 0)
            if m.any():
                rr = spread_99[m] / np.maximum(spread_90[m], 1e-12)
                rr = rr[np.isfinite(rr) & (rr > 0)]
                if rr.size:
                    r_med = float(np.nanmedian(rr))
                    # Normal ratio ≈ 1.815; heavier tails -> larger
                    if r_med > 1.95:
                        use_studentt = True
                        target = r_med
                        lo, hi = 2.1, 200.0
                        for _ in range(40):
                            mid = 0.5 * (lo + hi)
                            r_mid = float(_student_t.ppf(0.99, mid) / _student_t.ppf(0.90, mid))
                            if r_mid > target:
                                lo = mid
                            else:
                                hi = mid
                        nu_val = float(0.5 * (lo + hi))
        except Exception:
            use_studentt = False
            nu_val = None

    if use_studentt and (nu_val is None or not np.isfinite(nu_val)):
        return False, None
    return use_studentt, nu_val


def _predictive_params_at_obs(
    *,
    time: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per1: np.ndarray,
    per99: np.ndarray,
    obs_time: np.ndarray,
    trace=None,
) -> dict:
    """Compute predictive parameters at obs times using quantile-based approximation."""
    q10 = _interp_to_obs(time, per10, obs_time)
    q50 = _interp_to_obs(time, per50, obs_time)
    q90 = _interp_to_obs(time, per90, obs_time)
    q01 = _interp_to_obs(time, per1, obs_time)
    q99 = _interp_to_obs(time, per99, obs_time)

    use_studentt, nu_val = _infer_studentt_nu_from_trace_or_spread(q10=q10, q50=q50, q90=q90, q01=q01, q99=q99, trace=trace)

    # scale from 90-10 width
    width = np.maximum(q90 - q10, 1e-9)
    if use_studentt:
        t90 = float(_student_t.ppf(0.90, float(nu_val)))
        sig = width / np.maximum(2.0 * t90, 1e-9)
        like_name = f"studentt (nu≈{float(nu_val):.2g})"
    else:
        z90 = float(_norm.ppf(0.90))
        sig = width / np.maximum(2.0 * z90, 1e-9)
        like_name = "normal"
    sig = np.maximum(sig, 1e-9)

    return {
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "q01": q01,
        "q99": q99,
        "mu": q50,
        "sig": sig,
        "use_studentt": use_studentt,
        "nu": nu_val,
        "like_name": like_name,
    }


def _logpdf_at_obs(y: np.ndarray, mu: np.ndarray, sig: np.ndarray, use_studentt: bool, nu: float | None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-9)
    z = (y - mu) / sig
    if use_studentt:
        nuv = float(nu)
        return (
            _gammaln((nuv + 1.0) / 2.0)
            - _gammaln(nuv / 2.0)
            - 0.5 * np.log(nuv * np.pi)
            - np.log(sig)
            - ((nuv + 1.0) / 2.0) * np.log1p((z * z) / nuv)
        )
    return -0.5 * (z * z) - np.log(sig) - 0.5 * np.log(2.0 * np.pi)


def _cdf_at_obs(y: np.ndarray, mu: np.ndarray, sig: np.ndarray, use_studentt: bool, nu: float | None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-9)
    z = (y - mu) / sig
    if use_studentt:
        return _student_t.cdf(z, float(nu))
    return _norm.cdf(z)


def _crps_from_ensemble(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS from ensemble draws.

    x: (S, N) draws at obs times.
    y: (N,) observations.
    Returns (N,) CRPS.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    S, N = x.shape
    # term1 = mean |x - y|
    term1 = np.mean(np.abs(x - y[None, :]), axis=0)

    # term2 = 0.5 * Gini mean difference of x
    xs = np.sort(x, axis=0)
    i = np.arange(1, S + 1, dtype=float)[:, None]
    w = (2.0 * i - S - 1.0)
    gmd = (2.0 / (S * S)) * np.sum(w * xs, axis=0)
    return term1 - 0.5 * gmd


def _crps_normal(y: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-9)
    z = (y - mu) / sig
    # CRPS for Normal
    return sig * (z * (2.0 * _norm.cdf(z) - 1.0) + 2.0 * _norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def _crps_studentt_mc(y: np.ndarray, mu: np.ndarray, sig: np.ndarray, nu: float, *, n_mc: int = 250, seed: int = 123) -> np.ndarray:
    """Approximate CRPS for Student-t via Monte Carlo using ensemble formula."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-9)
    N = y.size
    # sample (n_mc, N)
    x = rng.standard_t(df=float(nu), size=(n_mc, N)) * sig[None, :] + mu[None, :]
    return _crps_from_ensemble(x, y)


def _coverage_and_width_from_dist(mu: np.ndarray, sig: np.ndarray, use_studentt: bool, nu: float | None, levels=_DEFAULT_COVERAGE_LEVELS) -> tuple[dict[float, float], dict[float, float], dict[float, tuple[np.ndarray, np.ndarray]]]:
    """Return per-level coverage helper structures (no obs needed) and quantile bounds."""
    bounds = {}
    widths = {}
    for lev in levels:
        a = 0.5 * (1.0 - lev)
        lo_p = a
        hi_p = 1.0 - a
        if use_studentt:
            qlo = mu + sig * _student_t.ppf(lo_p, float(nu))
            qhi = mu + sig * _student_t.ppf(hi_p, float(nu))
        else:
            qlo = mu + sig * _norm.ppf(lo_p)
            qhi = mu + sig * _norm.ppf(hi_p)
        bounds[lev] = (qlo, qhi)
        widths[lev] = float(np.nanmean(qhi - qlo))
    return widths, bounds


def save_probabilistic_diagnostics(
    *,
    time: np.ndarray,
    per1: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per99: np.ndarray,
    per5: np.ndarray,
    per95: np.ndarray,
    mini: np.ndarray,
    maxi: np.ndarray,
    draws: Optional[np.ndarray],
    obs_time: np.ndarray,
    obs: np.ndarray,
    label: str,
    out_dir: Path,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
    trace=None,
    gate_weights: Optional[np.ndarray] = None,
    gate_model_names: Sequence[str] = (),
    rotation_series: Optional[dict[str, np.ndarray]] = None,
) -> None:
    """Save a single table with deterministic + probabilistic diagnostics (cal vs val)."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)

    if obs_mask_cal is None:
        obs_mask_cal = np.isfinite(obs)
    else:
        obs_mask_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
    if obs_mask_val is None:
        obs_mask_val = np.zeros_like(obs_mask_cal, dtype=bool)
    else:
        obs_mask_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    pp = _predictive_params_at_obs(
        time=time,
        per10=per10,
        per50=per50,
        per90=per90,
        per1=per1,
        per99=per99,
        obs_time=obs_time,
        trace=trace,
    )
    mu_o = pp["mu"]
    sig_o = pp["sig"]
    use_t = bool(pp["use_studentt"])
    nu = pp["nu"]

    base = np.isfinite(obs) & np.isfinite(mu_o) & np.isfinite(sig_o)
    m_cal = base & obs_mask_cal
    m_val = base & obs_mask_val

    per50_o = mu_o
    lpd = _logpdf_at_obs(obs, mu_o, sig_o, use_t, nu)
    ign_bits = _ignorance_bits(lpd)

    pit = None
    crps = None
    if draws is not None:
        try:
            d = np.asarray(draws, dtype=float)
            if d.ndim == 2 and d.shape[1] == np.asarray(time).shape[0]:
                idx = _nearest_time_index(time, obs_time)
                d_obs = d[:, idx]
                pit = (np.sum(d_obs <= obs[None, :], axis=0) + 1.0) / (d_obs.shape[0] + 1.0)
                crps = _crps_from_ensemble(d_obs, obs)
        except Exception:
            pit = None
            crps = None

    if pit is None:
        pit = _cdf_at_obs(obs, mu_o, sig_o, use_t, nu)
    if crps is None:
        if use_t:
            crps = _crps_studentt_mc(obs, mu_o, sig_o, float(nu))
        else:
            crps = _crps_normal(obs, mu_o, sig_o)

    z = (obs - mu_o) / np.maximum(sig_o, 1e-9)

    levels = list(_DEFAULT_COVERAGE_LEVELS)
    cov_cal, cov_val, wid_cal, wid_val = {}, {}, {}, {}
    for lev in levels:
        qlo, qhi = _interval_bounds(mu_o, sig_o, lev, use_t, nu)
        cov_cal[lev] = float(np.mean((obs[m_cal] >= qlo[m_cal]) & (obs[m_cal] <= qhi[m_cal]))) if m_cal.sum() else float("nan")
        cov_val[lev] = float(np.mean((obs[m_val] >= qlo[m_val]) & (obs[m_val] <= qhi[m_val]))) if m_val.sum() else float("nan")
        wid_cal[lev] = float(np.nanmean(qhi[m_cal] - qlo[m_cal])) if m_cal.sum() else float("nan")
        wid_val[lev] = float(np.nanmean(qhi[m_val] - qlo[m_val])) if m_val.sum() else float("nan")

    per5_o = _interp_to_obs(time, per5, obs_time)
    per95_o = _interp_to_obs(time, per95, obs_time)
    mini_o = _interp_to_obs(time, mini, obs_time)
    maxi_o = _interp_to_obs(time, maxi, obs_time)

    ref_mask = m_cal if int(m_cal.sum()) >= 5 else base
    if int(ref_mask.sum()) >= 5:
        clim_lo10, clim_hi90 = np.nanpercentile(obs[ref_mask], [10.0, 90.0])
    else:
        clim_lo10, clim_hi90 = float("nan"), float("nan")
    prob_low10 = _dist_cdf_at_threshold(clim_lo10, mu_o, sig_o, use_t, nu) if np.isfinite(clim_lo10) else np.full_like(obs, np.nan, dtype=float)
    prob_high90 = 1.0 - _dist_cdf_at_threshold(clim_hi90, mu_o, sig_o, use_t, nu) if np.isfinite(clim_hi90) else np.full_like(obs, np.nan, dtype=float)
    evt_low10 = (obs <= clim_lo10).astype(float) if np.isfinite(clim_lo10) else np.full_like(obs, np.nan, dtype=float)
    evt_high90 = (obs >= clim_hi90).astype(float) if np.isfinite(clim_hi90) else np.full_like(obs, np.nan, dtype=float)

    def _compute(mask: np.ndarray):
        if mask.sum() < 3:
            return {}
        o = obs[mask]
        p50 = per50_o[mask]
        p5 = per5_o[mask]
        p95 = per95_o[mask]
        mi = mini_o[mask]
        ma = maxi_o[mask]
        q05, q25, _, q75, q95 = np.nanpercentile(o, [5, 25, 50, 75, 95])
        obs_iqr = q75 - q25
        obs_width_90 = q95 - q05
        out = {
            "rmse_median": _rmse(o, p50),
            "mae_median": _mae(o, p50),
            "bias_median": _bias(p50, o),
            "pearson_r": _pearsonr(o, p50),
            "nse_median": _nse(o, p50),
            "kge_median": _kge(o, p50),
            "coverage_p5_p95": _coverage(o, p5, p95),
            "coverage_min_max": _coverage(o, mi, ma),
            "mean_width_90": _mean_width(p5, p95),
            "elpd_sum": float(np.nansum(lpd[mask])),
            "elpd_mean": float(np.nanmean(lpd[mask])),
            "crps_mean": float(np.nanmean(crps[mask])),
            "ignorance_mean_bits": float(np.nanmean(ign_bits[mask])),
            "brier_low10": _brier_score(prob_low10[mask], evt_low10[mask]),
            "brier_high90": _brier_score(prob_high90[mask], evt_high90[mask]),
            "pit_mean": float(np.nanmean(pit[mask])),
            "pit_std": float(np.nanstd(pit[mask])),
            "z_mean": float(np.nanmean(z[mask])),
            "z_std": float(np.nanstd(z[mask])),
            "z_frac_abs1": float(np.mean(np.abs(z[mask]) <= 1.0)),
            "z_frac_abs2": float(np.mean(np.abs(z[mask]) <= 2.0)),
            "n_used": int(mask.sum()),
            "obs_q25": float(q25),
            "obs_q75": float(q75),
            "obs_iqr": float(obs_iqr),
            "obs_q05": float(q05),
            "obs_q95": float(q95),
            "obs_width_90": float(obs_width_90),
        }
        for lev in levels:
            key = int(round(lev * 100))
            out[f"coverage_nom_{key:02d}"] = cov_cal[lev] if mask is m_cal else cov_val[lev]
            out[f"sharpness_nom_{key:02d}"] = wid_cal[lev] if mask is m_cal else wid_val[lev]
            out[f"avg_width_{key:02d}"] = wid_cal[lev] if mask is m_cal else wid_val[lev]
        out["brier_extreme10_mean"] = float(np.nanmean([out.get("brier_low10", np.nan), out.get("brier_high90", np.nan)]))
        sh90 = wid_cal.get(0.90) if mask is m_cal else wid_val.get(0.90)
        out["sharp90_over_obs_iqr"] = float(sh90 / obs_iqr) if np.isfinite(obs_iqr) and obs_iqr > 0 and np.isfinite(sh90) else float("nan")
        out["sharp90_over_obs_width90"] = float(sh90 / obs_width_90) if np.isfinite(obs_width_90) and obs_width_90 > 0 and np.isfinite(sh90) else float("nan")
        return out

    met_cal = _compute(m_cal)
    met_val = _compute(m_val)

    ent_cal = ent_val = float("nan")
    if gate_weights is not None and len(gate_model_names) > 0:
        W = np.asarray(gate_weights, dtype=float)
        if W.ndim == 2:
            if W.shape[0] == len(gate_model_names):
                Wt = W.T
            elif W.shape[1] == len(gate_model_names):
                Wt = W
            else:
                Wt = W
            eps = 1e-12
            H = -np.sum(Wt * np.log(np.maximum(Wt, eps)), axis=1)
            H = H / np.log(max(2, Wt.shape[1]))
            idx = _nearest_time_index(time, obs_time)
            Hobs = H[idx]
            ent_cal = float(np.nanmean(Hobs[m_cal])) if m_cal.sum() else float("nan")
            ent_val = float(np.nanmean(Hobs[m_val])) if m_val.sum() else float("nan")

    metric_order = [
        "rmse_median", "mae_median", "bias_median", "pearson_r", "nse_median", "kge_median",
        "elpd_sum", "elpd_mean", "crps_mean", "ignorance_mean_bits",
        "brier_low10", "brier_high90", "brier_extreme10_mean",
        "pit_mean", "pit_std", "z_mean", "z_std", "z_frac_abs1", "z_frac_abs2",
        "coverage_p5_p95", "coverage_min_max",
        "avg_width_50", "avg_width_67", "avg_width_90", "avg_width_95", "avg_width_99",
        "mean_width_90",
        "obs_q25", "obs_q75", "obs_iqr", "obs_q05", "obs_q95", "obs_width_90",
        "sharp90_over_obs_iqr", "sharp90_over_obs_width90",
    ]
    for lev in levels:
        metric_order.append(f"coverage_nom_{int(round(lev * 100)):02d}")
    for lev in levels:
        metric_order.append(f"sharpness_nom_{int(round(lev * 100)):02d}")
    if gate_weights is not None and len(gate_model_names) > 0:
        metric_order += ["entropy_mean"]
        met_cal["entropy_mean"] = ent_cal
        met_val["entropy_mean"] = ent_val
    metric_order += ["n_used"]

    def _fmt(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        try:
            vf = float(v)
            return "" if not np.isfinite(vf) else f"{vf:.6g}"
        except Exception:
            return str(v)

    rows = [[k, _fmt(met_cal.get(k, float("nan"))), _fmt(met_val.get(k, float("nan")))] for k in metric_order]

    out_csv = out_dir / f"diagnostics_{label}.csv"
    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["metric", "calibration", "validation"])
        if split_date is not None:
            df.attrs["split_date"] = str(np.datetime64(split_date))
        df.attrs["predictive_family"] = pp["like_name"]
        df.to_csv(out_csv, index=False)
    except Exception:
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("metric,calibration,validation\n")
            for r in rows:
                f.write(",".join(r) + "\n")

    title = f"Probabilistic diagnostics (cal vs val): {label}\nPredictive family: {pp['like_name']}"
    if np.isfinite(clim_lo10) and np.isfinite(clim_hi90):
        title += f"\nBrier thresholds from calibration obs: p10={clim_lo10:.3g}, p90={clim_hi90:.3g}"
    if split_date is not None:
        try:
            title += f"\nSplit @ {np.datetime64(split_date)}"
        except Exception:
            pass

    _save_table_png(
        rows=rows,
        col_labels=["metric", "calibration", "validation"],
        out_png=out_dir / f"diagnostics_{label}.png",
        title=title,
        font_size=10,
        max_rows_per_page=72,
    )


def save_probabilistic_diagnostics_2d(
    *,
    time: np.ndarray,
    per1: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per99: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    label: str,
    out_dir: Path,
    obs_mask: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Global + per-transect diagnostics table for 2D (time, transect) outputs."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)
    tr_dir = out_dir / "transects"
    _safe_makedirs(tr_dir)

    t = np.asarray(time)
    ot = np.asarray(obs_time)
    O = np.asarray(obs, dtype=float)
    P1 = np.asarray(per1, dtype=float)
    P10 = np.asarray(per10, dtype=float)
    P50 = np.asarray(per50, dtype=float)
    P90 = np.asarray(per90, dtype=float)
    P99 = np.asarray(per99, dtype=float)

    if P50.ndim != 2 or O.ndim != 2:
        return
    if P50.shape[0] != t.size:
        raise ValueError("per50 shape must be (time, transect)")
    if O.shape[0] != ot.size:
        raise ValueError("obs must be (obs_time, transect)")
    n_tr = int(P50.shape[1])

    valid_obs = np.isfinite(O) if obs_mask is None else ((~np.asarray(obs_mask, dtype=bool)) & np.isfinite(O))
    if split_date is not None and np.issubdtype(ot.dtype, np.datetime64):
        sd = np.datetime64(split_date)
        is_cal_t = (ot < sd)
        is_val_t = ~is_cal_t
    else:
        is_cal_t = np.ones_like(ot, dtype=bool)
        is_val_t = np.zeros_like(ot, dtype=bool)

    q01 = _interp_to_obs_2d(t, P1, ot)
    q10 = _interp_to_obs_2d(t, P10, ot)
    q50 = _interp_to_obs_2d(t, P50, ot)
    q90 = _interp_to_obs_2d(t, P90, ot)
    q99 = _interp_to_obs_2d(t, P99, ot)

    finite_pred = np.isfinite(q50) & np.isfinite(q10) & np.isfinite(q90) & np.isfinite(q01) & np.isfinite(q99)
    ok = valid_obs & finite_pred
    cal_mask = ok & is_cal_t[:, None]
    val_mask = ok & is_val_t[:, None]
    levels = list(_DEFAULT_COVERAGE_LEVELS)

    def _metrics_from_arrays(o, q01_, q10_, q50_, q90_, q99_, ref_obs=None):
        msk = np.isfinite(o) & np.isfinite(q01_) & np.isfinite(q10_) & np.isfinite(q50_) & np.isfinite(q90_) & np.isfinite(q99_)
        if int(msk.sum()) < 3:
            return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "r": float("nan"), "nse": float("nan"), "kge": float("nan"), "crps_mean": float("nan"), "ignorance_mean_bits": float("nan"), "brier_low10": float("nan"), "brier_high90": float("nan"), "brier_extreme10_mean": float("nan"), "n_used": int(msk.sum())}
        oo = np.asarray(o, dtype=float)[msk]
        qq01 = np.asarray(q01_, dtype=float)[msk]
        qq10 = np.asarray(q10_, dtype=float)[msk]
        qq50 = np.asarray(q50_, dtype=float)[msk]
        qq90 = np.asarray(q90_, dtype=float)[msk]
        qq99 = np.asarray(q99_, dtype=float)[msk]
        use_t, nu = _infer_studentt_nu_from_trace_or_spread(q10=qq10, q50=qq50, q90=qq90, q01=qq01, q99=qq99, trace=None)
        width = np.maximum(qq90 - qq10, 1e-9)
        if use_t:
            sc = width / np.maximum(2.0 * float(_student_t.ppf(0.90, float(nu))), 1e-9)
        else:
            sc = width / np.maximum(2.0 * float(_norm.ppf(0.90)), 1e-9)
        sc = np.maximum(sc, 1e-9)
        lpd = _logpdf_at_obs(oo, qq50, sc, use_t, nu)
        ign = _ignorance_bits(lpd)
        crps = _crps_studentt_mc(oo, qq50, sc, float(nu)) if use_t else _crps_normal(oo, qq50, sc)
        ref = np.asarray(ref_obs if ref_obs is not None else oo, dtype=float)
        ref = ref[np.isfinite(ref)]
        if ref.size >= 5:
            thr_lo, thr_hi = np.nanpercentile(ref, [10.0, 90.0])
            p_lo = _dist_cdf_at_threshold(thr_lo, qq50, sc, use_t, nu)
            p_hi = 1.0 - _dist_cdf_at_threshold(thr_hi, qq50, sc, use_t, nu)
            e_lo = (oo <= thr_lo).astype(float)
            e_hi = (oo >= thr_hi).astype(float)
            bs_lo = _brier_score(p_lo, e_lo)
            bs_hi = _brier_score(p_hi, e_hi)
        else:
            bs_lo = bs_hi = float("nan")
        out = {
            "rmse": _rmse(oo, qq50),
            "mae": _mae(oo, qq50),
            "bias": _bias(qq50, oo),
            "r": _pearsonr(oo, qq50),
            "nse": _nse(oo, qq50),
            "kge": _kge(oo, qq50),
            "coverage_10_90": _coverage(oo, qq10, qq90),
            "coverage_01_99": _coverage(oo, qq01, qq99),
            "sharpness_10_90": _mean_width(qq10, qq90),
            "sharpness_01_99": _mean_width(qq01, qq99),
            "crps_mean": float(np.nanmean(crps)),
            "ignorance_mean_bits": float(np.nanmean(ign)),
            "brier_low10": bs_lo,
            "brier_high90": bs_hi,
            "brier_extreme10_mean": float(np.nanmean([bs_lo, bs_hi])),
            "n_used": int(msk.sum()),
        }
        for lev in levels:
            qlo, qhi = _interval_bounds(qq50, sc, lev, use_t, nu)
            key = int(round(lev * 100))
            out[f"coverage_nom_{key:02d}"] = _coverage(oo, qlo, qhi)
            out[f"avg_width_{key:02d}"] = _mean_width(qlo, qhi)
        return out

    def _fmt(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        try:
            vf = float(v)
            return "" if not np.isfinite(vf) else f"{vf:.6g}"
        except Exception:
            return str(v)

    ref_global = O[cal_mask] if int(np.sum(cal_mask)) >= 5 else O[ok]
    cal = _metrics_from_arrays(O[cal_mask], q01[cal_mask], q10[cal_mask], q50[cal_mask], q90[cal_mask], q99[cal_mask], ref_obs=ref_global)
    val = _metrics_from_arrays(O[val_mask], q01[val_mask], q10[val_mask], q50[val_mask], q90[val_mask], q99[val_mask], ref_obs=ref_global)

    order = [
        "rmse", "mae", "bias", "r", "nse", "kge",
        "crps_mean", "ignorance_mean_bits",
        "brier_low10", "brier_high90", "brier_extreme10_mean",
        "coverage_10_90", "coverage_01_99",
        "avg_width_50", "avg_width_67", "avg_width_90", "avg_width_95", "avg_width_99",
        "sharpness_10_90", "sharpness_01_99", "n_used",
    ]
    for lev in levels:
        order.append(f"coverage_nom_{int(round(lev * 100)):02d}")

    global_rows = [[k, _fmt(cal.get(k)), _fmt(val.get(k))] for k in order]
    out_csv = out_dir / f"diagnostics_{label}_global.csv"
    try:
        import pandas as pd
        pd.DataFrame(global_rows, columns=["metric", "calibration", "validation"]).to_csv(out_csv, index=False)
    except Exception:
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("metric,calibration,validation\n")
            for r in global_rows:
                f.write(",".join(r) + "\n")

    title = f"Diagnostics (global pooled): {label}"
    if split_date is not None:
        try:
            title += f"\nSplit @ {np.datetime64(split_date)}"
        except Exception:
            pass
    _save_table_png(global_rows, ["metric", "calibration", "validation"], out_dir / f"diagnostics_{label}_global.png", title=title, font_size=10, max_rows_per_page=72)

    by_rows = []
    summary_header = [
        "transect", "cal_rmse", "val_rmse", "cal_kge", "val_kge", "cal_r", "val_r",
        "cal_crps", "val_crps", "cal_ign_bits", "val_ign_bits",
        "cal_cov10_90", "val_cov10_90", "cal_avgw90", "val_avgw90", "cal_n", "val_n",
    ]

    for j in range(n_tr):
        ref_j = O[:, j][cal_mask[:, j]] if int(np.sum(cal_mask[:, j])) >= 5 else O[:, j][ok[:, j]]
        cal_j = _metrics_from_arrays(O[:, j][cal_mask[:, j]], q01[:, j][cal_mask[:, j]], q10[:, j][cal_mask[:, j]], q50[:, j][cal_mask[:, j]], q90[:, j][cal_mask[:, j]], q99[:, j][cal_mask[:, j]], ref_obs=ref_j)
        val_j = _metrics_from_arrays(O[:, j][val_mask[:, j]], q01[:, j][val_mask[:, j]], q10[:, j][val_mask[:, j]], q50[:, j][val_mask[:, j]], q90[:, j][val_mask[:, j]], q99[:, j][val_mask[:, j]], ref_obs=ref_j)

        rows_j = [[k, _fmt(cal_j.get(k)), _fmt(val_j.get(k))] for k in order]
        out_csv_j = tr_dir / f"diagnostics_{label}_transect_{j:02d}.csv"
        try:
            import pandas as pd
            pd.DataFrame(rows_j, columns=["metric", "calibration", "validation"]).to_csv(out_csv_j, index=False)
        except Exception:
            with out_csv_j.open("w", encoding="utf-8") as f:
                f.write("metric,calibration,validation\n")
                for r in rows_j:
                    f.write(",".join(r) + "\n")
        _save_table_png(rows_j, ["metric", "calibration", "validation"], tr_dir / f"diagnostics_{label}_transect_{j:02d}.png", title=f"Diagnostics (transect {j:02d}): {label}", font_size=10, max_rows_per_page=72)

        by_rows.append([
            f"{j:02d}", _fmt(cal_j.get("rmse")), _fmt(val_j.get("rmse")), _fmt(cal_j.get("kge")), _fmt(val_j.get("kge")),
            _fmt(cal_j.get("r")), _fmt(val_j.get("r")), _fmt(cal_j.get("crps_mean")), _fmt(val_j.get("crps_mean")),
            _fmt(cal_j.get("ignorance_mean_bits")), _fmt(val_j.get("ignorance_mean_bits")),
            _fmt(cal_j.get("coverage_10_90")), _fmt(val_j.get("coverage_10_90")),
            _fmt(cal_j.get("avg_width_90")), _fmt(val_j.get("avg_width_90")),
            _fmt(cal_j.get("n_used")), _fmt(val_j.get("n_used")),
        ])

    out_csv_bt = tr_dir / f"diagnostics_{label}_by_transect.csv"
    try:
        import pandas as pd
        pd.DataFrame(by_rows, columns=summary_header).to_csv(out_csv_bt, index=False)
    except Exception:
        with out_csv_bt.open("w", encoding="utf-8") as f:
            f.write(",".join(summary_header) + "\n")
            for r in by_rows:
                f.write(",".join(r) + "\n")
    _save_table_png(by_rows, summary_header, tr_dir / f"diagnostics_{label}_by_transect.png", title=f"Diagnostics by transect: {label}", font_size=9, max_rows_per_page=60)


def plot_coverage_reliability(
    *,
    time: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per1: np.ndarray,
    per99: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    draws: Optional[np.ndarray],
    trace=None,
    label: str,
    out_dir: Path,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Reliability diagram: nominal coverage vs empirical coverage."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)
    if obs_mask_cal is None:
        obs_mask_cal = np.isfinite(obs)
    else:
        obs_mask_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
    if obs_mask_val is None:
        obs_mask_val = np.zeros_like(obs_mask_cal, dtype=bool)
    else:
        obs_mask_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    pp = _predictive_params_at_obs(time=time, per10=per10, per50=per50, per90=per90, per1=per1, per99=per99, obs_time=obs_time, trace=trace)
    mu, sig, use_t, nu = pp["mu"], pp["sig"], pp["use_studentt"], pp["nu"]
    base = np.isfinite(obs) & np.isfinite(mu) & np.isfinite(sig)
    m_cal = base & obs_mask_cal
    m_val = base & obs_mask_val

    levs = np.asarray(_DEFAULT_COVERAGE_LEVELS, dtype=float)
    cov_cal, cov_val = [], []
    for lev in levs:
        qlo, qhi = _interval_bounds(mu, sig, float(lev), bool(use_t), nu)
        cov_cal.append(float(np.mean((obs[m_cal] >= qlo[m_cal]) & (obs[m_cal] <= qhi[m_cal]))) if m_cal.any() else np.nan)
        cov_val.append(float(np.mean((obs[m_val] >= qlo[m_val]) & (obs[m_val] <= qhi[m_val]))) if m_val.any() else np.nan)
    cov_cal = np.asarray(cov_cal, dtype=float)
    cov_val = np.asarray(cov_val, dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.fill_between([0.45, 1.0], [0.40, 0.95], [0.50, 1.05], color=_THEME["header"], alpha=0.65, zorder=0)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=_THEME["slate"], alpha=0.9, label="Ideal")
    if np.any(np.isfinite(cov_cal)):
        ax.plot(levs, cov_cal, marker="o", ms=5.5, lw=1.8, color=_THEME["teal"], label="Calibration")
    if np.any(np.isfinite(cov_val)):
        ax.plot(levs, cov_val, marker="o", ms=5.5, lw=1.8, color=_THEME["coral"], label="Validation")

    _apply_axes_style(ax, grid_axis="both")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Coverage reliability — {label}")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"coverage_reliability_{label}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def plot_pit_histogram(
    *,
    time: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per1: np.ndarray,
    per99: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    draws: Optional[np.ndarray],
    trace=None,
    label: str,
    out_dir: Path,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """PIT histogram (cal vs val)."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)
    if obs_mask_cal is None:
        obs_mask_cal = np.isfinite(obs)
    else:
        obs_mask_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
    if obs_mask_val is None:
        obs_mask_val = np.zeros_like(obs_mask_cal, dtype=bool)
    else:
        obs_mask_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    pp = _predictive_params_at_obs(time=time, per10=per10, per50=per50, per90=per90, per1=per1, per99=per99, obs_time=obs_time, trace=trace)
    mu, sig, use_t, nu = pp["mu"], pp["sig"], pp["use_studentt"], pp["nu"]
    base = np.isfinite(obs) & np.isfinite(mu) & np.isfinite(sig)
    m_cal = base & obs_mask_cal
    m_val = base & obs_mask_val

    pit = None
    if draws is not None:
        try:
            d = np.asarray(draws, dtype=float)
            if d.ndim == 2 and d.shape[1] == np.asarray(time).shape[0]:
                idx = _nearest_time_index(time, obs_time)
                d_obs = d[:, idx]
                pit = (np.sum(d_obs <= obs[None, :], axis=0) + 1.0) / (d_obs.shape[0] + 1.0)
        except Exception:
            pit = None
    if pit is None:
        pit = _cdf_at_obs(obs, mu, sig, use_t, nu)

    bins = np.linspace(0.0, 1.0, 11)
    bin_width = bins[1] - bins[0]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    def _draw(mask, color, label_txt):
        if not np.any(mask):
            return
        vals = pit[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        ax.hist(vals, bins=bins, density=True, alpha=0.45, color=color, edgecolor="white", linewidth=0.9, label=label_txt)
        n = vals.size
        p0 = 1.0 / (len(bins) - 1)
        se = np.sqrt(max(p0 * (1.0 - p0), 0.0) / max(n, 1)) / bin_width
        ax.fill_between([0.0, 1.0], [1.0 - 1.96 * se, 1.0 - 1.96 * se], [1.0 + 1.96 * se, 1.0 + 1.96 * se], color=color, alpha=0.10)

    _draw(m_cal, _THEME["teal"], "Calibration")
    _draw(m_val, _THEME["coral"], "Validation")
    ax.axhline(1.0, ls="--", lw=1.2, color=_THEME["slate"], alpha=0.95, label="Uniform")
    _apply_axes_style(ax, grid_axis="y")
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT histogram — {label}")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_dir / f"pit_hist_{label}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def plot_standardized_residuals_diagnostics(
    *,
    time: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per1: np.ndarray,
    per99: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    draws: Optional[np.ndarray],
    trace=None,
    label: str,
    out_dir: Path,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Standardized residuals (z) + histogram + QQ plot."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)
    if obs_mask_cal is None:
        obs_mask_cal = np.isfinite(obs)
    else:
        obs_mask_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
    if obs_mask_val is None:
        obs_mask_val = np.zeros_like(obs_mask_cal, dtype=bool)
    else:
        obs_mask_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    pp = _predictive_params_at_obs(time=time, per10=per10, per50=per50, per90=per90, per1=per1, per99=per99, obs_time=obs_time, trace=trace)
    mu, sig = pp["mu"], np.maximum(pp["sig"], 1e-9)
    use_t, nu = bool(pp["use_studentt"]), pp["nu"]
    base = np.isfinite(obs) & np.isfinite(mu) & np.isfinite(sig)
    m_cal = base & obs_mask_cal
    m_val = base & obs_mask_val
    z = (obs - mu) / sig

    def _one(mask: np.ndarray, suffix: str, color: str):
        if mask.sum() < 5:
            return
        zz = z[mask]
        tt = obs_time[mask]
        fig = plt.figure(figsize=(12.6, 4.5))
        gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.0, 1.0], wspace=0.28)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])

        ax0.plot(tt, zz, lw=1.0, color=color, alpha=0.9)
        ax0.scatter(tt, zz, s=12, alpha=0.55, color=color)
        ax0.axhline(0.0, lw=1.2, color=_THEME["slate"], ls="--")
        for v in (1.0, 2.0):
            ax0.axhline(v, lw=0.8, color=_THEME["gold"], alpha=0.55, ls=":")
            ax0.axhline(-v, lw=0.8, color=_THEME["gold"], alpha=0.55, ls=":")
        if split_date is not None:
            try:
                ax0.axvline(np.datetime64(split_date), ls="--", lw=1.0, c=_THEME["slate"], alpha=0.7)
            except Exception:
                pass
        ax0.set_title(f"Standardized residuals ({suffix})")
        ax0.set_ylabel("z")
        _apply_axes_style(ax0, grid_axis="both")
        if np.issubdtype(obs_time.dtype, np.datetime64):
            ax0.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
            ax0.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax0.xaxis.get_major_locator()))

        ax1.hist(zz, bins=min(28, max(12, int(np.sqrt(zz.size)))), density=True, alpha=0.60, color=color, edgecolor="white", linewidth=0.8)
        xg = np.linspace(np.nanpercentile(zz, 1), np.nanpercentile(zz, 99), 240)
        if use_t and (nu is not None):
            ax1.plot(xg, _student_t.pdf(xg, float(nu)), lw=1.5, color=_THEME["navy"])
            ax1.set_title(f"Distribution vs t (ν≈{float(nu):.2g})")
        else:
            ax1.plot(xg, _norm.pdf(xg), lw=1.5, color=_THEME["navy"])
            ax1.set_title("Distribution vs N(0,1)")
        _apply_axes_style(ax1, grid_axis="y")

        qs = np.linspace(0.01, 0.99, 100)
        emp = np.nanquantile(zz, qs)
        if use_t and (nu is not None):
            theo = _student_t.ppf(qs, float(nu))
            title = "QQ vs Student-t"
        else:
            theo = _norm.ppf(qs)
            title = "QQ vs Normal"
        ax2.scatter(theo, emp, s=14, alpha=0.75, color=_THEME["purple"])
        mn = min(np.nanmin(theo), np.nanmin(emp))
        mx = max(np.nanmax(theo), np.nanmax(emp))
        ax2.plot([mn, mx], [mn, mx], ls="--", lw=1.1, c=_THEME["slate"], alpha=0.9)
        ax2.set_title(title)
        ax2.set_xlabel("Theoretical")
        ax2.set_ylabel("Empirical")
        _apply_axes_style(ax2, grid_axis="both")

        txt = f"mean={np.nanmean(zz):.2f}\nstd={np.nanstd(zz):.2f}\n|z|≤1: {np.mean(np.abs(zz)<=1):.2f}\n|z|≤2: {np.mean(np.abs(zz)<=2):.2f}"
        ax2.text(0.05, 0.95, txt, transform=ax2.transAxes, va="top", ha="left", fontsize=8.5, color=_THEME["ink"], bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=_THEME["grid"], alpha=0.95))

        fig.suptitle(f"Bayesian residual diagnostics — {label} ({suffix})", y=1.02, fontsize=13, fontweight="semibold", color=_THEME["ink"])
        fig.tight_layout()
        fig.savefig(out_dir / f"z_residuals_{label}_{suffix}.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    _one(m_cal, "cal", _THEME["teal"])
    if m_val.any():
        _one(m_val, "val", _THEME["coral"])



def plot_residual_percentile_ribbons(
    *,
    time: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per1: np.ndarray,
    per99: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    label: str,
    out_dir: Path,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Residual percentiles at obs times (q - y_obs) with 10–90 and 1–99 ribbons."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)

    if obs_mask_cal is None:
        obs_mask_cal = np.isfinite(obs)
    else:
        obs_mask_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
    if obs_mask_val is None:
        obs_mask_val = np.zeros_like(obs_mask_cal, dtype=bool)
    else:
        obs_mask_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    q10 = _interp_to_obs(time, per10, obs_time)
    q50 = _interp_to_obs(time, per50, obs_time)
    q90 = _interp_to_obs(time, per90, obs_time)
    q01 = _interp_to_obs(time, per1, obs_time)
    q99 = _interp_to_obs(time, per99, obs_time)

    m = np.isfinite(obs) & np.isfinite(q50)
    if not m.any():
        return

    r50 = q50 - obs
    r10 = q10 - obs
    r90 = q90 - obs
    r01 = q01 - obs
    r99 = q99 - obs

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.fill_between(obs_time[m], r01[m], r99[m], alpha=0.18, label="1–99%", linewidth=0)
    ax.fill_between(obs_time[m], r10[m], r90[m], alpha=0.28, label="10–90%", linewidth=0)
    ax.plot(obs_time[m], r50[m], lw=1.2, label="median")
    ax.axhline(0.0, lw=1.0, c='k', alpha=0.6)

    if split_date is not None:
        try:
            ax.axvline(np.datetime64(split_date), ls='--', lw=1.0, c='k', alpha=0.6)
        except Exception:
            pass

    ax.set_title(f"Residual percentiles at obs — {label}")
    ax.set_ylabel("pred - obs")
    ax.grid(True, alpha=0.2)
    ax.legend(ncol=3, fontsize=9)

    if np.issubdtype(obs_time.dtype, np.datetime64):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    fig.tight_layout()
    fig.savefig(out_dir / f"residual_ribbons_{label}.png", dpi=200)
    plt.close(fig)


def plot_sharpness_over_time(
    *,
    time: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per1: np.ndarray,
    per99: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    label: str,
    out_dir: Path,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Sharpness (interval width) through time at obs points."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)
    q10 = _interp_to_obs(time, per10, obs_time)
    q90 = _interp_to_obs(time, per90, obs_time)
    q01 = _interp_to_obs(time, per1, obs_time)
    q99 = _interp_to_obs(time, per99, obs_time)

    m = np.isfinite(obs) & np.isfinite(q10) & np.isfinite(q90) & np.isfinite(q01) & np.isfinite(q99)
    if not m.any():
        return

    w90 = q90 - q10
    w99 = q99 - q01

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(obs_time[m], w90[m], lw=1.2, label="width 80% (10–90)")
    ax.plot(obs_time[m], w99[m], lw=1.2, label="width 98% (1–99)")

    if split_date is not None:
        try:
            ax.axvline(np.datetime64(split_date), ls='--', lw=1.0, c='k', alpha=0.6)
        except Exception:
            pass

    ax.set_title(f"Predictive sharpness at obs — {label}")
    ax.set_ylabel("interval width")
    ax.grid(True, alpha=0.2)
    ax.legend(ncol=2, fontsize=9)

    if np.issubdtype(obs_time.dtype, np.datetime64):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    fig.tight_layout()
    fig.savefig(out_dir / f"sharpness_{label}.png", dpi=200)
    plt.close(fig)


def plot_gate_entropy(
    *,
    time: np.ndarray,
    weights: np.ndarray,
    model_names: Sequence[str],
    label: str,
    out_dir: Path,
    obs_time: np.ndarray,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Gate entropy through time (normalized by log K) and at obs points."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        return

    if w.shape[0] == len(model_names):
        Wt = w.T
    elif w.shape[1] == len(model_names):
        Wt = w
    else:
        Wt = w

    eps = 1e-12
    H = -np.sum(Wt * np.log(np.maximum(Wt, eps)), axis=1)
    Hn = H / np.log(max(2, Wt.shape[1]))

    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.plot(np.asarray(time), Hn, lw=1.4)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Gate entropy (0=confident, 1=uniform) — {label}")
    ax.set_ylabel("entropy / log K")
    ax.grid(True, alpha=0.2)

    if split_date is not None:
        try:
            ax.axvline(np.datetime64(split_date), ls='--', lw=1.0, c='k', alpha=0.6)
        except Exception:
            pass

    if np.issubdtype(np.asarray(time).dtype, np.datetime64):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    fig.tight_layout()
    fig.savefig(out_dir / f"gate_entropy_{label}.png", dpi=200)
    plt.close(fig)


# ---------------------------
# Multi-model comparison helpers (MoE)
# ---------------------------


def _compute_metrics_split_from_envelope(
    *,
    time: np.ndarray,
    per5: np.ndarray,
    per50: np.ndarray,
    per95: np.ndarray,
    mini: np.ndarray,
    maxi: np.ndarray,
    draws: Optional[np.ndarray] = None,
    obs_time: np.ndarray,
    obs: np.ndarray,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
) -> tuple[dict, dict, dict]:
    """Compute (cal, val, counts) metrics for a predictive envelope."""

    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)

    if obs_mask_cal is None:
        obs_mask_cal = np.isfinite(obs)
    else:
        obs_mask_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
    if obs_mask_val is None:
        obs_mask_val = np.zeros_like(obs_mask_cal, dtype=bool)
    else:
        obs_mask_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    # Map envelopes to observation times
    per50_o = _interp_to_obs(time, per50, obs_time)
    per5_o = _interp_to_obs(time, per5, obs_time)
    per95_o = _interp_to_obs(time, per95, obs_time)
    mini_o = _interp_to_obs(time, mini, obs_time)
    maxi_o = _interp_to_obs(time, maxi, obs_time)

    base = np.isfinite(obs) & np.isfinite(per50_o) & np.isfinite(per5_o) & np.isfinite(per95_o)
    m_cal = base & obs_mask_cal
    m_val = base & obs_mask_val

    def _compute(mask: np.ndarray) -> dict:
        if int(mask.sum()) < 3:
            return {
                "rmse_median": float("nan"),
                "mae_median": float("nan"),
                "bias_median": float("nan"),
                "pearson_r": float("nan"),
                "nse_median": float("nan"),
                "kge_median": float("nan"),
                "coverage_95": float("nan"),
                "coverage_min_max": float("nan"),
                "mean_width_90": float("nan"),
                "n_used": int(mask.sum()),
            }
        o = obs[mask]
        p50 = per50_o[mask]
        p5 = per5_o[mask]
        p95 = per95_o[mask]
        mi = mini_o[mask]
        ma = maxi_o[mask]
        
        return {
            "rmse_median": _rmse(o, p50),
            "mae_median": _mae(o, p50),
            "bias_median": _bias(p50, o),
            "pearson_r": _pearsonr(o, p50),
            "nse_median": _nse(o, p50),
            "kge_median": _kge(o, p50),
            "coverage_p5_p95": _coverage(o, p5, p95),
            "coverage_min_max": _coverage(o, mi, ma),
            "mean_width_90": _mean_width(p5, p95),
            "n_used": int(mask.sum()),
        }

    met_cal = _compute(m_cal)
    met_val = _compute(m_val)
    counts = {
        "n_obs_total": int(np.isfinite(obs).sum()),
        "n_cal_total": int(np.asarray(obs_mask_cal, dtype=bool).sum()),
        "n_val_total": int(np.asarray(obs_mask_val, dtype=bool).sum()),
        "n_used_cal": int(met_cal.get("n_used", 0)),
        "n_used_val": int(met_val.get("n_used", 0)),
    }
    return met_cal, met_val, counts


def save_compare_metrics(
    *,
    time: np.ndarray,
    models: dict[str, dict[str, np.ndarray]],
    obs_time: np.ndarray,
    obs: np.ndarray,
    out_dir: Path,
    label: str,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Save a comparison metrics table for multiple models (cal vs val)."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    metric_order = [
        "rmse_median",
        "mae_median",
        "bias_median",
        "pearson_r",
        "nse_median",
        "kge_median",
        "elpd_sum",
        "elpd_mean",
        "crps_mean",
        "pit_mean",
        "z_mean",
        "z_std",
        "coverage_p5_p95",
        "coverage_min_max",
        "mean_width_90",
        "coverage_nom_50",
        "coverage_nom_80",
        "coverage_nom_90",
        "coverage_nom_95",
        "coverage_nom_98",
        "sharpness_nom_50",
        "sharpness_nom_80",
        "sharpness_nom_90",
        "sharpness_nom_95",
        "sharpness_nom_98",
        "n_used",
    ]

    # rows: one per model
    rows = []
    for name, env in models.items():
        met_cal, met_val, counts = _compute_metrics_split_from_envelope(
            time=time,
            per5=env["per5"],
            per50=env["per50"],
            per95=env["per95"],
            mini=env["mini"],
            maxi=env["maxi"],
            obs_time=obs_time,
            obs=obs,
            obs_mask_cal=obs_mask_cal,
            obs_mask_val=obs_mask_val,
        )

        # --- probabilistic metrics (quantile-based parametric approximation) ---
        try:
            pp = _predictive_params_at_obs(
                time=time,
                per10=env.get("per10"),
                per50=env.get("per50"),
                per90=env.get("per90"),
                per1=env.get("per1"),
                per99=env.get("per99"),
                obs_time=obs_time,
                trace=None,
            )
            mu_o = pp["mu"]
            sig_o = pp["sig"]
            use_t = bool(pp["use_studentt"])
            nu = pp["nu"]
            lpd = _logpdf_at_obs(np.asarray(obs, dtype=float), mu_o, sig_o, use_t, nu)
            pit = _cdf_at_obs(np.asarray(obs, dtype=float), mu_o, sig_o, use_t, nu)
            z = (np.asarray(obs, dtype=float) - mu_o) / np.maximum(sig_o, 1e-9)

            # masks consistent with envelope metrics
            if obs_mask_cal is None:
                om_cal = np.isfinite(obs)
            else:
                om_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
            if obs_mask_val is None:
                om_val = np.zeros_like(om_cal, dtype=bool)
            else:
                om_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)
            base = np.isfinite(mu_o) & np.isfinite(sig_o) & np.isfinite(obs)
            m_cal = base & om_cal
            m_val = base & om_val

            # CRPS
            if use_t:
                crps = _crps_studentt_mc(np.asarray(obs, dtype=float), mu_o, sig_o, float(nu), n_mc=250, seed=123)
            else:
                crps = _crps_normal(np.asarray(obs, dtype=float), mu_o, sig_o)

            met_cal.update({
                "elpd_sum": float(np.nansum(lpd[m_cal])) if m_cal.any() else float('nan'),
                "elpd_mean": float(np.nanmean(lpd[m_cal])) if m_cal.any() else float('nan'),
                "crps_mean": float(np.nanmean(crps[m_cal])) if m_cal.any() else float('nan'),
                "pit_mean": float(np.nanmean(pit[m_cal])) if m_cal.any() else float('nan'),
                "z_mean": float(np.nanmean(z[m_cal])) if m_cal.any() else float('nan'),
                "z_std": float(np.nanstd(z[m_cal])) if m_cal.any() else float('nan'),
            })
            met_val.update({
                "elpd_sum": float(np.nansum(lpd[m_val])) if m_val.any() else float('nan'),
                "elpd_mean": float(np.nanmean(lpd[m_val])) if m_val.any() else float('nan'),
                "crps_mean": float(np.nanmean(crps[m_val])) if m_val.any() else float('nan'),
                "pit_mean": float(np.nanmean(pit[m_val])) if m_val.any() else float('nan'),
                "z_mean": float(np.nanmean(z[m_val])) if m_val.any() else float('nan'),
                "z_std": float(np.nanstd(z[m_val])) if m_val.any() else float('nan'),
            })

            # coverages/sharpness at levels
            levels = (0.50, 0.80, 0.90, 0.95, 0.98)
            for lev in levels:
                a = 0.5 * (1.0 - lev)
                lo_p = a
                hi_p = 1.0 - a
                if use_t:
                    qlo = mu_o + sig_o * _student_t.ppf(lo_p, float(nu))
                    qhi = mu_o + sig_o * _student_t.ppf(hi_p, float(nu))
                else:
                    qlo = mu_o + sig_o * _norm.ppf(lo_p)
                    qhi = mu_o + sig_o * _norm.ppf(hi_p)
                met_cal[f"coverage_nom_{int(lev*100):02d}"] = float(np.mean((obs[m_cal] >= qlo[m_cal]) & (obs[m_cal] <= qhi[m_cal]))) if m_cal.any() else float('nan')
                met_val[f"coverage_nom_{int(lev*100):02d}"] = float(np.mean((obs[m_val] >= qlo[m_val]) & (obs[m_val] <= qhi[m_val]))) if m_val.any() else float('nan')
                met_cal[f"sharpness_nom_{int(lev*100):02d}"] = float(np.nanmean(qhi[m_cal] - qlo[m_cal])) if m_cal.any() else float('nan')
                met_val[f"sharpness_nom_{int(lev*100):02d}"] = float(np.nanmean(qhi[m_val] - qlo[m_val])) if m_val.any() else float('nan')


        except Exception:
            pass

        def _fmt(v):
            try:
                vf = float(v)
                return "" if not np.isfinite(vf) else f"{vf:.6g}"
            except Exception:
                return str(v)

        row = {
            "model": name,
            **{f"cal_{k}": met_cal.get(k, np.nan) for k in metric_order},
            **{f"val_{k}": met_val.get(k, np.nan) for k in metric_order},
            **counts,
        }
        rows.append(row)

    # Save CSV + PNG table
    out_csv = out_dir / f"compare_metrics_{label}.csv"
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        # ΔELPD on validation (relative to best)
        if 'val_elpd_sum' in df.columns:
            try:
                best = float(df['val_elpd_sum'].max())
                df['delta_val_elpd_sum'] = df['val_elpd_sum'] - best
            except Exception:
                pass
        if split_date is not None:
            df.attrs["split_date"] = str(np.datetime64(split_date))
        df.to_csv(out_csv, index=False)

        # compact table view (a few key metrics)
        show_cols = [
            "model",
            "val_rmse_median",
            "val_kge_median",
            "val_elpd_sum",
            "delta_val_elpd_sum",
            "val_crps_mean",
            "val_coverage_95",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        df_show = df[show_cols].copy()
        table_rows = [df_show.columns.tolist()] + [[str(x) for x in r] for r in df_show.values.tolist()]

        title = f"Model comparison metrics: {label}"
        if split_date is not None:
            title += f"\nSplit @ {np.datetime64(split_date)}"
        _save_table_png(
            rows=table_rows[1:],
            col_labels=table_rows[0],
            out_png=out_dir / f"compare_metrics_{label}.png",
            title=title,
            font_size=10,
            max_rows_per_page=80,
        )
    except Exception:
        # minimal CSV fallback
        with out_csv.open("w", encoding="utf-8") as f:
            if rows:
                cols = list(rows[0].keys())
                f.write(",".join(cols) + "\n")
                for r in rows:
                    f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def plot_compare_medians_split(
    *,
    time: np.ndarray,
    models: dict[str, dict[str, np.ndarray]],
    obs_time: np.ndarray,
    obs: np.ndarray,
    out_dir: Path,
    label: str,
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Plot median predictions for multiple models on the same axis."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    t = np.asarray(time)
    fig, ax = plt.subplots(figsize=(12, 4))

    for name, env in models.items():
        ax.plot(t, np.asarray(env["per50"], dtype=float), lw=1.6, label=name)

    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)
    if obs_mask_cal is None:
        m_cal = np.isfinite(obs)
    else:
        m_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs)
    if obs_mask_val is None:
        m_val = np.zeros_like(m_cal, dtype=bool)
    else:
        m_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs)

    ax.scatter(obs_time[m_cal], obs[m_cal], s=14, c="k", alpha=0.9, label="Obs (cal)", marker="s", zorder=3)
    if m_val.any():
        ax.scatter(obs_time[m_val], obs[m_val], s=18, c="r", alpha=0.9, label="Obs (val)", zorder=3)

    if split_date is not None:
        try:
            ax.axvline(np.datetime64(split_date), ls="--", lw=1.0, c="k", alpha=0.6)
        except Exception:
            pass

    ax.set_title(f"Median comparison: {label}")
    ax.set_xlabel("time")
    ax.set_ylabel("shoreline")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_medians_{label}.png", dpi=200)
    plt.close(fig)


def plot_gate_weights_heatmap(
    *,
    time: np.ndarray,
    weights: np.ndarray,
    model_names: Sequence[str],
    out_dir: Path,
    label: str,
    split_date: Optional[np.datetime64] = None,
    # --- new options ---
    overall_mode: str = "mean",          # "mean" (time-average) or "sum"
    overall_period: str = "all",         # "all" or "post_split"
    bars_width_ratio: float = 0.28,      # fraction of width reserved for bars
) -> None:
    """Heatmap of expert weights through time + right panel with overall contribution (%)."""
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)

    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        return

    # ensure (K, T)
    if w.shape[0] == len(model_names):
        W = w
    elif w.shape[1] == len(model_names):
        W = w.T
    else:
        # best effort: assume (T, K)
        W = w.T

    t = np.asarray(time)

    # --- decide which timesteps count toward the "overall contribution" ---
    mask = np.ones(W.shape[1], dtype=bool)
    if overall_period == "post_split" and split_date is not None and np.issubdtype(t.dtype, np.datetime64):
        try:
            sd = np.datetime64(split_date)
            mask = t >= sd
        except Exception:
            mask = np.ones(W.shape[1], dtype=bool)

    Wm = W[:, mask] if mask.size == W.shape[1] else W

    # --- overall contribution (percent) ---
    if Wm.size == 0:
        overall = np.full(W.shape[0], np.nan, dtype=float)
    else:
        if overall_mode == "sum":
            overall = np.nansum(Wm, axis=1)
            denom = np.nansum(overall)
            overall = overall / denom if denom > 0 else overall
        else:
            overall = np.nanmean(Wm, axis=1)
            denom = np.nansum(overall)
            overall = overall / denom if denom > 0 else overall

    overall_pct = 100.0 * overall

    # --- x-axis mapping ---
    if np.issubdtype(t.dtype, np.datetime64):
        t_num = mdates.date2num(t.astype("datetime64[ns]").astype("datetime64[ms]").astype(object))
        extent = [float(t_num[0]), float(t_num[-1]), -0.5, W.shape[0] - 0.5]
        x_is_date = True
        split_x = None
        if split_date is not None:
            try:
                sd = np.datetime64(split_date)
                split_x = mdates.date2num(sd.astype("datetime64[ms]").astype(object))
            except Exception:
                split_x = None
    else:
        extent = [0.0, float(W.shape[1] - 1), -0.5, W.shape[0] - 0.5]
        x_is_date = False
        split_x = None

    # --- layout: heatmap + small bars axis ---
    fig = plt.figure(figsize=(12, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, bars_width_ratio], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1], sharey=ax)

    im = ax.imshow(
        W,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("weight")

    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(list(model_names))
    ax.set_title(f"MoE gate weights: {label}")
    ax.set_xlabel("time")
    ax.set_ylabel("expert")

    if x_is_date:
        ax.xaxis_date()
        loc = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        if split_x is not None:
            ax.axvline(split_x, ls="--", lw=1.0, c="k", alpha=0.7)

    # --- right panel: overall contribution bars (percent) ---
    y = np.arange(len(model_names))
    vals = np.nan_to_num(overall_pct, nan=0.0)

    axb.barh(y, vals, height=0.8)
    axb.set_xlim(0.0, max(1.0, float(np.nanmax(vals) * 1.15)))
    axb.set_xlabel("%")
    axb.set_title("Overall")

    # tidy: keep y labels only on left
    axb.tick_params(axis="y", left=False, labelleft=False)
    axb.grid(axis="x", alpha=0.25, lw=0.6)

    # annotate % values
    for yi, v in zip(y, vals):
        axb.text(v + 0.5, yi, f"{v:.1f}%", va="center", ha="left", fontsize=9)

    # If we computed post-split contributions, add a tiny note
    if overall_period == "post_split" and split_date is not None:
        axb.text(
            0.0, 1.02, "post-split",
            transform=axb.transAxes, ha="left", va="bottom", fontsize=9, alpha=0.8
        )

    fig.tight_layout()
    fig.savefig(out_dir / f"moe_gate_weights_heatmap_{label}.png", dpi=200)
    plt.close(fig)




def _softmax_with_optional_floor(logits: np.ndarray, weight_floor: float = 0.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        logits = logits[None, :]
    logits = logits - np.nanmax(logits, axis=1, keepdims=True)
    ez = np.exp(logits)
    denom = np.nansum(ez, axis=1, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    w = ez / denom
    eps_floor = float(weight_floor or 0.0)
    if eps_floor > 0.0:
        K = w.shape[1]
        w = (1.0 - eps_floor) * w + (eps_floor / float(K))
    return w


def _flatten_categorical_gate_posterior(idata):
    if idata is None or getattr(idata, "posterior", None) is None:
        return None
    post = idata.posterior
    if ("mu_class" not in post) or ("gamma" not in post):
        return None
    mu_class = np.asarray(post["mu_class"].values, dtype=float)
    gamma = np.asarray(post["gamma"].values, dtype=float)
    mu_class_f = mu_class.reshape(mu_class.shape[0] * mu_class.shape[1], mu_class.shape[2])
    gamma_f = gamma.reshape(gamma.shape[0] * gamma.shape[1], gamma.shape[2], gamma.shape[3])
    out = {"mu_class": mu_class_f, "gamma": gamma_f}
    if "rho_dwell" in post:
        rho = np.asarray(post["rho_dwell"].values, dtype=float)
        out["rho_dwell"] = rho.reshape(rho.shape[0] * rho.shape[1], rho.shape[2])
    if "sigma_class" in post:
        sig = np.asarray(post["sigma_class"].values, dtype=float)
        out["sigma_class"] = sig.reshape(sig.shape[0] * sig.shape[1], sig.shape[2])
    return out


def summarize_categorical_bmu_gate(
    *,
    weights: np.ndarray,
    bmu_full_idx: np.ndarray,
    bmu_obs_idx: np.ndarray | None,
    bmu_levels: Sequence[object],
    model_names: Sequence[str],
    idata=None,
    weight_floor: float = 0.0,
) -> list[dict[str, object]]:
    """Summarize categorical-BMU gate behavior by category.

    Returns one dictionary per BMU plus optional ``__UNKNOWN__`` and ``__GLOBAL__``
    rows. The summary is designed for diagnostics tables and heatmaps.
    """
    W = np.asarray(weights, dtype=float)
    bmu_full_idx = np.asarray(bmu_full_idx, dtype=int)
    bmu_obs_idx = np.asarray([] if bmu_obs_idx is None else bmu_obs_idx, dtype=int)
    K = W.shape[1]
    levels = list(bmu_levels)
    n_time_total = int(W.shape[0])
    n_obs_total = int(bmu_obs_idx.shape[0])

    post = _flatten_categorical_gate_posterior(idata)
    rows: list[dict[str, object]] = []

    def _entry_summary_for_category(cat_idx: int | None):
        if post is None:
            return None
        mu_class = post["mu_class"]
        gamma = post["gamma"]
        S = mu_class.shape[0]
        logits = np.zeros((S, K), dtype=float)
        if cat_idx is None:
            logits[:, : K - 1] = mu_class
        else:
            logits[:, : K - 1] = gamma[:, :, int(cat_idx)]
        w = _softmax_with_optional_floor(logits, weight_floor=weight_floor)
        return {
            "mean": np.nanmean(w, axis=0),
            "p05": np.nanpercentile(w, 5.0, axis=0),
            "p50": np.nanpercentile(w, 50.0, axis=0),
            "p95": np.nanpercentile(w, 95.0, axis=0),
        }

    for i, lvl in enumerate(levels):
        mask_full = bmu_full_idx == i
        mask_obs = bmu_obs_idx == i if bmu_obs_idx.size else np.zeros(0, dtype=bool)
        n_time = int(np.sum(mask_full))
        n_obs = int(np.sum(mask_obs))
        if n_time > 0:
            w_mean_time = np.nanmean(W[mask_full, :], axis=0)
            w_std_time = np.nanstd(W[mask_full, :], axis=0)
        else:
            w_mean_time = np.full(K, np.nan, dtype=float)
            w_std_time = np.full(K, np.nan, dtype=float)
        entry = _entry_summary_for_category(i)
        base = {
            "row_type": "bmu",
            "bmu": lvl,
            "bmu_index": int(i),
            "n_time": n_time,
            "n_obs": n_obs,
            "frac_time_pct": 100.0 * n_time / max(n_time_total, 1),
            "frac_obs_pct": 100.0 * n_obs / max(n_obs_total, 1),
        }
        dominant_source = entry["mean"] if entry is not None else w_mean_time
        dom_idx = int(np.nanargmax(dominant_source)) if np.any(np.isfinite(dominant_source)) else -1
        base["dominant_expert"] = model_names[dom_idx] if dom_idx >= 0 else ""
        base["dominant_weight_pct"] = float(100.0 * dominant_source[dom_idx]) if dom_idx >= 0 else np.nan
        for k, name in enumerate(model_names):
            safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or f"expert_{k}"
            base[f"w_time_mean_{safe}"] = float(w_mean_time[k]) if np.isfinite(w_mean_time[k]) else np.nan
            base[f"w_time_std_{safe}"] = float(w_std_time[k]) if np.isfinite(w_std_time[k]) else np.nan
            if entry is not None:
                base[f"w_entry_mean_{safe}"] = float(entry["mean"][k])
                base[f"w_entry_p05_{safe}"] = float(entry["p05"][k])
                base[f"w_entry_p50_{safe}"] = float(entry["p50"][k])
                base[f"w_entry_p95_{safe}"] = float(entry["p95"][k])
        rows.append(base)

    # Unknown categories on the full grid are informative because they explain
    # why the time heatmap may collapse to global weights.
    unk_mask_full = bmu_full_idx < 0
    unk_mask_obs = bmu_obs_idx < 0 if bmu_obs_idx.size else np.zeros(0, dtype=bool)
    if np.any(unk_mask_full) or np.any(unk_mask_obs):
        n_time = int(np.sum(unk_mask_full))
        n_obs = int(np.sum(unk_mask_obs))
        if n_time > 0:
            w_mean_time = np.nanmean(W[unk_mask_full, :], axis=0)
            w_std_time = np.nanstd(W[unk_mask_full, :], axis=0)
        else:
            w_mean_time = np.full(K, np.nan, dtype=float)
            w_std_time = np.full(K, np.nan, dtype=float)
        entry = _entry_summary_for_category(None)
        base = {
            "row_type": "unknown",
            "bmu": "__UNKNOWN__",
            "bmu_index": -1,
            "n_time": n_time,
            "n_obs": n_obs,
            "frac_time_pct": 100.0 * n_time / max(n_time_total, 1),
            "frac_obs_pct": 100.0 * n_obs / max(n_obs_total, 1),
        }
        dominant_source = entry["mean"] if entry is not None else w_mean_time
        dom_idx = int(np.nanargmax(dominant_source)) if np.any(np.isfinite(dominant_source)) else -1
        base["dominant_expert"] = model_names[dom_idx] if dom_idx >= 0 else ""
        base["dominant_weight_pct"] = float(100.0 * dominant_source[dom_idx]) if dom_idx >= 0 else np.nan
        for k, name in enumerate(model_names):
            safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or f"expert_{k}"
            base[f"w_time_mean_{safe}"] = float(w_mean_time[k]) if np.isfinite(w_mean_time[k]) else np.nan
            base[f"w_time_std_{safe}"] = float(w_std_time[k]) if np.isfinite(w_std_time[k]) else np.nan
            if entry is not None:
                base[f"w_entry_mean_{safe}"] = float(entry["mean"][k])
                base[f"w_entry_p05_{safe}"] = float(entry["p05"][k])
                base[f"w_entry_p50_{safe}"] = float(entry["p50"][k])
                base[f"w_entry_p95_{safe}"] = float(entry["p95"][k])
        rows.append(base)

    # Global pooled row: useful baseline even when every BMU is known.
    global_entry = _entry_summary_for_category(None)
    if global_entry is not None:
        base = {
            "row_type": "global",
            "bmu": "__GLOBAL__",
            "bmu_index": -999,
            "n_time": n_time_total,
            "n_obs": n_obs_total,
            "frac_time_pct": 100.0,
            "frac_obs_pct": 100.0,
        }
        dom_idx = int(np.nanargmax(global_entry["mean"]))
        base["dominant_expert"] = model_names[dom_idx]
        base["dominant_weight_pct"] = float(100.0 * global_entry["mean"][dom_idx])
        for k, name in enumerate(model_names):
            safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or f"expert_{k}"
            base[f"w_entry_mean_{safe}"] = float(global_entry["mean"][k])
            base[f"w_entry_p05_{safe}"] = float(global_entry["p05"][k])
            base[f"w_entry_p50_{safe}"] = float(global_entry["p50"][k])
            base[f"w_entry_p95_{safe}"] = float(global_entry["p95"][k])
        rows.append(base)

    return rows


def save_categorical_bmu_gate_summary(
    *,
    weights: np.ndarray,
    bmu_full_idx: np.ndarray,
    bmu_obs_idx: np.ndarray | None,
    bmu_levels: Sequence[object],
    model_names: Sequence[str],
    out_dir: Path,
    label: str,
    idata=None,
    weight_floor: float = 0.0,
) -> list[dict[str, object]]:
    rows = summarize_categorical_bmu_gate(
        weights=weights,
        bmu_full_idx=bmu_full_idx,
        bmu_obs_idx=bmu_obs_idx,
        bmu_levels=bmu_levels,
        model_names=model_names,
        idata=idata,
        weight_floor=weight_floor,
    )
    if not rows:
        return rows
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)
    import csv
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(out_dir / f"categorical_bmu_gate_summary_{label}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_categorical_bmu_gate_heatmap(
    *,
    weights: np.ndarray,
    bmu_full_idx: np.ndarray,
    bmu_obs_idx: np.ndarray | None,
    bmu_levels: Sequence[object],
    model_names: Sequence[str],
    out_dir: Path,
    label: str,
    idata=None,
    weight_floor: float = 0.0,
    sort_by: str = "n_obs",
) -> None:
    """Heatmap of posterior mean entry weights for each BMU category."""
    rows = summarize_categorical_bmu_gate(
        weights=weights,
        bmu_full_idx=bmu_full_idx,
        bmu_obs_idx=bmu_obs_idx,
        bmu_levels=bmu_levels,
        model_names=model_names,
        idata=idata,
        weight_floor=weight_floor,
    )
    cat_rows = [r for r in rows if r.get("row_type") == "bmu"]
    if not cat_rows:
        return

    if sort_by == "dominant_weight":
        cat_rows = sorted(cat_rows, key=lambda r: (-(r.get("dominant_weight_pct") or 0.0), -(r.get("n_obs") or 0), str(r.get("bmu"))))
    elif sort_by == "bmu":
        cat_rows = sorted(cat_rows, key=lambda r: str(r.get("bmu")))
    else:
        cat_rows = sorted(cat_rows, key=lambda r: (-(r.get("n_obs") or 0), -(r.get("n_time") or 0), str(r.get("bmu"))))

    expert_keys = [re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or f"expert_{k}" for k, name in enumerate(model_names)]
    W = np.array([[r.get(f"w_entry_mean_{k}", np.nan) for k in expert_keys] for r in cat_rows], dtype=float)
    n_obs = np.array([r.get("n_obs", 0) for r in cat_rows], dtype=float)
    labels = [str(r.get("bmu")) for r in cat_rows]

    fig_h = max(4.5, 0.28 * len(cat_rows) + 1.2)
    fig = plt.figure(figsize=(10.8, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.32], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1], sharey=ax)

    im = ax.imshow(W, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("posterior mean weight")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(list(model_names), rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("expert")
    ax.set_ylabel("BMU / weather type")
    ax.set_title(f"Categorical BMU gate: weights by weather type ({label})")

    if len(cat_rows) <= 24:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                val = W[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{100.0*val:.0f}", ha="center", va="center", fontsize=7)

    y = np.arange(len(labels))
    axb.barh(y, n_obs, height=0.8)
    axb.set_xlabel("n obs")
    axb.set_title("Calibration support")
    axb.tick_params(axis="y", left=False, labelleft=False)
    axb.grid(axis="x", alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"categorical_bmu_gate_heatmap_{label}.png", dpi=200)
    plt.close(fig)


def plot_categorical_bmu_gate_timeline(
    *,
    time: np.ndarray,
    weights: np.ndarray,
    bmu_full_idx: np.ndarray,
    bmu_levels: Sequence[object],
    model_names: Sequence[str],
    out_dir: Path,
    label: str,
    split_date: Optional[np.datetime64] = None,
) -> None:
    """Timeline diagnostic: BMU strip + expert weights through time."""
    t = np.asarray(time)
    W = np.asarray(weights, dtype=float)
    bmu_full_idx = np.asarray(bmu_full_idx, dtype=int)
    if W.ndim != 2 or W.shape[0] != t.shape[0]:
        return

    fig = plt.figure(figsize=(12.0, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.32, 1.0], hspace=0.12)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

    # top strip: BMU index (unknown -> -1)
    strip = np.asarray(bmu_full_idx, dtype=float)[None, :]
    if np.issubdtype(t.dtype, np.datetime64):
        t_num = mdates.date2num(t.astype("datetime64[ns]").astype("datetime64[ms]").astype(object))
        extent = [float(t_num[0]), float(t_num[-1]), -0.5, 0.5]
        im = ax0.imshow(strip, aspect="auto", origin="lower", interpolation="nearest", extent=extent)
    else:
        extent = [0.0, float(max(len(t) - 1, 1)), -0.5, 0.5]
        im = ax0.imshow(strip, aspect="auto", origin="lower", interpolation="nearest", extent=extent)
    cbar = fig.colorbar(im, ax=ax0, pad=0.02)
    cbar.set_label("BMU index")
    ax0.set_yticks([])
    ax0.set_ylabel("BMU")
    ax0.set_title(f"BMU states and gate weights: {label}")

    for k, name in enumerate(model_names):
        ax1.plot(t, W[:, k], lw=1.6, label=str(name))
    ax1.set_ylabel("weight")
    ax1.set_xlabel("time")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False, ncol=max(1, min(len(model_names), 4)), loc="upper right")

    if np.issubdtype(t.dtype, np.datetime64):
        ax0.xaxis_date()
        ax1.xaxis_date()
        loc = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax1.xaxis.set_major_locator(loc)
        ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        if split_date is not None:
            try:
                sd = np.datetime64(split_date)
                split_num = mdates.date2num(sd.astype("datetime64[ms]").astype(object))
                ax0.axvline(split_num, ls="--", lw=1.0, c="k", alpha=0.7)
                ax1.axvline(sd, ls="--", lw=1.0, c="k", alpha=0.7)
            except Exception:
                pass

    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"categorical_bmu_gate_timeline_{label}.png", dpi=200)
    plt.close(fig)


def plot_categorical_bmu_effect_strength(
    *,
    idata,
    model_names: Sequence[str],
    out_dir: Path,
    label: str,
) -> None:
    """Posterior summary of sigma_class to diagnose whether BMU effects exist."""
    flat = _flatten_categorical_gate_posterior(idata)
    if flat is None or "sigma_class" not in flat:
        return
    s = np.asarray(flat["sigma_class"], dtype=float)
    if s.ndim != 2 or s.shape[1] == 0:
        return
    q05 = np.nanpercentile(s, 5.0, axis=0)
    q50 = np.nanpercentile(s, 50.0, axis=0)
    q95 = np.nanpercentile(s, 95.0, axis=0)
    x = np.arange(s.shape[1])

    fig, ax = plt.subplots(figsize=(max(5.2, 1.8 + 1.4 * len(x)), 4.2))
    ax.errorbar(x, q50, yerr=np.vstack([q50 - q05, q95 - q50]), fmt="o", capsize=4)
    labels = [f"{model_names[i]} vs {model_names[-1]}" if i < len(model_names)-1 else f"class_{i}" for i in x]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("sigma_class")
    ax.set_title(f"Strength of BMU-specific deviations: {label}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"categorical_bmu_effect_strength_{label}.png", dpi=200)
    plt.close(fig)
def save_trace_diagnostics_table(
    trace,
    *,
    label: str,
    out_dir: Path,
    max_total_elements: int = 200,
) -> None:
    """Save r_hat and ESS summary for all (non-huge) posterior variables."""
    import numpy as np
    import re
    try:
        import arviz as az
    except Exception:
        return

    if trace is None:
        return

    post = trace.posterior
    var_names: list[str] = []
    for v in list(post.data_vars):
        try:
            shape = tuple(post[v].shape)  # (chain, draw, ...)
            extra = int(np.prod(shape[2:])) if len(shape) > 2 else 1
            if extra <= max_total_elements:
                var_names.append(v)
        except Exception:
            continue

    if not var_names:
        return

    try:
        summ = az.summary(trace, var_names=var_names, round_to=None)
    except Exception:
        return

    cols = [c for c in ["r_hat", "ess_bulk", "ess_tail"] if c in summ.columns]
    if not cols:
        return
    summ_small = summ[cols].copy()

    # --- feature importance (%), derived from posterior beta ---
    try:
        if "beta" in post.data_vars:
            beta = post["beta"].values  # (chain, draw, class, feature)

            # importance per feature (averaged over chains, draws, classes)
            imp = np.mean(np.abs(beta), axis=(0, 1, 2))  # (feature,)

            s = float(np.sum(imp))
            imp_pct = (100.0 * imp / s) if s > 0 else np.zeros_like(imp)

            feat_names = None
            if "feature" in post["beta"].coords:
                feat_names = [str(x) for x in post["beta"].coords["feature"].values]

            feat_to_pct = {}
            if feat_names is not None and len(feat_names) == len(imp_pct):
                feat_to_pct = {fn: float(p) for fn, p in zip(feat_names, imp_pct)}

            # Match rows like: beta[0, E]   beta[1, Hs_ewma30] ...
            feat_weight_col = []
            for idx in summ_small.index.astype(str):
                m = re.match(r"^beta\[\d+,\s*(.+)\]$", idx)
                if m and feat_to_pct:
                    feat = m.group(1)
                    feat_weight_col.append(feat_to_pct.get(feat, np.nan))
                else:
                    feat_weight_col.append(np.nan)

            summ_small["feat_weight_pct"] = feat_weight_col
            cols = cols + ["feat_weight_pct"]
    except Exception:
        pass

    # Save CSV
    out_csv = out_dir / f"trace_diagnostics_{label}.csv"
    try:
        summ_small.to_csv(out_csv)
    except Exception:
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("name," + ",".join(cols) + "\n")
            for idx, row in summ_small.iterrows():
                f.write(str(idx) + "," + ",".join(str(row[c]) for c in cols) + "\n")

    # Save PNG table
    rows = []
    for name, row in summ_small.iterrows():
        rr = [name]
        for c in cols:
            v = row[c]
            rr.append("" if not np.isfinite(v) else f"{float(v):.4g}")
        rows.append(rr)

    _save_table_png(
        rows=rows,
        col_labels=["param", *cols],
        out_png=out_dir / f"trace_diagnostics_{label}.png",
        title=f"Sampler diagnostics: {label}",
        font_size=9,
        max_rows_per_page=55,
    )


# ---------------------------
# Main public API
# ---------------------------


def make_default_calibration_plots(
    *,
    out_dir: Path,
    prior: Optional[object] = None,
    prior_raw: Optional[np.ndarray] = None,
    posterior_raw: Optional[np.ndarray] = None,
    n_prior_samples: int = 5000,
    n_posterior_samples: int = 5000,
    random_seed: int = 123,
    time: np.ndarray,
    per5: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per95: np.ndarray,
    per1: np.ndarray,
    per99: np.ndarray,
    mini: np.ndarray,
    maxi: np.ndarray,
    draws: Optional[np.ndarray] = None,
    obs_time: np.ndarray,
    obs: np.ndarray,
    trace=None,
    ppc=None,
    param_names: Sequence[str] = (),
    label: str = "calibration",
    obs_mask_cal: Optional[np.ndarray] = None,
    obs_mask_val: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
    compare_models: Optional[dict[str, dict[str, np.ndarray]]] = None,
    gate_weights: Optional[np.ndarray] = None,
    gate_model_names: Sequence[str] = (),
    rotation_series: Optional[dict[str, np.ndarray]] = None,
) -> None:
    """Generate default plots + diagnostics for a calibration run.

    Parameters
    ----------
    trace:
        Optional ArviZ/PyMC inference data. When provided, r_hat and ESS tables
        are saved automatically.
    ppc:
        Optional posterior predictive InferenceData. When provided, additional
        PPC plots are saved (time-series + distribution KDE).
    """

    # Ensure output directory exists
    out_dir = Path(out_dir)
    _safe_makedirs(out_dir)
    diag_dir = out_dir / "diagnostics"
    _safe_makedirs(diag_dir)

    # Helper: sample from prior (raw space)
    def _draw_prior_raw(pr: object, n: int, seed: int) -> Optional[np.ndarray]:
        rng = np.random.default_rng(int(seed))
        if pr is None:
            return None

        # PriorProduct-like: has "priors" and "dims"
        if hasattr(pr, "priors") and hasattr(pr, "dims"):
            blocks = []
            for pblk, dblk in zip(getattr(pr, "priors"), getattr(pr, "dims")):
                s = _draw_prior_raw(pblk, n, seed + 17)
                if s is None:
                    return None
                if s.shape[1] != int(dblk):
                    # fallback: trim/pad
                    s = s[:, : int(dblk)]
                blocks.append(s)
            return np.concatenate(blocks, axis=1)

        # MVN-like: has mean/cov
        if hasattr(pr, "mean") and hasattr(pr, "cov"):
            mu = np.asarray(getattr(pr, "mean"), dtype=float)
            cov = np.asarray(getattr(pr, "cov"), dtype=float)
            try:
                return rng.multivariate_normal(mu, cov, size=int(n)).astype("float64")
            except Exception:
                # regularize if needed
                cov = cov + 1e-6 * np.eye(mu.size)
                return rng.multivariate_normal(mu, cov, size=int(n)).astype("float64")

        # KDE-like: has samples
        if hasattr(pr, "samples"):
            samp = np.asarray(getattr(pr, "samples"), dtype=float)
            if samp.ndim == 2 and samp.shape[0] > 0:
                idx = rng.choice(samp.shape[0], size=int(n), replace=samp.shape[0] < int(n))
                return samp[idx].astype("float64")

        return None

    # Helper: extract posterior raw samples from trace (InferenceData)
    def _extract_posterior_raw(tr, n: int, seed: int) -> Optional[np.ndarray]:
        if tr is None:
            return None
        try:
            posterior = tr.posterior
        except Exception:
            return None

        def _to_np(x):
            try:
                return np.asarray(x)
            except Exception:
                return np.array(x)

        if "raw_par" in posterior:
            arr = _to_np(posterior["raw_par"])  # (chain, draw, dim)
        else:
            blocks = []
            bi = 0
            while f"raw_par_{bi}" in posterior:
                blocks.append(_to_np(posterior[f"raw_par_{bi}"]))
                bi += 1
            if len(blocks) == 0:
                return None
            arr = np.concatenate(blocks, axis=-1)

        if arr.ndim < 3:
            return None
        flat = arr.reshape(-1, arr.shape[-1]).astype("float64")
        if flat.shape[0] <= int(n):
            return flat
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(flat.shape[0], size=int(n), replace=False)
        return flat[idx]

    # Auto-build prior/posterior samples if requested
    if prior_raw is None and prior is not None:
        prior_raw = _draw_prior_raw(prior, int(n_prior_samples), int(random_seed))
    if posterior_raw is None and trace is not None:
        posterior_raw = _extract_posterior_raw(trace, int(n_posterior_samples), int(random_seed) + 11)

    # --- 2D IH-MOOSE envelopes (per transect) ---
    if np.asarray(per50).ndim == 2 or np.asarray(obs).ndim == 2:
        # per-transect time series envelope
        tran_dir = out_dir / "transects"
        plot_transect_posterior_envelopes(
            out_dir=tran_dir,
            time=np.asarray(time),
            per1=np.asarray(per1),
            per10=np.asarray(per10),
            per50=np.asarray(per50),
            per90=np.asarray(per90),
            per99=np.asarray(per99),
            obs_time=np.asarray(obs_time),
            obs=np.asarray(obs),
            obs_mask=obs_mask_cal,
            split_date=split_date,
            label=label,
        )

        # optional rotation plot
        if rotation_series is not None:
            rot_dir = out_dir / "rotation"
            plot_rotation_envelope(
                out_path=rot_dir / f"{label}_rotation_posterior.png",
                time=np.asarray(rotation_series["time"]),
                per1=np.asarray(rotation_series["per1"]),
                per10=np.asarray(rotation_series["per10"]),
                per50=np.asarray(rotation_series["per50"]),
                per90=np.asarray(rotation_series["per90"]),
                per99=np.asarray(rotation_series["per99"]),
                obs_time=np.asarray(rotation_series["obs_time"]),
                obs=np.asarray(rotation_series["obs"]),
                obs_mask=rotation_series.get("obs_mask", None),
                split_date=split_date,
                label=f"{label} – rotation",
                ref_rot=rotation_series.get("ref_rot", None),
                wrap=str(rotation_series.get("wrap", "360")),
                obs_centered=rotation_series.get("obs_centered", None),
                model_centered=rotation_series.get("model_centered", None),
            )

            # Residual diagnostics for rotation (median at observation times), aligned using ref_rot if needed.
            try:
                rot_time = np.asarray(rotation_series["time"])
                rot_per50 = np.asarray(rotation_series["per50"], dtype=float)
                rot_obs_time = np.asarray(rotation_series["obs_time"])
                rot_obs = np.asarray(rotation_series["obs"], dtype=float)
                rot_obs_mask = rotation_series.get("obs_mask", None)

                ref_rot_val = rotation_series.get("ref_rot", None)
                wrap_mode = str(rotation_series.get("wrap", "360"))
                obs_centered = rotation_series.get("obs_centered", None)
                model_centered = rotation_series.get("model_centered", None)

                # Heuristic defaults: obs near 0 while model far from 0 => obs likely centered
                med_obs = float(np.nanmedian(rot_obs)) if rot_obs.size else 0.0
                med_mod = float(np.nanmedian(rot_per50)) if rot_per50.size else 0.0
                if obs_centered is None:
                    obs_centered = (abs(med_obs) < 60.0) and (abs(med_mod) > 60.0)
                if model_centered is None:
                    model_centered = False

                # Resolve reference angle
                ref_f = None
                if ref_rot_val is not None:
                    try:
                        ref_f = float(ref_rot_val)
                    except Exception:
                        ref_f = None
                    if ref_f is not None and (not np.isfinite(ref_f)):
                        ref_f = None

                # If missing but obs seems centered, estimate ref from circular mean of model median
                if ref_f is None and bool(obs_centered) and (not bool(model_centered)):
                    try:
                        ang = np.deg2rad(np.asarray(rot_per50, dtype=float))
                        s_ = float(np.nanmean(np.sin(ang)))
                        c_ = float(np.nanmean(np.cos(ang)))
                        if np.isfinite(s_) and np.isfinite(c_) and (abs(s_) + abs(c_) > 0):
                            ref_f = float(np.rad2deg(np.arctan2(s_, c_)))
                    except Exception:
                        ref_f = None

                # Apply de-centering if possible
                if ref_f is not None:
                    try:
                        from slmcal.utils.angles import uncenter_deg
                    except Exception:
                        uncenter_deg = None
                    if uncenter_deg is not None:
                        if bool(obs_centered):
                            rot_obs = uncenter_deg(rot_obs, ref_f, wrap=wrap_mode)
                        if bool(model_centered):
                            rot_per50 = uncenter_deg(rot_per50, ref_f, wrap=wrap_mode)

                rot_pred_obs = _interp_to_obs(rot_time, rot_per50, rot_obs_time)

                if rot_obs_mask is None:
                    good = np.isfinite(rot_obs) & np.isfinite(rot_pred_obs)
                else:
                    good = (~np.asarray(rot_obs_mask, dtype=bool)) & np.isfinite(rot_obs) & np.isfinite(rot_pred_obs)

                if np.any(good):
                    plot_residual_diagnostics(
                        obs_time=rot_obs_time[good],
                        obs=rot_obs[good],
                        pred_med_obs=rot_pred_obs[good],
                        label=f"{label}_rotation",
                        out_dir=diag_dir,
                    )
            except Exception:
                pass

        # Prior vs posterior plot (raw space)
        if prior_raw is not None and posterior_raw is not None and len(param_names) > 0:
            plot_prior_posterior(prior_raw, posterior_raw, param_names, label, diag_dir)

        

        # 2D diagnostics tables (global pooled + per-transect)
        try:
            save_probabilistic_diagnostics_2d(
                time=np.asarray(time),
                per1=np.asarray(per1),
                per10=np.asarray(per10),
                per50=np.asarray(per50),
                per90=np.asarray(per90),
                per99=np.asarray(per99),
                obs_time=np.asarray(obs_time),
                obs=np.asarray(obs),
                obs_mask=obs_mask_cal,
                split_date=split_date,
                label=label,
                out_dir=diag_dir,
            )
        except Exception:
            pass# Gate weights (MoE)
        if gate_weights is not None and len(gate_model_names) > 0:
            plot_gate_weights_heatmap(
                time=time,
                weights=gate_weights,
                model_names=gate_model_names,
                out_dir=diag_dir,
                label=label,
                split_date=split_date,
            )

        # Sampler diagnostics
        save_trace_diagnostics_table(trace, label=label, out_dir=diag_dir)
        return

    # (1D) continues below

    # --- baseline figures (existing behavior) ---
    if prior_raw is not None and posterior_raw is not None and len(param_names) > 0:
        plot_prior_posterior(prior_raw, posterior_raw, param_names, label, diag_dir)
    # Posterior predictive envelope over full time (with calibration/validation markers)
    plot_posterior_predictive_split(
        time=time,
        per5=per5,
        per10=per10,
        per50=per50,
        per90=per90,
        per95=per95,
        per1=per1,
        per99=per99,
        mini=mini,
        maxi=maxi,
        obs_time=obs_time,
        obs=obs,
        label=label,
        out_dir=out_dir,
        obs_mask_cal=obs_mask_cal,
        obs_mask_val=obs_mask_val,
        split_date=split_date,
    )

    # Posterior predictive + LPD at observation times (Normal/Student-t auto)
    plot_posterior_predictive_with_likelihood(
        time=time,
        per10=per10,
        per50=per50,
        per90=per90,
        per1=per1,
        per99=per99,
        obs_time=obs_time,
        obs=obs,
        label=label,
        out_dir=out_dir,
        trace=trace,
        obs_mask_cal=obs_mask_cal,
        obs_mask_val=obs_mask_val,
        split_date=split_date,
    )

    # --- probabilistic diagnostics plots ---
    # (Uses draws when provided; otherwise uses quantile-based parametric approximation)
    plot_coverage_reliability(
        time=time,
        per10=per10,
        per50=per50,
        per90=per90,
        per1=per1,
        per99=per99,
        obs_time=obs_time,
        obs=np.asarray(obs, dtype=float),
        draws=draws,
        trace=trace,
        label=label,
        out_dir=diag_dir,
        obs_mask_cal=obs_mask_cal,
        obs_mask_val=obs_mask_val,
        split_date=split_date,
    )

    plot_pit_histogram(
        time=time,
        per10=per10,
        per50=per50,
        per90=per90,
        per1=per1,
        per99=per99,
        obs_time=obs_time,
        obs=np.asarray(obs, dtype=float),
        draws=draws,
        trace=trace,
        label=label,
        out_dir=diag_dir,
        obs_mask_cal=obs_mask_cal,
        obs_mask_val=obs_mask_val,
        split_date=split_date,
    )

    plot_standardized_residuals_diagnostics(
        time=time,
        per10=per10,
        per50=per50,
        per90=per90,
        per1=per1,
        per99=per99,
        obs_time=obs_time,
        obs=np.asarray(obs, dtype=float),
        draws=draws,
        trace=trace,
        label=label,
        out_dir=diag_dir,
        obs_mask_cal=obs_mask_cal,
        obs_mask_val=obs_mask_val,
        split_date=split_date,
    )

    plot_residual_percentile_ribbons(
        time=time,
        per10=per10,
        per50=per50,
        per90=per90,
        per1=per1,
        per99=per99,
        obs_time=obs_time,
        obs=np.asarray(obs, dtype=float),
        label=label,
        out_dir=diag_dir,
        obs_mask_cal=obs_mask_cal,
        obs_mask_val=obs_mask_val,
        split_date=split_date,
    )

    plot_sharpness_over_time(
        time=time,
        per10=per10,
        per50=per50,
        per90=per90,
        per1=per1,
        per99=per99,
        obs_time=obs_time,
        obs=np.asarray(obs, dtype=float),
        label=label,
        out_dir=diag_dir,
        obs_mask_cal=obs_mask_cal,
        obs_mask_val=obs_mask_val,
        split_date=split_date,
    )

    plot_density_kde_bands_cal_val(
        draws=draws,                 # (S,T)
        time=time,
        obs_time=obs_time,
        obs=obs,
        out_dir=out_dir,
        label=label,
        split_date=split_date,
        max_draws=600,
        n_bins=220,
        smooth_bw_frac=0.03,
    )

    if gate_weights is not None and len(gate_model_names) > 0:
        plot_gate_entropy(
            time=time,
            weights=np.asarray(gate_weights, dtype=float),
            model_names=list(gate_model_names),
            label=label,
            out_dir=diag_dir,
            obs_time=obs_time,
            obs_mask_cal=obs_mask_cal,
            obs_mask_val=obs_mask_val,
            split_date=split_date,
        )

    # --- added diagnostics (Option B) ---
    # Probabilistic diagnostics table (single table: deterministic + probabilistic)
    save_probabilistic_diagnostics(
        time=time,
        per1=per1,
        per10=per10,
        per50=per50,
        per90=per90,
        per99=per99,
        per5=per5,
        per95=per95,
        mini=mini,
        maxi=maxi,
        draws=draws,
        obs_time=obs_time,
        obs=np.asarray(obs, dtype=float),
        label=label,
        out_dir=diag_dir,
        obs_mask_cal=obs_mask_cal,
        obs_mask_val=obs_mask_val,
        split_date=split_date,
        trace=trace,
        gate_weights=gate_weights,
        gate_model_names=gate_model_names,
    )

    # Residual diagnostics (median at observation times)
    per50_o = _interp_to_obs(time, per50, obs_time)
    obs_time_a = np.asarray(obs_time)
    obs_a = np.asarray(obs, dtype=float)
    if obs_mask_cal is not None:
        m_cal = np.asarray(obs_mask_cal, dtype=bool) & np.isfinite(obs_a)
        plot_residual_diagnostics(obs_time_a[m_cal], obs_a[m_cal], per50_o[m_cal], f"{label}_cal", diag_dir)
        if obs_mask_val is not None:
            m_val = np.asarray(obs_mask_val, dtype=bool) & np.isfinite(obs_a)
            if m_val.any():
                plot_residual_diagnostics(obs_time_a[m_val], obs_a[m_val], per50_o[m_val], f"{label}_val", diag_dir)
    else:
        plot_residual_diagnostics(obs_time_a, obs_a, per50_o, label, diag_dir)

    # PPC plots
    plot_ppc_timeseries(ppc, obs_time, obs, label, out_dir, obs_mask_cal=obs_mask_cal, obs_mask_val=obs_mask_val, split_date=split_date)
    plot_likelihood_distribution_kde(ppc, obs, label, out_dir, obs_mask_cal=obs_mask_cal)

    # Comparative plots for MoE / multi-model runs
    if compare_models:
        plot_compare_medians_split(
            time=time,
            models=compare_models,
            obs_time=obs_time,
            obs=obs,
            out_dir=out_dir,
            label=label,
            obs_mask_cal=obs_mask_cal,
            obs_mask_val=obs_mask_val,
            split_date=split_date,
        )
        save_compare_metrics(
            time=time,
            models=compare_models,
            obs_time=obs_time,
            obs=obs,
            out_dir=out_dir,
            label=label,
            obs_mask_cal=obs_mask_cal,
            obs_mask_val=obs_mask_val,
            split_date=split_date,
        )

    if gate_weights is not None and len(gate_model_names) > 0:
        plot_gate_weights_heatmap(
            time=time,
            weights=gate_weights,
            model_names=gate_model_names,
            out_dir=out_dir,
            label=label,
            split_date=split_date,
        )

    # Sampler diagnostics
    save_trace_diagnostics_table(trace, label=label, out_dir=diag_dir)



def plot_transect_posterior_envelopes(
    *,
    out_dir: Path,
    time: np.ndarray,
    per1: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per99: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    obs_mask: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
    label: str = "calibration",
):
    """Save one posterior predictive envelope plot per transect.

    Style matches `plot_posterior_predictive_split`:
    - Outer envelope (1–99%) shown as two red fills: [p1,p10] and [p90,p99] (no overlap).
    - Inner envelope (10–90%) shown as green fill.
    - Median as blue line.
    - Obs(cal)=black, Obs(val)=red if `split_date` provided.
    - Validation region is shaded and split date marked with a dashed vertical line.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    time = np.asarray(time)
    per1 = np.asarray(per1, dtype=float)
    per10 = np.asarray(per10, dtype=float)
    per50 = np.asarray(per50, dtype=float)
    per90 = np.asarray(per90, dtype=float)
    per99 = np.asarray(per99, dtype=float)
    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)

    if per50.ndim != 2:
        raise ValueError("Expected per50 to be 2D (time, transect) in plot_transect_posterior_envelopes.")

    n_tr = per50.shape[1]

    # Build validity mask: obs_mask is expected to mark invalid points (True=invalid)
    if obs_mask is None:
        valid = np.isfinite(obs)
    else:
        obs_mask = np.asarray(obs_mask, dtype=bool)
        valid = (~obs_mask) & np.isfinite(obs)

    # Calibration/validation split on observation timestamps
    if split_date is not None:
        sd = np.datetime64(split_date)
        is_cal_t = obs_time < sd
        is_val_t = ~is_cal_t
    else:
        sd = None
        is_cal_t = np.ones_like(obs_time, dtype=bool)
        is_val_t = np.zeros_like(obs_time, dtype=bool)

    for j in range(n_tr):
        fig, ax = plt.subplots(figsize=(12, 4))

        # Validation shading + split marker
        if sd is not None:
            try:
                ax.axvspan(sd, time.max(), alpha=0.15, color="gray", zorder=0)
                ax.axvline(sd, lw=1.2, ls="--", color="black", label="Split date", zorder=5)
            except Exception:
                pass

        # Envelopes (zorder: back)
        ax.fill_between(time, per1[:, j], per10[:, j], alpha=0.30, color="red",
                        label="Posterior 1–99%", linewidth=0.0, zorder=1)
        ax.fill_between(time, per90[:, j], per99[:, j], alpha=0.30, color="red",
                        label="", linewidth=0.0, zorder=1)
        ax.fill_between(time, per10[:, j], per90[:, j], alpha=0.45, color="green",
                        label="Posterior 10–90%", linewidth=0.0, zorder=2)

        # Median
        ax.plot(time, per50[:, j], lw=1.0, color="blue", label="Posterior median", zorder=3)

        # Observations
        good_j = valid[:, j]
        cal_j = good_j & is_cal_t
        val_j = good_j & is_val_t

        if np.any(cal_j):
            ax.scatter(obs_time[cal_j], obs[cal_j, j], s=10, c="k", label="Obs (cal)", zorder=4)
        if np.any(val_j):
            ax.scatter(obs_time[val_j], obs[val_j, j], s=10, c="r", label="Obs (val)", zorder=4, alpha=0.9)

        ax.set_title(f"Posterior predictive envelope — {label} (transect {j:02d})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Shoreline position")
        ax.set_xlim(time.min(), time.max())
        ax.grid(True, alpha=0.3)

        # Compact legend similar to the 1D plot
        ax.legend(loc="best", ncols=6)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_transect_{j:02d}_posterior.png", dpi=200)
        plt.close(fig)

def plot_rotation_envelope(
    *,
    out_path: Path,
    time: np.ndarray,
    per1: np.ndarray,
    per10: np.ndarray,
    per50: np.ndarray,
    per90: np.ndarray,
    per99: np.ndarray,
    obs_time: np.ndarray,
    obs: np.ndarray,
    obs_mask: Optional[np.ndarray] = None,
    split_date: Optional[np.datetime64] = None,
    label: str = "rotation",
    # Optional metadata to de-center centered rotation obs/model for plotting
    ref_rot: float | None = None,
    wrap: str = "360",
    obs_centered: bool | None = None,
    model_centered: bool | None = None,
) -> None:
    """Plot rotation obs vs model with uncertainty bands.

    Notes
    -----
    - Style matches `plot_posterior_predictive_split`:
        * 1–99% as two red fills: [p1,p10] and [p90,p99] (no overlap)
        * 10–90% as green fill
        * median as blue line
        * obs(cal)=black, obs(val)=red
        * split date dashed + validation region shaded
    - If observations were stored in a centered angular frame, this function can
      de-center them for plotting using `ref_rot` (or an automatic estimate).
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    time = np.asarray(time)
    per1 = np.asarray(per1, dtype=float)
    per10 = np.asarray(per10, dtype=float)
    per50 = np.asarray(per50, dtype=float)
    per90 = np.asarray(per90, dtype=float)
    per99 = np.asarray(per99, dtype=float)
    obs_time = np.asarray(obs_time)
    obs = np.asarray(obs, dtype=float)

    # -------------------------
    # Decide whether to de-center
    # -------------------------
    try:
        med_obs = float(np.nanmedian(obs))
    except Exception:
        med_obs = 0.0
    try:
        med_mod = float(np.nanmedian(per50))
    except Exception:
        med_mod = 0.0

    # Heuristic: obs clustered near 0 but model far from 0 => obs likely centered
    if obs_centered is None:
        obs_centered = (abs(med_obs) < 60.0) and (abs(med_mod) > 60.0)

    # By default, model bands are assumed to already be in absolute frame
    if model_centered is None:
        model_centered = False

    # Resolve reference angle
    ref_f: float | None = None
    if ref_rot is not None:
        try:
            ref_f = float(ref_rot)
        except Exception:
            ref_f = None
        if ref_f is not None and (not np.isfinite(ref_f)):
            ref_f = None

    # If ref_rot missing but obs looks centered and model looks absolute,
    # estimate reference from circular mean of model median.
    if ref_f is None and bool(obs_centered) and (not bool(model_centered)):
        try:
            ang = np.deg2rad(np.asarray(per50, dtype=float))
            s = float(np.nanmean(np.sin(ang)))
            c = float(np.nanmean(np.cos(ang)))
            if np.isfinite(s) and np.isfinite(c) and (abs(s) + abs(c) > 0):
                ref_f = float(np.rad2deg(np.arctan2(s, c)))
        except Exception:
            ref_f = None

    # Apply de-centering if possible
    if ref_f is not None:
        try:
            from slmcal.utils.angles import uncenter_deg
        except Exception:
            uncenter_deg = None

        if uncenter_deg is not None:
            if bool(obs_centered):
                obs = uncenter_deg(obs, ref_f, wrap=str(wrap))
            if bool(model_centered):
                per1 = uncenter_deg(per1, ref_f, wrap=str(wrap))
                per10 = uncenter_deg(per10, ref_f, wrap=str(wrap))
                per50 = uncenter_deg(per50, ref_f, wrap=str(wrap))
                per90 = uncenter_deg(per90, ref_f, wrap=str(wrap))
                per99 = uncenter_deg(per99, ref_f, wrap=str(wrap))

    # -------------------------
    # Masks: obs_mask expected True=invalid
    # -------------------------
    if obs_mask is None:
        good = np.isfinite(obs)
    else:
        good = (~np.asarray(obs_mask, dtype=bool)) & np.isfinite(obs)

    # Calibration/validation split for obs points
    if split_date is not None:
        sd = np.datetime64(split_date)
        is_cal = obs_time < sd
        is_val = ~is_cal
    else:
        sd = None
        is_cal = np.ones_like(obs_time, dtype=bool)
        is_val = np.zeros_like(obs_time, dtype=bool)

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(12, 4))

    # Validation shading + split marker (background)
    if sd is not None:
        try:
            ax.axvspan(sd, time.max(), alpha=0.15, color="gray", zorder=0)
            ax.axvline(sd, lw=1.2, ls="--", color="black", label="Split date", zorder=5)
        except Exception:
            pass

    # Envelopes (back)
    ax.fill_between(
        time, per1, per10,
        alpha=0.30, color="red", linewidth=0.0,
        label="Posterior 1–99%", zorder=1,
    )
    ax.fill_between(
        time, per90, per99,
        alpha=0.30, color="red", linewidth=0.0,
        label="", zorder=1,
    )
    ax.fill_between(
        time, per10, per90,
        alpha=0.45, color="green", linewidth=0.0,
        label="Posterior 10–90%", zorder=2,
    )

    # Median
    ax.plot(time, per50, lw=1.0, color="blue", label="Posterior median", zorder=3)

    # Obs points
    cal_pts = good & is_cal
    val_pts = good & is_val
    if np.any(cal_pts):
        ax.scatter(obs_time[cal_pts], obs[cal_pts], s=12, c="k", label="Obs (cal)", zorder=4)
    if np.any(val_pts):
        ax.scatter(obs_time[val_pts], obs[val_pts], s=12, c="r", label="Obs (val)", zorder=4, alpha=0.9)

    ax.set_title(label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Rotation (deg)")
    ax.set_xlim(time.min(), time.max())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=6)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
