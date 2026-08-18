"""IH-MOOSE hybrid shoreline model (fast par2 projection).

This is your **Bayesian-ready** IH-MOOSE wrapper. It couples:

- a cross-shore model providing a reference shoreline signal ``S_ref(t)``;
- a rotation model (Jaramillo21a) providing ``alpha(t)`` in degrees;
- a deterministic, dynamic double-parabola planform (González par2);
- a fast Numba projection from (dX, delta_alpha) to shoreline distances.

Core model idea
---------------
We treat the beach planform as a double parabola that is:

1) **Shifted** by a planform-axis displacement ``dX(t)`` derived from ``S_ref(t)``.
2) **Rotated** by the rotation anomaly ``delta_alpha(t) = alpha(t) - mean(alpha)``.

The planform is rotated about a center that is computed from a *pivot profile*:
we find the point on the planform closest to the pivot-profile line and use its
(distance to the pivot origin) to define the rotation center.

Why this implementation is fast
-------------------------------
Compared to the legacy IHSet workflow, we do NOT re-sample the coastline along a
large family of profiles. We rotate the planform once and project it directly to
transects using a segment-aware selector (exact crossing if available, otherwise
minimum perpendicular distance).

Bayesian speed
--------------
- ``simulate_obs`` evaluates only the forcing indices needed by the likelihood
  (``dataset.unique_t``), then gathers the flattened observation layout.
- Heavy geometry kernels are compiled with Numba.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from slmcal.models.moose_geometry import (
    get_or_build_par2_prof_cache,
    line_ABC_from_points,
    project_par2_direct_subset,
    project_par2_direct_full,
    project_par2_direct_gather,
)

# We try to use the existing ParameterSpec in your package.
try:
    from slmcal.models.base import ParameterSpec
except Exception:  # pragma: no cover
    ParameterSpec = None  # type: ignore


@dataclass
class IHMoosePar2Params:
    """Fixed planform parameters for par2."""

    Fmean: float
    Cp1: tuple[float, float]
    Cp2: tuple[float, float]
    Cl: tuple[float, float]   # only used to build the equilibrium pivot profiles
    T: float
    depth: float
    Lr: float
    gamd: float

    # kept for API compatibility
    n_bridge: int = 50
    lpro: float = 500.0


@dataclass
class IHMooseConfig:
    """Keys used in dataset forcings."""

    P_key: str = "P"          # wave power
    dir_key: str = "dir"      # nautical direction (deg)
    Sref_key: str = "S_ref"   # used if cross_shore_model is None

    # How to interpret the reference cross-shore signal.
    #
    # IH-MOOSE's par2 coupling expects a *reference shoreline signal* that can be
    # converted into a planform-axis shift dX(t). Depending on how the dataset is
    # built, this signal can be provided as an already-centered displacement (raw)
    # or as an absolute shoreline position along the reference transect.
    #
    # - "raw":           use S_ref as provided.
    # - "anomaly_y0":    subtract dataset.y0 (fallback: first S_ref value).
    # - "anomaly_first": subtract first S_ref value.
    # - "anomaly_mean":  subtract mean(S_ref).
    Sref_mode: str = "raw"


def _prefix_params(specs: list[ParameterSpec], prefix: str) -> list[ParameterSpec]:
    out: list[ParameterSpec] = []
    for s in specs:
        out.append(ParameterSpec(name=f"{prefix}{s.name}", bounds_raw=s.bounds_raw, transform=s.transform))
    return out


class IHMooseModel:
    """IH-MOOSE hybrid model with fast par2 projection."""

    def __init__(
        self,
        *,
        planform: IHMoosePar2Params,
        cross_shore_model=None,
        cfg: IHMooseConfig = IHMooseConfig(),
        ref_transect: int | None = None,
    ) -> None:
        if ParameterSpec is None:
            raise ImportError("slmcal.models.base.ParameterSpec not found")

        self.planform = planform
        self.cross_shore_model = cross_shore_model
        self.cfg = cfg
        self.ref_transect = ref_transect

        # Rotation parameters specs (match your Jaramillo21a model)
        try:
            from slmcal.models.jaramillo21 import Jaramillo21aModel
        except Exception as e:  # pragma: no cover
            raise ImportError("IHMooseModel requires slmcal.models.jaramillo21.Jaramillo21aModel") from e

        rot_model = Jaramillo21aModel(P_key=self.cfg.P_key, theta_key=self.cfg.dir_key)
        self._rot_model = rot_model
        rot_specs = list(rot_model.parameters)
        self._rot_specs = _prefix_params(rot_specs, "rot_")

        # Cross-shore parameter specs
        if self.cross_shore_model is None:
            self._cs_specs = []
        else:
            self._cs_specs = _prefix_params(list(self.cross_shore_model.parameters), "cs_")

        self._parameters = self._cs_specs + self._rot_specs

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> list[ParameterSpec]:
        return list(self._parameters)

    @property
    def n_cs(self) -> int:
        return len(self._cs_specs)

    @property
    def n_rot(self) -> int:
        return len(self._rot_specs)

    def _get_ref_transect(self, dataset) -> int:
        if self.ref_transect is not None:
            return int(self.ref_transect)
        if getattr(dataset, "ref_transect", None) is not None:
            return int(dataset.ref_transect)
        raise ValueError("ref_transect must be provided (model or dataset)")

    def _get_lpro(self, dataset) -> float:
        """Best-effort lpro retrieval."""
        lp = float(getattr(self.planform, "lpro", 500.0))
        if np.isfinite(lp) and lp > 0:
            return lp
        # fallback to dataset attrs (if present)
        try:
            attrs = getattr(dataset, "attrs", {})
            if isinstance(attrs, dict) and "IH_MOOSE" in attrs:
                import json

                cfg = json.loads(attrs["IH_MOOSE"]) if isinstance(attrs["IH_MOOSE"], str) else attrs["IH_MOOSE"]
                if isinstance(cfg, dict) and "lpro" in cfg:
                    lp2 = float(cfg["lpro"])
                    if np.isfinite(lp2) and lp2 > 0:
                        return lp2
        except Exception:
            pass
        return 500.0

    # ------------------------------------------------------------------
    # Coupling signals
    # ------------------------------------------------------------------

    def _sref_from_cross_shore(self, cs_params: np.ndarray, dataset, y0: float | None) -> np.ndarray:
        r = self._get_ref_transect(dataset)
        ds1d = dataset.to_timeseries_ref() if hasattr(dataset, "to_timeseries_ref") else dataset.to_timeseries(r)
        return np.asarray(self.cross_shore_model.simulate(cs_params, ds1d, y0), dtype=float)

    def _sref_from_forcing(self, dataset) -> np.ndarray:
        key = self.cfg.Sref_key
        if key not in dataset.forcings:
            raise KeyError(f"IHMooseModel needs dataset.forcings['{key}'] when cross_shore_model=None")
        return np.asarray(dataset.forcings[key], dtype=float)

    def _alpha_from_rotation(self, rot_params: np.ndarray, dataset) -> np.ndarray:
        # Delegate to the Jaramillo21aModel wrapper which handles circular
        # centering of wave directions using the circular mean of rot.
        y0 = None
        if getattr(dataset, "alpha0", None) is not None:
            y0 = float(dataset.alpha0)
        return np.asarray(self._rot_model.simulate(np.asarray(rot_params, dtype=float), dataset, y0=y0), dtype=float)

    # ------------------------------------------------------------------
    # Cross-shore -> planform-axis shift (IHSet-style)
    # ------------------------------------------------------------------

    def _dx_from_sref(self, S_ref: np.ndarray, dataset) -> np.ndarray:
        """Convert S_ref(t) into planform shift dX(t).

        This follows the same coupling convention you validated:

        - compute the reference transect direction DirN (nautical degrees)
        - take the *raw* absolute difference to Fmean (no wrap/min-angle)
        - apply the cosine projection and sign
        """
        # In Bayesian sampling this is called *a lot*. Cache the geometric
        # factor that maps S_ref -> dX.
        r = int(self._get_ref_transect(dataset))
        fmean = float(self.planform.Fmean)

        pair = getattr(dataset, "_moose_dx_fac", None)
        if pair is not None:
            k0, fac = pair
            if k0 == (r, fmean) and np.isfinite(fac):
                return (-(np.asarray(S_ref, dtype=np.float64) * float(fac))).astype(np.float64)

        xi = np.asarray(dataset.xi, dtype=np.float64)
        yi = np.asarray(dataset.yi, dtype=np.float64)
        xf = np.asarray(dataset.xf, dtype=np.float64)
        yf = np.asarray(dataset.yf, dtype=np.float64)

        dirN = 90.0 - np.arctan((yi[r] - yf[r]) / ((xi[r] - xf[r]) + 1e-12)) * 180.0 / np.pi
        dif = np.abs(dirN - fmean)
        fac = float(np.cos(np.deg2rad(dif)))

        try:
            setattr(dataset, "_moose_dx_fac", ((r, fmean), fac))
        except Exception:
            pass

        return (-(np.asarray(S_ref, dtype=np.float64) * fac)).astype(np.float64)

    def _normalize_sref(self, S_ref: np.ndarray, dataset) -> np.ndarray:
        """Normalize S_ref according to cfg.Sref_mode.

        This is mainly to support using data-driven cross-shore signals (e.g. SPADS)
        that may be produced as absolute shoreline positions.
        """
        s = np.asarray(S_ref, dtype=np.float64)
        mode = str(getattr(self.cfg, "Sref_mode", "raw")).lower().strip()
        if mode in ("", "raw", "none"):
            return s

        if s.size == 0:
            return s

        if mode in ("anomaly_y0", "y0"):
            base = getattr(dataset, "y0", None)
            try:
                base_f = float(base) if base is not None else np.nan
            except Exception:
                base_f = np.nan
            if not np.isfinite(base_f):
                base_f = float(s[0])
            return (s - base_f).astype(np.float64)

        if mode in ("anomaly_first", "first"):
            return (s - float(s[0])).astype(np.float64)

        if mode in ("anomaly_mean", "mean"):
            base = float(np.nanmean(s))
            if not np.isfinite(base):
                base = float(s[0])
            return (s - base).astype(np.float64)

        raise ValueError(f"Unknown IHMooseConfig.Sref_mode='{self.cfg.Sref_mode}'")

    @staticmethod
    def _delta_alpha(alpha: np.ndarray) -> np.ndarray:
        """Rotation anomaly using circular mean and wrapping.

        Using a linear mean on degrees creates discontinuities near 0/360.
        """
        from slmcal.utils.angles import circmean_deg, wrap180_deg

        a = np.asarray(alpha, dtype=np.float64)
        if a.size == 0:
            return a
        m = circmean_deg(a)
        if not np.isfinite(m):
            m = float(np.mean(a))
        return wrap180_deg(a - float(m)).astype(np.float64)

    # ------------------------------------------------------------------
    # Geometry cache
    # ------------------------------------------------------------------

    def _prep_geometry_cache(self, dataset):
        """Prepare cached geometry for fast projections."""
        r = int(self._get_ref_transect(dataset))

        # Cache the transect coordinates arrays as float64 to avoid repeated
        # xarray -> numpy conversions on every likelihood evaluation.
        xy = getattr(dataset, "_moose_transect_xy", None)
        if xy is None:
            xi = np.asarray(dataset.xi, dtype=np.float64)
            yi = np.asarray(dataset.yi, dtype=np.float64)
            xf = np.asarray(dataset.xf, dtype=np.float64)
            yf = np.asarray(dataset.yf, dtype=np.float64)
            xy = (xi, yi, xf, yf)
            try:
                setattr(dataset, "_moose_transect_xy", xy)
            except Exception:
                pass
        else:
            xi, yi, xf, yf = xy

        # Dynamic Cl override (kept because it stabilizes the planform reference
        # for the Angourie deterministic tests).
        cl_dyn_x = float(xi[r])
        cl_dyn_y = float(yi[r])

        pf = self.planform
        lpro = float(self._get_lpro(dataset))

        # Pivot profiles depend only on fixed planform settings
        planform_key = (
            float(pf.Fmean),
            float(pf.Cp1[0]), float(pf.Cp1[1]),
            float(pf.Cp2[0]), float(pf.Cp2[1]),
            float(pf.Cl[0]),  float(pf.Cl[1]),
            float(pf.T),
            float(pf.depth),
            float(lpro),
        )
        prof_cache = get_or_build_par2_prof_cache(dataset, planform_key)

        # Pivot profile index (cache)
        pivotN = getattr(dataset, "_moose_par2_pivotN", None)
        if pivotN is None:
            dxp = prof_cache.x0 - float(dataset.x_pivotal)
            dyp = prof_cache.y0 - float(dataset.y_pivotal)
            pivotN = int(np.argmin(dxp * dxp + dyp * dyp))
            try:
                setattr(dataset, "_moose_par2_pivotN", int(pivotN))
            except Exception:
                pass

        # Transect line coefficients (cache)
        tr_ABC = getattr(dataset, "_moose_transect_ABC", None)
        if tr_ABC is None:
            A_tr = np.empty(xi.size, dtype=np.float64)
            B_tr = np.empty(xi.size, dtype=np.float64)
            C_tr = np.empty(xi.size, dtype=np.float64)
            for j in range(xi.size):
                Aj, Bj, Cj = line_ABC_from_points(float(xi[j]), float(yi[j]), float(xf[j]), float(yf[j]))
                A_tr[j] = Aj
                B_tr[j] = Bj
                C_tr[j] = Cj
            tr_ABC = (A_tr, B_tr, C_tr)
            try:
                setattr(dataset, "_moose_transect_ABC", tr_ABC)
            except Exception:
                pass

        return xi, yi, xf, yf, cl_dyn_x, cl_dyn_y, prof_cache, int(pivotN), tr_ABC

    # ------------------------------------------------------------------
    # Convenience predictors
    # ------------------------------------------------------------------

    def predict_alpha(self, physical_params: np.ndarray, dataset) -> np.ndarray:
        physical_params = np.asarray(physical_params, dtype=float)
        rot_params = physical_params[self.n_cs :]
        return self._alpha_from_rotation(rot_params, dataset)

    def predict_sref(self, physical_params: np.ndarray, dataset, y0: float | None = None) -> np.ndarray:
        physical_params = np.asarray(physical_params, dtype=float)
        cs_params = physical_params[: self.n_cs]
        if self.cross_shore_model is None:
            return self._sref_from_forcing(dataset)
        return self._sref_from_cross_shore(cs_params, dataset, y0)

    # ------------------------------------------------------------------
    # Public simulation
    # ------------------------------------------------------------------

    def simulate(self, physical_params: np.ndarray, dataset, y0: float | None = None) -> np.ndarray:
        """Return full distances array (time, n_transects)."""

        out, _alpha = self.simulate_with_alpha(physical_params, dataset, y0=y0)
        return out

    def simulate_with_alpha(
        self, physical_params: np.ndarray, dataset, y0: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, alpha_abs).

        This is a performance helper for posterior predictive sampling: it avoids
        computing the rotation signal twice (once in simulate and again in
        predict_alpha).
        """

        physical_params = np.asarray(physical_params, dtype=float)
        if physical_params.size != self.n_cs + self.n_rot:
            raise ValueError("physical_params length mismatch")

        cs_params = physical_params[: self.n_cs]
        rot_params = physical_params[self.n_cs :]

        # cross-shore
        if self.cross_shore_model is None:
            S_ref = self._sref_from_forcing(dataset)
        else:
            S_ref = self._sref_from_cross_shore(cs_params, dataset, y0)

        # Optional normalization (kept as a config switch to preserve legacy behaviour)
        S_ref = self._normalize_sref(S_ref, dataset)

        # rotation (absolute)
        alpha = self._alpha_from_rotation(rot_params, dataset)
        delta_alpha = self._delta_alpha(alpha)

        dX = self._dx_from_sref(S_ref, dataset)

        xi, yi, _xf, _yf, cl_dyn_x, cl_dyn_y, prof_cache, pivotN, tr_ABC = self._prep_geometry_cache(dataset)
        A_tr, B_tr, C_tr = tr_ABC

        out = project_par2_direct_full(
            float(self.planform.Fmean),
            float(self.planform.Cp1[0]), float(self.planform.Cp1[1]),
            float(self.planform.Cp2[0]), float(self.planform.Cp2[1]),
            float(cl_dyn_x), float(cl_dyn_y),
            float(self.planform.T),
            float(self.planform.depth),
            prof_cache.x0,
            prof_cache.y0,
            prof_cache.dir_deg,
            prof_cache.A,
            prof_cache.B,
            prof_cache.C,
            int(pivotN),
            xi,
            yi,
            A_tr,
            B_tr,
            C_tr,
            dX,
            delta_alpha,
        )

        return np.asarray(out, dtype=float), np.asarray(alpha, dtype=float)

    def simulate_obs(self, physical_params: np.ndarray, dataset, y0: float | None = None) -> np.ndarray:
        """Fast path: return only predictions aligned with dataset.obs_flat."""

        if getattr(dataset, "obs_flat", None) is None:
            raise TypeError("simulate_obs requires a 2D dataset with obs_flat")
        if getattr(dataset, "unique_t", None) is None or getattr(dataset, "obs_unique_row", None) is None:
            raise TypeError("dataset missing unique_t/obs_unique_row")

        physical_params = np.asarray(physical_params, dtype=float)
        cs_params = physical_params[: self.n_cs]
        rot_params = physical_params[self.n_cs :]

        if self.cross_shore_model is None:
            S_ref = self._sref_from_forcing(dataset)
        else:
            S_ref = self._sref_from_cross_shore(cs_params, dataset, y0)

        S_ref = self._normalize_sref(S_ref, dataset)

        alpha = self._alpha_from_rotation(rot_params, dataset)
        delta_alpha = self._delta_alpha(alpha)

        dX = self._dx_from_sref(S_ref, dataset)

        xi, yi, _xf, _yf, cl_dyn_x, cl_dyn_y, prof_cache, pivotN, tr_ABC = self._prep_geometry_cache(dataset)
        A_tr, B_tr, C_tr = tr_ABC

        unique_t = np.asarray(dataset.unique_t, dtype=np.int64)

        # If the obs layout is sparse, avoid allocating a full (n_u, n_tr)
        # matrix by gathering only the required entries.
        try:
            gather_cache = getattr(dataset, "_moose_obs_gather", None)
        except Exception:
            gather_cache = None

        if gather_cache is None:
            # Build CSR-like mapping once
            obs_row = np.asarray(dataset.obs_unique_row, dtype=np.int64)
            obs_tr = np.asarray(dataset.idx_obs_tr_flat, dtype=np.int64)

            # Stable grouping by obs_row (then obs_tr)
            order = np.lexsort((obs_tr, obs_row))
            row_s = obs_row[order]
            tr_s = obs_tr[order]

            # ptr into sorted obs arrays for each unique row
            n_u = int(unique_t.size)
            counts = np.bincount(row_s, minlength=n_u).astype(np.int64)
            ptr = np.empty(n_u + 1, dtype=np.int64)
            ptr[0] = 0
            ptr[1:] = np.cumsum(counts)

            # output positions back to original obs order
            outpos = order.astype(np.int64)

            gather_cache = (ptr, tr_s.astype(np.int64), outpos)
            try:
                setattr(dataset, "_moose_obs_gather", gather_cache)
            except Exception:
                pass

        obs_ptr, obs_tr_sorted, obs_outpos = gather_cache

        y_pred = project_par2_direct_gather(
            float(self.planform.Fmean),
            float(self.planform.Cp1[0]), float(self.planform.Cp1[1]),
            float(self.planform.Cp2[0]), float(self.planform.Cp2[1]),
            float(cl_dyn_x), float(cl_dyn_y),
            float(self.planform.T),
            float(self.planform.depth),
            prof_cache.x0,
            prof_cache.y0,
            prof_cache.dir_deg,
            prof_cache.A,
            prof_cache.B,
            prof_cache.C,
            int(pivotN),
            xi,
            yi,
            A_tr,
            B_tr,
            C_tr,
            dX,
            delta_alpha,
            unique_t,
            np.asarray(obs_ptr, dtype=np.int64),
            np.asarray(obs_tr_sorted, dtype=np.int64),
            np.asarray(obs_outpos, dtype=np.int64),
        )

        # y_pred is already in obs_flat order
        return np.asarray(y_pred, dtype=float)


__all__ = [
    "IHMooseModel",
    "IHMooseConfig",
    "IHMoosePar2Params",
]
