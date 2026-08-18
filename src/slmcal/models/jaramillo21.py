from __future__ import annotations
from weakref import ref

"""Rotation model from Jaramillo et al. (2021a).

Ported from the IH-MOOSE Julia implementation.

alpha_eq = (theta - b) / a
alpha[0] = alpha0
alpha[i+1] = (alpha[i]-alpha_eq[i+1]) * exp(-L*P[i+1]*dt) + alpha_eq[i+1]
where L = Lcw if alpha[i] < alpha_eq[i+1] else Lccw
"""

import numpy as np

from slmcal.models.base import ParameterSpec, ShorelineModel
from slmcal.utils.angles import center_deg, circmean_deg, uncenter_deg, wrap180_deg

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore


if njit is not None:

    @njit(cache=True)
    def jaramillo21a_numba(P, theta, dt, a, b, Lcw, Lccw, alpha0):
        n = P.size
        alpha = np.empty(n, dtype=np.float64)
        alpha_eq = np.empty(n, dtype=np.float64)
        for i in range(n):
            alpha_eq[i] = (theta[i] - b) / a

        alpha[0] = alpha0

        # dt can be scalar or length n-1
        dt_is_scalar = dt.size == 1
        for i in range(n - 1):
            dti = float(dt[0]) if dt_is_scalar else float(dt[i])
            target = alpha_eq[i + 1]
            if alpha[i] < target:
                alpha[i + 1] = (alpha[i] - target) * np.exp(-Lcw * P[i + 1] * dti) + target
            else:
                alpha[i + 1] = (alpha[i] - target) * np.exp(-Lccw * P[i + 1] * dti) + target

        return alpha, alpha_eq

else:

    def jaramillo21a_numba(*args, **kwargs):  # pragma: no cover
        raise ImportError("Numba is required for jaramillo21a_numba.")


class Jaramillo21aModel(ShorelineModel):
    """1D rotation model wrapper compatible with the framework."""

    def __init__(self, P_key: str = "P", theta_key: str = "dir"):
        self.P_key = P_key
        self.theta_key = theta_key

        # Cache a reference angle so that this model can be used on datasets
        # that do not carry rotation observations (e.g. 2D IH-MOOSE datasets).
        self._ref_deg_default: float | None = None
        self._wrap_default: str = "360"

        def _exp(x: np.ndarray) -> np.ndarray:
            return np.exp(x)
        

        self._parameters = [
            ParameterSpec("a", (np.log(1e-2), np.log(20)), _exp),
            # IMPORTANT: b is calibrated in a *centered* angular frame.
            # Keeping it in (-180, 180] avoids wrap pathologies in both NSGA-II
            # and Bayesian sampling.
            ParameterSpec("b", (-180.0, 180.0), lambda x: x),
            ParameterSpec("logLcw", (np.log(1e-7), np.log(3e-5)), _exp),
            ParameterSpec("logLccw", (np.log(1e-7), np.log(3e-5)), _exp),
            # ParameterSpec("alpha0", (-360.0, 360.0), lambda x: x),
        ]

    # ------------------------------------------------------------------
    # Dataset preparation / centering
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ref_rot(dataset) -> float | None:
        """Return explicit reference angle if present on dataset (new API).
        Preprocess may attach it as `ref_rot`.
        """
        ref = getattr(dataset, "ref_rot", None)
        if ref is None:
            return None
        try:
            ref_f = float(ref)
        except Exception:
            return None
        return ref_f if np.isfinite(ref_f) else None


    def _resolve_ref_deg(self, dataset) -> float:
        """Resolve the reference angle (deg) robustly.

        Priority:
        1) dataset.ref_rot (explicit from preprocess)
        2) dataset._rot_ref_deg (legacy)
        3) cached self._ref_deg_default (from previous prepare_dataset call)
        4) circular mean of observed rotation (dataset.rot / dataset.obs)
        5) 0.0 fallback
        """
        # 1) Explicit from preprocess
        ref = self._get_ref_rot(dataset)
        if ref is not None:
            return float(ref)

        # 2) Legacy attribute
        ref = getattr(dataset, "_rot_ref_deg", None)
        try:
            ref_f = float(ref) if ref is not None else None
        except Exception:
            ref_f = None
        if ref_f is not None and np.isfinite(ref_f):
            return float(ref_f)

        # 3) Cached default
        if self._ref_deg_default is not None:
            try:
                ref_f = float(self._ref_deg_default)
            except Exception:
                ref_f = None
            if ref_f is not None and np.isfinite(ref_f):
                return float(ref_f)

        # 4) Compute from obs
        try:
            rot_obs = self._get_rot_obs(dataset)
            ref = circmean_deg(rot_obs)
            ref_f = float(ref)
            if np.isfinite(ref_f):
                return float(ref_f)
        except Exception:
            pass

        # 5) Last resort
        return 0.0

    @staticmethod
    def _get_rot_obs(dataset) -> np.ndarray:
        """Return the rotation observations used to define the angular reference."""
        if getattr(dataset, "rot", None) is not None:
            r = np.asarray(dataset.rot, dtype=float)
            return r
        # Fallback: for 1D datasets rot may be stored in obs
        if getattr(dataset, "obs", None) is not None:
            o = np.asarray(dataset.obs, dtype=float)
            if o.ndim == 1:
                return o
        raise ValueError("Dataset must provide rotation observations in 'rot' (preferred) or 1D 'obs'.")

    def prepare_dataset(self, dataset, *, inplace: bool = False):
        """Attach a rotation-based reference and optionally pre-center forcings.

        The wave direction forcing (theta) is centered using the *circular mean*
        of the rotation observations (rot). Centering is **only** a numerical
        artifact to keep the calibration of ``b`` well-conditioned.

        **Important:** the model output returned by :meth:`simulate` is always
        **de-centered** (absolute angles). Therefore this method **does not**
        modify ``dataset.rot``.

        Notes
        -----
        - This method is intended to be used by scripts/workflows before NSGA-II
          and Bayesian calibration so that priors/search are well-conditioned.
        - `simulate()` will still work if the dataset is not pre-centered.
        """

        from dataclasses import replace

        rot = self._get_rot_obs(dataset)
        ref = self._resolve_ref_deg(dataset)

        # Infer output wrap convention from observations.
        wrap_mode = getattr(dataset, "_rot_wrap_mode", None)
        if wrap_mode is None:
            r = rot[np.isfinite(rot)]
            if r.size and float(np.nanmin(r)) >= 0.0 and float(np.nanmax(r)) > 180.0:
                wrap_mode = "360"
            else:
                wrap_mode = "180"

        # Cache defaults for later calls on datasets without rot (e.g. 2D datasets)
        self._ref_deg_default = float(ref)
        self._wrap_default = str(wrap_mode)

        # Pre-center theta forcing (optional), but DO NOT change dataset.rot.
        forc = dict(getattr(dataset, "forcings", {}))
        if self.theta_key in forc:
            forc[self.theta_key] = center_deg(np.asarray(forc[self.theta_key], dtype=float), ref)

        # alpha0: store a centered version so the kernel runs in the centered frame
        alpha0 = getattr(dataset, "alpha0", None)
        if alpha0 is None and rot.size:
            alpha0 = float(rot[0])
        alpha0_c = float(center_deg(float(alpha0) if alpha0 is not None else 0.0, ref))

        if inplace:
            try:
                setattr(dataset, "forcings", forc)
            except Exception:
                pass
            try:
                setattr(dataset, "alpha0", alpha0_c)
            except Exception:
                pass
            try:
                setattr(dataset, "_rot_ref_deg", float(ref))
                setattr(dataset, "ref_rot", float(ref))
                setattr(dataset, "_rot_wrap_mode", str(wrap_mode))
                setattr(dataset, "_theta_centered", True)
                setattr(dataset, "_alpha0_centered", True)
            except Exception:
                pass
            return dataset

        ds2 = replace(dataset, forcings=forc)
        try:
            setattr(ds2, "alpha0", alpha0_c)
        except Exception:
            pass
        try:
            setattr(ds2, "_rot_ref_deg", float(ref))
            setattr(ds2, "ref_rot", float(ref))
            setattr(ds2, "_rot_wrap_mode", str(wrap_mode))
            setattr(ds2, "_theta_centered", True)
            setattr(ds2, "_alpha0_centered", True)
        except Exception:
            pass
        return ds2

    def attach_reference(self, dataset, ref_deg: float) -> None:
        """Attach a reference angle (in degrees) to an existing dataset."""
        try:
            setattr(dataset, "_rot_ref_deg", float(ref_deg))
            setattr(dataset, "ref_rot", float(ref_deg))
            if getattr(dataset, "_rot_wrap_mode", None) is None:
                setattr(dataset, "_rot_wrap_mode", str(self._wrap_default))
        except Exception:
            return

    @property
    def parameters(self) -> list[ParameterSpec]:
        return list(self._parameters)

    def simulate(self, physical_params: np.ndarray, dataset, y0=None) -> np.ndarray:
        a = float(physical_params[0])
        # b is interpreted in the *centered* angular frame
        b = float(wrap180_deg(float(physical_params[1])))
        Lcw = float(physical_params[2])
        Lccw = float(physical_params[3])

        # Reference angle: circular mean of rotation observations (preferred)
        ref = self._resolve_ref_deg(dataset)

        wrap_mode = getattr(dataset, "_rot_wrap_mode", None)
        if wrap_mode is None:
            wrap_mode = str(self._wrap_default)

        # alpha0: prefer explicit y0, else dataset.alpha0, else first rot/obs
        if y0 is not None:
            alpha0_raw = float(y0)
        elif getattr(dataset, "alpha0", None) is not None:
            alpha0_raw = float(dataset.alpha0)
        else:
            try:
                rot_obs = self._get_rot_obs(dataset)
                alpha0_raw = float(rot_obs[0]) if rot_obs.size else 0.0
            except Exception:
                alpha0_raw = 0.0

        # Center forcing direction and alpha0 if needed
        P = np.asarray(dataset.forcings[self.P_key], dtype=float)
        theta_raw = np.asarray(dataset.forcings[self.theta_key], dtype=float)
        if getattr(dataset, "_theta_centered", False):
            theta = theta_raw
        else:
            theta = center_deg(theta_raw, ref)

        if getattr(dataset, "_alpha0_centered", False):
            alpha0 = float(wrap180_deg(alpha0_raw))
        else:
            alpha0 = float(center_deg(alpha0_raw, ref))

        if getattr(dataset, "dt", None) is None:
            dt = np.array([1.0], dtype=float)
        else:
            dt = np.asarray(dataset.dt, dtype=float)
            if dt.ndim == 0:
                dt = np.array([float(dt)], dtype=float)
            if dt.size == P.size:
                dt = dt[:-1]
            if dt.size not in (1, P.size - 1):
                dt = np.array([float(np.median(dt))], dtype=float)

        alpha, _ = jaramillo21a_numba(P.astype(np.float64), theta.astype(np.float64), dt.astype(np.float64),
                                     a, b, Lcw, Lccw, alpha0)
        # Return in the original (absolute) angular frame.
        return uncenter_deg(alpha, ref, wrap=str(wrap_mode)).astype(float)


__all__ = ["jaramillo21a_numba", "Jaramillo21aModel"]
