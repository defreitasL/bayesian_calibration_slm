"""Fast IH-MOOSE *par2* geometry (Numba).

This is the **optimized** planform-projection kernel used by your Bayesian
IH-MOOSE implementation.

Model idea (as implemented here)
-------------------------------
At each time step we:

1) Recompute the double-parabola planform (González formulation) for the current
   cross-shore shift ``dX(t)``.

2) Compute a *pivot distance* ``pivotS`` by projecting the planform onto the
   **pivot profile** line (built once from the equilibrium planform).  The pivot
   distance is the Euclidean distance between the projected point and the pivot
   profile origin.

3) Define the rotation center by walking ``pivotS`` from the pivot origin along
   the pivot-profile direction.

4) Rotate the planform around that center by ``-delta_alpha(t)``.

5) Project the rotated planform **directly** onto each real transect using a
   robust, segment-aware selector:

   - if any polyline segment crosses the transect line, return the *exact*
     intersection point (distance-to-line is 0);
   - otherwise, return the polyline vertex with minimum perpendicular distance
     to the line.

Why this is faster than IHSet's discrete workflow
------------------------------------------------
The legacy IHSet implementation re-samples the rotated planform on a large
family of pivot profiles (~410) and then projects that cloud onto transects.
That is expensive and adds discretisation noise.

Here we keep the *same* physical idea (pivot-based rotation center) but remove
that intermediate re-sampling: we rotate once and project directly.

Performance notes
-----------------
- All heavy kernels are compiled with Numba (``@njit``).
- The segment-aware selector avoids ray casting and is both faster and more
  stable.
- The rotation is done **in-place** to halve memory traffic.

Conventions
-----------
- Coordinates are cartesian (meters).
- Directions are degrees (trigonometry uses radians internally).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore

from .gonzalez_par2 import N_CURVE, double_parabola_polyline_into


# Keep the par2 "bridge" discretisation fixed for stability & speed.
# (IHSet uses np.linspace default -> 50 points; this value also matches your
# deterministic tests that now work.)
N_BRIDGE_PAR2 = 50


@dataclass
class Par2ProfCache:
    """Cached pivot-profile geometry for the par2 rotation center."""

    # profile endpoints
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray

    # direction (degrees) and line coefficients for each profile
    dir_deg: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray


def _require_numba() -> None:  # pragma: no cover
    if njit is None:
        raise ImportError("Numba is required for slmcal.models.moose_geometry")


def build_par2_profiles(
    *,
    fmean_nautical_deg: float,
    cp1: tuple[float, float],
    cp2: tuple[float, float],
    cl_cfg: tuple[float, float],
    T: float,
    depth: float,
    lpro: float,
) -> Par2ProfCache:
    """Build the *equilibrium* pivot-profile family.

    We evaluate the planform at ``dX = +lpro/2`` and ``dX = -lpro/2`` and treat
    the resulting paired points as profile endpoints.

    Notes
    -----
    - We set (Lr, gamd) = (0, 0) here because we only need a stable *pivot
      direction* reference; using extensions for profiles adds little physical
      value but increases the risk of discrete artefacts.
    """

    n_poly = 2 * N_CURVE + N_BRIDGE_PAR2

    x_i = np.empty(n_poly, dtype=np.float64)
    y_i = np.empty(n_poly, dtype=np.float64)
    x_f = np.empty(n_poly, dtype=np.float64)
    y_f = np.empty(n_poly, dtype=np.float64)

    tmp1_x = np.empty(N_CURVE, dtype=np.float64)
    tmp1_y = np.empty(N_CURVE, dtype=np.float64)
    tmp2_x = np.empty(N_CURVE, dtype=np.float64)
    tmp2_y = np.empty(N_CURVE, dtype=np.float64)

    # +lpro/2
    double_parabola_polyline_into(
        float(fmean_nautical_deg),
        (float(cp1[0]), float(cp1[1])),
        (float(cp2[0]), float(cp2[1])),
        (float(cl_cfg[0]), float(cl_cfg[1])),
        float(T),
        float(depth),
        0.0,
        0.0,
        float(+lpro / 2.0),
        int(N_BRIDGE_PAR2),
        x_i,
        y_i,
        tmp1_x,
        tmp1_y,
        tmp2_x,
        tmp2_y,
    )

    # -lpro/2
    double_parabola_polyline_into(
        float(fmean_nautical_deg),
        (float(cp1[0]), float(cp1[1])),
        (float(cp2[0]), float(cp2[1])),
        (float(cl_cfg[0]), float(cl_cfg[1])),
        float(T),
        float(depth),
        0.0,
        0.0,
        float(-lpro / 2.0),
        int(N_BRIDGE_PAR2),
        x_f,
        y_f,
        tmp1_x,
        tmp1_y,
        tmp2_x,
        tmp2_y,
    )

    # Direction in degrees (kept as IHSet-style expression for consistency)
    dir_deg = 90.0 - np.arctan((y_i - y_f) / ((x_i - x_f) + 1e-10)) * 180.0 / np.pi

    # Line coefficients for each profile
    A = y_i - y_f
    B = x_f - x_i
    C = x_i * y_f - x_f * y_i

    return Par2ProfCache(
        x0=x_i,
        y0=y_i,
        x1=x_f,
        y1=y_f,
        dir_deg=dir_deg.astype(np.float64),
        A=A.astype(np.float64),
        B=B.astype(np.float64),
        C=C.astype(np.float64),
    )


def get_or_build_par2_prof_cache(dataset, planform_key: tuple) -> Par2ProfCache:
    """Dataset-level cache for the par2 pivot-profile family."""

    try:
        pair = getattr(dataset, "_moose_par2_prof_cache", None)
        if pair is not None:
            k0, cache0 = pair
            if k0 == planform_key and isinstance(cache0, Par2ProfCache):
                return cache0
    except Exception:
        pass

    cache = build_par2_profiles(
        fmean_nautical_deg=float(planform_key[0]),
        cp1=(float(planform_key[1]), float(planform_key[2])),
        cp2=(float(planform_key[3]), float(planform_key[4])),
        cl_cfg=(float(planform_key[5]), float(planform_key[6])),
        T=float(planform_key[7]),
        depth=float(planform_key[8]),
        lpro=float(planform_key[9]),
    )

    try:
        setattr(dataset, "_moose_par2_prof_cache", (planform_key, cache))
    except Exception:
        pass

    return cache


# -----------------------------------------------------------------------------
# Numba kernels
# -----------------------------------------------------------------------------

if njit is not None:

    @njit(cache=True, fastmath=True)
    def dist2(xa: float, ya: float, xb: float, yb: float) -> float:
        dx = xa - xb
        dy = ya - yb
        return dx * dx + dy * dy


    @njit(cache=True, fastmath=True)
    def line_ABC_from_points(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float]:
        """Line coefficients (A,B,C) for the infinite line through 2 points."""
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        return A, B, C


    @njit(cache=True, fastmath=True)
    def closest_point_polyline_to_line(
        x_poly: np.ndarray,
        y_poly: np.ndarray,
        A: float,
        B: float,
        C: float,
        x_ref: float,
        y_ref: float,
    ) -> tuple[float, float]:
        """Return the polyline point closest to an infinite line (segment-aware).

        Behaviour
        ---------
        - If any segment crosses the line -> return the exact crossing point.
          If multiple crossings exist, return the one closest to (x_ref, y_ref).
        - Otherwise -> return the polyline *vertex* with minimum |A x + B y + C|,
          breaking ties by proximity to (x_ref, y_ref).

        This is both more stable and typically faster than ray casting.
        """
        n = x_poly.size
        if n == 0:
            return np.nan, np.nan

        # --- best vertex (fallback) ---
        x0 = x_poly[0]
        y0 = y_poly[0]
        v0 = A * x0 + B * y0 + C
        av0 = v0 if v0 >= 0.0 else -v0
        best_abs = av0
        best_x = x0
        best_y = y0
        best_dref = dist2(x0, y0, x_ref, y_ref)

        # --- best intersection ---
        has_int = False
        best_int_dref = 1.0e308
        best_ix = np.nan
        best_iy = np.nan

        # walk segments (reuse v_prev)
        x_prev = x0
        y_prev = y0
        v_prev = v0

        eps = 1e-14

        for k in range(1, n):
            xk = x_poly[k]
            yk = y_poly[k]
            v = A * xk + B * yk + C

            # Track best vertex only if we still have no intersection
            if not has_int:
                av = v if v >= 0.0 else -v
                dref = dist2(xk, yk, x_ref, y_ref)
                if (av < best_abs - eps) or ((av - best_abs if av >= best_abs else best_abs - av) <= eps and dref < best_dref):
                    best_abs = av
                    best_x = xk
                    best_y = yk
                    best_dref = dref

            # Exact hit at a vertex
            if v == 0.0:
                dref = dist2(xk, yk, x_ref, y_ref)
                if (not has_int) or dref < best_int_dref:
                    has_int = True
                    best_int_dref = dref
                    best_ix = xk
                    best_iy = yk

            # Segment crossing (strict sign change)
            if v_prev * v < 0.0:
                # Robustness: with fastmath, some degenerate cases can still
                # lead to v_prev == v numerically. Guard the division.
                den = v_prev - v
                if den == 0.0:
                    t = 0.5
                else:
                    t = v_prev / den  # in (0,1)
                ix = x_prev + t * (xk - x_prev)
                iy = y_prev + t * (yk - y_prev)
                dref = dist2(ix, iy, x_ref, y_ref)
                if (not has_int) or dref < best_int_dref:
                    has_int = True
                    best_int_dref = dref
                    best_ix = ix
                    best_iy = iy

            x_prev = xk
            y_prev = yk
            v_prev = v

        if has_int:
            return best_ix, best_iy
        return best_x, best_y


    @njit(cache=True, fastmath=True)
    def project_par2_direct_subset(
        # planform params
        fmean_nautical_deg: float,
        cp1_x: float,
        cp1_y: float,
        cp2_x: float,
        cp2_y: float,
        # dynamic Cl override
        cl_dyn_x: float,
        cl_dyn_y: float,
        T: float,
        depth: float,
        # pivot profiles
        prof_x0: np.ndarray,
        prof_y0: np.ndarray,
        prof_dir_deg: np.ndarray,
        prof_A: np.ndarray,
        prof_B: np.ndarray,
        prof_C: np.ndarray,
        pivotN: int,
        # transects
        tr_xi: np.ndarray,
        tr_yi: np.ndarray,
        tr_A: np.ndarray,
        tr_B: np.ndarray,
        tr_C: np.ndarray,
        # dynamic signals
        dX: np.ndarray,
        delta_alpha_deg: np.ndarray,
        # subset indices
        unique_t: np.ndarray,
    ) -> np.ndarray:
        """Fast par2 projection for selected time indices.

        - Builds planform per time.
        - Computes pivot-based rotation center.
        - Rotates **in-place**.
        - Projects directly to transects using segment-aware selector.
        """
        n_u = unique_t.size
        n_tr = tr_xi.size

        n_poly = 2 * N_CURVE + N_BRIDGE_PAR2
        x_poly = np.empty(n_poly, dtype=np.float64)
        y_poly = np.empty(n_poly, dtype=np.float64)
        tmp1_x = np.empty(N_CURVE, dtype=np.float64)
        tmp1_y = np.empty(N_CURVE, dtype=np.float64)
        tmp2_x = np.empty(N_CURVE, dtype=np.float64)
        tmp2_y = np.empty(N_CURVE, dtype=np.float64)

        out = np.empty((n_u, n_tr), dtype=np.float64)

        # Pivot profile constants (cached outside the loop)
        pN = int(pivotN)
        pA = prof_A[pN]
        pB = prof_B[pN]
        pC = prof_C[pN]
        px0 = prof_x0[pN]
        py0 = prof_y0[pN]

        # Direction vector for the pivot profile (constant)
        ang0 = (prof_dir_deg[pN] - 90.0) * np.pi / 180.0
        c0 = np.cos(ang0)
        s0 = np.sin(ang0)

        for iu in range(n_u):
            it = int(unique_t[iu])

            # 1) dynamic planform for this dX
            double_parabola_polyline_into(
                fmean_nautical_deg,
                (cp1_x, cp1_y),
                (cp2_x, cp2_y),
                (cl_dyn_x, cl_dyn_y),
                T,
                depth,
                0.0,  # Lr fixed for stability
                0.0,  # gamd fixed for stability
                float(dX[it]),
                int(N_BRIDGE_PAR2),
                x_poly,
                y_poly,
                tmp1_x,
                tmp1_y,
                tmp2_x,
                tmp2_y,
            )

            # 2) pivotS from planform -> pivot profile
            xp, yp = closest_point_polyline_to_line(x_poly, y_poly, pA, pB, pC, px0, py0)
            if np.isnan(xp) or np.isnan(yp):
                # Ensure row is defined even for degenerate cases
                for tr in range(n_tr):
                    out[iu, tr] = np.nan
                continue
            pivotS = np.sqrt(dist2(xp, yp, px0, py0))

            # 3) rotation center along pivot direction
            cx = px0 + pivotS * c0
            cy = py0 - pivotS * s0

            # 4) rotate in-place by -delta_alpha
            a = float(delta_alpha_deg[it]) * np.pi / 180.0
            ca = np.cos(a)
            sa = np.sin(a)
            for k in range(n_poly):
                dxk = x_poly[k] - cx
                dyk = y_poly[k] - cy
                x_poly[k] = cx + ca * dxk + sa * dyk
                y_poly[k] = cy - sa * dxk + ca * dyk

            # 5) project to each transect
            for tr in range(n_tr):
                xq, yq = closest_point_polyline_to_line(
                    x_poly,
                    y_poly,
                    tr_A[tr],
                    tr_B[tr],
                    tr_C[tr],
                    tr_xi[tr],
                    tr_yi[tr],
                )
                if np.isnan(xq) or np.isnan(yq):
                    out[iu, tr] = np.nan
                else:
                    out[iu, tr] = np.sqrt(dist2(xq, yq, tr_xi[tr], tr_yi[tr]))

        return out


    @njit(cache=True, fastmath=True)
    def project_par2_direct_gather(
        # planform params
        fmean_nautical_deg: float,
        cp1_x: float,
        cp1_y: float,
        cp2_x: float,
        cp2_y: float,
        # dynamic Cl override
        cl_dyn_x: float,
        cl_dyn_y: float,
        T: float,
        depth: float,
        # pivot profiles
        prof_x0: np.ndarray,
        prof_y0: np.ndarray,
        prof_dir_deg: np.ndarray,
        prof_A: np.ndarray,
        prof_B: np.ndarray,
        prof_C: np.ndarray,
        pivotN: int,
        # transects
        tr_xi: np.ndarray,
        tr_yi: np.ndarray,
        tr_A: np.ndarray,
        tr_B: np.ndarray,
        tr_C: np.ndarray,
        # dynamic signals
        dX: np.ndarray,
        delta_alpha_deg: np.ndarray,
        # observation gather (CSR-like)
        unique_t: np.ndarray,
        obs_ptr: np.ndarray,
        obs_tr_sorted: np.ndarray,
        obs_outpos: np.ndarray,
    ) -> np.ndarray:
        """Compute only the entries needed for obs_flat.

        This avoids allocating a full (n_unique_t, n_tr) matrix when the
        observation layout is sparse.

        Inputs
        ------
        unique_t: (n_u,) global time indices
        obs_ptr:  (n_u+1,) start/end pointers into obs_tr_sorted/obs_outpos
        obs_tr_sorted: (n_obs,) transect index for each obs, grouped by unique row
        obs_outpos:    (n_obs,) output position in obs_flat order
        """

        n_u = unique_t.size
        n_obs = obs_tr_sorted.size

        n_poly = 2 * N_CURVE + N_BRIDGE_PAR2
        x_poly = np.empty(n_poly, dtype=np.float64)
        y_poly = np.empty(n_poly, dtype=np.float64)
        tmp1_x = np.empty(N_CURVE, dtype=np.float64)
        tmp1_y = np.empty(N_CURVE, dtype=np.float64)
        tmp2_x = np.empty(N_CURVE, dtype=np.float64)
        tmp2_y = np.empty(N_CURVE, dtype=np.float64)

        out = np.empty(n_obs, dtype=np.float64)

        # Pivot profile constants
        pN = int(pivotN)
        pA = prof_A[pN]
        pB = prof_B[pN]
        pC = prof_C[pN]
        px0 = prof_x0[pN]
        py0 = prof_y0[pN]

        ang0 = (prof_dir_deg[pN] - 90.0) * np.pi / 180.0
        c0 = np.cos(ang0)
        s0 = np.sin(ang0)

        for iu in range(n_u):
            it = int(unique_t[iu])

            # 1) dynamic planform for this dX
            double_parabola_polyline_into(
                fmean_nautical_deg,
                (cp1_x, cp1_y),
                (cp2_x, cp2_y),
                (cl_dyn_x, cl_dyn_y),
                T,
                depth,
                0.0,
                0.0,
                float(dX[it]),
                int(N_BRIDGE_PAR2),
                x_poly,
                y_poly,
                tmp1_x,
                tmp1_y,
                tmp2_x,
                tmp2_y,
            )

            # 2) pivotS from planform -> pivot profile
            xp, yp = closest_point_polyline_to_line(x_poly, y_poly, pA, pB, pC, px0, py0)
            if np.isnan(xp) or np.isnan(yp):
                j0 = int(obs_ptr[iu])
                j1 = int(obs_ptr[iu + 1])
                for j in range(j0, j1):
                    out[int(obs_outpos[j])] = np.nan
                continue
            pivotS = np.sqrt(dist2(xp, yp, px0, py0))

            # 3) rotation center
            cx = px0 + pivotS * c0
            cy = py0 - pivotS * s0

            # 4) rotate in-place by -delta_alpha
            a = float(delta_alpha_deg[it]) * np.pi / 180.0
            ca = np.cos(a)
            sa = np.sin(a)
            for k in range(n_poly):
                dxk = x_poly[k] - cx
                dyk = y_poly[k] - cy
                x_poly[k] = cx + ca * dxk + sa * dyk
                y_poly[k] = cy - sa * dxk + ca * dyk

            # 5) only evaluate requested transects for this unique time row
            j0 = int(obs_ptr[iu])
            j1 = int(obs_ptr[iu + 1])
            for j in range(j0, j1):
                tr = int(obs_tr_sorted[j])
                xq, yq = closest_point_polyline_to_line(
                    x_poly,
                    y_poly,
                    tr_A[tr],
                    tr_B[tr],
                    tr_C[tr],
                    tr_xi[tr],
                    tr_yi[tr],
                )
                if np.isnan(xq) or np.isnan(yq):
                    out[int(obs_outpos[j])] = np.nan
                else:
                    out[int(obs_outpos[j])] = np.sqrt(dist2(xq, yq, tr_xi[tr], tr_yi[tr]))

        return out


    @njit(cache=True, fastmath=True)
    def project_par2_direct_full(
        fmean_nautical_deg: float,
        cp1_x: float,
        cp1_y: float,
        cp2_x: float,
        cp2_y: float,
        cl_dyn_x: float,
        cl_dyn_y: float,
        T: float,
        depth: float,
        prof_x0: np.ndarray,
        prof_y0: np.ndarray,
        prof_dir_deg: np.ndarray,
        prof_A: np.ndarray,
        prof_B: np.ndarray,
        prof_C: np.ndarray,
        pivotN: int,
        tr_xi: np.ndarray,
        tr_yi: np.ndarray,
        tr_A: np.ndarray,
        tr_B: np.ndarray,
        tr_C: np.ndarray,
        dX: np.ndarray,
        delta_alpha_deg: np.ndarray,
    ) -> np.ndarray:
        unique_t = np.arange(dX.size, dtype=np.int64)
        return project_par2_direct_subset(
            fmean_nautical_deg,
            cp1_x,
            cp1_y,
            cp2_x,
            cp2_y,
            cl_dyn_x,
            cl_dyn_y,
            T,
            depth,
            prof_x0,
            prof_y0,
            prof_dir_deg,
            prof_A,
            prof_B,
            prof_C,
            pivotN,
            tr_xi,
            tr_yi,
            tr_A,
            tr_B,
            tr_C,
            dX,
            delta_alpha_deg,
            unique_t,
        )


else:

    def line_ABC_from_points(*args, **kwargs):  # pragma: no cover
        _require_numba()

    def get_or_build_par2_prof_cache(*args, **kwargs):  # pragma: no cover
        _require_numba()

    def project_par2_direct_subset(*args, **kwargs):  # pragma: no cover
        _require_numba()

    def project_par2_direct_full(*args, **kwargs):  # pragma: no cover
        _require_numba()


__all__ = [
    "N_BRIDGE_PAR2",
    "Par2ProfCache",
    "build_par2_profiles",
    "get_or_build_par2_prof_cache",
    "line_ABC_from_points",
    "project_par2_direct_subset",
    "project_par2_direct_full",
]
