"""González-style *double parabola* planform for IH-MOOSE (Numba).

This module implements the dynamic double-parabola planform used by the IH-MOOSE
"par2" projection.

Key properties
--------------
- Control points (Cp1, Cp2, Cl) are **fixed** inputs.
- The cross-shore signal enters as a scalar shift ``dX`` **inside** the parabola
  formulation (``X = ... + dX``), following the reference implementation you shared.
- All heavy routines are ``@njit`` to be safe and fast inside Bayesian black-box
  likelihood evaluation.

Output shapes
-------------
- A single parabola is written into preallocated arrays of length ``N_CURVE``.
- A double-parabola polyline is written as:
    parabola(Cp1) + bridge + reversed(parabola(Cp2))

Angles
------
Inputs are in **degrees**. Internally, trig uses radians.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore


# Fixed curve length used by the reference algorithm
N_CURVE = 180


def _require_numba() -> None:  # pragma: no cover
    if njit is None:
        raise ImportError("Numba is required for slmcal.models.gonzalez_par2")


if njit is not None:

    @njit(cache=True)
    def clamp_fmean_deg(fmean: float) -> float:
        """Clamp Fmean away from singular direction limits (degrees)."""
        if fmean < 0.2:
            fmean = 0.2
        elif fmean > 359.8:
            fmean = 359.8
        elif 179.8 < fmean <= 180.0:
            fmean = 179.8
        elif 180.0 < fmean < 180.2:
            fmean = 180.2
        elif 89.8 < fmean <= 90.0:
            fmean = 89.8
        elif 90.0 < fmean < 90.2:
            fmean = 90.2
        elif 269.8 < fmean <= 270.0:
            fmean = 269.8
        elif 270.0 < fmean < 270.2:
            fmean = 270.2
        return fmean


    @njit(cache=True)
    def nautical_to_cartesian_deg(nautical_deg: float) -> float:
        """Convert nautical direction (0=N, clockwise) to cartesian (0=E, CCW)."""
        c = 90.0 - nautical_deg
        if c < -180.0:
            c += 360.0
        return c


    @njit(cache=True)
    def hunt_wavelength(T: float, depth: float) -> float:
        """Hunt approximation of wavelength (meters)."""
        g = 9.81
        G = (2.0 * np.pi / T) ** 2 * (depth / g)
        F = G + 1.0 / (1.0 + 0.6522 * G + 0.4622 * G ** 2 + 0.0864 * G ** 4 + 0.0675 * G ** 5)
        return T * (g * depth / F) ** 0.5


    @njit(cache=True)
    def reflect_point_over_line(x: float, y: float, m: float, b: float) -> tuple[float, float]:
        """Reflect (x,y) over the line y = m x + b."""
        perp_m = -1.0 / m
        perp_b = y - perp_m * x
        x_int = (perp_b - b) / (m - perp_m)
        y_int = m * x_int + b
        return 2.0 * x_int - x, 2.0 * y_int - y


    @njit(cache=True)
    def gonzalez_single_parabola_into(
        fmean_nautical_deg: float,
        cp_xy: tuple[float, float],
        cl_xy: tuple[float, float],
        T: float,
        depth: float,
        Lr: float,
        gamd: float,  # kept for API compatibility (unused in your reference snippet)
        dX: float,
        out_x: np.ndarray,
        out_y: np.ndarray,
    ) -> None:
        """Compute one González parabola into preallocated arrays of length ``N_CURVE``."""
        _ = gamd

        fmean_nautical_deg = clamp_fmean_deg(fmean_nautical_deg)
        fmean_o = fmean_nautical_deg

        Ld = hunt_wavelength(T, depth)
        Xd, Yd = cp_xy
        Xc, Yc = cl_xy

        fmean = nautical_to_cartesian_deg(fmean_nautical_deg) + 90.0
        if fmean < 0.0:
            fmean += 360.0

        # Reference line through Cp
        xc = Xd + 100.0 * np.cos(np.deg2rad(fmean - 90.0))
        yc = Yd + 100.0 * np.sin(np.deg2rad(fmean - 90.0))
        m = (yc - Yd) / (xc - Xd + 1e-12)
        if m == 0.0:
            m = 1e-12
        b = Yd - m * Xd
        b2 = Yc - m * Xc
        flag_dir = 1 if b2 > b else -1

        # X-distance proxy + dynamic shift
        Rl = ((Xd - Xc) ** 2 + (Yd - Yc) ** 2) ** 0.5
        the = np.arctan2(Yc - Yd, Xc - Xd)
        X = np.abs(Rl * np.sin(the - fmean * np.pi / 180.0)) + dX

        beta_r = 2.13
        XL = X / (Ld + 1e-12)
        alpha_min = (
            np.arctan((((beta_r ** 4) / 16.0 + ((beta_r ** 2) / 2.0) * XL) ** 0.5) / (XL + 1e-12))
            * 180.0
            / np.pi
        )
        beta = 90.0 - alpha_min

        # Static equilibrium coefficients
        btmp = 10.0 if beta <= 10.0 else beta
        C0 = 0.0707 - 0.0047 * btmp + 0.000349 * (btmp**2) - 0.00000875 * (btmp**3) + 0.00000004765 * (btmp**4)
        C1 = 0.9536 + 0.0078 * btmp - 0.0004879 * (btmp**2) + 0.0000182 * (btmp**3) - 0.0000001281 * (btmp**4)
        C2 = 1.0 - C0 - C1

        # Beta adjustment using Cp->Cl bearing
        thed = np.arctan2(Yd - Yc, Xd - Xc) * 180.0 / np.pi
        thed = 90.0 - thed
        if thed < 0.0:
            thed += 360.0

        if flag_dir == 1:
            if fmean_o >= 270.0 and thed <= 90.0:
                thed += 360.0
            if fmean_o <= 90.0 and thed >= 270.0:
                thed -= 360.0
            bt_ref = 90.0 - abs(thed - fmean_o)
        else:
            if fmean_o >= 270.0 and thed <= 90.0:
                thed += 360.0
            if fmean_o <= 90.0 and thed >= 270.0:
                thed -= 360.0
            bt_ref = 90.0 - abs(fmean_o - thed)

        if bt_ref >= beta:
            beta = bt_ref

        # Main arc
        Ro = (XL / (np.sin(beta * np.pi / 180.0) + 1e-12)) * Ld
        bceil = int(np.ceil(beta))
        n_theta = 1 + (181 - bceil)
        if n_theta < 1:
            n_theta = 1

        n_ext = N_CURVE - n_theta
        if n_ext < 0:
            n_ext = 0

        ux = np.cos(fmean * np.pi / 180.0)
        uy = np.sin(fmean * np.pi / 180.0)

        # write reversed
        for k in range(n_theta):
            if k == 0:
                theta_deg = beta
            else:
                theta_deg = float(bceil + (k - 1))
            theta_rad = np.deg2rad(theta_deg + fmean)
            ratio = beta / (theta_deg + 1e-12)
            R = Ro * (C0 + C1 * ratio + C2 * (ratio * ratio))
            xk = Xd + R * np.cos(theta_rad)
            yk = Yd + R * np.sin(theta_rad)
            out_x[n_theta - 1 - k] = xk
            out_y[n_theta - 1 - k] = yk

        # Reflection + extension sign
        if 0.0 < fmean_o <= 180.0:
            do_reflect = flag_dir == -1
            ext_sign = -1.0 if flag_dir == -1 else 1.0
        else:
            do_reflect = flag_dir == 1
            ext_sign = -1.0 if flag_dir == 1 else 1.0

        if do_reflect:
            for k in range(n_theta):
                out_x[k], out_y[k] = reflect_point_over_line(out_x[k], out_y[k], m, b)

        x_last = out_x[n_theta - 1]
        y_last = out_y[n_theta - 1]
        for k in range(n_ext):
            s = 0.0 if n_ext <= 1 else (Lr * k / (n_ext - 1))
            out_x[n_theta + k] = x_last + ext_sign * s * ux
            out_y[n_theta + k] = y_last + ext_sign * s * uy


    @njit(cache=True)
    def double_parabola_polyline_into(
        fmean_nautical_deg: float,
        cp1_xy: tuple[float, float],
        cp2_xy: tuple[float, float],
        cl_xy: tuple[float, float],
        T: float,
        depth: float,
        Lr: float,
        gamd: float,
        dX: float,
        n_bridge: int,
        out_x: np.ndarray,
        out_y: np.ndarray,
        tmp1_x: np.ndarray,
        tmp1_y: np.ndarray,
        tmp2_x: np.ndarray,
        tmp2_y: np.ndarray,
    ) -> int:
        """Write the double-parabola polyline into ``out_x/out_y``.

        Returns
        -------
        n_points : int
            Number of valid points written.
        """
        if n_bridge < 2:
            n_bridge = 2

        gonzalez_single_parabola_into(fmean_nautical_deg, cp1_xy, cl_xy, T, depth, Lr, gamd, dX, tmp1_x, tmp1_y)
        gonzalez_single_parabola_into(fmean_nautical_deg, cp2_xy, cl_xy, T, depth, Lr, gamd, dX, tmp2_x, tmp2_y)

        # x1
        for k in range(N_CURVE):
            out_x[k] = tmp1_x[k]
            out_y[k] = tmp1_y[k]

        # bridge between end points
        x1e = tmp1_x[N_CURVE - 1]
        y1e = tmp1_y[N_CURVE - 1]
        x2e = tmp2_x[N_CURVE - 1]
        y2e = tmp2_y[N_CURVE - 1]
        for k in range(n_bridge):
            s = k / (n_bridge - 1)
            out_x[N_CURVE + k] = x1e + s * (x2e - x1e)
            out_y[N_CURVE + k] = y1e + s * (y2e - y1e)

        # reversed x2
        base = N_CURVE + n_bridge
        for k in range(N_CURVE):
            out_x[base + k] = tmp2_x[N_CURVE - 1 - k]
            out_y[base + k] = tmp2_y[N_CURVE - 1 - k]

        return base + N_CURVE

else:

    def gonzalez_single_parabola_into(*args, **kwargs):  # pragma: no cover
        _require_numba()

    def double_parabola_polyline_into(*args, **kwargs):  # pragma: no cover
        _require_numba()


__all__ = [
    "N_CURVE",
    "gonzalez_single_parabola_into",
    "double_parabola_polyline_into",
]
