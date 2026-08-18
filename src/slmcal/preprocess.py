from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr

from IHSetUtils.libjit.morfology import wast, wMOORE, deanSlope
from IHSetUtils.libjit.waves import RU2_Stockdon2006

@dataclass(frozen=True)
class PreprocessResult:
    # Full series (cropped so forcing starts at start_date)
    time: np.ndarray
    dt: np.ndarray

    time_obs: np.ndarray
    Obs: np.ndarray

    # Calibration/validation split
    start_date: np.datetime64
    # `split_date` is the end of calibration and start of validation.
    # (Kept as `end_date` for legacy compatibility.)
    end_date: np.datetime64
    # End of the available record (forcing time).
    end_record: np.datetime64

    time_splited: np.ndarray
    time_obs_splited: np.ndarray
    Obs_splited: np.ndarray

    # Index helpers (legacy-compatible)
    idx_obs: np.ndarray          # indices into Obs/time_obs within [start_date, end_date)
    idx_obs_splited: np.ndarray  # indices mapping time_obs_splited -> time_splited (nearest)

    # -------------------------
    # NEW: validation support
    # -------------------------
    # Masks are defined on the *full* observation arrays (time_obs/Obs)
    # and follow the convention:
    #   calibration: [start_date, end_date)
    #   validation:  [end_date, end_of_record]
    #
    # obs_in_time_mask selects observations that fall inside the model forcing
    # time window (after any forcing cropping).
    obs_in_time_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    mask_cal: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    mask_val: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    # For each observation, the nearest forcing index in `time`.
    # Out-of-range observations get -1.
    idx_time_obs: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    #  -------------------------
    #  OPIONAL VARTIABLES: Depending on model type
    #  -------------------------
    E: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    P: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hs: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    tp: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    dir: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sl: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hb: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    depthb: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    wast: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hberm: float | np.float64 = 0.0
    tm: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    mslp: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sp: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sst: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    bmus: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    E_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    P_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hs_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    tp_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    dir_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sl_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hb_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    depthb_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    wast_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    dt_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    Omega: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    Omega_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    cos_dir: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sin_dir: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    cos_dir_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sin_dir_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    tm_splitted: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    mslp_splitted: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sp_splitted: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sst_splitted: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    bmus_splitted: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def _mk_dt(time: np.ndarray) -> np.ndarray:
    # hours between consecutive timesteps
    t = pd.to_datetime(time)
    dt_hours = np.asarray([(t[i + 1] - t[i]).total_seconds() / 3600.0 for i in range(len(t) - 1)], dtype=float)
    return dt_hours


def _mk_nearest_idx(time: np.ndarray, query_time: np.ndarray) -> np.ndarray:
    # vectorized nearest neighbour indices
    t = pd.to_datetime(time).values.astype("datetime64[ns]")
    q = pd.to_datetime(query_time).values.astype("datetime64[ns]")
    # compute via argmin of abs difference
    # Using broadcasting is fine here; lengths are manageable.
    # For huge arrays, replace with searchsorted-based nearest.
    idx = np.argmin(np.abs(t[:, None] - q[None, :]), axis=0)
    return idx.astype(int)


def preprocess_legacy_yates(
    ds: xr.Dataset,
    start_date: str | np.datetime64 | pd.Timestamp,
    split_date: str | np.datetime64 | pd.Timestamp,
    end_date: str | np.datetime64 | pd.Timestamp | None = None,
    hs_var: str = "hs",
    tp_var: str = "tp",
    obs_var: str = "obs",
    time_var: str = "time",
    time_obs_var: str = "time_obs",
) -> PreprocessResult:
    """
    - build E = hs**2
    - crop forcing time series so `time >= start_date`
    - split calibration window [start_date, split_date)
    - validation window [split_date, end_of_record]
    - compute `idx_obs` and `idx_obs_splited` using nearest-neighbour argmin.
    """
    start = pd.to_datetime(start_date)
    split = pd.to_datetime(split_date)

    time = pd.to_datetime(ds[time_var].values)
    hs = np.asarray(ds[hs_var].values, dtype=float)
    E = hs ** 2
    tp = np.asarray(ds[tp_var].values, dtype=float)

    # crop forcing to start_date (legacy)
    idx0 = np.where(time >= start)[0]
    time = time[idx0]
    E = E[idx0]
    hs = hs[idx0]
    tp = tp[idx0]

    Obs = np.asarray(ds[obs_var].values, dtype=float)
    time_obs = pd.to_datetime(ds[time_obs_var].values)

    # Optional crop to an explicit end_date (end of record)
    if end_date is not None:
        end_rec = pd.to_datetime(end_date)
        idx1 = np.where(time <= end_rec)[0]
        if idx1.size:
            time = time[idx1]
            E = E[idx1]
            hs = hs[idx1]
            tp = tp[idx1]
        else:
            # no forcing points in window
            time = time[:0]
            E = E[:0]
            hs = hs[:0]
            tp = tp[:0]

    dt = _mk_dt(time.values)

    # --- masks on full observation arrays ---
    time_full = time.values.astype("datetime64[ns]")
    time_obs_full = time_obs.values.astype("datetime64[ns]")
    if time_full.size:
        obs_in_time_mask = (time_obs_full >= time_full[0]) & (time_obs_full <= time_full[-1])
    else:
        obs_in_time_mask = np.zeros_like(time_obs_full, dtype=bool)

    split64 = np.datetime64(split.to_datetime64())
    start64 = np.datetime64(start.to_datetime64())

    mask_cal = obs_in_time_mask & (time_obs_full >= start64) & (time_obs_full < split64)
    # Validation goes from split_date to the end of record
    mask_val = obs_in_time_mask & (time_obs_full >= split64)

    # Nearest forcing index for each observation (full forcing window)
    idx_time_obs = np.full(time_obs_full.shape[0], -1, dtype=int)
    if obs_in_time_mask.any():
        idx_time_obs[obs_in_time_mask] = _mk_nearest_idx(time_full, time_obs_full[obs_in_time_mask])

    # split calibration window
    idx = np.where((time >= start) & (time < split))[0]
    time_spl = time[idx]
    E_spl = E[idx]
    hs_spl = hs[idx]
    tp_spl = tp[idx]
    dt_spl = _mk_dt(time_spl.values)

    # Calibration observations (within forcing window + inside [start,end))
    idx_obs = np.where(mask_cal)[0]
    Obs_spl = Obs[idx_obs]
    time_obs_spl = time_obs[idx_obs]

    idx_obs_spl = _mk_nearest_idx(time_spl.values, time_obs_spl.values)

    return PreprocessResult(
        time=time_full,
        E=E,
        hs=hs,
        tp=tp,
        dt=dt,
        time_obs=time_obs_full,
        Obs=Obs,
        start_date=np.datetime64(start.to_datetime64()),
        end_date=np.datetime64(split.to_datetime64()),
        end_record=(time_full[-1] if time_full.size else np.datetime64("NaT")),
        time_splited=time_spl.values.astype("datetime64[ns]"),
        E_splited=E_spl,
        hs_splited=hs_spl,
        tp_splited=tp_spl,
        dt_splited=dt_spl,
        time_obs_splited=time_obs_spl.values.astype("datetime64[ns]"),
        Obs_splited=Obs_spl,
        idx_obs=idx_obs.astype(int),
        idx_obs_splited=idx_obs_spl.astype(int),
        obs_in_time_mask=np.asarray(obs_in_time_mask, dtype=bool),
        mask_cal=np.asarray(mask_cal, dtype=bool),
        mask_val=np.asarray(mask_val, dtype=bool),
        idx_time_obs=np.asarray(idx_time_obs, dtype=int),
    )





def preprocess_legacy_no_index(
    ds: xr.Dataset,
    end_date: str | np.datetime64 | pd.Timestamp | None = None,
    hs_var: str = "hs",
    tp_var: str = "tp",
    dir_var: str = "dir",
    sl_var: str = "sl",
    params: Dict[str, any] = {"depth": 100.0, "hberm": 0.0, "alpha_baty": 0.0, 'bcoef': 0.55, 'break_it': True, 'D50': 0.31e-3},
    obs_var: str = "obs",
    time_var: str = "time",
    time_obs_var: str = "time_obs",
    cop_vars: bool = False,
    add_bmu_info: bool = False,
) -> PreprocessResult:
    """
    Legacy Miller & Dean/SPADS preprocess **without** splitting by dates.

    This helper computes all derived forcings on the full available forcing
    record (optionally cropped only by ``end_date``), keeps all observations
    that fall inside that forcing window, and builds the observation-to-time
    mapping on the full record.

    It is useful for workflows where slow covariates (e.g. EWMAs) must be
    computed before any calibration/validation window is selected.
    """
    time = pd.to_datetime(ds[time_var].values)
    hs = np.asarray(ds[hs_var].values, dtype=float)
    tp = np.asarray(ds[tp_var].values, dtype=float)
    dir_ = np.asarray(ds[dir_var].values, dtype=float)
    sl = np.asarray(ds[sl_var].values, dtype=float)

    depth = params.get("depth", 100.0)
    depth = np.repeat(depth, hs.shape[0])
    hberm = params.get("hberm", 0.0)
    alpha_baty = params.get("alpha_baty", 0.0)
    alpha_baty = np.repeat(alpha_baty, hs.shape[0])
    bcoef = params.get("bcoef", 0.55)
    break_it = params.get("break_it", True)
    D50 = params.get("D50", 0.3e-3)

    E = hs ** 2
    P = E * tp
    ws = wMOORE(D50)
    cos_dir = np.cos(np.radians(dir_))
    sin_dir = np.sin(np.radians(dir_))

    if cop_vars:
        tm = np.asarray(ds["tm"].values, dtype=float)
        mslp = np.asarray(ds["mslp"].values, dtype=float)
        sp = np.asarray(ds["sp"].values, dtype=float)
        sst = np.asarray(ds["sst"].values, dtype=float)

    if add_bmu_info:
        bmus = np.asarray(ds["kma_bmus"].values, dtype=int)

    if break_it:
        from IHSetUtils.libjit.waves import BreakingPropagation
        from IHSetUtils.libjit.geometry import nauticalDir2cartesianDir

        dir_model = nauticalDir2cartesianDir(dir_)
        hb, _, depthb = BreakingPropagation(hs, tp, dir_model, depth, alpha_baty, bcoef)
        depthb[hb < 0.1] = 0.2
        hb[hb < 0.1] = 0.1
        dir_out = dir_model
    else:
        hb = hs
        depthb = np.repeat(depth, hs.shape[0])
        dir_out = dir_

    dir_out = np.mod(dir_out, 360.0) - np.mean(dir_out)  # demean direction to help with training

    wast_ = wast(hb, D50)
    Omega = hb / (ws * tp)

    runnup = np.zeros_like(hs)
    slope = deanSlope(depth[0], D50)
    print(slope)
    for i in range(hs.shape[0]):
        runnup[i] = RU2_Stockdon2006(slope, hs[i], tp[i])

    sl = sl + runnup

    if end_date is not None:
        end_rec = pd.to_datetime(end_date)
        idx1 = np.where(time <= end_rec)[0]
        if idx1.size:
            time = time[idx1]
            hs = hs[idx1]
            tp = tp[idx1]
            dir_out = dir_out[idx1]
            sl = sl[idx1]
            hb = hb[idx1]
            depthb = depthb[idx1]
            wast_ = wast_[idx1]
            E = E[idx1]
            P = P[idx1]
            Omega = Omega[idx1]
            cos_dir = cos_dir[idx1]
            sin_dir = sin_dir[idx1]
            if cop_vars:
                tm = tm[idx1]
                mslp = mslp[idx1]
                sp = sp[idx1]
                sst = sst[idx1]
            if add_bmu_info:
                bmus = bmus[idx1]
        else:
            time = time[:0]
            hs = hs[:0]
            tp = tp[:0]
            dir_out = dir_out[:0]
            sl = sl[:0]
            hb = hb[:0]
            depthb = depthb[:0]
            wast_ = wast_[:0]
            E = E[:0]
            P = P[:0]
            Omega = Omega[:0]
            cos_dir = cos_dir[:0]
            sin_dir = sin_dir[:0]
            if cop_vars:
                tm = tm[:0]
                mslp = mslp[:0]
                sp = sp[:0]
                sst = sst[:0]
            if add_bmu_info:
                bmus = bmus[:0]

    dt = _mk_dt(time.values)

    Obs = np.asarray(ds[obs_var].values, dtype=float)
    time_obs = pd.to_datetime(ds[time_obs_var].values)

    time_full = time.values.astype("datetime64[ns]")
    time_obs_full = time_obs.values.astype("datetime64[ns]")
    if time_full.size:
        obs_in_time_mask = (time_obs_full >= time_full[0]) & (time_obs_full <= time_full[-1])
    else:
        obs_in_time_mask = np.zeros_like(time_obs_full, dtype=bool)

    idx_time_obs = np.full(time_obs_full.shape[0], -1, dtype=int)
    if obs_in_time_mask.any():
        idx_time_obs[obs_in_time_mask] = _mk_nearest_idx(time_full, time_obs_full[obs_in_time_mask])

    idx_obs = np.where(obs_in_time_mask)[0]
    time_obs_spl = time_obs[idx_obs]
    Obs_spl = Obs[idx_obs]
    idx_obs_spl = idx_time_obs[idx_obs].astype(int) if idx_obs.size else np.array([], dtype=int)

    if not cop_vars:
        tm = np.zeros_like(hs)
        mslp = np.zeros_like(hs)
        sp = np.zeros_like(hs)
        sst = np.zeros_like(hs)

    if not add_bmu_info:
        bmus = np.zeros_like(hs)

    start64 = time_full[0] if time_full.size else np.datetime64("NaT")
    end64 = time_full[-1] if time_full.size else np.datetime64("NaT")

    return PreprocessResult(
        time=time_full,
        hs=hs,
        tp=tp,
        dir=dir_out,
        sl=sl,
        hb=hb,
        depthb=depthb,
        wast=wast_,
        E=E,
        P=P,
        Omega=Omega,
        cos_dir=cos_dir,
        sin_dir=sin_dir,
        tm=tm,
        mslp=mslp,
        sp=sp,
        sst=sst,
        bmus=bmus,
        hberm=hberm,
        dt=dt,
        time_obs=time_obs_full,
        Obs=Obs,
        start_date=start64,
        end_date=end64,
        end_record=end64,
        time_splited=time_full,
        hs_splited=hs,
        tp_splited=tp,
        dir_splited=dir_out,
        sl_splited=sl,
        hb_splited=hb,
        depthb_splited=depthb,
        wast_splited=wast_,
        E_splited=E,
        P_splited=P,
        Omega_splited=Omega,
        cos_dir_splited=cos_dir,
        sin_dir_splited=sin_dir,
        tm_splitted=tm,
        mslp_splitted=mslp,
        sp_splitted=sp,
        sst_splitted=sst,
        bmus_splitted=bmus,
        dt_splited=dt,
        time_obs_splited=time_obs_spl.values.astype("datetime64[ns]"),
        Obs_splited=Obs_spl,
        idx_obs=idx_obs.astype(int),
        idx_obs_splited=idx_obs_spl,
        obs_in_time_mask=np.asarray(obs_in_time_mask, dtype=bool),
        mask_cal=np.asarray(obs_in_time_mask, dtype=bool),
        mask_val=np.zeros_like(obs_in_time_mask, dtype=bool),
        idx_time_obs=np.asarray(idx_time_obs, dtype=int),
    )


def preprocess_legacy_millerDean(
    ds: xr.Dataset,
    start_date: str | np.datetime64 | pd.Timestamp,
    split_date: str | np.datetime64 | pd.Timestamp,
    end_date: str | np.datetime64 | pd.Timestamp | None = None,
    hs_var: str = "hs",
    tp_var: str = "tp",
    dir_var: str = "dir",
    sl_var: str = "sl",
    params: Dict[str, any] = {"depth": 100.0, "hberm": 0.0, "alpha_baty": 0.0, 'bcoef': 0.55, 'break_it': True, 'D50': 0.31e-3},
    obs_var: str = "obs",
    time_var: str = "time",
    time_obs_var: str = "time_obs",
    cop_vars: bool = False,
) -> PreprocessResult:
    """
    - build E = hs**2
    - crop forcing time series so `time >= start_date`
    - split calibration window [start_date, split_date)
    - validation window [split_date, end_of_record]
    - compute `idx_obs` and `idx_obs_splited` using nearest-neighbour argmin.
    """
    start = pd.to_datetime(start_date)
    split = pd.to_datetime(split_date)

    time = pd.to_datetime(ds[time_var].values)
    hs = np.asarray(ds[hs_var].values, dtype=float)
    tp = np.asarray(ds[tp_var].values, dtype=float)
    dir_ = np.asarray(ds[dir_var].values, dtype=float)
    sl = np.asarray(ds[sl_var].values, dtype=float)
    depth = params.get("depth", 100.0)
    depth = np.repeat(depth, hs.shape[0])
    hberm = params.get("hberm", 0.0)
    alpha_baty = params.get("alpha_baty", 0.0)
    alpha_baty = np.repeat(alpha_baty, hs.shape[0])
    bcoef = params.get("bcoef", 0.55)
    break_it = params.get("break_it", True)
    D50 = params.get("D50", 0.3e-3)
    E = hs ** 2
    P = E * tp
    ws = wMOORE(D50)
    cos_dir = np.cos(np.radians(dir_))
    sin_dir = np.sin(np.radians(dir_))

    if cop_vars:
        tm = np.asarray(ds["tm"].values, dtype=float)
        mslp = np.asarray(ds["mslp"].values, dtype=float)
        sp = np.asarray(ds["sp"].values, dtype=float)
        sst = np.asarray(ds["sst"].values, dtype=float)

    if break_it:
        from IHSetUtils.libjit.waves import BreakingPropagation
        from IHSetUtils.libjit.geometry import nauticalDir2cartesianDir
        dir_ = nauticalDir2cartesianDir(dir_)
        hb, _, depthb = BreakingPropagation(hs, tp, dir_, depth, alpha_baty, bcoef)
        depthb[hb < 0.1] = 0.2
        hb[hb < 0.1] = 0.1
        
    else:    
        hb = hs
        depthb = np.repeat(depth, hs.shape[0])
    
    wast_ = wast(hb, D50)
    Omega = hb / (ws * tp)

    runnup = np.zeros_like(hs)
    slope = deanSlope(depth[0], D50)
    print(slope)
    for i in range(hs.shape[0]):
        runnup[i] = RU2_Stockdon2006(slope, hs[i], tp[i])

    sl = sl + runnup
    # crop forcing to start_date (legacy)
    idx0 = np.where(time >= start)[0]
    time = time[idx0]
    hs = hs[idx0]
    tp = tp[idx0]
    dir_ = dir_[idx0]
    sl = sl[idx0]
    hb = hb[idx0]
    depthb = depthb[idx0]
    wast_ = wast_[idx0]
    E = E[idx0]
    P = P[idx0]
    Omega = Omega[idx0]
    cos_dir = cos_dir[idx0]
    sin_dir = sin_dir[idx0]
    if cop_vars:
        tm = tm[idx0]
        mslp = mslp[idx0]
        sp = sp[idx0]
        sst = sst[idx0]


    Obs = np.asarray(ds[obs_var].values, dtype=float)
    time_obs = pd.to_datetime(ds[time_obs_var].values)

    # Optional crop to an explicit end_date (end of record)
    if end_date is not None:
        end_rec = pd.to_datetime(end_date)
        idx1 = np.where(time <= end_rec)[0]
        if idx1.size:
            time = time[idx1]
            hs = hs[idx1]
            tp = tp[idx1]
            dir_ = dir_[idx1]
            sl = sl[idx1]
            hb = hb[idx1]
            depthb = depthb[idx1]
            wast_ = wast_[idx1]
            E = E[idx1]
            P = P[idx1]
            Omega = Omega[idx1]
            cos_dir = cos_dir[idx1]
            sin_dir = sin_dir[idx1]
            if cop_vars:
                tm = tm[idx1]
                mslp = mslp[idx1]
                sp = sp[idx1]
                sst = sst[idx1]
        else:
            # no forcing points in window
            time = time[:0]
            hs = hs[:0]
            tp = tp[:0]
            dir_ = dir_[:0]
            sl = sl[:0]
            hb = hb[:0]
            depthb = depthb[:0]
            wast_ = wast_[:0]
            E = E[:0]
            P = P[:0]
            Omega = Omega[:0]
            cos_dir = cos_dir[:0]
            sin_dir = sin_dir[:0]
            if cop_vars:
                tm = tm[:0]
                mslp = mslp[:0]
                sp = sp[:0]
                sst = sst[:0]

    dt = _mk_dt(time.values)

    # --- masks on full observation arrays ---
    time_full = time.values.astype("datetime64[ns]")
    time_obs_full = time_obs.values.astype("datetime64[ns]")
    if time_full.size:
        obs_in_time_mask = (time_obs_full >= time_full[0]) & (time_obs_full <= time_full[-1])
    else:
        obs_in_time_mask = np.zeros_like(time_obs_full, dtype=bool)

    split64 = np.datetime64(split.to_datetime64())
    start64 = np.datetime64(start.to_datetime64())

    mask_cal = obs_in_time_mask & (time_obs_full >= start64) & (time_obs_full < split64)
    # Validation goes from split_date to the end of record
    mask_val = obs_in_time_mask & (time_obs_full >= split64)

    # Nearest forcing index for each observation (full forcing window)
    idx_time_obs = np.full(time_obs_full.shape[0], -1, dtype=int)
    if obs_in_time_mask.any():
        idx_time_obs[obs_in_time_mask] = _mk_nearest_idx(time_full, time_obs_full[obs_in_time_mask])

    # split calibration window
    idx = np.where((time >= start) & (time < split))[0]
    time_spl = time[idx]
    hs_spl = hs[idx]
    tp_spl = tp[idx]
    dir_spl = dir_[idx]
    sl_spl = sl[idx]
    hb_spl = hb[idx]
    depthb_spl = depthb[idx]
    wast_spl = wast_[idx]
    E_spl = E[idx]
    P_spl = P[idx]
    Omega_spl = Omega[idx]
    cos_dir_spl = cos_dir[idx]
    sin_dir_spl = sin_dir[idx]
    dt_spl = _mk_dt(time_spl.values)
    if cop_vars:
        tm_spl = tm[idx]
        mslp_spl = mslp[idx]
        sp_spl = sp[idx]
        sst_spl = sst[idx]
    else:
        tm = np.zeros_like(hs)
        mslp = np.zeros_like(hs)
        sp = np.zeros_like(hs)
        sst = np.zeros_like(hs)
        tm_spl = np.zeros_like(hs)
        mslp_spl = np.zeros_like(hs)
        sp_spl = np.zeros_like(hs)
        sst_spl = np.zeros_like(hs)


    # Calibration observations (within forcing window + inside [start,end))
    idx_obs = np.where(mask_cal)[0]
    Obs_spl = Obs[idx_obs]
    time_obs_spl = time_obs[idx_obs]

    idx_obs_spl = _mk_nearest_idx(time_spl.values, time_obs_spl.values)

    return PreprocessResult(
        time=time_full,
        hs=hs,
        tp=tp,
        dir=dir_,
        sl=sl,
        hb=hb,
        depthb=depthb,
        wast=wast_,
        E=E,
        P=P,
        Omega=Omega,
        cos_dir=cos_dir,
        sin_dir=sin_dir,
        tm=tm,
        mslp=mslp,
        sp=sp,
        sst=sst,
        hberm=hberm,
        dt=dt,
        time_obs=time_obs_full,
        Obs=Obs,
        start_date=np.datetime64(start.to_datetime64()),
        end_date=np.datetime64(split.to_datetime64()),
        end_record=(time_full[-1] if time_full.size else np.datetime64("NaT")),
        time_splited=time_spl.values.astype("datetime64[ns]"),
        hs_splited=hs_spl,
        tp_splited=tp_spl,
        dir_splited=dir_spl,
        sl_splited=sl_spl,
        hb_splited=hb_spl,
        depthb_splited=depthb_spl,
        wast_splited=wast_spl,
        E_splited=E_spl,
        P_splited=P_spl,
        Omega_splited=Omega_spl,
        cos_dir_splited=cos_dir_spl,
        sin_dir_splited=sin_dir_spl,
        tm_splitted=tm_spl,
        mslp_splitted=mslp_spl,
        sp_splitted=sp_spl,
        sst_splitted=sst_spl,
        dt_splited=dt_spl,
        time_obs_splited=time_obs_spl.values.astype("datetime64[ns]"),
        Obs_splited=Obs_spl,
        idx_obs=idx_obs.astype(int),
        idx_obs_splited=idx_obs_spl.astype(int),
        obs_in_time_mask=np.asarray(obs_in_time_mask, dtype=bool),
        mask_cal=np.asarray(mask_cal, dtype=bool),
        mask_val=np.asarray(mask_val, dtype=bool),
        idx_time_obs=np.asarray(idx_time_obs, dtype=int),
    )


# ============================================================
# IH-MOOSE 2D legacy preprocess (1D forcings + 2D obs)
# ============================================================

@dataclass(frozen=True)
class PreprocessResultMoose2D:
    """Legacy-style preprocess output for IH-MOOSE Bayesian workflow."""

    # full forcing grid
    time: np.ndarray
    dt: np.ndarray

    # 1D forcings (time,)
    hs: np.ndarray
    tp: np.ndarray
    dir: np.ndarray
    sl: np.ndarray
    E: np.ndarray
    P: np.ndarray
    hb: np.ndarray
    depthb: np.ndarray
    wast: np.ndarray
    Omega: np.ndarray
    cos_dir: np.ndarray
    sin_dir: np.ndarray

    # 2D obs (time_obs, ntrs)
    time_obs: np.ndarray
    Obs: np.ndarray
    mask_nan_obs: np.ndarray

    # optional rotation obs (time_obs,)
    rot: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    mask_nan_rot: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    ref_rot: float | None = None
    alpha0: float | None = None

    # split
    start_date: np.datetime64 = np.datetime64("NaT")
    end_date: np.datetime64 = np.datetime64("NaT")
    end_record: np.datetime64 = np.datetime64("NaT")

    time_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[ns]"))
    dt_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hs_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    tp_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    dir_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sl_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    E_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hb_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    depthb_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    wast_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    Omega_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    P_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    cos_dir_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sin_dir_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    time_obs_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[ns]"))
    Obs_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    mask_nan_obs_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    rot_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    mask_nan_rot_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    idx_obs_splited: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # geometry
    xi: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    yi: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    xf: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    yf: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    phi: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    x_pivotal: float = np.nan
    y_pivotal: float = np.nan
    phi_pivotal: float = np.nan
    ref_transect: int = 0

    ref_i1: int | None = None
    ref_i2: int | None = None
    ref_w1: float | None = None
    ref_w2: float | None = None
    xi_ref: float | None = None
    yi_ref: float | None = None
    xf_ref: float | None = None
    yf_ref: float | None = None

    hberm: float = 0.0


def _circmean_deg(x_deg: np.ndarray, axis: int = 1) -> np.ndarray:
    ang = np.deg2rad(np.asarray(x_deg, dtype=float))
    s = np.nanmean(np.sin(ang), axis=axis)
    c = np.nanmean(np.cos(ang), axis=axis)
    out = np.rad2deg(np.arctan2(s, c))
    return (out + 360.0) % 360.0

def _wrap180_deg(x: np.ndarray | float) -> np.ndarray:
    """Wrap angles to (-180, 180]."""
    a = np.asarray(x, dtype=float)
    out = np.mod(a + 180.0, 360.0) - 180.0
    # Map -180 -> +180 to obtain (-180, 180]
    out = np.where(out <= -180.0, out + 360.0, out)
    return out


def _dist_point_to_segment(px, py, x1, y1, x2, y2):
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    vv = vx * vx + vy * vy
    if vv <= 0.0:
        dx = px - x1
        dy = py - y1
        return (dx * dx + dy * dy) ** 0.5
    t = (wx * vx + wy * vy) / vv
    if t < 0.0:
        cx, cy = x1, y1
    elif t > 1.0:
        cx, cy = x2, y2
    else:
        cx, cy = x1 + t * vx, y1 + t * vy
    dx = px - cx
    dy = py - cy
    return (dx * dx + dy * dy) ** 0.5


def preprocess_legacy_moose2d(
    ds: xr.Dataset,
    start_date: str | np.datetime64 | pd.Timestamp,
    split_date: str | np.datetime64 | pd.Timestamp,
    end_date: str | np.datetime64 | pd.Timestamp | None = None,
    hs_var: str = "hs",
    tp_var: str = "tp",
    dir_var: str = "dir",
    sl_var: str = "sl",
    obs_var: str = "obs",
    mask_obs_var: str = "mask_nan_obs",
    rot_var: str = "rot",
    mask_rot_var: str = "mask_nan_rot",
    time_var: str = "time",
    time_obs_var: str = "time_obs",
    params: Dict[str, any] = {"depth": 100.0, "hberm": 0.0, "alpha_baty": 0.0, "bcoef": 0.55, "break_it": True, "D50": 0.31e-3},
    forcing_1d_kind = "closest_transect",  # or "average_transects"
) -> PreprocessResultMoose2D:
    """Preprocess an Angourie-like 2D shoreline dataset for IH-MOOSE.

    - collapses wave forcings to 1D by averaging over transects (dir uses circular mean)
    - computes E, breaking (hb, depthb), wast, Omega (using IHSetUtils) like MillerDean legacy
    - crops forcing to [start_date, end_record] and builds calibration split [start, split)
    - keeps obs 2D (time_obs, ntrs) and its NaN mask
    - computes idx_obs_splited: mapping time_obs_splited -> nearest time_splited index
    - extracts pivotal + transects and chooses ref_transect as the closest to pivotal
    """
    start = pd.to_datetime(start_date)
    split = pd.to_datetime(split_date)

    time = pd.to_datetime(ds[time_var].values)

    hs2d = np.asarray(ds[hs_var].values, dtype=float)
    tp2d = np.asarray(ds[tp_var].values, dtype=float)
    dir2d = np.asarray(ds[dir_var].values, dtype=float)

    
    # geometry
    xi = np.asarray(ds.get("xi").values, dtype=float) if "xi" in ds else np.array([], dtype=float)
    yi = np.asarray(ds.get("yi").values, dtype=float) if "yi" in ds else np.array([], dtype=float)
    xf = np.asarray(ds.get("xf").values, dtype=float) if "xf" in ds else np.array([], dtype=float)
    yf = np.asarray(ds.get("yf").values, dtype=float) if "yf" in ds else np.array([], dtype=float)
    phi = np.asarray(ds.get("phi").values, dtype=float) if "phi" in ds else np.array([], dtype=float)

    x_piv = float(ds.get("x_pivotal").values) if "x_pivotal" in ds else np.nan
    y_piv = float(ds.get("y_pivotal").values) if "y_pivotal" in ds else np.nan
    phi_piv = float(ds.get("phi_pivotal").values) if "phi_pivotal" in ds else np.nan

    # --- reference transect: interpolate between the two closest transects to pivotal ---
    ref_tr = 0
    ref_i1 = None
    ref_i2 = None
    ref_w1 = None
    ref_w2 = None
    xi_ref = yi_ref = xf_ref = yf_ref = None

    if xi.size and np.isfinite(x_piv) and np.isfinite(y_piv):
        # IHSet reference transect selection:
        # choose the transect whose *infinite line* is closest to the pivotal point.
        # (IHSetMOOSE.ih_moose.find_min_distance)
        dists = np.empty(xi.size, dtype=float)
        for j in range(xi.size):
            A = yi[j] - yf[j]
            B = xf[j] - xi[j]
            C = xi[j] * yf[j] - xf[j] * yi[j]
            den = (A * A + B * B) ** 0.5
            if den <= 0.0:
                den = 1e-12
            dists[j] = abs(A * x_piv + B * y_piv + C) / den

        order = np.argsort(dists)
        i1 = int(order[0])
        i2 = int(order[1]) if order.size > 1 else int(order[0])
        d1 = float(dists[i1])
        d2 = float(dists[i2])

        if (d1 + d2) > 0.0:
            w1 = d2 / (d1 + d2)
            w2 = d1 / (d1 + d2)
        else:
            w1, w2 = 1.0, 0.0

        ref_tr = i1  # keep backward-compatible "closest" index
        ref_i1, ref_i2 = i1, i2
        ref_w1, ref_w2 = float(w1), float(w2)

        # virtual transect endpoints (linear interpolation)
        xi_ref = w1 * xi[i1] + w2 * xi[i2]
        yi_ref = w1 * yi[i1] + w2 * yi[i2]
        xf_ref = w1 * xf[i1] + w2 * xf[i2]
        yf_ref = w1 * yf[i1] + w2 * yf[i2]

    # forcing - must be 1D -> use closest to pivotal transect (or average if no geometry)

    if forcing_1d_kind == "closest_transect":
        hs = hs2d[:, ref_tr]
        tp = tp2d[:, ref_tr]
        dir_ = dir2d[:, ref_tr]
    elif forcing_1d_kind == "average_transects":
        hs = np.nanmean(hs2d, axis=1)
        tp = np.nanmean(tp2d, axis=1)
        dir_ = _circmean_deg(dir2d, axis=1)

    if sl_var in ds:
        sl2d = np.asarray(ds[sl_var].values, dtype=float)
        sl = np.nanmean(sl2d, axis=1)
    else:
        # fallback: try tide+surge+slr if present
        sl = np.zeros_like(hs)
        for k in ("tide", "surge", "slr"):
            if k in ds:
                sl += np.nanmean(np.asarray(ds[k].values, dtype=float), axis=1)

    # homogenize time dimension before computing derived variables
    dt = float(params.get("dt", 24.0))
    hmg = homogenize_time(time[0], time[-1], dt, time_unit="h")
    hs = hmg.go(time, hs)
    tp = hmg.go(time, tp)
    dir_ = hmg.go(time, dir_)
    sl = hmg.go(time, sl)
    time = hmg.regular_grid.astype("datetime64[ns]")


    depth = float(params.get("depth", 100.0))
    depth_arr = np.repeat(depth, hs.shape[0])
    hberm = float(params.get("hberm", 0.0))
    alpha_baty = float(params.get("alpha_baty", 0.0))
    alpha_baty_arr = np.repeat(alpha_baty, hs.shape[0])
    bcoef = float(params.get("bcoef", 0.55))
    break_it = bool(params.get("break_it", True))
    D50 = float(params.get("D50", 0.31e-3))

    E = hs ** 2
    P = E * tp
    ws = wMOORE(D50)
    cos_dir = np.cos(np.radians(dir_))
    sin_dir = np.sin(np.radians(dir_))

    if break_it:
        from IHSetUtils.libjit.waves import BreakingPropagation
        from IHSetUtils.libjit.geometry import nauticalDir2cartesianDir

        dir_cart = nauticalDir2cartesianDir(dir_.copy())
        hb, _, depthb = BreakingPropagation(hs, tp, dir_cart, depth_arr, alpha_baty_arr, bcoef)
        depthb[hb < 0.1] = 0.2
        hb[hb < 0.1] = 0.1
    else:
        hb = hs.copy()
        depthb = depth_arr.copy()

    wast_ = wast(hb, D50)
    Omega = hb / (ws * tp)

    runnup = np.zeros_like(hs)
    slope = deanSlope(depth, D50)
    print(slope)
    for i in range(hs.shape[0]):
        runnup[i] = RU2_Stockdon2006(slope, hs[i], tp[i])

    sl = sl + runnup

    # crop forcing to start_date
    idx0 = np.where(time >= start)[0]
    time = time[idx0]
    hs = hs[idx0]; tp = tp[idx0]; dir_ = dir_[idx0]; sl = sl[idx0]
    hb = hb[idx0]; depthb = depthb[idx0]; wast_ = wast_[idx0]
    E = E[idx0]; Omega = Omega[idx0]; cos_dir = cos_dir[idx0]; sin_dir = sin_dir[idx0]
    P = P[idx0]

    # optional crop end_date
    if end_date is not None:
        end_rec = pd.to_datetime(end_date)
        idx1 = np.where(time <= end_rec)[0]
        if idx1.size:
            time = time[idx1]
            hs = hs[idx1]; tp = tp[idx1]; dir_ = dir_[idx1]; sl = sl[idx1]
            hb = hb[idx1]; depthb = depthb[idx1]; wast_ = wast_[idx1]
            E = E[idx1]; Omega = Omega[idx1]; cos_dir = cos_dir[idx1]; sin_dir = sin_dir[idx1]
            P = P[idx1]

    end_record = np.datetime64(time[-1])

    # dt in hours (robusto para numpy datetime64 o pandas DatetimeIndex)
    time_np = np.asarray(time).astype("datetime64[ns]")
    dt_ns = np.diff(time_np).astype("timedelta64[ns]").astype(np.int64)
    dt = dt_ns / (3600.0 * 1e9)  # hours as float64

    # Safety: replace non-positive/invalid steps (shouldn't happen, but avoids crashes)
    if dt.size and (not np.all(np.isfinite(dt)) or np.any(dt <= 0)):
        good = np.isfinite(dt) & (dt > 0)
        fallback = float(np.median(dt[good])) if np.any(good) else 24.0
        dt = np.where(good, dt, fallback).astype(np.float64)
    else:
        dt = dt.astype(np.float64)

    # obs 2D
    Obs = np.asarray(ds[obs_var].values, dtype=float)
    time_obs = pd.to_datetime(ds[time_obs_var].values)

    if mask_obs_var in ds:
        mask_nan_obs = np.asarray(ds[mask_obs_var].values, dtype=bool)
    else:
        mask_nan_obs = ~np.isfinite(Obs)

    # rotation obs (optional)
    rot = np.array([], dtype=float)
    mask_nan_rot = np.array([], dtype=bool)
    if rot_var in ds:
        rot = np.asarray(ds[rot_var].values, dtype=float)
        if mask_rot_var in ds:
            mask_nan_rot = np.asarray(ds[mask_rot_var].values, dtype=bool)
        else:
            mask_nan_rot = ~np.isfinite(rot)

    # --- ensure numpy datetime64[ns] everywhere (avoid pandas TimedeltaIndex issues)
    time_np = np.asarray(time).astype("datetime64[ns]")
    time_obs_np = np.asarray(time_obs).astype("datetime64[ns]")

    # split calibration forcing grid [start, split)
    split_dt = np.datetime64(split_date)

    mask_forc_cal = time_np < split_dt
    time_spl = time_np[mask_forc_cal]

    hs_spl = hs[mask_forc_cal]
    tp_spl = tp[mask_forc_cal]
    dir_spl = dir_[mask_forc_cal]
    sl_spl = sl[mask_forc_cal]
    hb_spl = hb[mask_forc_cal]
    depthb_spl = depthb[mask_forc_cal]
    wast_spl = wast_[mask_forc_cal]
    E_spl = E[mask_forc_cal]
    P_spl = P[mask_forc_cal]
    Omega_spl = Omega[mask_forc_cal]
    cos_spl = cos_dir[mask_forc_cal]
    sin_spl = sin_dir[mask_forc_cal]

    # dt in hours (robust)
    if time_spl.size > 1:
        dt_ns = np.diff(time_spl).astype("timedelta64[ns]").astype(np.int64)
        dt_spl = (dt_ns / (3600.0 * 1e9)).astype(np.float64)
        # safety
        if np.any(~np.isfinite(dt_spl)) or np.any(dt_spl <= 0):
            good = np.isfinite(dt_spl) & (dt_spl > 0)
            fallback = float(np.median(dt_spl[good])) if np.any(good) else 24.0
            dt_spl = np.where(good, dt_spl, fallback).astype(np.float64)
    else:
        dt_spl = np.array([], dtype=np.float64)

    # observations in calibration window (time_obs in [start, split))
    start_dt = np.datetime64(start)
    mask_cal = (time_obs_np >= start_dt) & (time_obs_np < split_dt)

    time_obs_spl = time_obs_np[mask_cal]
    Obs_spl = Obs[mask_cal, :]
    mask_obs_spl = mask_nan_obs[mask_cal, :]

    # map time_obs_spl -> nearest forcing index in time_spl (fast + robust)
    if time_spl.size == 0:
        idx_obs_spl = np.array([], dtype=int)
    elif time_spl.size == 1:
        idx_obs_spl = np.zeros(time_obs_spl.shape[0], dtype=int)
    else:
        pos = np.searchsorted(time_spl, time_obs_spl, side="left")
        pos = np.clip(pos, 1, time_spl.size - 1)
        left = pos - 1
        right = pos
        choose_left = (time_obs_spl - time_spl[left]) <= (time_spl[right] - time_obs_spl)
        idx_obs_spl = np.where(choose_left, left, right).astype(int)


    # split rotation obs if present
    rot_spl = np.array([], dtype=float)
    mask_rot_spl = np.array([], dtype=bool)
    if rot.size:
        rot_spl = rot[mask_cal]
        mask_rot_spl = mask_nan_rot[mask_cal]
    
    alpha0 = rot_spl[~mask_rot_spl][0] if np.any(~mask_rot_spl) else None

    ref_rot = _wrap180_deg(_circmean_deg(rot, axis=0))

    return PreprocessResultMoose2D(
        time=np.asarray(time, dtype='datetime64[ns]'),
        dt=np.asarray(dt, dtype=float),
        hs=hs, tp=tp, dir=dir_, sl=sl,
        E=E, hb=hb, depthb=depthb, wast=wast_, Omega=Omega, P=P,
        cos_dir=cos_dir, sin_dir=sin_dir,
        time_obs=np.asarray(time_obs, dtype='datetime64[ns]'),
        Obs=Obs,
        mask_nan_obs=mask_nan_obs,
        rot=rot, ref_rot=ref_rot,
        alpha0=alpha0,
        mask_nan_rot=mask_nan_rot,
        start_date=np.datetime64(start),
        end_date=np.datetime64(split_dt),
        end_record=np.datetime64(end_record),
        time_splited=np.asarray(time_spl, dtype='datetime64[ns]'),
        dt_splited=np.asarray(dt_spl, dtype=float),
        hs_splited=hs_spl, tp_splited=tp_spl, dir_splited=dir_spl, sl_splited=sl_spl,
        E_splited=E_spl, hb_splited=hb_spl, depthb_splited=depthb_spl, wast_splited=wast_spl,
        Omega_splited=Omega_spl, P_splited=P_spl, cos_dir_splited=cos_spl, sin_dir_splited=sin_spl,
        time_obs_splited=np.asarray(time_obs_spl, dtype='datetime64[ns]'),
        Obs_splited=Obs_spl,
        mask_nan_obs_splited=mask_obs_spl,
        rot_splited=rot_spl,
        mask_nan_rot_splited=mask_rot_spl,
        idx_obs_splited=idx_obs_spl,
        xi=xi, yi=yi, xf=xf, yf=yf, phi=phi,
        x_pivotal=x_piv, y_pivotal=y_piv, phi_pivotal=phi_piv,
        ref_transect=ref_tr,
        hberm=hberm,
        ref_i1=ref_i1, ref_i2=ref_i2, ref_w1=ref_w1, ref_w2=ref_w2,
        xi_ref=xi_ref, yi_ref=yi_ref, xf_ref=xf_ref, yf_ref=yf_ref,
    )


class homogenize_time():
    """Helper class to homogenize irregular time series to a regular grid using interpolation."""

    def __init__(self, start: np.datetime64, end: np.datetime64, dt: float, time_unit: str = "h"):
        self.start = np.datetime64(start)
        self.end = np.datetime64(end)
        self.dt = float(dt)
        self.time_unit = time_unit

        # build regular grid
        num_points = int(np.ceil((self.end - self.start) / np.timedelta64(int(self.dt * 3600), 's'))) + 1
        self.regular_grid = self.start + np.arange(num_points) * np.timedelta64(int(self.dt * 3600), 's')

    def go(self, time: np.ndarray, values: np.ndarray) -> np.ndarray:
        if time.size == 0:
            return np.array([], dtype=float)

        # interpolate to regular grid
        interp_values = np.interp(self.regular_grid.astype('datetime64[ns]').astype(np.int64),
                                  time.astype('datetime64[ns]').astype(np.int64),
                                  values.astype(float))
        return interp_values