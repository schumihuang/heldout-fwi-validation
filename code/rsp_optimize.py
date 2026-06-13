"""Residual-split proximal FWI v0 drivers."""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from fwi_core import Acquisition, misfit_and_gradient
from optimize import model_r2, model_rmse, model_ssim, run_fwi_lbfgs
from tv_prox import tv_prox_chambolle, tv_value


def _data_step_residual(c_bg, r, obs_data, acq, max_iter=5):
    """Short L-BFGS-B data step on residual r with c = c_bg + r."""
    shape = c_bg.shape
    t0 = time.time()
    history = []

    def obj(r_flat):
        rr = r_flat.reshape(shape)
        c = np.clip(c_bg + rr, 1500.0, 5500.0)
        J, g_c = misfit_and_gradient(c, obs_data, acq)
        history.append({"J_data": float(J), "elapsed": time.time() - t0})
        return float(J), g_c.ravel()

    bounds = [(1500.0 - c0, 5500.0 - c0) for c0 in c_bg.ravel()]
    res = minimize(
        obj,
        r.ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-12,
                 "maxcor": 12, "disp": False},
    )
    return res.x.reshape(shape), history


def _data_step_model(c, obs_data, acq, max_iter=5):
    """Short L-BFGS-B data step on full model c."""
    v, hist = run_fwi_lbfgs(c, obs_data, acq, max_iter=max_iter, lambda_smo=0.0)
    return v, hist


def run_rsp_fwi(
    c_bg,
    obs_by_band,
    acq_by_band,
    tv_weight=0.0,
    blocks_per_band=2,
    data_iter=5,
    tv_iter=60,
):
    """Residual-only TV-prox FWI: c = c_bg + r."""
    r = np.zeros_like(c_bg)
    history = []
    t0 = time.time()
    for band_idx, (obs, acq) in enumerate(zip(obs_by_band, acq_by_band)):
        for block in range(blocks_per_band):
            r, h = _data_step_residual(c_bg, r, obs, acq, max_iter=data_iter)
            if tv_weight > 0.0:
                r = tv_prox_chambolle(
                    r, weight=tv_weight, dx=acq.dx, dz=acq.dz, n_iter=tv_iter
                )
            c = np.clip(c_bg + r, 1500.0, 5500.0)
            r = c - c_bg
            J, _ = misfit_and_gradient(c, obs, acq)
            history.append({
                "band": band_idx,
                "block": block,
                "J_data": float(J),
                "tv_r": tv_value(r, acq.dx, acq.dz),
                "fevals": len(h),
                "elapsed": time.time() - t0,
            })
    return np.clip(c_bg + r, 1500.0, 5500.0), history


def run_full_model_tv_fwi(
    c_init,
    obs_by_band,
    acq_by_band,
    tv_weight=0.0,
    blocks_per_band=2,
    data_iter=5,
    tv_iter=60,
):
    """Full-model TV-prox baseline: TV is applied to c rather than residual r."""
    c = c_init.copy()
    history = []
    t0 = time.time()
    for band_idx, (obs, acq) in enumerate(zip(obs_by_band, acq_by_band)):
        for block in range(blocks_per_band):
            c, h = _data_step_model(c, obs, acq, max_iter=data_iter)
            if tv_weight > 0.0:
                c = tv_prox_chambolle(
                    c, weight=tv_weight, dx=acq.dx, dz=acq.dz, n_iter=tv_iter
                )
            c = np.clip(c, 1500.0, 5500.0)
            J, _ = misfit_and_gradient(c, obs, acq)
            history.append({
                "band": band_idx,
                "block": block,
                "J_data": float(J),
                "tv_c": tv_value(c, acq.dx, acq.dz),
                "fevals": len(h),
                "elapsed": time.time() - t0,
            })
    return c, history


def run_multiscale_pixel_fwi(c_init, obs_by_band, acq_by_band, max_iter_per_band=10):
    """Sequential frequency-continuation pixel L-BFGS-B baseline."""
    c = c_init.copy()
    history = []
    for band_idx, (obs, acq) in enumerate(zip(obs_by_band, acq_by_band)):
        c, h = run_fwi_lbfgs(c, obs, acq, max_iter=max_iter_per_band, lambda_smo=1e-8)
        for item in h:
            row = dict(item)
            row["band"] = band_idx
            history.append(row)
    return c, history


def summarize_model(v, true):
    return {
        "rmse": model_rmse(v, true),
        "ssim": model_ssim(v, true),
        "r2": model_r2(v, true),
    }
