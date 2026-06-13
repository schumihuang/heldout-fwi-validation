"""Structure-gated hybrid proximal FWI utilities.

This module implements the structure-gated residual split optimization
utilities used in the Marmousi2 cross-grid validation experiments. It keeps
the background model fixed and optimizes a gated pixel residual:

    c = c_bg + g * r

where the structure gate g is derived from an early FWI gradient, not from
the true model. A TV proximal step is applied to the effective gated
residual, which is the simplest deterministic proxy for the proposed
structure-aware edge-preserving residual regularization.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

from fwi_core import misfit_and_gradient
from optimize import model_r2, model_rmse, model_ssim
from tv_prox import tv_prox_chambolle, tv_value, weighted_tv_prox_chambolle


def gradient_structure_gate(
    c_init,
    obs_data,
    acq,
    smooth_sigma=1.5,
    quantile=0.65,
    sharpness=8.0,
    floor=0.05,
):
    """Build a deterministic structure gate from early FWI gradient energy."""
    _, grad = misfit_and_gradient(c_init, obs_data, acq)
    energy = gaussian_filter(np.abs(grad), sigma=smooth_sigma)
    scale = np.percentile(energy, 95.0)
    if not np.isfinite(scale) or scale <= 0.0:
        return np.ones_like(c_init), energy
    norm = np.clip(energy / scale, 0.0, 1.0)
    tau = float(np.quantile(norm, quantile))
    gate = 1.0 / (1.0 + np.exp(-sharpness * (norm - tau)))
    gate = floor + (1.0 - floor) * gate
    return gate.astype(np.float64), energy.astype(np.float64)


def _data_step_gated_residual(c_bg, r, gate, obs_data, acq, max_iter=5):
    """Short L-BFGS-B step on r with c = c_bg + gate * r."""
    shape = c_bg.shape
    t0 = time.time()
    history = []

    def obj(r_flat):
        rr = r_flat.reshape(shape)
        c = np.clip(c_bg + gate * rr, 1500.0, 5500.0)
        J, g_c = misfit_and_gradient(c, obs_data, acq)
        g_r = gate * g_c
        history.append({"J_data": float(J), "elapsed": time.time() - t0})
        return float(J), g_r.ravel()

    bounds = [(-2500.0, 2500.0)] * r.size
    res = minimize(
        obj,
        r.ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": max_iter,
            "ftol": 1e-12,
            "gtol": 1e-12,
            "maxcor": 12,
            "disp": False,
        },
    )
    return res.x.reshape(shape), history


def run_sgh_vsp_fwi(
    c_bg,
    obs_by_band,
    acq_by_band,
    gate=None,
    gate_from_band=0,
    tv_weight=0.0,
    edge_weights=None,
    blocks_per_band=2,
    data_iter=5,
    tv_iter=60,
):
    """Run structure-gated residual split proximal FWI."""
    if gate is None:
        gate, _ = gradient_structure_gate(
            c_bg, obs_by_band[gate_from_band], acq_by_band[gate_from_band]
        )

    r = np.zeros_like(c_bg)
    history = []
    t0 = time.time()
    for band_idx, (obs, acq) in enumerate(zip(obs_by_band, acq_by_band)):
        for block in range(blocks_per_band):
            r, h = _data_step_gated_residual(
                c_bg, r, gate, obs, acq, max_iter=data_iter
            )
            if tv_weight > 0.0:
                effective = gate * r
                if edge_weights is None:
                    effective = tv_prox_chambolle(
                        effective, weight=tv_weight, dx=acq.dx, dz=acq.dz,
                        n_iter=tv_iter
                    )
                else:
                    effective = weighted_tv_prox_chambolle(
                        effective, weight=tv_weight, local_weights=edge_weights,
                        dx=acq.dx, dz=acq.dz, n_iter=tv_iter
                    )
                r = effective / np.maximum(gate, 1e-3)

            c = np.clip(c_bg + gate * r, 1500.0, 5500.0)
            r = (c - c_bg) / np.maximum(gate, 1e-3)
            J, _ = misfit_and_gradient(c, obs, acq)
            history.append({
                "band": band_idx,
                "block": block,
                "J_data": float(J),
                "tv_gated_r": tv_value(gate * r, acq.dx, acq.dz),
                "fevals": len(h),
                "elapsed": time.time() - t0,
            })
    return np.clip(c_bg + gate * r, 1500.0, 5500.0), history, gate


def edge_correlation(v_est, v_true, dx=1.0, dz=1.0):
    """Correlation between velocity-gradient magnitudes."""
    gz_e, gx_e = np.gradient(v_est, dz, dx)
    gz_t, gx_t = np.gradient(v_true, dz, dx)
    mag_e = np.sqrt(gx_e * gx_e + gz_e * gz_e).ravel()
    mag_t = np.sqrt(gx_t * gx_t + gz_t * gz_t).ravel()
    if np.std(mag_e) <= 0.0 or np.std(mag_t) <= 0.0:
        return 0.0
    return float(np.corrcoef(mag_e, mag_t)[0, 1])


def summarize_model_extended(v, true, dx=1.0, dz=1.0):
    return {
        "rmse": model_rmse(v, true),
        "mae": float(np.mean(np.abs(v - true))),
        "ssim": model_ssim(v, true),
        "r2": model_r2(v, true),
        "edge_corr": edge_correlation(v, true, dx=dx, dz=dz),
    }
