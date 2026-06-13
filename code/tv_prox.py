"""Deterministic isotropic TV proximal utilities."""
from __future__ import annotations

import numpy as np


def _grad(u: np.ndarray, dx: float, dz: float) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(u)
    gz = np.zeros_like(u)
    gx[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx
    gz[:-1, :] = (u[1:, :] - u[:-1, :]) / dz
    return gx, gz


def _div(px: np.ndarray, pz: np.ndarray, dx: float, dz: float) -> np.ndarray:
    out = np.zeros_like(px)
    out[:, :-1] -= px[:, :-1] / dx
    out[:, 1:] += px[:, :-1] / dx
    out[:-1, :] -= pz[:-1, :] / dz
    out[1:, :] += pz[:-1, :] / dz
    return out


def tv_value(u: np.ndarray, dx: float = 1.0, dz: float = 1.0) -> float:
    """Isotropic 2-D TV value with physical grid spacing."""
    gx, gz = _grad(u, dx, dz)
    return float(np.sum(np.sqrt(gx * gx + gz * gz + 1e-12)) * dx * dz)


def tv_prox_chambolle(
    y: np.ndarray,
    weight: float,
    dx: float = 1.0,
    dz: float = 1.0,
    n_iter: int = 50,
) -> np.ndarray:
    """Approximate prox_{weight TV}(y) with Chambolle's projection method.

    This solves approximately:
        argmin_x 0.5 ||x - y||_2^2 + weight * TV(x)

    The implementation is intentionally small and deterministic for CPU
    reproducibility. `weight` has the same units as the residual values.
    """
    if weight <= 0.0:
        return y.copy()
    px = np.zeros_like(y)
    pz = np.zeros_like(y)
    tau = 0.125
    for _ in range(n_iter):
        divp = _div(px, pz, dx, dz)
        u = y - weight * divp
        gx, gz = _grad(u, dx, dz)
        px_new = px + (tau / weight) * gx
        pz_new = pz + (tau / weight) * gz
        norm = np.maximum(1.0, np.sqrt(px_new * px_new + pz_new * pz_new))
        px = px_new / norm
        pz = pz_new / norm
    return y - weight * _div(px, pz, dx, dz)


def edge_preserving_weights(
    reference: np.ndarray,
    dx: float = 1.0,
    dz: float = 1.0,
    alpha: float = 8.0,
    floor: float = 0.15,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Return TV weights that relax smoothing near existing structure."""
    from scipy.ndimage import gaussian_filter

    gz, gx = np.gradient(reference, dz, dx)
    mag = np.sqrt(gx * gx + gz * gz)
    if smooth_sigma > 0.0:
        mag = gaussian_filter(mag, sigma=smooth_sigma)
    scale = np.percentile(mag, 95.0)
    if not np.isfinite(scale) or scale <= 0.0:
        return np.ones_like(reference)
    norm = np.clip(mag / scale, 0.0, 1.0)
    weights = floor + (1.0 - floor) / (1.0 + alpha * norm)
    return weights.astype(np.float64)


def weighted_tv_prox_chambolle(
    y: np.ndarray,
    weight: float,
    local_weights: np.ndarray,
    dx: float = 1.0,
    dz: float = 1.0,
    n_iter: int = 50,
) -> np.ndarray:
    """Approximate prox for a spatially weighted isotropic TV penalty.

    This solves approximately:
        argmin_x 0.5 ||x - y||_2^2 + weight * sum_i w_i |grad x_i|

    `local_weights` should be small near edges to preserve discontinuities
    and larger in smooth regions. With all weights equal to one, this is
    equivalent to `tv_prox_chambolle` up to numerical iteration effects.
    """
    if weight <= 0.0:
        return y.copy()
    if local_weights.shape != y.shape:
        raise ValueError("local_weights must have the same shape as y")

    radius = np.maximum(weight * local_weights, 1e-12)
    qx = np.zeros_like(y)
    qz = np.zeros_like(y)
    tau = 0.125
    for _ in range(n_iter):
        divq = _div(qx, qz, dx, dz)
        u = y - divq
        gx, gz = _grad(u, dx, dz)
        qx_new = qx + tau * gx
        qz_new = qz + tau * gz
        norm = np.maximum(radius, np.sqrt(qx_new * qx_new + qz_new * qz_new))
        qx = qx_new * radius / norm
        qz = qz_new * radius / norm
    return y - _div(qx, qz, dx, dz)
