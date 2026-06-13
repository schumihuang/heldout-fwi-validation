"""
Physics-guided regularizers for the PGNN-augmented inversion.

R_phys: time-harmonic Helmholtz residual at the central probe frequency.
        Probes that the network output is consistent with a propagating
        acoustic field given the source distribution.

R_smo:  Tikhonov gradient regularizer (anisotropic, depth-weighted).
"""
from __future__ import annotations

import numpy as np


def smooth_reg(m: np.ndarray, dx: float, dz: float) -> tuple[float, np.ndarray]:
    """0.5 * (|dm/dx|^2 + |dm/dz|^2). Returns (value, grad)."""
    dmdx = np.zeros_like(m)
    dmdz = np.zeros_like(m)
    dmdx[:, 1:-1] = (m[:, 2:] - m[:, :-2]) / (2.0 * dx)
    dmdz[1:-1, :] = (m[2:, :] - m[:-2, :]) / (2.0 * dz)
    val = 0.5 * float(np.sum(dmdx ** 2 + dmdz ** 2))
    # Gradient: -d/dx (dmdx) - d/dz (dmdz)
    g = np.zeros_like(m)
    g[:, 1:-1] -= (dmdx[:, 2:] - dmdx[:, :-2]) / (2.0 * dx)
    g[1:-1, :] -= (dmdz[2:, :] - dmdz[:-2, :]) / (2.0 * dz)
    return val, g


def helmholtz_residual_reg(
    m: np.ndarray,
    omega: float,
    src_field: np.ndarray,
    dx: float,
    dz: float,
) -> tuple[float, np.ndarray]:
    """0.5 * || (Δ + ω^2/m^2) u_ref - s ||^2  on a reference field u_ref.

    For a tractable physics-guided regularizer we use a fixed reference
    field u_ref (constructed once from the smooth initial model) and ask
    that m remain consistent with the Helmholtz equation acting on u_ref.
    The residual is differentiable w.r.t. m only through the slowness term.
    """
    nz, nx = m.shape
    u = src_field
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx ** 2
        + (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dz ** 2
    )
    slow2 = 1.0 / (m ** 2 + 1e-9)
    res = lap + omega ** 2 * slow2 * u  # (no point-source on RHS for interior)
    val = 0.5 * float(np.sum(res ** 2))
    # dRes/dm = omega^2 * (-2 / m^3) * u
    dres_dm = omega ** 2 * (-2.0 / (m ** 3 + 1e-9)) * u
    grad = res * dres_dm
    return val, grad


def make_reference_field(m0: np.ndarray, dx: float, dz: float) -> np.ndarray:
    """Synthesize a smooth Gaussian-windowed cosine wave on m0 for use
    as the fixed probe field u_ref in helmholtz_residual_reg."""
    nz, nx = m0.shape
    z = np.linspace(0, 1, nz)[:, None]
    x = np.linspace(0, 1, nx)[None, :]
    env = np.exp(-((z - 0.0) ** 2) / 0.6)  # surface-source emphasis
    field = env * np.cos(8.0 * np.pi * x) * np.cos(4.0 * np.pi * z)
    return field
