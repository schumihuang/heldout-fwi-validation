"""Core 2D acoustic FWI utilities (numpy + scipy only).

Forward solver uses an explicit padded grid. Inside the computational box,
the velocity field is padded with an absorbing layer of low-frequency
sponge damping that is applied identically in forward and adjoint passes.
The bundled gradient tests are finite-difference sanity checks rather than
a proof of exact discrete adjointness.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

NPAD = 12  # absorbing layer thickness (cells)


def marmousi_subset(nx=200, nz=80, seed=0):
    rng = np.random.default_rng(seed)
    z = np.linspace(0, 1, nz); x = np.linspace(0, 1, nx)
    Z, X = np.meshgrid(z, x, indexing="ij")
    v = 1500.0 + 3000.0 * Z
    layers = [(0.20,0.04,250.),(0.32,0.04,-200.),(0.45,0.05,350.),
              (0.60,0.05,-150.),(0.75,0.05,300.)]
    for z0, sigma, dv in layers:
        v += dv * np.exp(-((Z - z0) ** 2) / (2 * sigma ** 2))
    fault = (0.55 - 0.35 * X)
    wedge = np.exp(-((Z - fault) ** 2) / (2 * 0.04 ** 2)) * (X > 0.25) * (X < 0.75)
    v += 600.0 * wedge
    salt_r2 = ((Z - 0.78) / 0.10) ** 2 + ((X - 0.70) / 0.18) ** 2
    v += 800.0 * np.exp(-salt_r2)
    noise = gaussian_filter(rng.standard_normal(v.shape), sigma=2.0)
    v += 60.0 * noise
    return np.clip(v, 1500.0, 5500.0).astype(np.float64)


def marmousi2_real_subset(
    nx=100,
    nz=36,
    step=24,
    x0_m=6500.0,
    z0_m=700.0,
    native_dx=1.249,
    npy_path=None,
):
    """Load a true Marmousi2 VP subset from a cached SEGY-derived .npy file.

    The full Marmousi2 VP grid is expected in m/s with shape (depth, x).
    Defaults select a water-free, structurally complex compact patch with
    approximately 30 m grid spacing.
    """
    import os

    if npy_path is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        npy_path = os.path.join(root, "data", "marmousi2", "vp_marmousi2.npy")
    full = np.load(npy_path, mmap_mode="r")
    z0 = int(round(z0_m / native_dx))
    x0 = int(round(x0_m / native_dx))
    z1 = z0 + nz * step
    x1 = x0 + nx * step
    if z1 > full.shape[0] or x1 > full.shape[1]:
        raise ValueError("Requested Marmousi2 subset exceeds full model bounds")
    subset = np.array(full[z0:z1:step, x0:x1:step], dtype=np.float64)
    return subset, float(native_dx * step), dict(
        source="Marmousi2 VP SEGY",
        native_dx_m=float(native_dx),
        step=int(step),
        x0_m=float(x0_m),
        z0_m=float(z0_m),
        full_shape=[int(full.shape[0]), int(full.shape[1])],
    )


def smooth_initial_model(v_true, sigma=12.0):
    return gaussian_filter(v_true, sigma=sigma)


def ricker(nt, dt, f0, t0=None):
    t = np.arange(nt) * dt
    if t0 is None:
        t0 = 1.2 / f0
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _laplacian(p, inv_dx2, inv_dz2):
    lap = np.zeros_like(p)
    lap[1:-1, 1:-1] = (
        (p[1:-1, 2:] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2]) * inv_dx2
        + (p[2:, 1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2, 1:-1]) * inv_dz2)
    return lap


def _sponge(nz, nx, npad=NPAD, sigma=0.005):
    """Outer cells: damp by exp(-sigma * (npad-d)^2). Identical fwd/adj."""
    if npad <= 0:
        return np.ones((nz, nx))
    d = np.zeros((nz, nx))
    for k in range(npad):
        w = (npad - k) ** 2
        d[k, :] = np.maximum(d[k, :], w)
        d[-1 - k, :] = np.maximum(d[-1 - k, :], w)
        d[:, k] = np.maximum(d[:, k], w)
        d[:, -1 - k] = np.maximum(d[:, -1 - k], w)
    return np.exp(-sigma * d)


def _pad_velocity(v, npad=NPAD):
    return np.pad(v, npad, mode="edge")


def _unpad(g, npad=NPAD):
    return g[npad:-npad, npad:-npad] if npad > 0 else g


def forward_shot(v, src_pos, rec_z, src_wav, dt, dx, dz,
                  save_wavefield=True, npad=NPAD):
    v_p = _pad_velocity(v, npad)
    nz_p, nx_p = v_p.shape
    nt = src_wav.size
    inv_dx2 = 1.0 / dx ** 2
    inv_dz2 = 1.0 / dz ** 2
    sp = _sponge(nz_p, nx_p, npad=npad)
    c2dt2 = (v_p ** 2) * dt ** 2
    p_prev = np.zeros_like(v_p)
    p_curr = np.zeros_like(v_p)
    wave = np.zeros((nt, nz_p, nx_p)) if save_wavefield else None
    seis = np.zeros((nt, v.shape[1]))
    sz, sx = src_pos
    sz_p, sx_p = sz + npad, sx + npad
    rec_z_p = rec_z + npad
    for it in range(nt):
        lap = _laplacian(p_curr, inv_dx2, inv_dz2)
        p_next = 2.0 * p_curr - p_prev + c2dt2 * lap
        p_next[sz_p, sx_p] += c2dt2[sz_p, sx_p] * src_wav[it]
        p_next *= sp
        p_prev, p_curr = p_curr, p_next
        seis[it, :] = p_curr[rec_z_p, npad:nx_p-npad]
        if save_wavefield:
            wave[it] = p_curr
    return seis, wave


def adjoint_shot(v, residual, rec_z, dt, dx, dz, fwd_wave, npad=NPAD):
    v_p = _pad_velocity(v, npad)
    nz_p, nx_p = v_p.shape
    nt = residual.shape[0]
    inv_dx2 = 1.0 / dx ** 2
    inv_dz2 = 1.0 / dz ** 2
    sp = _sponge(nz_p, nx_p, npad=npad)
    c2dt2 = (v_p ** 2) * dt ** 2
    a_prev = np.zeros_like(v_p)
    a_curr = np.zeros_like(v_p)
    grad_p = np.zeros_like(v_p)
    rec_z_p = rec_z + npad
    nx_inner = v.shape[1]
    for it in range(nt - 1, -1, -1):
        lap = _laplacian(a_curr, inv_dx2, inv_dz2)
        a_next = 2.0 * a_curr - a_prev + c2dt2 * lap
        # adjoint of "seis = p[rec_z_p, npad:nx-npad]" is injecting residual at
        # the same locations
        a_next[rec_z_p, npad:npad+nx_inner] += c2dt2[rec_z_p, npad:npad+nx_inner] * residual[it, :]
        a_next *= sp
        a_prev, a_curr = a_curr, a_next
        if 1 <= it <= nt - 2:
            d2p = (fwd_wave[it + 1] - 2.0 * fwd_wave[it] + fwd_wave[it - 1]) / dt ** 2
            grad_p += (2.0 / (v_p ** 3)) * d2p * a_curr * dt
    return _unpad(grad_p, npad)


class Acquisition:
    def __init__(self, nz, nx, n_src=8, src_z=2, rec_z=2, f0=5.0,
                 nt=1500, dt=0.0015, dx=25.0, dz=25.0,
                 rec_mask=None):
        self.nz, self.nx = nz, nx
        self.n_src = n_src
        self.src_z = src_z; self.rec_z = rec_z
        self.f0 = f0; self.nt = nt; self.dt = dt
        self.dx = dx; self.dz = dz
        self.src_x = np.linspace(int(0.10 * nx), int(0.90 * nx), n_src).astype(int)
        self.wavelet = ricker(nt, dt, f0)
        if rec_mask is None:
            self.rec_mask = None
        else:
            rec_mask = np.asarray(rec_mask, dtype=np.float64)
            if rec_mask.shape != (nx,):
                raise ValueError("rec_mask must have shape (nx,)")
            self.rec_mask = rec_mask


def synthesize_observed(v_true, acq):
    data = []
    for sx in acq.src_x:
        seis, _ = forward_shot(v_true, (acq.src_z, sx), acq.rec_z,
                                acq.wavelet, acq.dt, acq.dx, acq.dz, save_wavefield=False)
        data.append(seis)
    return data


def misfit_and_gradient(v, obs_data, acq):
    total_J = 0.0
    total_grad = np.zeros_like(v)
    rec_mask = getattr(acq, "rec_mask", None)
    for sx, dobs in zip(acq.src_x, obs_data):
        seis, wave = forward_shot(v, (acq.src_z, sx), acq.rec_z, acq.wavelet,
                                    acq.dt, acq.dx, acq.dz, save_wavefield=True)
        residual = seis - dobs
        if rec_mask is not None:
            residual = residual * rec_mask[None, :]
        total_J += 0.5 * float(np.sum(residual ** 2)) * acq.dt
        g = adjoint_shot(v, residual, acq.rec_z, acq.dt, acq.dx, acq.dz, wave)
        total_grad += g
    return total_J, total_grad
