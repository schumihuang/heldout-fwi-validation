"""L-BFGS drivers."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize

from fwi_core import Acquisition, misfit_and_gradient
from pgnn import PGNNModel
from regularizers import smooth_reg, helmholtz_residual_reg, make_reference_field


def run_fwi_lbfgs(m_init, obs_data, acq, max_iter=25, lambda_smo=1.0e-6):
    history = []
    m_shape = m_init.shape
    t0 = time.time()
    def obj(m_flat):
        m = m_flat.reshape(m_shape)
        J_d, g_d = misfit_and_gradient(m, obs_data, acq)
        J_s, g_s = smooth_reg(m, acq.dx, acq.dz)
        J = J_d + lambda_smo * J_s
        g = g_d + lambda_smo * g_s
        history.append({"J": float(J), "J_data": float(J_d),
                        "J_smo": float(J_s), "elapsed": time.time() - t0})
        return float(J), g.ravel()
    res = minimize(obj, m_init.ravel(), jac=True, method="L-BFGS-B",
                   bounds=[(1500.0, 5500.0)] * m_init.size,
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-12,
                            "maxcor": 12, "disp": False})
    return res.x.reshape(m_shape), history


def pretrain_pgnn(pgnn, target, max_iter=400):
    """Pre-train PGNN parameters so pgnn.forward() matches target."""
    def obj(theta):
        pgnn.set_params(theta)
        m = pgnn.forward()
        diff = m - target
        J = 0.5 * float(np.sum(diff ** 2))
        # gradient via vjp
        g = pgnn.vjp(diff)
        return J, g
    res = minimize(obj, pgnn.get_params(), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-14,
                            "maxcor": 20, "disp": False})
    pgnn.set_params(res.x)
    return res.fun


def run_pgnn_lbfgs(m_init, obs_data, acq, max_iter=25,
                    lambda_smo=1.0e-6, lambda_phys=1.0e-12,
                    v_scale=1200.0, d_hidden=48, n_freq=4, seed=1,
                    pretrain_iter=400):
    pgnn = PGNNModel(m0=m_init, v_scale=v_scale, d_hidden=d_hidden,
                     n_freq=n_freq, seed=seed)
    # Pretrain so PGNN initially reproduces m_init (zero perturbation)
    target_for_pretrain = m_init  # network output should match m_init
    fit_err = pretrain_pgnn(pgnn, target_for_pretrain, max_iter=pretrain_iter)
    init_resid = float(np.sqrt(2.0 * fit_err / m_init.size))
    omega = 2.0 * np.pi * acq.f0
    u_ref = make_reference_field(m_init, acq.dx, acq.dz)
    history = []
    t0 = time.time()
    def obj(theta):
        pgnn.set_params(theta)
        m = pgnn.forward()
        m = np.clip(m, 1500.0, 5500.0)
        J_d, g_d = misfit_and_gradient(m, obs_data, acq)
        J_s, g_s = smooth_reg(m, acq.dx, acq.dz)
        J_p, g_p = helmholtz_residual_reg(m, omega, u_ref, acq.dx, acq.dz)
        J = J_d + lambda_smo * J_s + lambda_phys * J_p
        g_m = g_d + lambda_smo * g_s + lambda_phys * g_p
        g_theta = pgnn.vjp(g_m)
        history.append({"J": float(J), "J_data": float(J_d),
                        "J_smo": float(J_s), "J_phys": float(J_p),
                        "elapsed": time.time() - t0})
        return float(J), g_theta
    theta0 = pgnn.get_params()
    res = minimize(obj, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-12,
                            "maxcor": 12, "disp": False})
    pgnn.set_params(res.x)
    m_final = np.clip(pgnn.forward(), 1500.0, 5500.0)
    return m_final, history, pgnn, init_resid


def model_rmse(m_est, m_true):
    return float(np.sqrt(np.mean((m_est - m_true) ** 2)))


def model_ssim(m_est, m_true):
    from scipy.ndimage import gaussian_filter
    sigma = 1.5
    mu1 = gaussian_filter(m_est, sigma); mu2 = gaussian_filter(m_true, sigma)
    s1 = gaussian_filter(m_est ** 2, sigma) - mu1 ** 2
    s2 = gaussian_filter(m_true ** 2, sigma) - mu2 ** 2
    s12 = gaussian_filter(m_est * m_true, sigma) - mu1 * mu2
    L = float(m_true.max() - m_true.min())
    C1 = (0.01 * L) ** 2; C2 = (0.03 * L) ** 2
    num = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return float(np.mean(num / den))


def model_r2(m_est, m_true):
    ss_res = float(np.sum((m_true - m_est) ** 2))
    ss_tot = float(np.sum((m_true - m_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot
