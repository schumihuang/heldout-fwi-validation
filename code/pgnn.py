"""
Physics-Guided Neural Network (PGNN) reparameterization for FWI.

The PGNN replaces the explicit grid-of-velocities parameterization with a
small coordinate-based MLP with two hidden tanh layers and a linear
output:

    m(x, z) = m_0(x, z) + v_scale * tanh( N_theta(x, z) )

where N_theta is a two-hidden-layer MLP with sinusoidal positional encoding. The
reparameterization is trained by L-BFGS jointly minimizing the data misfit
(propagated via the adjoint state) and a physics-guided regularizer:

    L(theta) = L_data(m(theta)) + lambda_phys * R_phys(m(theta))
                                 + lambda_smo * R_smo(m(theta))

R_phys is a fixed-probe residual inspired by the time-harmonic
Helmholtz equation at the reference frequency. In the current
experiments it is a weak auxiliary penalty rather than the dominant
source of the structural effect. R_smo is a Tikhonov gradient
regularizer.

Manual gradients are written out so the implementation is self-contained
(no autodiff framework). Dimensions are intentionally small to keep the
joint optimization tractable.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Positional encoding & MLP
# ----------------------------------------------------------------------

def positional_encoding(coords: np.ndarray, n_freq: int = 4) -> np.ndarray:
    """coords: (N, 2) in [0, 1]. Returns (N, 2 + 4*n_freq)."""
    feats = [coords]
    for k in range(n_freq):
        f = 2.0 ** k * np.pi
        feats.append(np.sin(f * coords))
        feats.append(np.cos(f * coords))
    return np.concatenate(feats, axis=1)


class CoordMLP:
    """Compact two-hidden-layer MLP. Input (N, d_in) -> output (N, 1)."""

    def __init__(self, d_in: int, d_hidden: int = 32, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((d_in, d_hidden)) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.standard_normal((d_hidden, d_hidden)) * np.sqrt(2.0 / d_hidden)
        self.b2 = np.zeros(d_hidden)
        self.W3 = rng.standard_normal((d_hidden, 1)) * 0.05
        self.b3 = np.zeros(1)
        self.d_in = d_in
        self.d_hidden = d_hidden

    # --- parameter packing ---
    def get_params(self) -> np.ndarray:
        return np.concatenate(
            [self.W1.ravel(), self.b1, self.W2.ravel(), self.b2,
             self.W3.ravel(), self.b3]
        )

    def set_params(self, p: np.ndarray) -> None:
        n1 = self.d_in * self.d_hidden
        n2 = self.d_hidden
        n3 = self.d_hidden * self.d_hidden
        n4 = self.d_hidden
        n5 = self.d_hidden * 1
        n6 = 1
        idx = 0
        self.W1 = p[idx: idx + n1].reshape(self.d_in, self.d_hidden); idx += n1
        self.b1 = p[idx: idx + n2]; idx += n2
        self.W2 = p[idx: idx + n3].reshape(self.d_hidden, self.d_hidden); idx += n3
        self.b2 = p[idx: idx + n4]; idx += n4
        self.W3 = p[idx: idx + n5].reshape(self.d_hidden, 1); idx += n5
        self.b3 = p[idx: idx + n6]; idx += n6

    @property
    def n_params(self) -> int:
        return self.get_params().size

    # --- forward / backward ---
    def forward(self, X: np.ndarray) -> np.ndarray:
        h1 = np.tanh(X @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        # cache
        self._cache = (X, h1, h2)
        return out  # (N, 1)

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Given dL/dout (N,1), returns flattened gradient w.r.t. parameters."""
        X, h1, h2 = self._cache
        # output layer
        dW3 = h2.T @ dL_dout
        db3 = dL_dout.sum(axis=0)
        dh2 = dL_dout @ self.W3.T
        # layer 2
        dz2 = dh2 * (1.0 - h2 ** 2)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T
        # layer 1
        dz1 = dh1 * (1.0 - h1 ** 2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2, dW3.ravel(), db3])


# ----------------------------------------------------------------------
# PGNN parameterization wrapper
# ----------------------------------------------------------------------

class PGNNModel:
    """m(x,z) = m_0 + v_scale * tanh( N(p_enc(x,z); theta) )."""

    def __init__(
        self,
        m0: np.ndarray,
        v_scale: float = 1500.0,
        n_freq: int = 4,
        d_hidden: int = 32,
        seed: int = 0,
    ) -> None:
        self.m0 = m0.copy()
        self.v_scale = v_scale
        self.nz, self.nx = m0.shape

        zs = np.linspace(0.0, 1.0, self.nz)
        xs = np.linspace(0.0, 1.0, self.nx)
        Z, X = np.meshgrid(zs, xs, indexing="ij")
        coords = np.stack([Z.ravel(), X.ravel()], axis=1)
        self.coords = coords
        self.coord_feat = positional_encoding(coords, n_freq=n_freq)
        d_in = self.coord_feat.shape[1]
        self.mlp = CoordMLP(d_in=d_in, d_hidden=d_hidden, seed=seed)

    @property
    def n_params(self) -> int:
        return self.mlp.n_params

    def get_params(self) -> np.ndarray:
        return self.mlp.get_params()

    def set_params(self, p: np.ndarray) -> None:
        self.mlp.set_params(p)

    def forward(self) -> np.ndarray:
        raw = self.mlp.forward(self.coord_feat)  # (N,1)
        delta = self.v_scale * np.tanh(raw[:, 0]).reshape(self.nz, self.nx)
        return self.m0 + delta

    def vjp(self, dL_dm: np.ndarray) -> np.ndarray:
        """Pull gradient dL/dm (nz,nx) back to parameter space."""
        # m = m0 + v_scale * tanh(raw)  =>  dm/draw = v_scale * (1 - tanh(raw)^2)
        raw = self.mlp.forward(self.coord_feat)[:, 0]
        # tanh from cache (h-eq tanh forward); recompute outer tanh
        tanh_raw = np.tanh(raw)
        dL_draw = (dL_dm.ravel() * self.v_scale * (1.0 - tanh_raw ** 2))[:, None]
        return self.mlp.backward(dL_draw)
