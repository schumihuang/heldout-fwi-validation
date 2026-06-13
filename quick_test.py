"""Quick repository smoke test for the validation-controlled FWI code."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from tv_prox import tv_prox_chambolle, tv_value  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(20260613)
    z = np.linspace(0.0, 1.0, 32)
    x = np.linspace(0.0, 1.0, 40)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    model = 2.0 + 0.25 * np.sin(2.0 * np.pi * xx) + 0.15 * (zz > 0.55)
    noisy = model + 0.04 * rng.standard_normal(model.shape)

    before = tv_value(noisy)
    denoised = tv_prox_chambolle(noisy, weight=0.08, n_iter=25)
    after = tv_value(denoised)

    if denoised.shape != noisy.shape:
        raise AssertionError("TV prox changed the array shape")
    if not np.all(np.isfinite(denoised)):
        raise AssertionError("TV prox produced non-finite values")
    if not after < before:
        raise AssertionError(f"TV value did not decrease: before={before}, after={after}")
    if abs(float(denoised.mean() - noisy.mean())) > 1e-10:
        raise AssertionError("TV prox should preserve the image mean for this test")

    print(f"TV before: {before:.6f}")
    print(f"TV after:  {after:.6f}")
    print("quick_test passed")


if __name__ == "__main__":
    main()
