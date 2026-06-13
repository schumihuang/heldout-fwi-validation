"""Noisy calibrated cross-grid held-out shot validation."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from fwi_core import marmousi2_real_subset, smooth_initial_model
from optimize import run_fwi_lbfgs
from run_sgh_crossgrid_calibrated import calibrate_obs_to_initial, make_crossgrid_band_data
from run_validation_shot_split_crossgrid import split_shots, total_misfit
from sgh_optimize import summarize_model_extended


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results")


def add_noise(obs_by_band, snr_db, seed):
    rng = np.random.default_rng(seed)
    noisy = []
    for band in obs_by_band:
        noisy_band = []
        for shot in band:
            rms = float(np.sqrt(np.mean(shot * shot)))
            sigma = rms / (10.0 ** (snr_db / 20.0))
            noisy_band.append(shot + rng.normal(scale=sigma, size=shot.shape))
        noisy.append(noisy_band)
    return noisy


def run_noise_setting(snr_db, seed, v_init, v_true, dx, clean_obs, acq_full):
    print(f"\n=== noisy SNR={snr_db:g} dB ===", flush=True)
    obs_by_band = add_noise(clean_obs, snr_db=snr_db, seed=seed)
    train_idx = np.array([0, 2, 4])
    val_idx = np.array([1, 3, 5])
    obs_train, acq_train, obs_val, acq_val = split_shots(
        obs_by_band, acq_full, train_idx, val_idx
    )

    t0 = time.time()
    c = v_init.copy()
    warmup = []
    print("init", summarize_model_extended(c, v_true, dx, dx), flush=True)
    for band_idx, (obs, acq) in enumerate(zip(obs_train, acq_train)):
        c, h = run_fwi_lbfgs(c, obs, acq, max_iter=5, lambda_smo=1e-8)
        row = {
            "band": int(band_idx),
            "history_len": len(h),
            "train_misfit": total_misfit(c, obs_train, acq_train),
            "val_misfit": total_misfit(c, obs_val, acq_val),
            **summarize_model_extended(c, v_true, dx, dx),
            "elapsed": time.time() - t0,
        }
        warmup.append(row)
        print("warmup", row, flush=True)

    best = {
        "chunk": -1,
        "train_misfit": total_misfit(c, obs_train, acq_train),
        "val_misfit": total_misfit(c, obs_val, acq_val),
        **summarize_model_extended(c, v_true, dx, dx),
    }
    best_model = c.copy()
    chunks = []
    for chunk in range(8):
        c, h = run_fwi_lbfgs(c, obs_train[-1], acq_train[-1], max_iter=5, lambda_smo=1e-8)
        row = {
            "chunk": int(chunk),
            "history_len": len(h),
            "train_misfit": total_misfit(c, obs_train, acq_train),
            "val_misfit": total_misfit(c, obs_val, acq_val),
            **summarize_model_extended(c, v_true, dx, dx),
            "elapsed": time.time() - t0,
        }
        chunks.append(row)
        if row["val_misfit"] < best["val_misfit"]:
            best = {k: row[k] for k in row if k != "history_len"}
            best_model = c.copy()
        print("chunk", row, flush=True)

    final = {
        "chunk": 7,
        "train_misfit": total_misfit(c, obs_train, acq_train),
        "val_misfit": total_misfit(c, obs_val, acq_val),
        **summarize_model_extended(c, v_true, dx, dx),
        "elapsed": time.time() - t0,
    }
    result = {
        "snr_db": float(snr_db),
        "seed": int(seed),
        "warmup": warmup,
        "chunks": chunks,
        "best_by_validation": best,
        "final_pixel": final,
    }
    print("best_by_validation", best, flush=True)
    print("final_pixel", final, flush=True)
    return result, best_model, c


def main():
    np.random.seed(0)
    os.makedirs(OUT, exist_ok=True)
    v_true, dx_coarse, meta_coarse = marmousi2_real_subset(nx=100, nz=36, step=24)
    v_fine, dx_fine, meta_fine = marmousi2_real_subset(nx=200, nz=72, step=12)
    v_init = smooth_initial_model(v_true, sigma=8.0)
    raw_obs, acq_full = make_crossgrid_band_data(
        v_fine, v_true, dx_fine, dx_coarse,
        f0_values=(2.0, 3.0, 4.0), n_src=6
    )
    clean_obs, scales = calibrate_obs_to_initial(v_init, raw_obs, acq_full)

    results = {
        "dataset": "Marmousi2 true VP subset",
        "experiment": "noisy calibrated cross-grid held-out shot validation",
        "crop": {"x0_m": 6500.0, "z0_m": 700.0, "sigma": 8.0},
        "coarse_meta": meta_coarse,
        "fine_meta": meta_fine,
        "dx_coarse": float(dx_coarse),
        "dx_fine": float(dx_fine),
        "calibration_scales": scales,
        "settings": [],
    }
    models = {
        "true_coarse": v_true,
        "true_fine": v_fine,
        "init_sigma8": v_init,
    }
    for snr_db, seed in [(20.0, 2020), (10.0, 1010)]:
        result, best_model, final_model = run_noise_setting(
            snr_db, seed, v_init, v_true, dx_coarse, clean_obs, acq_full
        )
        results["settings"].append(result)
        models[f"snr{int(snr_db)}_best_by_validation"] = best_model
        models[f"snr{int(snr_db)}_final_pixel"] = final_model

    with open(os.path.join(OUT, "validation_noisy_crossgrid_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(OUT, "validation_noisy_crossgrid_models.npz"), **models)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
