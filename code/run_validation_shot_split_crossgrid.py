"""Shot-held-out validation for calibrated cross-grid Marmousi2 FWI."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from fwi_core import Acquisition, marmousi2_real_subset, misfit_and_gradient, smooth_initial_model
from optimize import run_fwi_lbfgs
from run_sgh_crossgrid_calibrated import calibrate_obs_to_initial, make_crossgrid_band_data
from sgh_optimize import gradient_structure_gate, run_sgh_vsp_fwi, summarize_model_extended
from tv_prox import edge_preserving_weights


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results")


def acq_with_src_subset(acq, src_x):
    out = Acquisition(
        nz=acq.nz, nx=acq.nx, n_src=len(src_x),
        src_z=acq.src_z, rec_z=acq.rec_z, f0=acq.f0,
        nt=acq.nt, dt=acq.dt, dx=acq.dx, dz=acq.dz,
        rec_mask=acq.rec_mask,
    )
    out.src_x = np.array(src_x, dtype=int)
    return out


def split_shots(obs_by_band, acq_by_band, train_idx, val_idx):
    obs_train = []
    obs_val = []
    acq_train = []
    acq_val = []
    for obs, acq in zip(obs_by_band, acq_by_band):
        obs_train.append([obs[i] for i in train_idx])
        obs_val.append([obs[i] for i in val_idx])
        acq_train.append(acq_with_src_subset(acq, acq.src_x[train_idx]))
        acq_val.append(acq_with_src_subset(acq, acq.src_x[val_idx]))
    return obs_train, acq_train, obs_val, acq_val


def total_misfit(v, obs_by_band, acq_by_band):
    total = 0.0
    for obs, acq in zip(obs_by_band, acq_by_band):
        J, _ = misfit_and_gradient(v, obs, acq)
        total += J
    return float(total)


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
    obs_by_band, scales = calibrate_obs_to_initial(v_init, raw_obs, acq_full)

    train_idx = np.array([0, 2, 4])
    val_idx = np.array([1, 3, 5])
    obs_train, acq_train, obs_val, acq_val = split_shots(
        obs_by_band, acq_full, train_idx, val_idx
    )

    results = {
        "dataset": "Marmousi2 true VP subset",
        "experiment": "shot-held-out calibrated cross-grid validation",
        "coarse_meta": meta_coarse,
        "fine_meta": meta_fine,
        "dx_coarse": float(dx_coarse),
        "dx_fine": float(dx_fine),
        "train_src_x": [int(x) for x in acq_train[0].src_x],
        "validation_src_x": [int(x) for x in acq_val[0].src_x],
        "calibration_scales": scales,
        "init": summarize_model_extended(v_init, v_true, dx_coarse, dx_coarse),
        "warmup": [],
        "chunks": [],
        "sgh_candidates": [],
    }
    models = {
        "true_coarse": v_true,
        "true_fine": v_fine,
        "init_sigma8": v_init,
    }

    t0 = time.time()
    c = v_init.copy()
    print("init", results["init"], flush=True)
    print("train src", results["train_src_x"], "val src", results["validation_src_x"], flush=True)

    for band_idx, (obs, acq) in enumerate(zip(obs_train, acq_train)):
        c, h = run_fwi_lbfgs(c, obs, acq, max_iter=5, lambda_smo=1e-8)
        row = {
            "band": int(band_idx),
            "history_len": len(h),
            "train_misfit": total_misfit(c, obs_train, acq_train),
            "val_misfit": total_misfit(c, obs_val, acq_val),
            **summarize_model_extended(c, v_true, dx_coarse, dx_coarse),
            "elapsed": time.time() - t0,
        }
        results["warmup"].append(row)
        print("warmup", row, flush=True)
    models["warmup_final"] = c

    best = {
        "chunk": -1,
        "train_misfit": total_misfit(c, obs_train, acq_train),
        "val_misfit": total_misfit(c, obs_val, acq_val),
        **summarize_model_extended(c, v_true, dx_coarse, dx_coarse),
    }
    best_model = c.copy()
    for chunk in range(8):
        c, h = run_fwi_lbfgs(c, obs_train[-1], acq_train[-1], max_iter=5, lambda_smo=1e-8)
        row = {
            "chunk": int(chunk),
            "history_len": len(h),
            "train_misfit": total_misfit(c, obs_train, acq_train),
            "val_misfit": total_misfit(c, obs_val, acq_val),
            **summarize_model_extended(c, v_true, dx_coarse, dx_coarse),
            "elapsed": time.time() - t0,
        }
        results["chunks"].append(row)
        if row["val_misfit"] < best["val_misfit"]:
            best = {k: row[k] for k in row if k != "history_len"}
            best_model = c.copy()
        print("chunk", row, flush=True)

    # Test whether SGH can improve the validation-selected pixel model.
    gate, _ = gradient_structure_gate(
        best_model, obs_train[0], acq_train[0],
        smooth_sigma=1.5, quantile=0.65, sharpness=8.0, floor=0.9
    )
    edge_w = edge_preserving_weights(
        best_model, dx=dx_coarse, dz=dx_coarse,
        alpha=16.0, floor=0.15, smooth_sigma=1.0
    )
    for blocks, tv_weight in [(1, 1.0), (2, 1.0), (2, 5.0)]:
        v_sgh, h_sgh, _ = run_sgh_vsp_fwi(
            best_model, obs_train, acq_train,
            gate=gate, tv_weight=tv_weight, edge_weights=edge_w,
            blocks_per_band=blocks, data_iter=3, tv_iter=60
        )
        row = {
            "blocks": int(blocks),
            "tv_weight": float(tv_weight),
            "history_len": len(h_sgh),
            "train_misfit": total_misfit(v_sgh, obs_train, acq_train),
            "val_misfit": total_misfit(v_sgh, obs_val, acq_val),
            **summarize_model_extended(v_sgh, v_true, dx_coarse, dx_coarse),
            "elapsed": time.time() - t0,
        }
        results["sgh_candidates"].append(row)
        models[f"sgh_from_best_b{blocks}_w{tv_weight:g}"] = v_sgh
        print("sgh", row, flush=True)

    results["best_by_validation"] = best
    results["final_pixel"] = {
        "chunk": 7,
        "train_misfit": total_misfit(c, obs_train, acq_train),
        "val_misfit": total_misfit(c, obs_val, acq_val),
        **summarize_model_extended(c, v_true, dx_coarse, dx_coarse),
    }
    models["best_by_validation"] = best_model
    models["final_pixel"] = c

    with open(os.path.join(OUT, "validation_shot_split_crossgrid_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(OUT, "validation_shot_split_crossgrid_models.npz"), **models)
    print("best_by_validation", best, flush=True)
    print("final_pixel", results["final_pixel"], flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
