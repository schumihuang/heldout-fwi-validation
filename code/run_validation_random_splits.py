"""Held-out shot validation repetitions for one Marmousi2 crop."""
import argparse
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


def run_split(
    split_label, train_idx, val_idx, v_init, v_true, dx, obs_by_band,
    acq_full, n_chunks=8, chunk_iter=5
):
    print(f"\n=== split {split_label} train={train_idx} val={val_idx} ===", flush=True)
    obs_train, acq_train, obs_val, acq_val = split_shots(
        obs_by_band, acq_full, np.array(train_idx), np.array(val_idx)
    )
    t0 = time.time()
    c = v_init.copy()
    warmup = []
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
    for chunk in range(n_chunks):
        c, h = run_fwi_lbfgs(
            c, obs_train[-1], acq_train[-1],
            max_iter=chunk_iter, lambda_smo=1e-8
        )
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
        "chunk": int(n_chunks - 1),
        "train_misfit": total_misfit(c, obs_train, acq_train),
        "val_misfit": total_misfit(c, obs_val, acq_val),
        **summarize_model_extended(c, v_true, dx, dx),
        "elapsed": time.time() - t0,
    }
    result = {
        "split_label": split_label,
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
        "train_src_x": [int(x) for x in acq_train[0].src_x],
        "validation_src_x": [int(x) for x in acq_val[0].src_x],
        "warmup": warmup,
        "chunks": chunks,
        "best_by_validation": best,
        "final_pixel": final,
    }
    print("best_by_validation", best, flush=True)
    print("final_pixel", final, flush=True)
    return result, best_model, c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="cropA_sigma8")
    parser.add_argument("--x0-m", type=float, default=6500.0)
    parser.add_argument("--z0-m", type=float, default=700.0)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--chunk-iter", type=int, default=5)
    parser.add_argument("--out-prefix", default="validation_random_splits")
    args = parser.parse_args()

    np.random.seed(0)
    os.makedirs(OUT, exist_ok=True)
    v_true, dx_coarse, meta_coarse = marmousi2_real_subset(
        nx=100, nz=36, step=24, x0_m=args.x0_m, z0_m=args.z0_m
    )
    v_fine, dx_fine, meta_fine = marmousi2_real_subset(
        nx=200, nz=72, step=12, x0_m=args.x0_m, z0_m=args.z0_m
    )
    v_init = smooth_initial_model(v_true, sigma=args.sigma)
    raw_obs, acq_full = make_crossgrid_band_data(
        v_fine, v_true, dx_fine, dx_coarse,
        f0_values=(2.0, 3.0, 4.0), n_src=6
    )
    obs_by_band, scales = calibrate_obs_to_initial(v_init, raw_obs, acq_full)

    splits = [
        ("even_odd", [0, 2, 4], [1, 3, 5]),
        ("left_center", [0, 1, 4], [2, 3, 5]),
        ("outer_inner", [0, 3, 5], [1, 2, 4]),
    ]
    results = {
        "dataset": "Marmousi2 true VP subset",
        "experiment": "held-out shot validation split repetitions",
        "crop": {"label": args.label, "x0_m": args.x0_m, "z0_m": args.z0_m, "sigma": args.sigma},
        "coarse_meta": meta_coarse,
        "fine_meta": meta_fine,
        "dx_coarse": float(dx_coarse),
        "dx_fine": float(dx_fine),
        "calibration_scales": scales,
        "init": summarize_model_extended(v_init, v_true, dx_coarse, dx_coarse),
        "splits": [],
    }
    models = {
        "true_coarse": v_true,
        "true_fine": v_fine,
        "init": v_init,
    }
    for label, train_idx, val_idx in splits:
        result, best_model, final_model = run_split(
            label, train_idx, val_idx, v_init, v_true, dx_coarse,
            obs_by_band, acq_full,
            n_chunks=args.n_chunks,
            chunk_iter=args.chunk_iter,
        )
        results["splits"].append(result)
        models[f"{label}_best_by_validation"] = best_model
        models[f"{label}_final_pixel"] = final_model

    summary_name = f"{args.out_prefix}_summary.json"
    models_name = f"{args.out_prefix}_models.npz"
    with open(os.path.join(OUT, summary_name), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(OUT, models_name), **models)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
