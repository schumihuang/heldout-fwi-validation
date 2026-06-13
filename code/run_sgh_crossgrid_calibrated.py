"""Calibrated cross-grid Marmousi2 pilot.

Fine-grid shot gathers are scaled shot-by-shot to match the coarse solver
response at the initial model. This removes source-amplitude/discretization
scale mismatch while keeping phase and modeling mismatch.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from fwi_core import Acquisition, forward_shot, marmousi2_real_subset, smooth_initial_model
from optimize import run_fwi_lbfgs
from rsp_optimize import run_multiscale_pixel_fwi
from sgh_optimize import gradient_structure_gate, run_sgh_vsp_fwi, summarize_model_extended
from tv_prox import edge_preserving_weights


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results")


def make_crossgrid_band_data(
    v_fine,
    v_coarse,
    dx_fine,
    dx_coarse,
    f0_values=(2.0, 3.0, 4.0),
    n_src=3,
):
    obs_by_band = []
    acq_coarse_by_band = []
    dt = 0.5 * dx_fine / (np.sqrt(2.0) * v_fine.max())
    nt = int(1.6 / dt)
    rec_idx_fine = np.rint(np.arange(v_coarse.shape[1]) * dx_coarse / dx_fine).astype(int)
    rec_idx_fine = np.clip(rec_idx_fine, 0, v_fine.shape[1] - 1)
    for f0 in f0_values:
        acq_fine = Acquisition(
            nz=v_fine.shape[0], nx=v_fine.shape[1], n_src=n_src,
            src_z=4, rec_z=4, f0=f0, nt=nt, dt=dt,
            dx=dx_fine, dz=dx_fine
        )
        acq_coarse = Acquisition(
            nz=v_coarse.shape[0], nx=v_coarse.shape[1], n_src=n_src,
            src_z=2, rec_z=2, f0=f0, nt=nt, dt=dt,
            dx=dx_coarse, dz=dx_coarse
        )
        band = []
        for sx in acq_fine.src_x:
            seis_fine, _ = forward_shot(
                v_fine, (acq_fine.src_z, sx), acq_fine.rec_z,
                acq_fine.wavelet, acq_fine.dt, acq_fine.dx, acq_fine.dz,
                save_wavefield=False
            )
            band.append(seis_fine[:, rec_idx_fine])
        obs_by_band.append(band)
        acq_coarse_by_band.append(acq_coarse)
    return obs_by_band, acq_coarse_by_band


def calibrate_obs_to_initial(v_init, obs_by_band, acq_by_band):
    calibrated = []
    scales = []
    for obs_band, acq in zip(obs_by_band, acq_by_band):
        out_band = []
        scale_band = []
        for sx, dobs in zip(acq.src_x, obs_band):
            pred, _ = forward_shot(
                v_init, (acq.src_z, sx), acq.rec_z,
                acq.wavelet, acq.dt, acq.dx, acq.dz,
                save_wavefield=False
            )
            denom = float(np.sum(dobs * dobs))
            scale = float(np.sum(pred * dobs) / denom) if denom > 0.0 else 1.0
            out_band.append(scale * dobs)
            scale_band.append(scale)
        calibrated.append(out_band)
        scales.append(scale_band)
    return calibrated, scales


def main():
    np.random.seed(0)
    os.makedirs(OUT, exist_ok=True)
    v_coarse, dx_coarse, meta_coarse = marmousi2_real_subset(nx=100, nz=36, step=24)
    v_fine, dx_fine, meta_fine = marmousi2_real_subset(nx=200, nz=72, step=12)
    v_init = smooth_initial_model(v_coarse, sigma=8.0)

    raw_obs, acq_by_band = make_crossgrid_band_data(
        v_fine, v_coarse, dx_fine, dx_coarse,
        f0_values=(2.0, 3.0, 4.0), n_src=3
    )
    obs_by_band, scales = calibrate_obs_to_initial(v_init, raw_obs, acq_by_band)

    results = {
        "dataset": "Marmousi2 true VP subset",
        "experiment": "calibrated fine-grid data, coarse-grid inversion",
        "coarse_meta": meta_coarse,
        "fine_meta": meta_fine,
        "coarse_grid": [int(v_coarse.shape[0]), int(v_coarse.shape[1])],
        "fine_grid": [int(v_fine.shape[0]), int(v_fine.shape[1])],
        "dx_coarse": float(dx_coarse),
        "dx_fine": float(dx_fine),
        "n_src": 3,
        "bands_hz": [float(a.f0) for a in acq_by_band],
        "calibration_scales": scales,
        "methods": {
            "init_sigma8": summarize_model_extended(v_init, v_coarse, dx_coarse, dx_coarse),
        },
        "runs": [],
    }
    models = {"true_coarse": v_coarse, "true_fine": v_fine, "init_sigma8": v_init}
    print("init", results["methods"]["init_sigma8"], flush=True)
    print("calibration scale range", float(np.min(scales)), float(np.max(scales)), flush=True)

    print("Calibrated cross-grid multiscale pixel...", flush=True)
    t0 = time.time()
    v_ms, h_ms = run_multiscale_pixel_fwi(
        v_init, obs_by_band, acq_by_band, max_iter_per_band=5
    )
    ms = {
        **summarize_model_extended(v_ms, v_coarse, dx_coarse, dx_coarse),
        "time": time.time() - t0,
        "history_len": len(h_ms),
        "max_iter_per_band": 5,
    }
    results["methods"]["multiscale_pixel_5"] = ms
    models["multiscale_pixel_5"] = v_ms
    print(ms, flush=True)

    print("Calibrated cross-grid pixel last-band 10...", flush=True)
    t0 = time.time()
    v_px10, h_px10 = run_fwi_lbfgs(
        v_ms, obs_by_band[-1], acq_by_band[-1], max_iter=10, lambda_smo=1e-8
    )
    px10 = {
        **summarize_model_extended(v_px10, v_coarse, dx_coarse, dx_coarse),
        "time": time.time() - t0,
        "history_len": len(h_px10),
    }
    results["methods"]["pixel_lastband10"] = px10
    models["pixel_lastband10"] = v_px10
    print(px10, flush=True)

    print("Calibrated cross-grid pixel continuation 20...", flush=True)
    t0 = time.time()
    v_px20, h_px20 = run_fwi_lbfgs(
        v_px10, obs_by_band[-1], acq_by_band[-1], max_iter=20, lambda_smo=1e-8
    )
    px20 = {
        **summarize_model_extended(v_px20, v_coarse, dx_coarse, dx_coarse),
        "time": time.time() - t0,
        "history_len": len(h_px20),
    }
    results["methods"]["pixel_continue20"] = px20
    models["pixel_continue20"] = v_px20
    print(px20, flush=True)

    for start_name, start_model in [("stage1", v_ms), ("pixel_lastband10", v_px10)]:
        gate, gate_energy = gradient_structure_gate(
            start_model, obs_by_band[0], acq_by_band[0],
            smooth_sigma=1.5, quantile=0.65, sharpness=8.0, floor=0.9
        )
        edge_w = edge_preserving_weights(
            start_model, dx=dx_coarse, dz=dx_coarse,
            alpha=16.0, floor=0.15, smooth_sigma=1.0
        )
        print(f"Calibrated cross-grid SGH start={start_name}...", flush=True)
        t0 = time.time()
        v_sgh, h_sgh, _ = run_sgh_vsp_fwi(
            start_model, obs_by_band, acq_by_band,
            gate=gate, tv_weight=1.0, edge_weights=edge_w,
            blocks_per_band=2, data_iter=5, tv_iter=80
        )
        row = {
            "method": "sgh_weighted",
            "start": start_name,
            "blocks": 2,
            "tv_weight": 1.0,
            **summarize_model_extended(v_sgh, v_coarse, dx_coarse, dx_coarse),
            "time": time.time() - t0,
            "history_len": len(h_sgh),
        }
        results["runs"].append(row)
        models[f"sgh_{start_name}"] = v_sgh
        if start_name == "stage1":
            models["gate_stage1"] = gate
            models["gate_energy_stage1"] = gate_energy
            models["edge_weights_stage1"] = edge_w
        print(row, flush=True)

    with open(os.path.join(OUT, "sgh_crossgrid_calibrated_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(OUT, "sgh_crossgrid_calibrated_models.npz"), **models)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
