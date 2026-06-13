"""Create actual model-comparison figures for validation-controlled FWI."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def _load_json(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def _velocity_limits(true_model, *models):
    stack = np.concatenate([true_model.ravel()] + [m.ravel() for m in models])
    return float(np.percentile(stack, 1.0)), float(np.percentile(stack, 99.0))


def _error_limit(true_model, *models):
    errs = np.concatenate([np.abs(m - true_model).ravel() for m in models])
    return float(np.percentile(errs, 98.0))


def _format_metric(row):
    return f"RMSE {row['rmse']:.1f} m/s, SSIM {row['ssim']:.3f}"


def _imshow(ax, img, title, vmin, vmax, cmap="viridis"):
    h = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return h


def _save(fig, stem):
    os.makedirs(FIGURES, exist_ok=True)
    png = os.path.join(FIGURES, f"{stem}.png")
    pdf = os.path.join(FIGURES, f"{stem}.pdf")
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    print(png)
    print(pdf)


def primary_model_figure():
    summary = _load_json("validation_shot_split_crossgrid_summary.json")
    z = np.load(os.path.join(RESULTS, "validation_shot_split_crossgrid_models.npz"))
    true = z["true_coarse"]
    init = z["init_sigma8"]
    selected = z["best_by_validation"]
    final = z["final_pixel"]

    vmin, vmax = _velocity_limits(true, init, selected, final)
    emax = _error_limit(true, init, selected, final)
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.4), constrained_layout=True)

    vel_images = [
        (true, "True Marmousi2 crop"),
        (init, f"Initial\n{_format_metric(summary['init'])}"),
        (selected, f"Validation-selected\n{_format_metric(summary['best_by_validation'])}"),
        (final, f"Final continuation\n{_format_metric(summary['final_pixel'])}"),
    ]
    last_vel = None
    for ax, (img, title) in zip(axes[0], vel_images):
        last_vel = _imshow(ax, img, title, vmin, vmax)

    err_images = [
        (init - true, "Initial error"),
        (selected - true, "Selected error"),
        (final - true, "Final error"),
        (final - selected, "Final minus selected"),
    ]
    last_err = None
    for ax, (img, title) in zip(axes[1], err_images):
        last_err = _imshow(ax, img, title, -emax, emax, cmap="coolwarm")

    fig.colorbar(last_vel, ax=axes[0].ravel().tolist(), shrink=0.85, label="Velocity (m/s)")
    fig.colorbar(last_err, ax=axes[1].ravel().tolist(), shrink=0.85, label="Difference (m/s)")
    _save(fig, "fig_validation_primary_models")


def split_model_figure():
    summary = _load_json("validation_random_splits_summary.json")
    z = np.load(os.path.join(RESULTS, "validation_random_splits_models.npz"))
    true = z["true_coarse"]
    rows = summary["splits"]
    selected_models = [z[f"{r['split_label']}_best_by_validation"] for r in rows]
    final_models = [z[f"{r['split_label']}_final_pixel"] for r in rows]
    vmin, vmax = _velocity_limits(true, *(selected_models + final_models))
    emax = _error_limit(true, *(selected_models + final_models))

    fig, axes = plt.subplots(3, 4, figsize=(12.5, 7.6), constrained_layout=True)
    last_vel = None
    last_err = None
    for i, row in enumerate(rows):
        label = row["split_label"].replace("_", "-")
        selected = selected_models[i]
        final = final_models[i]
        b = row["best_by_validation"]
        f = row["final_pixel"]
        last_vel = _imshow(axes[i, 0], selected, f"{label}: selected\n{_format_metric(b)}", vmin, vmax)
        last_vel = _imshow(axes[i, 1], final, f"{label}: final\n{_format_metric(f)}", vmin, vmax)
        last_err = _imshow(axes[i, 2], selected - true, "Selected error", -emax, emax, cmap="coolwarm")
        last_err = _imshow(axes[i, 3], final - true, "Final error", -emax, emax, cmap="coolwarm")

    fig.colorbar(last_vel, ax=axes[:, :2].ravel().tolist(), shrink=0.8, label="Velocity (m/s)")
    fig.colorbar(last_err, ax=axes[:, 2:].ravel().tolist(), shrink=0.8, label="Error (m/s)")
    _save(fig, "fig_validation_split_models")


def noisy_model_figure():
    summary = _load_json("validation_noisy_crossgrid_summary.json")
    z = np.load(os.path.join(RESULTS, "validation_noisy_crossgrid_models.npz"))
    true = z["true_coarse"]
    settings = summary["settings"]
    selected_models = [z[f"snr{int(s['snr_db'])}_best_by_validation"] for s in settings]
    final_models = [z[f"snr{int(s['snr_db'])}_final_pixel"] for s in settings]
    vmin, vmax = _velocity_limits(true, *(selected_models + final_models))
    emax = _error_limit(true, *(selected_models + final_models))

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.4), constrained_layout=True)
    last_vel = None
    last_err = None
    for i, setting in enumerate(settings):
        snr = int(setting["snr_db"])
        selected = selected_models[i]
        final = final_models[i]
        b = setting["best_by_validation"]
        f = setting["final_pixel"]
        last_vel = _imshow(axes[i, 0], selected, f"{snr} dB selected\n{_format_metric(b)}", vmin, vmax)
        last_vel = _imshow(axes[i, 1], final, f"{snr} dB final\n{_format_metric(f)}", vmin, vmax)
        last_err = _imshow(axes[i, 2], selected - true, "Selected error", -emax, emax, cmap="coolwarm")
        last_err = _imshow(axes[i, 3], final - true, "Final error", -emax, emax, cmap="coolwarm")

    fig.colorbar(last_vel, ax=axes[:, :2].ravel().tolist(), shrink=0.82, label="Velocity (m/s)")
    fig.colorbar(last_err, ax=axes[:, 2:].ravel().tolist(), shrink=0.82, label="Error (m/s)")
    _save(fig, "fig_validation_noisy_models")


def cropb_model_figure():
    summary = _load_json("validation_random_splits_cropB_long_summary.json")
    z = np.load(os.path.join(RESULTS, "validation_random_splits_cropB_long_models.npz"))
    true = z["true_coarse"]
    rows = summary["splits"]
    selected_models = [z[f"{r['split_label']}_best_by_validation"] for r in rows]
    final_models = [z[f"{r['split_label']}_final_pixel"] for r in rows]
    vmin, vmax = _velocity_limits(true, *(selected_models + final_models))
    emax = _error_limit(true, *(selected_models + final_models))

    fig, axes = plt.subplots(3, 4, figsize=(12.5, 7.6), constrained_layout=True)
    last_vel = None
    last_err = None
    for i, row in enumerate(rows):
        label = row["split_label"].replace("_", "-")
        selected = selected_models[i]
        final = final_models[i]
        b = row["best_by_validation"]
        f = row["final_pixel"]
        last_vel = _imshow(axes[i, 0], selected, f"{label}: selected\n{_format_metric(b)}", vmin, vmax)
        last_vel = _imshow(axes[i, 1], final, f"{label}: final\n{_format_metric(f)}", vmin, vmax)
        last_err = _imshow(axes[i, 2], selected - true, "Selected error", -emax, emax, cmap="coolwarm")
        last_err = _imshow(axes[i, 3], final - true, "Final error", -emax, emax, cmap="coolwarm")

    fig.colorbar(last_vel, ax=axes[:, :2].ravel().tolist(), shrink=0.8, label="Velocity (m/s)")
    fig.colorbar(last_err, ax=axes[:, 2:].ravel().tolist(), shrink=0.8, label="Error (m/s)")
    _save(fig, "fig_validation_cropB_models")


def main():
    primary_model_figure()
    split_model_figure()
    noisy_model_figure()
    cropb_model_figure()


if __name__ == "__main__":
    main()
