"""Create summary figure for noisy held-out shot validation."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(RESULTS, "validation_noisy_crossgrid_summary.json")) as f:
        data = json.load(f)

    labels = [f"{int(s['snr_db'])} dB" for s in data["settings"]]
    best = [s["best_by_validation"] for s in data["settings"]]
    final = [s["final_pixel"] for s in data["settings"]]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    specs = [
        ("rmse", "RMSE (m/s)", True),
        ("ssim", "SSIM", False),
        ("edge_corr", "Edge correlation", False),
    ]
    for ax, (key, ylabel, lower_better) in zip(axes, specs):
        b_vals = [r[key] for r in best]
        f_vals = [r[key] for r in final]
        ax.bar(x - width / 2, b_vals, width, label="Validation-selected",
               color="#177245")
        ax.bar(x + width / 2, f_vals, width, label="Final pixel",
               color="#b23a48")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title("Lower is better" if lower_better else "Higher is better")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    png = os.path.join(FIGURES, "fig_validation_noisy.png")
    pdf = os.path.join(FIGURES, "fig_validation_noisy.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
