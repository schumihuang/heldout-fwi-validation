"""Create validation-control diagnostic figures."""
import json
import os

import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(RESULTS, "validation_shot_split_crossgrid_summary.json")) as f:
        data = json.load(f)

    chunks = data["chunks"]
    x = [c["chunk"] for c in chunks]
    val = [c["val_misfit"] for c in chunks]
    train = [c["train_misfit"] for c in chunks]
    rmse = [c["rmse"] for c in chunks]
    ssim = [c["ssim"] for c in chunks]
    best_chunk = data["best_by_validation"]["chunk"]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True)

    ax = axes[0]
    ax.plot(x, train, marker="o", label="Training misfit", color="#3066be")
    ax.plot(x, val, marker="s", label="Held-out shot misfit", color="#b23a48")
    ax.axvline(best_chunk, color="#222222", linestyle="--", linewidth=1.2,
               label=f"Selected chunk {best_chunk}")
    ax.set_ylabel("Waveform misfit")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(x, rmse, marker="o", label="RMSE", color="#177245")
    ax.set_ylabel("RMSE (m/s)", color="#177245")
    ax.tick_params(axis="y", labelcolor="#177245")
    ax.grid(True, alpha=0.25)
    ax.axvline(best_chunk, color="#222222", linestyle="--", linewidth=1.2)

    ax2 = ax.twinx()
    ax2.plot(x, ssim, marker="s", label="SSIM", color="#7a4cc2")
    ax2.set_ylabel("SSIM", color="#7a4cc2")
    ax2.tick_params(axis="y", labelcolor="#7a4cc2")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    ax.set_xlabel("Last-band continuation chunk")

    fig.tight_layout()
    png = os.path.join(FIGURES, "fig_validation_shot_split.png")
    pdf = os.path.join(FIGURES, "fig_validation_shot_split.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
