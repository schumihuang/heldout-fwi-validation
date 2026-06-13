"""Create bar summary for random shot-split validation repetitions."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="validation_random_splits_summary.json")
    parser.add_argument("--out-stem", default="fig_validation_random_splits")
    args = parser.parse_args()

    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(RESULTS, args.summary)) as f:
        data = json.load(f)

    labels = [s["split_label"] for s in data["splits"]]
    best = [s["best_by_validation"] for s in data["splits"]]
    final = [s["final_pixel"] for s in data["splits"]]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
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
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        if lower_better:
            ax.set_title("Lower is better")
        else:
            ax.set_title("Higher is better")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    png = os.path.join(FIGURES, f"{args.out_stem}.png")
    pdf = os.path.join(FIGURES, f"{args.out_stem}.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
