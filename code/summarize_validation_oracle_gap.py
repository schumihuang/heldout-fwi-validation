"""Summarize validation-selected stopping versus oracle true-metric stopping."""
import json
import os
from glob import glob


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")


def best_by(rows, key, reverse=False):
    return min(rows, key=lambda r: r[key]) if not reverse else max(rows, key=lambda r: r[key])


def summarize_rows(label, rows, selected, final):
    oracle_rmse = best_by(rows, "rmse")
    oracle_ssim = best_by(rows, "ssim", reverse=True)
    return {
        "label": label,
        "validation_selected": selected,
        "final_pixel": final,
        "oracle_best_rmse": oracle_rmse,
        "oracle_best_ssim": oracle_ssim,
        "rmse_gap_validation_to_oracle": selected["rmse"] - oracle_rmse["rmse"],
        "rmse_gain_validation_over_final": final["rmse"] - selected["rmse"],
        "ssim_gap_validation_to_oracle": oracle_ssim["ssim"] - selected["ssim"],
        "ssim_gain_validation_over_final": selected["ssim"] - final["ssim"],
    }


def main():
    output = {
        "experiment": "validation-selected stopping versus oracle stopping",
        "items": [],
    }

    for random_path in sorted(glob(os.path.join(RESULTS, "validation_random_splits*_summary.json"))):
        with open(random_path) as f:
            data = json.load(f)
        crop_label = data.get("crop", {}).get("label", os.path.basename(random_path).replace("_summary.json", ""))
        for split in data["splits"]:
            output["items"].append(summarize_rows(
                f"{crop_label}_{split['split_label']}",
                split["chunks"],
                split["best_by_validation"],
                split["final_pixel"],
            ))

    noisy_path = os.path.join(RESULTS, "validation_noisy_crossgrid_summary.json")
    if os.path.exists(noisy_path):
        with open(noisy_path) as f:
            data = json.load(f)
        for setting in data["settings"]:
            output["items"].append(summarize_rows(
                f"noisy_{int(setting['snr_db'])}db",
                setting["chunks"],
                setting["best_by_validation"],
                setting["final_pixel"],
            ))

    suite_path = os.path.join(RESULTS, "validation_shot_split_suite_summary.json")
    if os.path.exists(suite_path):
        with open(suite_path) as f:
            data = json.load(f)
        for setting in data["settings"]:
            output["items"].append(summarize_rows(
                f"suite_{setting['label']}",
                setting["chunks"],
                setting["best_by_validation"],
                setting["final_pixel"],
            ))

    out_path = os.path.join(RESULTS, "validation_oracle_gap_summary.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(out_path)
    for item in output["items"]:
        print(
            item["label"],
            "rmse_gain_vs_final=", f"{item['rmse_gain_validation_over_final']:.3f}",
            "rmse_gap_to_oracle=", f"{item['rmse_gap_validation_to_oracle']:.3f}",
            "ssim_gain_vs_final=", f"{item['ssim_gain_validation_over_final']:.4f}",
            "ssim_gap_to_oracle=", f"{item['ssim_gap_validation_to_oracle']:.4f}",
        )


if __name__ == "__main__":
    main()
