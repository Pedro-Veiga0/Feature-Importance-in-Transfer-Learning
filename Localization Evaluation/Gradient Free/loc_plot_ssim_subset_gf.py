"""Plot gradient-free EBPG results for the SSIM-selected experiments."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUTPUT_DIR = Path(__file__).resolve().parent / "CAM-Loc-Eval" / "SSIM-10"
SUMMARY_CSV = OUTPUT_DIR / "localization_summary_gf_final.csv"

METHODS = ("EigenCAM", "ScoreCAM")
DISPLAY_NAMES = {
    "EigenCAM": "Eigen-CAM",
    "ScoreCAM": "Score-CAM",
}
COLORS = {
    "EigenCAM": "#ff7f0e",
    "ScoreCAM": "#2ca02c",
}
SETTINGS = (
    {
        "name": "Base",
        "dataset": "Arnold",
        "checkpoint": "arnold.pt",
        "setting": "Base",
        "output": "ebpg_overlay_gf_base.png",
    },
    {
        "name": "UDA",
        "dataset": "Rapid",
        "checkpoint": "arnold2rapid.pt",
        "setting": "UDA",
        "output": "ebpg_overlay_gf_uda.png",
    },
    {
        "name": "BaseT",
        "dataset": "Rapid",
        "checkpoint": "arnold.pt",
        "setting": "BaseT",
        "output": "ebpg_overlay_gf_baset.png",
    },
)


def load_summary_rows():
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(
            f"Gradient-free summary CSV not found: {SUMMARY_CSV}"
        )
    with SUMMARY_CSV.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "dataset",
            "setting",
            "checkpoint",
            "cam_method",
            "layer",
            "ebpg_mean",
            "ebpg_std",
            "pg_mean",
        }
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Expected columns {sorted(required_columns)} in {SUMMARY_CSV}"
            )
        return list(reader)


def select_setting_rows(rows, setting):
    selected = [
        row
        for row in rows
        if row["dataset"] == setting["dataset"]
        and row["setting"] == setting["setting"]
        and Path(row["checkpoint"]).name == setting["checkpoint"]
    ]
    if not selected:
        raise ValueError(
            f"No rows found for {setting['name']}: dataset={setting['dataset']}, "
            f"checkpoint={setting['checkpoint']}."
        )
    return selected


def make_plot(rows, setting):
    method_data = {method: {} for method in METHODS}
    method_pg_scores = {method: {} for method in METHODS}
    for row in rows:
        method = row["cam_method"]
        if method not in method_data:
            continue
        layer = int(row["layer"])
        if layer in method_data[method]:
            raise ValueError(
                f"Duplicate {method} layer {layer} rows for {setting['name']}."
            )
        mean_score = float(row["ebpg_mean"])
        std_score = float(row["ebpg_std"])
        if std_score < 0:
            raise ValueError(
                f"Negative EBPG standard deviation for {method}, layer {layer}."
            )
        method_data[method][layer] = (mean_score, std_score)
        method_pg_scores[method][layer] = float(row["pg_mean"])

    missing_methods = [method for method, values in method_data.items() if not values]
    if missing_methods:
        raise ValueError(
            f"Missing gradient-free methods for {setting['name']}: "
            f"{', '.join(missing_methods)}."
        )

    fig, ax = plt.subplots(figsize=(13, 6))
    for method in METHODS:
        values = method_data[method]
        layers = sorted(values)
        means = [values[layer][0] for layer in layers]
        stds = [values[layer][1] for layer in layers]
        ax.errorbar(
            layers,
            means,
            yerr=stds,
            label=DISPLAY_NAMES[method],
            color=COLORS[method],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            elinewidth=0.9,
            capsize=2,
        )
    for method in METHODS:
        hit_layers = [
            layer for layer, score in method_pg_scores[method].items() if score > 0
        ]
        ax.scatter(
            hit_layers,
            [method_data[method][layer][0] for layer in hit_layers],
            color=COLORS[method],
            marker="^",
            s=48,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

    all_layers = sorted(
        set.union(*(set(values) for values in method_data.values()))
    )
    ax.set_title(f"EBPG by layer - {setting['name']}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("EBPG mean (error bars: +/- 1 SD)")
    ax.set_xticks(all_layers[::2])
    ax.set_xlim(min(all_layers) - 0.5, max(all_layers) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)

    handles, _ = ax.get_legend_handles_labels()
    pg_labels = [
        f"{DISPLAY_NAMES[method]}: {sum(method_pg_scores[method].values()) / len(method_pg_scores[method]):.2f}"
        for method in METHODS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=7,
            label="PG hit (at least 1 image)",
        )
    )
    pg_labels.append("PG hit (at least 1 image)")
    score_legend = ax.legend(
        handles,
        pg_labels,
        title="Overall PG score",
        loc="upper left",
        edgecolor="gray",
    )
    score_legend.get_frame().set_alpha(0.8)
    fig.tight_layout()

    output_path = OUTPUT_DIR / setting["output"]
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {setting['name']} plot to: {output_path}")


def main():
    rows = load_summary_rows()
    for setting in SETTINGS:
        make_plot(select_setting_rows(rows, setting), setting)


if __name__ == "__main__":
    main()
