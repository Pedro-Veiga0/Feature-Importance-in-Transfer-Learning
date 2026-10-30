"""Plot SSIM-10 sanity-check means with +/- 1 SD error bars."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D


OUTPUT_DIR = Path(__file__).resolve().parent / "CAM-SanityCheck" / "SSIM-10"
GRADIENT_BASED_CSV = OUTPUT_DIR / "sanity_check_all_settings_summary_gradient_based.csv"
GRADIENT_FREE_CSV = OUTPUT_DIR / "sanity_check_all_settings_summary_gf.csv"

SETTINGS = (
    {"name": "Base", "color": "#4C72B0"},
    {"name": "UDA", "color": "#DD8452"},
    {"name": "BaseT", "color": "#55A868"},
)
METHOD_GROUPS = {
    "gradient_based": (
        ("GradCAM", "Grad-CAM"),
        ("GradCAM++", "Grad-CAM++"),
        ("LayerCAM", "LayerCAM"),
        ("ShapleyCAM", "ShapleyCAM"),
    ),
    "gradient_free": (("EigenCAM", "Eigen-CAM"), ("ScoreCAM", "Score-CAM")),
}
METRICS = {
    "spearman": ("Spearman Rank Correlation", "Spearman"),
    "ssim": ("SSIM", "SSIM"),
}
TESTS = {
    "cascading": "Cascading Randomization (layer 17 to 0)",
    "independent": "Independent Layer Randomization",
}
MARKERS = {"spearman": "o", "ssim": "s"}


def load_summary(path):
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    required = {"setting", "cam_method", "test", "layer", "metric", "mean", "std"}
    values = {}
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Expected columns {sorted(required)} in {path}")

        for row in reader:
            key = (
                row["setting"],
                row["cam_method"],
                row["test"],
                row["metric"],
                int(row["layer"]),
            )
            if key in values:
                raise ValueError(f"Duplicate summary row in {path}: {key}")
            mean = float(row["mean"])
            std = float(row["std"])
            if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
                raise ValueError(f"Invalid mean or standard deviation in {path}: {key}")
            values[key] = (mean, std)
    return values


def make_figure(summary, method_group, test, metric, output_path):
    methods = METHOD_GROUPS[method_group]
    fig, axes = plt.subplots(
        2 if len(methods) > 2 else 1,
        2,
        figsize=(10, 7.4) if len(methods) > 2 else (8, 4.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()
    fig.suptitle(
        f"{TESTS[test]} - {METRICS[metric][0]}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    for ax, (method, display_name) in zip(axes, methods):
        has_data = False
        for setting in SETTINGS:
            layers = sorted(
                layer
                for name, cam_method, row_test, row_metric, layer in summary
                if name == setting["name"]
                and cam_method == method
                and row_test == test
                and row_metric == metric
            )
            if not layers:
                continue

            has_data = True
            means = [
                summary[(setting["name"], method, test, metric, layer)][0]
                for layer in layers
            ]
            stds = [
                summary[(setting["name"], method, test, metric, layer)][1]
                for layer in layers
            ]
            ax.errorbar(
                layers,
                means,
                yerr=stds,
                color=setting["color"],
                marker=MARKERS[metric],
                markersize=3.5,
                linewidth=1.8,
                elinewidth=0.9,
                capsize=2,
                label=setting["name"],
                zorder=3,
            )

        if not has_data:
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="grey",
            )
        else:
            ax.set_xlim(max(layers) + 0.5, min(layers) - 0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.axhline(1, color="black", linewidth=0.5, linestyle=":", alpha=0.3)
        ax.set_title(display_name, fontsize=10, fontweight="bold")
        ax.set_ylim(-1.05, 1.05)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", color="grey", linewidth=0.4, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(methods) :]:
        ax.set_visible(False)

    axes[0].set_ylabel(METRICS[metric][0], fontsize=9)
    if len(methods) > 2:
        axes[2].set_ylabel(METRICS[metric][0], fontsize=9)
        axes[2].set_xlabel("Layer randomized", fontsize=9)
        axes[3].set_xlabel("Layer randomized", fontsize=9)
    else:
        axes[0].set_xlabel("Layer randomized", fontsize=9)
        axes[1].set_xlabel("Layer randomized", fontsize=9)

    handles = [
        Line2D(
            [0],
            [0],
            color=setting["color"],
            marker=MARKERS[metric],
            linewidth=1.8,
            markersize=3.5,
            label=setting["name"],
        )
        for setting in SETTINGS
        if any(key[0] == setting["name"] for key in summary)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        title="Setting; error bars: +/- 1 SD",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    summaries = {
        "gradient_based": load_summary(GRADIENT_BASED_CSV),
        "gradient_free": load_summary(GRADIENT_FREE_CSV),
    }
    for method_group, summary in summaries.items():
        for test in TESTS:
            for metric in METRICS:
                filename = f"metrics_{test}_{method_group}_{metric}.png"
                make_figure(
                    summary,
                    method_group,
                    test,
                    metric,
                    OUTPUT_DIR / filename,
                )


if __name__ == "__main__":
    main()
