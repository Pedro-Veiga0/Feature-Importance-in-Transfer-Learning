"""
plot_sanity_metrics.py

Reads sanity_check_all_methods.csv produced by sanity_check.py and generates
correlation plots inspired by Figures 3 & 4 of Adebayo et al. (2018).

One figure per test type (cascading / independent), each with one subplot
per CAM method. Spearman and SSIM are plotted as two lines on the same axes.
"""

import os
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

OUT_DIR = BASE_DIR / "CAM-SanityCheck" / "GradientBased" / "Cityescapes"

CSV_PATH = os.path.join(OUT_DIR, "sanity_check_all_methods_ce.csv")

CAM_NAMES = ["GradCAM", "GradCAM++", "LayerCAM", "ShapleyCAM"]

DISPLAY_NAMES = {
    "GradCAM": "Grad-CAM",
    "GradCAM++": "Grad-CAM++",
    "LayerCAM": "LayerCAM",
    "ShapleyCAM": "ShapleyCAM",
}

# Colors following the paper's style — two distinct lines per plot
COLOR_SPEARMAN = "#E05C2A"  # warm orange-red
COLOR_SSIM = "#2A7DB5"  # cool blue

LINEWIDTH = 1.8
MARKERSIZE = 4

# ── load CSV ──────────────────────────────────────────────────────────────────
# Expected columns: cam_method, test, step, layer, spearman, ssim
data = {
    "cascading": {n: [] for n in CAM_NAMES},
    "independent": {n: [] for n in CAM_NAMES},
}

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        test = row["test"]
        method = row["cam_method"]
        layer = row["layer"]
        if layer == "-":
            continue  # header or cascading placeholder
        layer = int(layer)
        sp = float(row["spearman"])
        sv = float(row["ssim"])
        if test in data and method in data[test]:
            data[test][method].append((layer, sp, sv))

# Sort each series by layer descending (17 → 0) so x-axis reads left to right
for test in data:
    for method in data[test]:
        data[test][method].sort(key=lambda t: t[0])


# ── plot factory ──────────────────────────────────────────────────────────────
def make_figure(test_key, title, fname):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7.4),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    for ax, method in zip(axes, CAM_NAMES):
        rows = data[test_key][method]

        if not rows:
            ax.set_title(DISPLAY_NAMES[method], fontsize=10, fontweight="bold")
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="grey",
            )
            continue

        layers = [r[0] for r in rows]
        spearmans = [r[1] for r in rows]
        ssims = [r[2] for r in rows]

        ax.plot(
            layers,
            spearmans,
            color=COLOR_SPEARMAN,
            linewidth=LINEWIDTH,
            marker="o",
            markersize=MARKERSIZE,
            label="Spearman",
            zorder=3,
        )
        ax.plot(
            layers,
            ssims,
            color=COLOR_SSIM,
            linewidth=LINEWIDTH,
            marker="s",
            markersize=MARKERSIZE,
            label="SSIM",
            zorder=3,
        )

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.axhline(1, color="black", linewidth=0.5, linestyle=":", alpha=0.3)

        ax.set_title(DISPLAY_NAMES[method], fontsize=10, fontweight="bold")
        ax.set_xlim(layers[0] + 0.5, layers[-1] - 0.5)
        ax.invert_xaxis()
        ax.set_xticks(layers)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.tick_params(axis="both", labelsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.grid(axis="y", color="grey", linewidth=0.4, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    # Axis labels only on outer plots
    axes[0].set_ylabel("Correlation", fontsize=9)
    axes[2].set_ylabel("Correlation", fontsize=9)
    axes[2].set_xlabel("Layer randomized", fontsize=9)
    axes[3].set_xlabel("Layer randomized", fontsize=9)

    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_path = os.path.join(OUT_DIR, fname)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

# ── generate both figures ─────────────────────────────────────────────────────
make_figure(
    "cascading",
    "Cascading Randomization  (layer 17 → 0)",
    "metrics_cascading_ce.png",
)

make_figure(
    "independent",
    "Independent Layer Randomization",
    "metrics_independent_ce.png",
)