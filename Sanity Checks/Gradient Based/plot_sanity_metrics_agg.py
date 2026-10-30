import os
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

EXP_SETTINGS = [
    {
        "dir": BASE_DIR / "CAM-SanityCheck" / "GradientBased" / "Base",
        "label": "Base",
        "color": "#4C72B0",  # muted blue
    },
    {
        "dir": BASE_DIR / "CAM-SanityCheck" / "GradientBased" / "UDA",
        "label": "UDA",
        "color": "#DD8452",  # warm orange
    },
    {
        "dir": BASE_DIR / "CAM-SanityCheck" / "GradientBased" / "BaseT",
        "label": "BaseT",
        "color": "#55A868",  # green
    },
]

# Shared line style — all lines identical, colour is the only differentiator
LINE_STYLE = "solid"
LINE_WIDTH = 1.8

OUT_DIR = BASE_DIR / "CAM-SanityCheck" / "GradientBased"
CSV_NAME = "sanity_check_all_methods.csv"

CAM_NAMES = ["GradCAM", "GradCAM++", "LayerCAM", "ShapleyCAM"]
DISPLAY_NAMES = ["Grad-CAM", "Grad-CAM++", "LayerCAM", "ShapleyCAM"]
MARKERSIZE = 3.5

METRIC_CFG = {
    "spearman": {
        "marker": "o",
        "label": "Spearman Rank Correlation",
        "idx": 1,
        "ylabel": "Spearman Rank Correlation",
    },
    "ssim": {
        "marker": "s",
        "label": "SSIM",
        "idx": 2,
        "ylabel": "SSIM",
    },
}

# ── load CSV for all experiments ──────────────────────────────────────────────
exp_data = {}

for exp in EXP_SETTINGS:
    label = exp["label"]
    csv_path = os.path.join(exp["dir"], CSV_NAME)
    exp_data[label] = {
        "cascading": {n: [] for n in CAM_NAMES},
        "independent": {n: [] for n in CAM_NAMES},
    }

    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        continue

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test = row["test"]
            method = row["cam_method"]
            layer = row["layer"]
            if layer == "-" or method not in CAM_NAMES:
                continue
            layer = int(layer)
            sp = float(row["spearman"])
            sv = float(row["ssim"])
            if test in exp_data[label]:
                exp_data[label][test][method].append((layer, sp, sv))

    for test in ("cascading", "independent"):
        for method in CAM_NAMES:
            exp_data[label][test][method].sort(key=lambda t: t[0])


# ── plot factory ──────────────────────────────────────────────────────────────
def make_figure(test_key, metric, title, fname):
    mcfg = METRIC_CFG[metric]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    for ax, method, dname in zip(axes, CAM_NAMES, DISPLAY_NAMES):
        has_any_data = False

        for exp in EXP_SETTINGS:
            label = exp["label"]
            color = exp["color"]  # colour now identifies the experiment
            rows = exp_data[label][test_key][method]

            if not rows:
                continue

            has_any_data = True
            layers = [r[0] for r in rows]
            values = [r[mcfg["idx"]] for r in rows]

            ax.plot(
                layers,
                values,
                color=color,
                linewidth=LINE_WIDTH,
                linestyle=LINE_STYLE,
                marker=mcfg["marker"],
                markersize=MARKERSIZE,
                zorder=3,
            )

        if not has_any_data:
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
            ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
            ax.invert_xaxis()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.axhline(1, color="black", linewidth=0.5, linestyle=":", alpha=0.3)

        ax.set_title(dname, fontsize=10, fontweight="bold")
        ax.set_ylim(-1.05, 1.05)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", color="grey", linewidth=0.4, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(mcfg["ylabel"], fontsize=9)
    axes[2].set_ylabel(mcfg["ylabel"], fontsize=9)
    axes[2].set_xlabel("Layer randomized", fontsize=9)
    axes[3].set_xlabel("Layer randomized", fontsize=9)

    # ── legend — one handle per experiment (colour only) ─────────────────────
    exp_handles = [
        Line2D(
            [0],
            [0],
            color=exp["color"],
            linewidth=LINE_WIDTH,
            linestyle=LINE_STYLE,
            marker=mcfg["marker"],
            markersize=MARKERSIZE,
            label=exp["label"],
        )
        for exp in EXP_SETTINGS
    ]

    fig.legend(
        handles=exp_handles,
        loc="lower center",
        ncol=len(exp_handles),
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.5,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    out_path = os.path.join(OUT_DIR, fname)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── generate all 4 figures ────────────────────────────────────────────────────
make_figure(
    "cascading",
    "spearman",
    "Cascading Randomization — Spearman Rank Correlation",
    "metrics_cascading_agg_spearman.png",
)
make_figure(
    "cascading",
    "ssim",
    "Cascading Randomization — SSIM",
    "metrics_cascading_agg_ssim.png",
)
make_figure(
    "independent",
    "spearman",
    "Independent Layer Randomization — Spearman Rank Correlation",
    "metrics_independent_agg_spearman.png",
)
make_figure(
    "independent",
    "ssim",
    "Independent Layer Randomization — SSIM",
    "metrics_independent_agg_ssim.png",
)