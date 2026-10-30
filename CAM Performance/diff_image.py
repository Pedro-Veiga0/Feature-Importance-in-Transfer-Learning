import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_CSV = "cam_difference_layer17.csv"
OUTPUT_PNG = "cam_agreement_layer17.png"
MODE = "difference"  # "ratio" for Grad-CAM / ShapleyCAM; "difference" for Grad-CAM - ShapleyCAM
EPS = 1e-12

A = np.genfromtxt(INPUT_CSV, delimiter=",", dtype=float)
if A.ndim != 2 or A.size == 0:
    raise ValueError("No numeric 2-D array was found in the CSV file.")

valid = np.isfinite(A)
if MODE == "ratio":
    error = np.abs(A - 1.0)
else:
    error = np.abs(A)

x = error[valid]
mae = np.mean(x)
rmse = np.sqrt(np.mean(x**2))
p95 = np.percentile(x, 95)
max_err = np.max(x)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
vmax = np.percentile(x, 99.9)
im = ax[0].imshow(
    error, cmap="magma", vmin=0, vmax=max(vmax, EPS), interpolation="nearest"
)
ax[0].set_title("Spatial agreement map")
ax[0].set_xlabel("x pixel")
ax[0].set_ylabel("y pixel")
cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

positive = x[x > 0]
left = max(positive.min() if positive.size else EPS, EPS)
bins = np.logspace(np.log10(left), np.log10(max(max_err, left) + EPS), 60)
ax[1].hist(
    np.maximum(x, EPS), bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.3
)
ax[1].set_xscale("log")
ax[1].set_title("Distribution of pixelwise error")
ax[1].set_ylabel("Number of pixels")
ax[1].grid(axis="y", alpha=0.25)

q = np.linspace(0, 100, 1001)
ax[2].plot(np.maximum(np.percentile(x, q), EPS), q, color="#E45756", lw=2.2)
ax[2].axvline(
    max(p95, EPS), color="black", ls="--", lw=1, label=f"95th pct. = {p95:.2e}"
)
ax[2].set_xscale("log")
ax[2].set_xlim(left=left)
ax[2].set_title("Cumulative error distribution")
ax[2].set_ylabel("Pixels at or below error (%)")
ax[2].legend(frameon=False, loc="lower right")
ax[2].grid(alpha=0.25)

fig.suptitle(
    "Grad-CAM vs. ShapleyCAM agreement — YOLOv5s layer 17",
    fontsize=14,
    fontweight="bold",
)
fig.text(
    0.5,
    -0.02,
    f"N = {x.size:,} pixels | max = {max_err:.2e}",
    ha="center",
    fontsize=9,
)
fig.savefig(OUTPUT_PNG, dpi=400, bbox_inches="tight")
print(f"Saved {Path(OUTPUT_PNG).resolve()}")
