r"""Evaluate CAM localization on the SSIM-selected Arnold and Rapid images.

For Google Colab, set CAM_EVAL_ROOT to the project directory containing
CAM-Arnold/, Rapid/, and Arnold Dataset/ before running this script.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import cv2
import numpy as np
import torch
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ShapleyCAM,
)
from ultralytics import YOLO

import pytorch_grad_cam.activations_and_gradients
original_clear = pytorch_grad_cam.activations_and_gradients.ActivationsAndGradients.clear

def patched_clear(self):
    for g in self.gradients:
        if isinstance(g, torch.Tensor):
            try:
                g.detach_()
            except RuntimeError:
                g.detach()
    self.activations.clear()
    self.gradients.clear()

pytorch_grad_cam.activations_and_gradients.ActivationsAndGradients.clear = patched_clear

PROJECT_DIR = Path(
    os.environ.get("CAM_EVAL_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser()
TOP_K = 10
TARGET_LAYER_IDX = 23
MODEL_DIR = Path("CAM-Arnold")
OUTPUT_DIR = PROJECT_DIR / "CAM-Loc-Eval" / "SSIM-10"
PER_IMAGE_CSV = OUTPUT_DIR / "localization_per_image_gb_v2.csv"
SUMMARY_CSV = OUTPUT_DIR / "localization_summary_gb_v2.csv"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

RUN_SETTINGS = {
    "Base": {
        "dataset": "Arnold",
        "image_dir": Path("Arnold Dataset") / "dataset" / "images" / "val",
        "label_dir": Path("Arnold Dataset") / "dataset" / "labels" / "val",
        "selection_csv": Path("CAM-Loc-Eval") / "SSIM" / "Arnold_top10.csv",
        "checkpoint": "arnold.pt",
    },
    "UDA": {
        "dataset": "Rapid",
        "image_dir": Path("Rapid") / "images" / "val",
        "label_dir": Path("Rapid") / "labels" / "val",
        "selection_csv": Path("CAM-Loc-Eval") / "SSIM" / "Rapid_top10.csv",
        "checkpoint": "arnold2rapid.pt",
    },
    "BaseT": {
        "dataset": "Rapid",
        "image_dir": Path("Rapid") / "images" / "val",
        "label_dir": Path("Rapid") / "labels" / "val",
        "selection_csv": Path("CAM-Loc-Eval") / "SSIM" / "Rapid_top10.csv",
        "checkpoint": "arnold.pt",
    }
}

CAM_METHODS = (
    ("GradCAM", GradCAM),
    ("GradCAM++", GradCAMPlusPlus),
    ("LayerCAM", LayerCAM),
    ("ShapleyCAM", ShapleyCAM),
)


def pad_to_stride(img, stride=32, value=114):
    height, width = img.shape[:2]
    padded_height = int(np.ceil(height / stride) * stride)
    padded_width = int(np.ceil(width / stride) * stride)
    padded = np.full((padded_height, padded_width, 3), value, dtype=img.dtype)
    padded[:height, :width] = img
    return padded


def load_gt_boxes(label_path, img_w, img_h):
    """Read YOLO xywh labels and return clipped pixel-coordinate xyxy boxes."""
    boxes = []
    with label_path.open(encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(
                    f"Invalid YOLO label at {label_path}:{line_number}: "
                    f" {line.rstrip()}"
                )
            _, cx, cy, box_w, box_h = map(float, parts[:5])
            x1 = int((cx - box_w / 2) * img_w)
            y1 = int((cy - box_h / 2) * img_h)
            x2 = int((cx + box_w / 2) * img_w)
            y2 = int((cy + box_h / 2) * img_h)
            boxes.append(
                (
                    max(0, x1),
                    max(0, y1),
                    min(img_w, x2),
                    min(img_h, y2),
                )
            )
    return boxes


def ebpg(cam, gt_boxes, orig_h, orig_w):
    """Return the fraction of CAM energy inside the union of GT boxes."""
    cam_content = np.maximum(cam[:orig_h, :orig_w], 0)
    total_energy = cam_content.sum()
    if total_energy <= 0:
        return 0.0

    box_mask = np.zeros_like(cam_content, dtype=np.float32)
    for x1, y1, x2, y2 in gt_boxes:
        box_mask[y1:y2, x1:x2] = 1.0
    return float((cam_content * box_mask).sum() / total_energy)


def pointing_game(cam, gt_boxes, orig_h, orig_w):
    """Return 1 if a maximum-activation pixel is inside a GT box."""
    cam_content = cam[:orig_h, :orig_w]
    max_value = cam_content.max()
    if max_value <= 0:
        return 0.0

    peak_ys, peak_xs = np.where(cam_content == max_value)
    for peak_y, peak_x in zip(peak_ys, peak_xs):
        for x1, y1, x2, y2 in gt_boxes:
            if x1 <= peak_x < x2 and y1 <= peak_y < y2:
                return 1.0
    return 0.0


class YOLOConfidenceTarget:
    def __call__(self, outputs):
        predictions = outputs[0]
        return predictions[..., 4].max()


def load_selected_images(dataset_name, image_dir, label_dir, selection_csv):
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not selection_csv.is_file():
        raise FileNotFoundError(f"SSIM selection CSV not found: {selection_csv}")

    images = []
    with selection_csv.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"dataset", "rank", "image"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Expected columns {sorted(required_columns)} in {selection_csv}"
            )
        selected_rows = [
            row for row in reader if row["dataset"] == dataset_name
        ]

    selected_rows.sort(key=lambda row: int(row["rank"]))
    if len(selected_rows) < TOP_K:
        raise ValueError(
            f"Expected {TOP_K} {dataset_name} selections in {selection_csv}; "
            f"found {len(selected_rows)}."
        )

    for row in selected_rows[:TOP_K]:
        image_name = Path(row["image"]).name
        if Path(image_name).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Invalid selected image name: {row['image']}")
        image_path = image_dir / image_name
        label_path = label_dir / f"{Path(image_name).stem}.txt"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Selected validation image not found: {image_path}"
            )
        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing label file for {image_name}: {label_path}"
            )
        images.append(image_path)

    return images


def summarize(values):
    return (
        mean(values),
        median(values),
        stdev(values) if len(values) > 1 else 0.0,
    )


def evaluate_checkpoint(checkpoint_path, dataset_name, image_dir, label_dir, selection_csv, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    images = load_selected_images(dataset_name, image_dir, label_dir, selection_csv)
    checkpoint_start = perf_counter()
    yolo = YOLO(str(checkpoint_path))
    model = yolo.model.to(device)
    model.eval()
    model.model[-1].inplace = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    last_feature_layer = len(model.model) - 2
    if TARGET_LAYER_IDX > last_feature_layer:
        raise ValueError(
            f"TARGET_LAYER_IDX={TARGET_LAYER_IDX} exceeds the last feature layer "
            f"({last_feature_layer}); the detection head is excluded."
        )

    targets = [YOLOConfidenceTarget()]
    image_rows = []
    summary_values = defaultdict(lambda: {"ebpg": [], "pg": []})

    print(
        f"\nEvaluating {checkpoint_path.name} on {dataset_name}: "
        f"{len(images)} selected images ({device})."
    )
    for cam_name, cam_method in CAM_METHODS:
        print(f"  Method: {cam_name}")
        for layer_idx in range(TARGET_LAYER_IDX + 1):
            with cam_method(
                model=model,
                target_layers=[model.model[layer_idx]],
            ) as cam:
                for image_index, image_path in enumerate(images, start=1):
                    image_bgr = cv2.imdecode(
                        np.frombuffer(image_path.read_bytes(), dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if image_bgr is None:
                        raise ValueError(
                            f"Could not read validation image: {image_path}"
                        )

                    orig_h, orig_w = image_bgr.shape[:2]
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    image_padded = pad_to_stride(image_rgb)
                    pad_h, pad_w = image_padded.shape[:2]
                    image_norm = image_padded.astype(np.float32) / 255.0
                    content_mask = np.zeros((pad_h, pad_w), dtype=np.float32)
                    content_mask[:orig_h, :orig_w] = 1.0
                    gt_boxes = load_gt_boxes(
                        label_dir / f"{image_path.stem}.txt",
                        orig_w,
                        orig_h,
                    )

                    input_tensor = (
                        torch.from_numpy(image_norm)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device)
                        .requires_grad_(True)
                    )
                    grayscale_cam = cam(
                        input_tensor=input_tensor,
                        targets=targets,
                    )[0]
                    cam_masked = grayscale_cam * content_mask
                    if cam_masked.max() > 0:
                        cam_masked = cam_masked / cam_masked.max()

                    ebpg_score = ebpg(cam_masked, gt_boxes, orig_h, orig_w)
                    pg_score = pointing_game(cam_masked, gt_boxes, orig_h, orig_w)
                    image_rows.append(
                        {
                            "dataset": dataset_name,
                            "checkpoint": str(checkpoint_path),
                            "cam_method": cam_name,
                            "layer": layer_idx,
                            "image": image_path.name,
                            "ebpg": ebpg_score,
                            "pg": pg_score,
                        }
                    )
                    summary_values[(cam_name, layer_idx)]["ebpg"].append(ebpg_score)
                    summary_values[(cam_name, layer_idx)]["pg"].append(pg_score)

                    if image_index == len(images):
                        print(
                            f"    {cam_name} layer {layer_idx}: "
                            f"{image_index}/{len(images)} images"
                        )

    summary_rows = []
    for cam_name, _ in CAM_METHODS:
        for layer_idx in range(TARGET_LAYER_IDX + 1):
            ebpg_mean, ebpg_median, ebpg_std = summarize(
                summary_values[(cam_name, layer_idx)]["ebpg"]
            )
            pg_mean, pg_median, pg_std = summarize(
                summary_values[(cam_name, layer_idx)]["pg"]
            )
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "checkpoint": str(checkpoint_path),
                    "cam_method": cam_name,
                    "layer": layer_idx,
                    "num_images": len(images),
                    "ebpg_mean": ebpg_mean,
                    "ebpg_median": ebpg_median,
                    "ebpg_std": ebpg_std,
                    "pg_mean": pg_mean,
                    "pg_median": pg_median,
                    "pg_std": pg_std,
                }
            )

    elapsed = perf_counter() - checkpoint_start
    del model, yolo
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Checkpoint evaluation completed in {elapsed:.2f} seconds.")
    return image_rows, summary_rows, elapsed


def main():
    print(f"Project directory: {PROJECT_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_rows = []
    summary_rows = []
    run_start = perf_counter()

    for setting_name, config in RUN_SETTINGS.items():
        dataset_name = config["dataset"]
        image_dir = PROJECT_DIR / config["image_dir"]
        label_dir = PROJECT_DIR / config["label_dir"]
        selection_csv = PROJECT_DIR / config["selection_csv"]
        checkpoint_path = (PROJECT_DIR / MODEL_DIR / config["checkpoint"]).resolve()

        checkpoint_images, checkpoint_summary, elapsed = evaluate_checkpoint(
            checkpoint_path,
            dataset_name,
            image_dir,
            label_dir,
            selection_csv,
            device,
        )
        for row in checkpoint_images:
            image_rows.append(
                {
                    "setting": setting_name,
                    **row,
                    "checkpoint_elapsed_seconds": f"{elapsed:.2f}",
                }
            )
        for row in checkpoint_summary:
            summary_rows.append(
                {
                    "setting": setting_name,
                    **row,
                    "checkpoint_elapsed_seconds": f"{elapsed:.2f}",
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_image_fields = (
        "dataset",
        "setting",
        "checkpoint",
        "cam_method",
        "layer",
        "image",
        "ebpg",
        "pg",
        "checkpoint_elapsed_seconds",
    )
    with PER_IMAGE_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=per_image_fields)
        writer.writeheader()
        for row in image_rows:
            row["ebpg"] = f"{row['ebpg']:.8f}"
            row["pg"] = f"{row['pg']:.8f}"
            writer.writerow(row)

    summary_fields = (
        "dataset",
        "setting",
        "checkpoint",
        "cam_method",
        "layer",
        "num_images",
        "ebpg_mean",
        "ebpg_median",
        "ebpg_std",
        "pg_mean",
        "pg_median",
        "pg_std",
        "checkpoint_elapsed_seconds",
    )
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            for metric in (
                "ebpg_mean",
                "ebpg_median",
                "ebpg_std",
                "pg_mean",
                "pg_median",
                "pg_std",
            ):
                row[metric] = f"{row[metric]:.8f}"
            writer.writerow(row)

    print(f"\nSaved per-image scores to: {PER_IMAGE_CSV}")
    print(f"Saved mean/median/std summary to: {SUMMARY_CSV}")
    print(f"Total elapsed time: {perf_counter() - run_start:.2f} seconds")

if __name__ == "__main__":
    main()