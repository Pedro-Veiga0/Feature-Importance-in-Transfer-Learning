r"""Run the Gradient-Free CAM sanity checks on the top-10 SSIM representative images in Colab.

Evaluates:
  - Base: Arnold dataset using the arnold.pt model checkpoint
  - BaseT: Rapid dataset using the arnold.pt model checkpoint
  - UDA: Rapid dataset using the arnold2rapid.pt model checkpoint
"""

import copy
import csv
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import (
    EigenCAM,
    ScoreCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.metrics import structural_similarity
from ultralytics import YOLO

PROJECT_DIR = Path(os.environ.get("CAM_EVAL_ROOT", "/content")).expanduser()
TOP_K = 10
TARGET_LAYER_IDX = 17
SAVE_OVERLAYS = False

# Configure all three study configurations
RUN_SETTINGS = {
    "Base": {
        "dataset": "Arnold",
        "image_dir": Path("Arnold Dataset") / "dataset" / "images" / "val",
        "selection_csv": Path("CAM-Loc-Eval") / "SSIM" / "Arnold_top10.csv",
        "checkpoint": "arnold.pt",
    },
    "BaseT": {
        "dataset": "Rapid",
        "image_dir": Path("Rapid") / "images" / "val",
        "selection_csv": Path("CAM-Loc-Eval") / "SSIM" / "Rapid_top10.csv",
        "checkpoint": "arnold.pt",
    },
    "UDA": {
        "dataset": "Rapid",
        "image_dir": Path("Rapid") / "images" / "val",
        "selection_csv": Path("CAM-Loc-Eval") / "SSIM" / "Rapid_top10.csv",
        "checkpoint": "arnold2rapid.pt",
    }
}

MODEL_DIR = Path("CAM-Arnold")
OUTPUT_DIR = Path("CAM-SanityCheck") / "SSIM-10"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
CAM_METHODS = (
    (EigenCAM, "EigenCAM"),
    (ScoreCAM, "ScoreCAM"),
)


class YOLOConfidenceTarget:
    def __call__(self, outputs):
        return outputs[0][..., 4].max()


class YOLOScoreCamWrapper(torch.nn.Module):
    """
    ScoreCAM internally does:
        outputs = [target(o).cpu().item() for o in batch_output]
    so it iterates over the first dimension of whatever the model returns.
    For a [batch, 1] tensor, each `o` has shape [1] -> target returns a scalar.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)          # tuple: (inference_out, train_out)
        # inference_out: [batch, num_detections, num_outputs]
        # index 4 is objectness; take the max detection per image
        scores = out[0][..., 4].max(dim=-1).values   # [batch]
        return scores.unsqueeze(-1)                   # [batch, 1]


class YOLOScoreCAMConfidenceTarget:
    """Receives a single-image slice [1] from the wrapper output."""
    def __call__(self, output):
        return output[0]   # scalar tensor


def load_selected_images(dataset_name, image_dir, selection_csv):
    if not selection_csv.is_file():
        raise FileNotFoundError(
            f"Top-{TOP_K} SSIM selection CSV not found: {selection_csv}."
        )
    with selection_csv.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"dataset", "rank", "image"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Expected columns {sorted(required)} in {selection_csv}"
            )
        rows = [
            row
            for row in reader
            if row["dataset"] == dataset_name and int(row["rank"]) <= TOP_K
        ]

    rows.sort(key=lambda row: int(row["rank"]))
    if len(rows) != TOP_K or [int(row["rank"]) for row in rows] != list(
        range(1, TOP_K + 1)
    ):
        raise ValueError(
            f"Expected exactly ranks 1-{TOP_K} for {dataset_name} in "
            f"{selection_csv}; found {[row["rank"] for row in rows]}"
        )

    image_paths = []
    for row in rows:
        image_name = row["image"]
        if Path(image_name).name != image_name:
            raise ValueError(f"Expected an image filename, not a path: {image_name}")
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Selected validation image does not exist: {image_path}"
            )
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported selected image format: {image_path}")
        image_paths.append(image_path)
    return image_paths


def pad_to_stride(image, stride=32, value=114):
    height, width = image.shape[:2]
    padded_height = math.ceil(height / stride) * stride
    padded_width = math.ceil(width / stride) * stride
    padded = np.full((padded_height, padded_width, 3), value, dtype=image.dtype)
    padded[:height, :width] = image
    return padded


def load_image(image_path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read selected image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    padded = pad_to_stride(image_rgb)
    image_norm = padded.astype(np.float32) / 255.0
    return image_bgr, image_norm, height, width


def normalize_cam(cam):
    cam = np.maximum(cam, 0)
    maximum = cam.max()
    if maximum > 0:
        cam = cam / maximum
    return cam.astype(np.float32)


def rankdata_average(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = 0.5 * (start + end) + 1.0
        start = end + 1
    return ranks


def spearman_score(reference, candidate):
    reference_ranks = rankdata_average(reference.ravel())
    candidate_ranks = rankdata_average(candidate.ravel())
    reference_centered = reference_ranks - reference_ranks.mean()
    candidate_centered = candidate_ranks - candidate_ranks.mean()
    numerator = np.mean(reference_centered * candidate_centered)
    denominator = np.sqrt(
        np.mean(reference_centered**2) * np.mean(candidate_centered**2)
    )
    if denominator < 1e-12:
        return 0.0
    return float(numerator / denominator)


def ssim_score(reference, candidate):
    return float(
        structural_similarity(reference, candidate, data_range=1.0, full=False)
    )


def get_model_layers(model):
    layers = getattr(model, "model", None)
    if layers is None:
        raise TypeError("Expected the YOLO model to expose a .model layer sequence")
    return layers


def disable_inplace(model):
    head = get_model_layers(model)[-1]
    if hasattr(head, "inplace"):
        head.inplace = False


def reinitialize_module(module):
    reset_parameters = getattr(module, "reset_parameters", None)
    if callable(reset_parameters):
        reset_parameters()


def restore_gradients(module):
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def compute_cams(cam_class, model, image_paths, device, layer_idx):
    score_cam = cam_class is ScoreCAM
    if score_cam:
        cam_model = YOLOScoreCamWrapper(model).eval()
        target_layers = [get_model_layers(model)[layer_idx]]
        targets = [YOLOScoreCAMConfidenceTarget()]
    else:
        cam_model = model
        target_layers = [get_model_layers(model)[layer_idx]]
        targets = [YOLOConfidenceTarget()]

    cams = []
    with cam_class(model=cam_model, target_layers=target_layers) as cam:
        for image_path in image_paths:
            _, image_norm, height, width = load_image(image_path)
            input_tensor = (
                torch.from_numpy(image_norm)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            grad_context = torch.no_grad() if score_cam else nullcontext()
            with grad_context:
                cam_map = cam(input_tensor=input_tensor, targets=targets)[0]
            cams.append(normalize_cam(cam_map[:height, :width]))
            del input_tensor
    return cams


def save_overlay(image_path, cam_map, output_path):
    _, image_norm, height, width = load_image(image_path)
    image_norm = image_norm[:height, :width]
    visualization = show_cam_on_image(image_norm, cam_map, use_rgb=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    ):
        raise OSError(f"Could not save CAM overlay: {output_path}")


def record_scores(
    per_image_writer,
    grouped_values,
    setting_name,
    cam_name,
    test_name,
    step,
    layer_idx,
    image_paths,
    reference_cams,
    randomized_cams,
    output_dir,
):
    for image_path, reference, randomized in zip(
        image_paths, reference_cams, randomized_cams
    ):
        spearman = spearman_score(reference, randomized)
        ssim = ssim_score(reference, randomized)
        per_image_writer.writerow(
            {
                "setting": setting_name,
                "cam_method": cam_name,
                "test": test_name,
                "step": "" if step is None else step,
                "layer": layer_idx,
                "image": image_path.name,
                "spearman": f"{spearman:.8f}",
                "ssim": f"{ssim:.8f}",
            }
        )
        grouped_values[(setting_name, cam_name, test_name, step, layer_idx, "spearman")].append(
            spearman
        )
        grouped_values[(setting_name, cam_name, test_name, step, layer_idx, "ssim")].append(ssim)

        if SAVE_OVERLAYS:
            image_output = output_dir / setting_name / cam_name / image_path.stem
            if test_name == "cascading":
                filename = f"T1_{cam_name}_step{step:02d}_randTo{layer_idx:02d}.png"
            else:
                filename = f"T2_{cam_name}_indep_layer{layer_idx:02d}.png"
            save_overlay(image_path, randomized, image_output / filename)


def write_summary(summary_path, grouped_values):
    fields = (
        "setting",
        "cam_method",
        "test",
        "step",
        "layer",
        "metric",
        "n_images",
        "valid_n",
        "mean",
        "median",
        "std",
    )
    with summary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for (setting_name, cam_name, test_name, step, layer_idx, metric), values in sorted(
            grouped_values.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2],
                -1 if item[0][3] is None else item[0][3],
                item[0][4],
                item[0][5],
            ),
        ):
            numeric_values = np.asarray(values, dtype=np.float64)
            valid_values = numeric_values[np.isfinite(numeric_values)]
            mean_val = float(np.mean(valid_values)) if valid_values.size else float("nan")
            median_val = (
                float(np.median(valid_values)) if valid_values.size else float("nan")
            )
            standard_deviation = (
                float(np.std(valid_values, ddof=1)) if valid_values.size > 1 else float("nan")
            )
            writer.writerow(
                {
                    "setting": setting_name,
                    "cam_method": cam_name,
                    "test": test_name,
                    "step": "" if step is None else step,
                    "layer": layer_idx,
                    "metric": metric,
                    "n_images": len(values),
                    "valid_n": len(valid_values),
                    "mean": f"{mean_val:.8f}",
                    "median": f"{median_val:.8f}",
                    "std": f"{standard_deviation:.8f}",
                }
            )


def main():
    if not PROJECT_DIR.is_dir():
        raise FileNotFoundError(f"Project directory not found: {PROJECT_DIR}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = PROJECT_DIR / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image_path = output_dir / "sanity_check_all_settings_per_image_gf.csv"
    summary_path = output_dir / "sanity_check_all_settings_summary_gf.csv"
    per_image_fields = (
        "setting",
        "cam_method",
        "test",
        "step",
        "layer",
        "image",
        "spearman",
        "ssim",
    )
    grouped_values = defaultdict(list)

    print(f"Project base: {PROJECT_DIR}; device: {device}; saving overlays: {SAVE_OVERLAYS}\n")

    with per_image_path.open("w", newline="", encoding="utf-8-sig") as per_image_file:
        writer = csv.DictWriter(per_image_file, fieldnames=per_image_fields)
        writer.writeheader()

        for setting_name, config in RUN_SETTINGS.items():
            dataset_name = config["dataset"]
            image_dir = PROJECT_DIR / config["image_dir"]
            selection_csv = PROJECT_DIR / config["selection_csv"]
            model_path = PROJECT_DIR / MODEL_DIR / config["checkpoint"]

            print(f"=================== Running {setting_name} ===================")
            print(f"Dataset: {dataset_name} | Checkpoint: {config["checkpoint"]}")

            if not image_dir.is_dir():
                print(f"Skipping {setting_name}: Image dir {image_dir} not found")
                continue
            if not model_path.is_file():
                print(f"Skipping {setting_name}: Model checkpoint {model_path} not found")
                continue

            image_paths = load_selected_images(dataset_name, image_dir, selection_csv)
            yolo = YOLO(str(model_path))
            if not isinstance(yolo.model, torch.nn.Module):
                raise TypeError(f"YOLO did not load a model from {model_path}")
            trained_model = yolo.model.to(device).eval()
            disable_inplace(trained_model)
            for parameter in trained_model.parameters():
                parameter.requires_grad_(True)

            for cam_class, cam_name in CAM_METHODS:
                print(f"\n  --- {cam_name}: baseline CAMs ---")
                reference_cams = compute_cams(
                    cam_class, trained_model, image_paths, device, TARGET_LAYER_IDX
                )

                print(f"  --- {cam_name}: cascading randomization ---")
                cascading_model = copy.deepcopy(trained_model).to(device).eval()
                disable_inplace(cascading_model)
                for step, layer_idx in enumerate(
                    range(TARGET_LAYER_IDX, -1, -1),
                    start=0 if cam_name == "ScoreCAM" else 1,
                ):
                    if cam_name == "EigenCAM" and layer_idx == 0:
                        print(f"    EigenCAM step {step} layer 0: Skipped to prevent RAM crash.")
                        continue

                    get_model_layers(cascading_model)[layer_idx].apply(reinitialize_module)
                    get_model_layers(cascading_model)[layer_idx].apply(restore_gradients)
                    randomized_cams = compute_cams(
                        cam_class, cascading_model, image_paths, device, TARGET_LAYER_IDX
                    )
                    record_scores(
                        writer,
                        grouped_values,
                        setting_name,
                        cam_name,
                        "cascading",
                        step,
                        layer_idx,
                        image_paths,
                        reference_cams,
                        randomized_cams,
                        output_dir,
                    )
                    del randomized_cams
                del cascading_model

                print(f"  --- {cam_name}: independent randomization ---")
                for layer_idx in range(TARGET_LAYER_IDX, -1, -1):
                    if cam_name == "EigenCAM" and layer_idx == 0:
                        print(f"    EigenCAM layer 0: Skipped to prevent RAM crash.")
                        continue

                    independent_model = copy.deepcopy(trained_model).to(device).eval()
                    disable_inplace(independent_model)
                    layers = get_model_layers(independent_model)
                    layers[layer_idx].apply(reinitialize_module)
                    layers[layer_idx].apply(restore_gradients)
                    randomized_cams = compute_cams(
                        cam_class, independent_model, image_paths, device, TARGET_LAYER_IDX
                    )
                    record_scores(
                        writer,
                        grouped_values,
                        setting_name,
                        cam_name,
                        "independent",
                        None,
                        layer_idx,
                        image_paths,
                        reference_cams,
                        randomized_cams,
                        output_dir,
                    )
                    del randomized_cams, independent_model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                del reference_cams
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del trained_model, yolo
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_summary(summary_path, grouped_values)
    print(f"\nSaved metrics to: {per_image_path}")
    print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()