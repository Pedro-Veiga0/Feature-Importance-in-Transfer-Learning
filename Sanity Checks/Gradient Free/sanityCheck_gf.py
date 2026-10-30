import copy
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "CAM-SanityCheck" / "GradientFree"
MODEL_PATH = Path(r"B:\UDA test\ConfMix-master\CAM-Arnold\best_cityescapes.pt")
IMAGE_PATH = Path(r"B:\UDA test\ConfMix-master\archive\train\img\train1.png")
TARGET_LAYER_IDX = 17
PROJECT_DIR = Path(r"B:\UDA test\ConfMix-master")
EXPERIMENTS = [
    (
        "Archive",
        PROJECT_DIR / "CAM-Arnold" / "arnold.pt",
        PROJECT_DIR / "archive" / "train" / "img" / "train1.png",
    ),
    (
        "Rapid",
        PROJECT_DIR / "CAM-Arnold" / "arnold2rapid.pt",
        PROJECT_DIR / "Rapid" / "images" / "train" / "image_1.png",
    ),
    (
        "Cityscapes",
        PROJECT_DIR / "CAM-Arnold" / "best_cityscapes.pt",
        Path(
            r"C:\Users\Pedro Afonso\Desktop\Tese\Código\Codigo Dissertacao\UDA test\ConfMix-master\CAM-SanityCheck\GradientBased\Cityscapes\input_image.png"
        ),
    ),
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAM_METHODS = [
    (EigenCAM, "EigenCAM"),
    (ScoreCAM, "ScoreCAM"),
]


def pad_to_stride(image, stride=32, value=114):
    height, width = image.shape[:2]
    padded_height = int(np.ceil(height / stride) * stride)
    padded_width = int(np.ceil(width / stride) * stride)
    padded = np.full(
        (padded_height, padded_width, 3), value, dtype=image.dtype
    )
    padded[:height, :width] = image
    return padded


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
        average_rank = 0.5 * (start + end) + 1.0
        ranks[order[start : end + 1]] = average_rank
        start = end + 1

    return ranks


def pearson_corr(first, second):
    first = first.flatten().astype(np.float64)
    second = second.flatten().astype(np.float64)
    first_std, second_std = first.std(), second.std()
    if first_std < 1e-12 or second_std < 1e-12:
        return 0.0
    return float(
        (
            (first - first.mean())
            / first_std
            * (second - second.mean())
            / second_std
        ).mean()
    )


def spearman_corr(first, second):
    return pearson_corr(
        rankdata_average(first.flatten()),
        rankdata_average(second.flatten()),
    )


def ssim_score(first, second):
    return float(ssim(first, second, data_range=1.0))


def reinitialize_module_weights(module):
    reset_parameters = getattr(module, "reset_parameters", None)
    if callable(reset_parameters):
        reset_parameters()


def restore_gradients(module):
    for parameter in module.parameters():
        parameter.requires_grad_(True)


class YOLOScoreCamWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_tensor):
        output = self.model(image_tensor)
        confidence_scores = output[0][..., 4].max(dim=-1).values
        return confidence_scores.unsqueeze(-1)


class YOLOConfidenceTarget:
    def __call__(self, outputs):
        return outputs[0][..., 4].max()


class YOLOScoreCamTarget:
    def __call__(self, output):
        return output[0]


def compute_cam(cam_class, model, input_tensor, layer_idx):
    if cam_class is ScoreCAM:
        wrapped_model = YOLOScoreCamWrapper(model).to(DEVICE).eval()
        cam = ScoreCAM(
            model=wrapped_model,
            target_layers=[wrapped_model.model.model[layer_idx]],
        )
        try:
            with torch.no_grad():
                result = cam(
                    input_tensor=input_tensor,
                    targets=[YOLOScoreCamTarget()],
                )
            return normalize_cam(result[0])
        finally:
            cam.activations_and_grads.release()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    cam = cam_class(model=model, target_layers=[model.model[layer_idx]])
    try:
        result = cam(
            input_tensor=input_tensor,
            targets=[YOLOConfidenceTarget()],
        )
        return normalize_cam(result[0])
    finally:
        cam.activations_and_grads.release()


def save_overlay(image, cam_map, output_path):
    visualization = show_cam_on_image(image, cam_map, use_rgb=True)
    if not cv2.imwrite(
        str(output_path),
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
    ):
        raise OSError(f"Could not save CAM overlay: {output_path}")


def prepare_model(model_path):
    yolo = YOLO(str(model_path))
    model = yolo.model.to(DEVICE).eval()
    model.model[-1].inplace = False
    return model


def run_experiment(experiment_name, model_path, image_path):
    output_dir = OUTPUT_ROOT / (
        f"{experiment_name}_{image_path.stem}_{model_path.stem}"
    )
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read input image: {image_path}")

    original_height, original_width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_normalized = (
        pad_to_stride(image_rgb).astype(np.float32) / 255.0
    )
    input_tensor = (
        torch.from_numpy(image_normalized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    if not cv2.imwrite(
        str(output_dir / "input_image.png"),
        image_bgr[:original_height, :original_width],
    ):
        raise OSError(f"Could not save input image: {output_dir / 'input_image.png'}")
    trained_model = prepare_model(model_path)
    all_rows = [("cam_method", "test", "step", "layer", "spearman", "ssim")]

    for cam_class, cam_name in CAM_METHODS:
        print(f"\n{'=' * 60}")
        print(f"  CAM METHOD: {cam_name}")
        print(f"{'=' * 60}")

        method_dir = output_dir / cam_name
        method_dir.mkdir(parents=True, exist_ok=True)

        original_cam = compute_cam(
            cam_class, trained_model, input_tensor, TARGET_LAYER_IDX
        )
        save_overlay(
            image_normalized,
            original_cam,
            method_dir / f"00_original_layer{TARGET_LAYER_IDX}.png",
        )

        print(f"\n  TEST 1 — Cascading ({cam_name})")
        print(f"  {'Step':>4} | {'Rand to':>7} | {'Spearman':>9} | {'SSIM':>7}")
        print(f"  {'-' * 38}")

        randomized_model = copy.deepcopy(trained_model).to(DEVICE).eval()
        randomized_model.model[-1].inplace = False

        first_step = 0 if cam_name == "ScoreCAM" else 1
        for step, randomized_layer in enumerate(
            range(TARGET_LAYER_IDX, -1, -1), start=first_step
        ):
            randomized_model.model[randomized_layer].apply(
                reinitialize_module_weights
            )
            randomized_model.model[randomized_layer].apply(restore_gradients)

            randomized_cam = compute_cam(
                cam_class,
                randomized_model,
                input_tensor,
                TARGET_LAYER_IDX,
            )
            spearman = spearman_corr(original_cam, randomized_cam)
            similarity = ssim_score(original_cam, randomized_cam)

            save_overlay(
                image_normalized,
                randomized_cam,
                method_dir
                / f"T1_{cam_name}_step{step:02d}_randTo{randomized_layer:02d}.png",
            )
            print(
                f"  {step:>4} | {randomized_layer:>7} | "
                f"{spearman:>+9.4f} | {similarity:>7.4f}"
            )
            all_rows.append(
                (
                    cam_name,
                    "cascading",
                    step,
                    randomized_layer,
                    f"{spearman:.6f}",
                    f"{similarity:.6f}",
                )
            )

        del randomized_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        print(f"\n  TEST 2 — Independent ({cam_name})")
        print(f"  {'Layer':>6} | {'Spearman':>9} | {'SSIM':>7}")
        print(f"  {'-' * 30}")

        for randomized_layer in range(TARGET_LAYER_IDX, -1, -1):
            independent_model = copy.deepcopy(trained_model).to(DEVICE).eval()
            independent_model.model[-1].inplace = False
            for parameter in independent_model.parameters():
                parameter.requires_grad_(True)

            independent_model.model[randomized_layer].apply(
                reinitialize_module_weights
            )
            independent_model.model[randomized_layer].apply(restore_gradients)

            independent_cam = compute_cam(
                cam_class,
                independent_model,
                input_tensor,
                TARGET_LAYER_IDX,
            )
            spearman = spearman_corr(original_cam, independent_cam)
            similarity = ssim_score(original_cam, independent_cam)

            save_overlay(
                image_normalized,
                independent_cam,
                method_dir
                / f"T2_{cam_name}_indep_layer{randomized_layer:02d}.png",
            )
            print(
                f"  {randomized_layer:>6} | "
                f"{spearman:>+9.4f} | {similarity:>7.4f}"
            )
            all_rows.append(
                (
                    cam_name,
                    "independent",
                    "-",
                    randomized_layer,
                    f"{spearman:.6f}",
                    f"{similarity:.6f}",
                )
            )

            del independent_model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    csv_path = output_dir / "sanity_check_all_methods_gf_ce.csv"
    with csv_path.open("w", newline="") as csv_file:
        csv.writer(csv_file).writerows(all_rows)
    print(f"\nAll results → {csv_path}")


def run_sanity_check():
    for experiment_name, model_path, image_path in EXPERIMENTS:
        print(
            f"\nRunning {experiment_name}: model={model_path.name}, "
            f"image={image_path.name}"
        )
        run_experiment(experiment_name, model_path, image_path)


if __name__ == "__main__":
    run_sanity_check()
