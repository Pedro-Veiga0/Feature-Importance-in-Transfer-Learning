from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os
import csv
from glob import glob

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ShapleyCAM

# =========================================================
# USER SETTINGS
# =========================================================
PROJECT_DIR = os.path.join(
    os.path.expanduser("~"), "Desktop", "Tese", "C\u00f3digo",
    "Codigo Dissertacao", "UDA test", "ConfMix-master"
)
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "CAM-FAITHFULNESS", "GradientBased")
RUN_SETTINGS = [
    (
        "Base",
        os.path.join(PROJECT_DIR, "CAM-Arnold", "arnold.pt"),
        os.path.join(PROJECT_DIR, "archive", "val", "img"),
        os.path.join(OUTPUT_ROOT, "Base"),
    ),
    (
        "UDA",
        os.path.join(PROJECT_DIR, "CAM-Arnold", "arnold2rapid.pt"),
        os.path.join(PROJECT_DIR, "Rapid", "images", "val"),
        os.path.join(OUTPUT_ROOT, "UDA"),
    ),
]

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
# MAX_IMAGES = 0
LAYER_IDX = 17

cam_methods = [GradCAM, GradCAMPlusPlus, LayerCAM, ShapleyCAM]
BASELINE_CAM = GradCAM

# =========================================================
# HELPERS
# =========================================================
def pad_to_stride(img, stride=32, value=114):
    h, w = img.shape[:2]
    new_h = int(np.ceil(h / stride) * stride)
    new_w = int(np.ceil(w / stride) * stride)
    padded = np.full((new_h, new_w, 3), value, dtype=img.dtype)
    padded[:h, :w] = img
    return padded

def get_image_paths(img_dir):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob(os.path.join(img_dir, ext)))
    return sorted(paths)

class YOLOConfidenceTarget:
    def __call__(self, outputs):
        preds = outputs[0]
        conf = preds[..., 4]
        return conf.max()

def get_model_score(model, input_tensor, target_fn):
    with torch.no_grad():
        outputs = model(input_tensor)
        return float(target_fn(outputs).item())

def cam_to_mask(cam_map):
    cam_map = np.maximum(cam_map, 0)
    if cam_map.max() > 0:
        cam_map = cam_map / cam_map.max()
    return cam_map.astype(np.float32)

def masked_input_from_cam(input_tensor, cam_map):
    cam_tensor = torch.from_numpy(cam_map).float().to(input_tensor.device)
    cam_tensor = cam_tensor.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    cam_tensor = cam_tensor.repeat(1, 3, 1, 1)         # [1,3,H,W]
    return input_tensor * cam_tensor

def compute_avg_drop_and_increase(original_score, masked_score):
    avg_drop = max(0.0, original_score - masked_score) / (original_score + 1e-8)
    avg_increase = 1.0 if masked_score > original_score else 0.0
    return float(avg_drop), float(avg_increase)

def run_evaluation(MODEL_PATH, IMG_DIR, OUT_DIR):
    # =========================================================
    # LOAD MODEL
    # =========================================================
    os.makedirs(OUT_DIR, exist_ok=True)

    yolo = YOLO(MODEL_PATH)
    model = yolo.model
    model.eval().cuda()
    model.model[-1].inplace = False

    for p in model.parameters():
        p.requires_grad_(True)

    target_fn = YOLOConfidenceTarget()

    # =========================================================
    # IMAGE LIST
    # =========================================================
    image_paths = get_image_paths(IMG_DIR)
    # if MAX_IMAGES is not None:
    # image_paths = image_paths[:MAX_IMAGES]
    image_paths = image_paths[:]

    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        raise RuntimeError("No images found.")

    # =========================================================
    # PRECOMPUTE BASELINE MASKED SCORES
    # =========================================================
    baseline_name = BASELINE_CAM.__name__
    baseline_scores = {}

    print(f"\nPrecomputing baseline masked scores using {baseline_name}...")

    baseline_cam_obj = BASELINE_CAM(model=model, target_layers=[model.model[LAYER_IDX]])

    for img_idx, img_path in enumerate(image_paths, 1):
        print(f"[Baseline {img_idx}/{len(image_paths)}] {os.path.basename(img_path)}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  Skipping: could not read image {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_padded = pad_to_stride(img_rgb, stride=32)
        img_norm = img_padded.astype(np.float32) / 255.0

        input_tensor = (
            torch.from_numpy(img_norm)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .cuda()
            .requires_grad_(True)
        )

        baseline_cam = baseline_cam_obj(input_tensor=input_tensor, targets=[target_fn])
        baseline_map = cam_to_mask(baseline_cam[0])
        baseline_masked_input = masked_input_from_cam(input_tensor, baseline_map)
        baseline_masked_score = get_model_score(model, baseline_masked_input, target_fn)
        baseline_scores[img_path] = round(baseline_masked_score, 4)

        del input_tensor
        torch.cuda.empty_cache()

    # =========================================================
    # MAIN EVALUATION
    # =========================================================
    comparison_rows = []

    for cam_model in cam_methods:
        cam_name = cam_model.__name__
        print(f"\n=== Evaluating {cam_name} ===")

        cam_obj = cam_model(model=model, target_layers=[model.model[LAYER_IDX]])

        per_image_rows = []
        avg_drop_list = []
        avg_increase_list = []
        sum_orig_score = 0.0
        sum_masked_score = 0.0
        sum_drop = 0.0
        sum_increase = 0.0
        valid_count = 0
        win_count = 0
        win_total = 0

        for img_idx, img_path in enumerate(image_paths, 1):
            print(f"[{img_idx}/{len(image_paths)}] {os.path.basename(img_path)}")

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print("  Skipping: could not read image")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_padded = pad_to_stride(img_rgb, stride=32)
            img_norm = img_padded.astype(np.float32) / 255.0

            input_tensor = (
                torch.from_numpy(img_norm)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .cuda()
                .requires_grad_(True)
            )

            original_score = get_model_score(model, input_tensor, target_fn)
            original_score = round(original_score, 4)

            grayscale_cam = cam_obj(input_tensor=input_tensor, targets=[target_fn])
            cam_map = cam_to_mask(grayscale_cam[0])

            masked_input = masked_input_from_cam(input_tensor, cam_map)
            masked_score = get_model_score(model, masked_input, target_fn)
            masked_score = round(masked_score, 4)

            # Previously low masked scores were skipped and not counted.
            # Keep images even if `masked_score` is very low so all images
            # from the validation set are recorded in the per-image CSV.
            # (No continue here.)

            avg_drop, avg_increase = compute_avg_drop_and_increase(original_score, masked_score)
            avg_drop = round(avg_drop, 4)
            avg_increase = round(avg_increase, 4)

            avg_drop_list.append(avg_drop)
            avg_increase_list.append(avg_increase)

            baseline_masked_score = baseline_scores.get(img_path, None)
            win = ""

            if baseline_masked_score is not None and cam_name != baseline_name:
                win = 1 if masked_score > baseline_masked_score else 0
                win_count += win
                win_total += 1

            sum_orig_score += original_score
            sum_masked_score += masked_score
            sum_drop += avg_drop
            sum_increase += avg_increase
            valid_count += 1

            per_image_rows.append({
                "image": os.path.basename(img_path),
                "method": cam_name,
                "layer": LAYER_IDX,
                "original_score": original_score,
                "masked_score": masked_score,
                "avg_drop": avg_drop,
                "avg_increase": avg_increase,
                "baseline_method": baseline_name if cam_name != baseline_name else "",
                "baseline_masked_score": baseline_masked_score if cam_name != baseline_name else "",
                "win": win,
                "avg_drop_mean_final": "",
                "avg_increase_mean_final": "",
            })

            # print(
            #     f"  Layer {LAYER_IDX:>2d} | "
            #     f"Masked={masked_score:.4f} | "
            #     f"Drop={avg_drop:.4f} | "
            #     f"AvgIncrease={avg_increase:.0f}"
            # )

            del input_tensor
            torch.cuda.empty_cache()

        mean_final = round(sum(avg_drop_list) / len(avg_drop_list), 4) if avg_drop_list else 0.0
        avg_increase_mean_final = round(sum(avg_increase_list) / len(avg_increase_list), 4) if avg_increase_list else 0.0
        mean_orig = round(sum_orig_score / valid_count, 4) if valid_count > 0 else 0.0
        mean_masked = round(sum_masked_score / valid_count, 4) if valid_count > 0 else 0.0
        mean_drop = round(sum_drop / valid_count, 4) if valid_count > 0 else 0.0
        mean_increase = round(sum_increase / valid_count, 4) if valid_count > 0 else 0.0
        win_pct = round((win_count / win_total) * 100.0, 2) if win_total > 0 else ""

        # Only add avg_drop_mean_final to the LAST processed row for this method
        if len(per_image_rows) > 0:
            per_image_rows[-1]["avg_drop_mean_final"] = mean_final
            per_image_rows[-1]["avg_increase_mean_final"] = avg_increase_mean_final

        detail_csv = os.path.join(
            OUT_DIR,
            f"{cam_name}_avgdrop_avgincrease_win_per_image.csv"
        )

        with open(detail_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "method",
                    "layer",
                    "original_score",
                    "masked_score",
                    "baseline_method",
                    "baseline_masked_score",
                    "win",
                    "avg_drop",
                    "avg_increase",
                    "avg_drop_mean_final",
                    "avg_increase_mean_final",
                ],
            )
            writer.writeheader()
            writer.writerows(per_image_rows)

        comparison_rows.append({
            "method": cam_name,
            "layer": LAYER_IDX,
            "num_images": valid_count,
            "mean_original_score": mean_orig,
            "mean_masked_score": mean_masked,
            "average_drop": mean_drop,
            "average_increase": mean_increase,
            "baseline_method": baseline_name if cam_name != baseline_name else "",
            "win_percentage": win_pct,
            "avg_drop_mean_final": mean_final,
            "avg_increase_mean_final": avg_increase_mean_final,
        })

        print(f"Saved per-image results to: {detail_csv}")
        print(f"{cam_name} | AvgDropMeanFinal={mean_final:.4f}")

    # =========================================================
    # SAVE FINAL COMPARISON CSV
    # =========================================================
    comparison_csv = os.path.join(
        OUT_DIR,
        "AllCam-basedMethods_avgdrop_avgincrease_win_summary.csv"
    )

    with open(comparison_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "layer",
                "num_images",
                "mean_original_score",
                "mean_masked_score",
                "average_drop",
                "average_increase",
                "baseline_method",
                "win_percentage",
                "avg_drop_mean_final",
                "avg_increase_mean_final",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"\nSaved final comparison summary to: {comparison_csv}")

    del yolo, model
    torch.cuda.empty_cache()


for setting_name, MODEL_PATH, IMG_DIR, OUT_DIR in RUN_SETTINGS:
    print(
        f"\nRunning {setting_name}: "
        f"model={os.path.basename(MODEL_PATH)}, images={IMG_DIR}"
    )
    run_evaluation(MODEL_PATH, IMG_DIR, OUT_DIR)
