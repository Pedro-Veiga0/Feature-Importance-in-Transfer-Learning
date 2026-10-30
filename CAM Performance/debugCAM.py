from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os
from pytorch_grad_cam import GradCAM, EigenCAM, GradCAMPlusPlus, ShapleyCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
path_models = [
    BASE_DIR / "CAM-Arnold" / "arnold.pt",
    # BASE_DIR / "CAM-Arnold" / "arnold2rapid.pt",
]
cam_methods = [GradCAM, ShapleyCAM]
cam_names = ["GradCAM", "ShapleyCAM"]
layer = 17

def pad_to_stride(img, stride=32, value=114):
    h, w = img.shape[:2]
    new_h = int(np.ceil(h / stride) * stride)
    new_w = int(np.ceil(w / stride) * stride)
    padded = np.full((new_h, new_w, 3), value, dtype=img.dtype)
    padded[:h, :w] = img
    return padded
all_cams = {}
for name, method in zip(cam_names, cam_methods):

    yolo = YOLO(f"{path_models[0]}")
    model = yolo.model
    model.eval().cuda()
    model.model[-1].inplace = False

    for p in model.parameters():
        p.requires_grad_(True)

    img = cv2.imread(
        BASE_DIR / "Arnold Dataset" / "dataset" / "images" / "train" / "20251215161829174_b.jpeg"
    )
    # img = cv2.imread(BASE_DIR / "Rapid" / "images" / "train" / "image_1.png")

    label_path = BASE_DIR / "Arnold Dataset" / "dataset" / "labels" / "train" / "20251215161829174_b.txt"
    # label_path = BASE_DIR / "Rapid" / "labels" / "train" / "image_1.txt"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = pad_to_stride(img_rgb, stride=32)
    img_norm = img_rgb.astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).cuda()
    input_tensor.requires_grad_(True)

    class YOLOConfidenceTarget:
        def __call__(self, outputs):
            preds = outputs[0]
            conf = preds[..., 4]
            max_conf = conf.max()
            # print(f"  Max confidence: {max_conf.item():.4f}")
            return max_conf

    targets = [YOLOConfidenceTarget()]

    # for layer in range(0, len(model.model) - 1):
    target_layer = model.model[layer]

    cam = method(
        model=model,
        target_layers=[target_layer],
    )

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # print(f"{name} - ", grayscale_cam)
    print(grayscale_cam.shape)
    cam_map = grayscale_cam[0]
    all_cams[name] = cam_map
    pd.DataFrame(cam_map).to_csv(
        BASE_DIR / name / f"{name}_layer{layer}_base_mask.csv",
        index=False,
        header=False,
    )
    pd.DataFrame(cam_map).to_excel(
        BASE_DIR / name / f"{name}_layer{layer}_base_mask.xlsx",
        index=False,
        header=False,
    )

    if "GradCAM" in all_cams and "ShapleyCAM" in all_cams:
        diff = all_cams["ShapleyCAM"] - all_cams["GradCAM"]
        ratio = np.divide(
            all_cams["ShapleyCAM"],
            all_cams["GradCAM"],
            out=np.full_like(all_cams["GradCAM"], np.nan),
            where=np.abs(all_cams["GradCAM"]) > 1e-12,
        )

        pd.DataFrame(diff).to_excel(
            BASE_DIR / f"cam_difference_layer{layer}.xlsx",
            index=False,
            header=False,
        )
        pd.DataFrame(ratio).to_excel(
            BASE_DIR / f"cam_ratio_layer{layer}.xlsx",
            index=False,
            header=False,
        )
    visualization = show_cam_on_image(img_norm, grayscale_cam[0], use_rgb=True)

    out_dir = BASE_DIR / name
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(
        BASE_DIR / name / f"{name}_layer{layer}_base.png",
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
    )

    # cam_df = pd.DataFrame(cam_map)
    # cam_df.to_excel(
    #     BASE_DIR / name / f"{name}_layer{layer}_baset_mask.xlsx", index=False, header=False
    # )
    # cam_df.to_csv(
    #     BASE_DIR / name / f"{name}_layer{layer}_baset_mask.csv", index=False, header=False
    # )

    # np.savetxt(
    #     BASE_DIR / name / f"{name}_layer{layer}_baset_mask_raw.csv",
    #     cam_map,
    #     delimiter=",",
    #     fmt="%.10f",
    # )