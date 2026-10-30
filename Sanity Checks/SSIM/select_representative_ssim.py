"""Rank Arnold and Rapid validation images by mean pairwise SSIM.

The highest-ranked images are medoids: each is, on average, most similar to
the other images in its validation split. Images are converted to grayscale
and resized to a square thumbnail before comparison so pairwise SSIM uses a
consistent, affordable resolution.

Run with:
    python select_representative_ssim.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from skimage.metrics import structural_similarity


BASE_DIR = Path(__file__).resolve().parent
DATASETS = {
    "Arnold": BASE_DIR / "Arnold Dataset" / "dataset" / "images" / "val",
    "Rapid": BASE_DIR / "Rapid" / "images" / "val",
}
OUTPUT_DIR = BASE_DIR / "CAM-Loc-Eval" / "SSIM"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def load_thumbnails(image_paths: list[Path], size: int) -> list[np.ndarray]:
    thumbnails = []
    for image_path in image_paths:
        image = cv2.imdecode(
            np.frombuffer(image_path.read_bytes(), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None:
            raise ValueError(f"Could not read validation image: {image_path}")
        thumbnails.append(
            cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        )
    return thumbnails


def rank_images(image_paths: list[Path], size: int) -> list[tuple[Path, float]]:
    if len(image_paths) < 2:
        raise ValueError("At least two validation images are required for SSIM ranking.")

    thumbnails = load_thumbnails(image_paths, size)
    score_totals = np.zeros(len(thumbnails), dtype=np.float64)

    for left_index, left_image in enumerate(thumbnails[:-1]):
        for right_index in range(left_index + 1, len(thumbnails)):
            score = structural_similarity(
                left_image,
                thumbnails[right_index],
                data_range=255,
            )
            score_totals[left_index] += score
            score_totals[right_index] += score
        if (left_index + 1) % 25 == 0:
            print(
                f"Compared pairs for {left_index + 1}/{len(thumbnails)} images."
            )

    mean_scores = score_totals / (len(thumbnails) - 1)
    ranked = sorted(
        zip(image_paths, mean_scores.tolist()),
        key=lambda item: (-item[1], item[0].name.casefold()),
    )
    return ranked


def write_ranking(
    dataset_name: str,
    ranked_images: list[tuple[Path, float]],
    top_k: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / f"{dataset_name}_ssim_ranking.csv"
    top_path = OUTPUT_DIR / f"{dataset_name}_top{top_k}.csv"

    for output_path, rows in (
        (all_path, ranked_images),
        (top_path, ranked_images[:top_k]),
    ):
        with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(("dataset", "rank", "image", "mean_pairwise_ssim"))
            writer.writerows(
                (dataset_name, rank, image_path.name, f"{score:.8f}")
                for rank, (image_path, score) in enumerate(rows, start=1)
            )

    print(f"Saved full SSIM ranking: {all_path}")
    print(f"Saved representative images: {top_path}")
    for rank, (image_path, score) in enumerate(ranked_images[:top_k], start=1):
        print(f"  {rank:2d}. {image_path.name} (mean SSIM={score:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="all",
        help="Validation split to rank (default: both datasets).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of representative images to save (default: 10).",
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=128,
        help="Square grayscale thumbnail side length for SSIM (default: 128).",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if args.thumbnail_size < 7:
        parser.error("--thumbnail-size must be at least 7 for SSIM.")

    dataset_names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset_name in dataset_names:
        image_dir = DATASETS[dataset_name]
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"{dataset_name} validation image directory not found: {image_dir}"
            )
        image_paths = sorted(
            (
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )
        if not image_paths:
            raise FileNotFoundError(f"No validation images found in {image_dir}")

        started_at = perf_counter()
        print(
            f"Ranking {len(image_paths)} {dataset_name} validation images "
            f"using {args.thumbnail_size}x{args.thumbnail_size} grayscale SSIM."
        )
        ranked = rank_images(image_paths, args.thumbnail_size)
        write_ranking(dataset_name, ranked, min(args.top_k, len(ranked)))
        print(f"Elapsed: {perf_counter() - started_at:.1f} seconds")


if __name__ == "__main__":
    main()
