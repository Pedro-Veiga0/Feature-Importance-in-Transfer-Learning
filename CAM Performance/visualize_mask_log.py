import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        data = np.genfromtxt(path, delimiter=",", dtype=float)
    elif suffix in {".xlsx", ".xls"}:
        try:
            data = pd.read_excel(path).to_numpy(dtype=float)
        except ImportError as exc:
            raise RuntimeError(
                "Excel support requires openpyxl. Install it with: pip install openpyxl"
            ) from exc
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return np.asarray(data, dtype=float)


def to_log_grayscale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    finite = np.isfinite(values)
    if not np.all(finite):
        values = np.where(finite, values, 0.0)

    if np.any(values < 0):
        transformed = np.sign(values) * np.log1p(np.abs(values))
    else:
        transformed = np.log1p(values)

    min_val = np.min(transformed)
    max_val = np.max(transformed)

    if np.isclose(min_val, max_val):
        return np.zeros(transformed.shape, dtype=np.uint8)

    normalized = (transformed - min_val) / (max_val - min_val)
    return (255 * normalized).astype(np.uint8)


def save_image(values: np.ndarray, output_path: Path) -> None:
    grayscale = to_log_grayscale(values)
    image = Image.fromarray(grayscale, mode="L")
    image.save(output_path)
    print(f"Saved grayscale image to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a mask array from CSV/XLSX into a black-and-white log-scale image."
    )
    parser.add_argument("input", type=Path, help="Path to the CSV or Excel file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: input stem + '_log.png')",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    array = load_array(args.input)
    output_path = args.output or args.input.with_suffix(".png")
    save_image(array, output_path)


if __name__ == "__main__":
    main()
