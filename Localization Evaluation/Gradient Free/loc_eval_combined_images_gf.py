"""Create 2x2 combined localization CAM images for gradient-free methods."""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(
    os.environ.get(
        "CAM_EVAL_ROOT",
        str(
            BASE_DIR.parents[2]
            / "Código"
            / "Codigo Dissertacao"
            / "UDA test"
            / "ConfMix-master"
        ),
    )
).expanduser()
SOURCE_ROOT = PROJECT_DIR / "CAM-Loc-Eval" / "GradientFree"
EVALUATION_SETTINGS = ("Base", "BaseT", "UDA")
METHODS = (
    ("EigenCAM", "Eigen-CAM"),
    ("ScoreCAM", "Score-CAM"),
)
LAYERS = (("4", "layer 04"), ("17", "layer 17"))

def _load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_labeled_grid(
    image_paths,
    labels,
    output_path,
    padding=20,
    label_height=64,
    font_size=32,
    background="white",
    text_color="black",
):
    if len(image_paths) != 4 or len(labels) != 4:
        raise ValueError("A 2x2 grid requires exactly four images and labels.")

    images = []
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))

    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    font = _load_font(font_size)

    grid_width = 2 * cell_width + 3 * padding
    row_height = cell_height + label_height
    grid_height = 2 * row_height + 3 * padding
    canvas = Image.new("RGB", (grid_width, grid_height), color=background)
    draw = ImageDraw.Draw(canvas)

    for index, (image, label) in enumerate(zip(images, labels)):
        row, column = divmod(index, 2)
        cell_x = padding + column * (cell_width + padding)
        cell_y = padding + row * (row_height + padding)

        image_x = cell_x + (cell_width - image.width) // 2
        canvas.paste(image, (image_x, cell_y))

        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        text_x = cell_x + (cell_width - text_width) // 2
        text_y = cell_y + cell_height + (label_height - text_height) // 2
        draw.text((text_x, text_y), label, fill=text_color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _find_layer_image(setting_dir, method, layer):
    layer_number = int(layer)
    candidates = (
        setting_dir / method / f"{method}_{layer}layer.png",
        setting_dir / method / f"{method}_layer{layer_number:02d}.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Layer-{layer_number} image not found for {method} in "
        f"{setting_dir / method}"
    )


def combine_setting(setting):
    setting_dir = SOURCE_ROOT / setting
    image_paths = [
        _find_layer_image(setting_dir, method, layer)
        for method, _ in METHODS
        for layer, _ in LAYERS
    ]
    labels = [
        f"{method_label} ({layer_label})"
        for _, method_label in METHODS
        for _, layer_label in LAYERS
    ]
    output_path = (
        setting_dir
        / "combined"
        / f"layer4&17_combined_images_{setting.lower()}.png"
    )
    return make_labeled_grid(image_paths, labels, output_path)


def main():
    missing_settings = [
        SOURCE_ROOT / setting
        for setting in EVALUATION_SETTINGS
        if not (SOURCE_ROOT / setting).is_dir()
    ]
    if missing_settings:
        raise FileNotFoundError(
            f"Gradient-free localization directory not found: {missing_settings[0]}"
        )

    outputs = [combine_setting(setting) for setting in EVALUATION_SETTINGS]
    for output_path in outputs:
        print(f"Saved combined 2x2 image to: {output_path}")


if __name__ == "__main__":
    main()
