from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

labels = [
    "Grad-CAM",
    "Grad-CAM++",
    "LayerCAM",
    "ShapleyCAM",
]

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = (
    BASE_DIR
    / "CAM-Loc-Eval"
    / "GradientBased"
    / "Base"
    / "Rapid_val_layer17"
)
OUTPUT_DIR = SOURCE_DIR / "combined"
CAM_IMAGE_SUFFIXES = (
    "_GradCAM_layer17.png",
    "_GradCAM++_layer17.png",
    "_LayerCAM_layer17.png",
    "_ShapleyCAM_layer17.png",
)


def make_labeled_strip_pil(
    image_paths,
    labels,
    out_path,
    pad=20,
    bottom_space=80,
    bg_color="white",
    text_color="black",
    font_size=48,
):
    if len(image_paths) != len(labels):
        raise ValueError("Each input image must have one corresponding label.")

    images = []
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    target_height = max(img.height for img in images)

    resized_images = []
    for img in images:
        if img.height != target_height:
            new_width = int(img.width * target_height / img.height)
            img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        resized_images.append(img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    total_width = sum(img.width for img in resized_images) + pad * (len(resized_images) + 1)
    canvas_height = target_height + bottom_space

    canvas = Image.new("RGB", (total_width, canvas_height), color=bg_color)
    draw = ImageDraw.Draw(canvas)

    x = pad
    for img, label in zip(resized_images, labels):
        canvas.paste(img, (x, 0))

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_x = x + (img.width - text_w) // 2
        text_y = target_height + (bottom_space - text_h) // 2

        draw.text((text_x, text_y), label, fill=text_color, font=font)
        x += img.width + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main():
    method_dirs = [SOURCE_DIR / method for method in ("GradCAM", "GradCAM++", "LayerCAM", "ShapleyCAM")]
    missing_dirs = [path for path in method_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"CAM output directory not found: {missing_dirs[0]}")

    source_images = sorted(method_dirs[0].glob(f"*{CAM_IMAGE_SUFFIXES[0]}"))
    if not source_images:
        raise FileNotFoundError(f"No Rapid layer-17 CAM images found in {method_dirs[0]}")

    for source_image in source_images:
        image_stem = source_image.name[: -len(CAM_IMAGE_SUFFIXES[0])]
        image_paths = [
            method_dir / f"{image_stem}{suffix}"
            for method_dir, suffix in zip(method_dirs, CAM_IMAGE_SUFFIXES)
        ]
        output_path = OUTPUT_DIR / f"{image_stem}_layer17_combined.png"
        make_labeled_strip_pil(image_paths, labels, output_path)

    print(f"Saved {len(source_images)} combined Rapid CAM images to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()