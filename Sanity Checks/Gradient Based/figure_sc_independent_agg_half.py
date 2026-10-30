import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "CAM-SanityCheck" / "GradientBased"

EXP_SETTINGS = [
    {
        "dir": OUTPUT_DIR / "Cityscapes",
        "label": "Cityscapes",
    },
]

COMPARISON_SETTINGS = [
    {"dir": OUTPUT_DIR / "Base", "label": "Base"},
    {"dir": OUTPUT_DIR / "UDA", "label": "UDA"},
    {"dir": OUTPUT_DIR / "BaseT", "label": "BaseT"},
]

TARGET_LAYER_IDX = 17
CAM_NAMES = ["GradCAM", "GradCAM++", "LayerCAM", "ShapleyCAM"]
CAM_FOLDER = {
    "GradCAM": "GradCAM",
    "GradCAM++": "GradCAM++",
    "LayerCAM": "LayerCAM",
    "ShapleyCAM": "ShapleyCAM",
}

CAM_LABELS = {
    "GradCAM": "Grad-CAM",
    "GradCAM++": "Grad-CAM++",
    "LayerCAM": "LayerCAM",
    "ShapleyCAM": "ShapleyCAM",
}

THUMB_W = 272
THUMB_H = 170
# Downsample to 192x120 and compensate for the common 96-DPI fallback of PNGs without DPI metadata.
OUTPUT_SCALE = 192 / THUMB_W
OUTPUT_DPI = 96 * OUTPUT_SCALE
EXP_W = 62  # experiment-label sub-column
CAM_W = 245  # CAM-name sub-column
LABEL_W = EXP_W + CAM_W
HEADER_H = 54
PAD = 8
HEATMAP_GAP = 12
DIVIDER_H = 6

TEXT_COLOR = (0, 0, 0)
HEADER_COLOR = (0, 0, 0)
TITLE_COLOR = (0, 0, 0)
EXP_COLOR = (80, 80, 180)
DIVIDER_COLOR = (200, 200, 200)
TITLE_TEXT = "Independent layer randomization"

NUM_METHODS = len(CAM_NAMES)


# ── fonts ─────────────────────────────────────────────────────────────────────
def _load_font(size, bold=False):
    candidates = (
        ["arialbd.ttf", "segoeuib.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else [
            "segoeui.ttf",
            "calibri.ttf",
            "arial.ttf",
            "DejaVuSans.ttf",
            "Helvetica.ttc",
        ]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


FONT_TITLE = _load_font(42, bold=True)
FONT_HEADER = _load_font(38, bold=True)
FONT_LABEL = _load_font(37)
FONT_EXP = _load_font(37, bold=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def load_thumb(path):
    if not path or not os.path.exists(str(path)):
        ph = Image.new("RGB", (THUMB_W, THUMB_H), (200, 200, 200))
        ImageDraw.Draw(ph).rectangle(
            [0, 0, THUMB_W - 1, THUMB_H - 1], outline=(180, 0, 0), width=2
        )
        return ph
    return (
        Image.open(path)
        .convert("RGB")
        .resize((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
    )


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def put_centered(draw, text, cx, cy, font, color):
    tw, th = text_size(draw, text, font)
    draw.text((cx - tw // 2, cy - th // 2), text, fill=color, font=font)


def col_x(col_idx):
    return LABEL_W + PAD + col_idx * (THUMB_W + PAD)


def row_y_top(exp_idx, cam_idx, top_section_h):
    global_row = exp_idx * NUM_METHODS + cam_idx
    return top_section_h + HEADER_H + global_row * (THUMB_H + PAD) + exp_idx * DIVIDER_H


def draw_vertical_text(canvas, text, cx, cy, font, color):
    """Render text rotated 90° CCW, centred at (cx, cy)."""
    bb = font.getbbox(text)
    tw, th = bb[2], bb[3]
    tmp = Image.new("RGBA", (tw + 4, th + 4), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((2, 2), text, fill=color, font=font)
    tmp = tmp.rotate(90, expand=True)
    canvas.paste(tmp, (cx - tmp.width // 2, cy - tmp.height // 2), tmp)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE FACTORY
# ═══════════════════════════════════════════════════════════════════════════════
def make_figure(layer_range, figure_path, exp_settings=None):
    if exp_settings is None:
        exp_settings = EXP_SETTINGS

    first_layer, last_layer = layer_range

    layers_in_range = list(range(last_layer, first_layer - 1, -1))
    n_range_cols = len(layers_in_range)
    total_cols = 1 + n_range_cols
    col_labels = ["Original"] + [str(l) for l in layers_in_range]

    top_section_h = THUMB_H + PAD + PAD * 2
    total_w = LABEL_W + PAD + total_cols * (THUMB_W + PAD)
    total_h = (
        top_section_h
        + HEADER_H
        + (len(exp_settings) * NUM_METHODS) * (THUMB_H + PAD)
        + (len(exp_settings) - 1) * DIVIDER_H
        + PAD
    )

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # ── title ─────────────────────────────────────────────────────────────────
    title = f"{TITLE_TEXT}  (layers {last_layer}–{first_layer})"
    left_x = col_x(0)
    right_x = col_x(total_cols - 1) + THUMB_W
    put_centered(
        draw,
        title,
        (left_x + right_x) // 2,
        top_section_h // 2,
        FONT_TITLE,
        TITLE_COLOR,
    )

    # ── header row ────────────────────────────────────────────────────────────
    header_y = top_section_h + HEADER_H // 2
    for ci, lbl in enumerate(col_labels):
        if ci == 0 or (ci % 2 == 1) or ci == total_cols - 1:
            put_centered(
                draw, lbl, col_x(ci) + THUMB_W // 2, header_y, FONT_HEADER, HEADER_COLOR
            )

    # ── vertical separator between EXP_W and CAM_W sub-columns ───────────────
    draw.line(
        [(EXP_W, top_section_h), (EXP_W, total_h - PAD)], fill=DIVIDER_COLOR, width=1
    )

    # ── data rows ─────────────────────────────────────────────────────────────
    for exp_idx, exp in enumerate(exp_settings):
        exp_dir, exp_label = exp["dir"], exp["label"]

        if exp_idx > 0:
            div_y = row_y_top(exp_idx, 0, top_section_h) - DIVIDER_H // 2 - 1
            draw.line([(0, div_y), (total_w, div_y)], fill=DIVIDER_COLOR, width=2)

        # ── experiment label: vertically centred across all method rows ───────
        group_top = row_y_top(exp_idx, 0, top_section_h)
        group_bot = row_y_top(exp_idx, NUM_METHODS - 1, top_section_h) + THUMB_H
        group_cy = (group_top + group_bot) // 2
        draw_vertical_text(canvas, exp_label, EXP_W // 2, group_cy, FONT_EXP, EXP_COLOR)

        for cam_idx, cam_name in enumerate(CAM_NAMES):
            y_top = row_y_top(exp_idx, cam_idx, top_section_h)
            folder = CAM_FOLDER[cam_name]
            method_dir = os.path.join(exp_dir, folder)

            # CAM name — centred in the CAM_W sub-column
            cam_tw, cam_th = text_size(draw, CAM_LABELS[cam_name], FONT_LABEL)
            cam_x = EXP_W + (CAM_W - cam_tw) // 2
            cam_y = y_top + (THUMB_H - cam_th) // 2
            draw.text(
                (cam_x, cam_y), CAM_LABELS[cam_name], fill=TEXT_COLOR, font=FONT_LABEL
            )

            # Col 0 — Original baseline
            orig_path = os.path.join(
                method_dir, f"00_original_layer{TARGET_LAYER_IDX}.png"
            )
            canvas.paste(load_thumb(orig_path), (col_x(0), y_top))

            # Independent columns — rand_idx IS the layer
            for ci, rand_idx in enumerate(layers_in_range, start=1):
                fname = f"T2_{folder}_indep_layer{rand_idx:02d}.png"
                canvas.paste(
                    load_thumb(os.path.join(method_dir, fname)), (col_x(ci), y_top)
                )

    output_size = tuple(round(dimension * OUTPUT_SCALE) for dimension in canvas.size)
    canvas.resize(output_size, Image.Resampling.LANCZOS).save(
        figure_path, "PNG", dpi=(OUTPUT_DPI, OUTPUT_DPI)
    )
    print(f"Figure saved → {figure_path}")


# ── generate both figures ─────────────────────────────────────────────────────
make_figure(
    layer_range=(9, 17),
    figure_path=os.path.join(OUTPUT_DIR / "Cityscapes", "sanity_check_independent_layers9to17_ce.png"),
)
make_figure(
    layer_range=(0, 8),
    figure_path=os.path.join(OUTPUT_DIR / "Cityscapes", "sanity_check_independent_layers0to8_ce.png"),
)
make_figure(
    layer_range=(9, 17),
    figure_path=os.path.join(OUTPUT_DIR, "sanity_check_independent_layers9to17_base_uda_baset_ce.png"),
    exp_settings=COMPARISON_SETTINGS,
)
make_figure(
    layer_range=(0, 8),
    figure_path=os.path.join(OUTPUT_DIR, "sanity_check_independent_layers0to8_base_uda_baset_ce.png"),
    exp_settings=COMPARISON_SETTINGS,
)
