"""Write before/after preprocessing strips to eval/preprocess_samples/ for a visual check.

Each strip shows, left to right:
    original | default pipeline | background removed | removed area in red | background removed + lighting

The red panel is the one to inspect: chains, prongs and thin settings must not be red. If they are,
keep PREPROCESS_REMOVE_BACKGROUND off.

Usage (from the project root; background panels need `pip install -r backend/requirements-bg.txt`):
    python scripts/preprocess_samples.py                       # every catalogue and query image
    python scripts/preprocess_samples.py --ids NK-005_01 q014  # selected images
    python scripts/preprocess_samples.py --no-background       # skip rembg
"""

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.preprocessing import ImagePreprocessError, PreprocessOptions, preprocess, remove_background  # noqa: E402

OUT_DIR = PROJECT_ROOT / "eval" / "preprocess_samples"
PANEL = 320
LABEL_HEIGHT = 24


def list_images() -> list[tuple[str, Path]]:
    images = []
    with (PROJECT_ROOT / "data" / "catalogue.csv").open(newline="", encoding="utf-8") as f:
        images += [(row["image_id"], PROJECT_ROOT / row["image_path"]) for row in csv.DictReader(f)]
    with (PROJECT_ROOT / "eval" / "queries.csv").open(newline="", encoding="utf-8") as f:
        images += [(row["query_id"], PROJECT_ROOT / row["query_path"]) for row in csv.DictReader(f)]
    return images


def removed_overlay(original: Image.Image, cutout: Image.Image) -> Image.Image:
    """The original with every pixel the background remover dropped tinted red."""
    removed = ImageOps.invert(cutout.getchannel("A")).point(lambda a: 255 if a > 127 else 0)
    red = Image.new("RGB", original.size, (230, 0, 0))
    return Image.composite(Image.blend(original, red, 0.6), original, removed)


def strip(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    font = ImageFont.load_default(size=14)
    sheet = Image.new("RGB", (PANEL * len(panels), PANEL + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(panels):
        sheet.paste(ImageOps.pad(image, (PANEL, PANEL), color=(255, 255, 255)), (i * PANEL, LABEL_HEIGHT))
        draw.text((i * PANEL + 6, 4), label, fill="black", font=font)
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", help="image_id or query_id values (default: all)")
    parser.add_argument("--no-background", action="store_true", help="skip the rembg panels")
    args = parser.parse_args()

    base = PreprocessOptions.from_settings(get_settings())
    default = PreprocessOptions(base.target_size, base.min_side, rembg_model=base.rembg_model)
    with_bg = PreprocessOptions(base.target_size, base.min_side, True, base.rembg_model)
    with_bg_norm = PreprocessOptions(base.target_size, base.min_side, True, base.rembg_model, True)

    images = list_images()
    if args.ids:
        wanted = set(args.ids)
        images = [item for item in images if item[0] in wanted]
        missing = wanted - {image_id for image_id, _ in images}
        if missing:
            print(f"Unknown ids: {', '.join(sorted(missing))}", file=sys.stderr)
            return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_id, path in images:
        data = path.read_bytes()
        try:
            panels = [("original", Image.open(path).convert("RGB")), ("default", preprocess(data, default))]
            if not args.no_background:
                upright = preprocess(data, PreprocessOptions(base.target_size, base.min_side))
                panels += [
                    (f"bg removed ({base.rembg_model})", preprocess(data, with_bg)),
                    ("removed area (red)", removed_overlay(upright, remove_background(upright, base.rembg_model))),
                    ("bg removed + lighting", preprocess(data, with_bg_norm)),
                ]
        except ImagePreprocessError as exc:
            print(f"{image_id}: rejected ({exc})")
            continue
        out = OUT_DIR / f"{image_id}.jpg"
        strip(panels).save(out, quality=90)
        print(f"{image_id}: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
