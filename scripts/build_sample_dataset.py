"""Build the public sample catalogue and evaluation queries (Stage 2 stand-in for client data).

Source: "AI Tool Pool Jewelry Vision" (Roboflow), CC BY 4.0, mirrored on Hugging Face at
Tarunhugging/ai-tool-pool-jewelry-vision, pinned to REVISION below. See data/ATTRIBUTION.md.

Inputs (tracked in git):
    data/products.csv       one row per product: metadata + catalogue_sources (dataset paths)
    eval/query_specs.csv    one row per query: expected product, source photo, transform, tags

Outputs (images are git-ignored; the CSVs are tracked):
    data/catalogue/<category>/<image_id>.jpg  +  data/catalogue.csv
    data/queries/<query_id>.jpg               +  eval/queries.csv

Every query starts from a different file than the catalogue image (another augmented copy of the
photo, or for "real:" specs a separate photo of the same product), then gets one synthetic edge-case
transform. Transforms are seeded per query, so re-running gives the same images.

Usage (from the project root):
    python scripts/build_sample_dataset.py
"""

import csv
import hashlib
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO = "Tarunhugging/ai-tool-pool-jewelry-vision"
REVISION = "71e72b548a2a2b44d73b40e4b0999b3e74f1b3d4"
MAX_QUERY_SIDE = 1024
GOLD_HUE = (0.05, 0.17)  # hue range (0-1) left alone by the colour transform so metal keeps its colour


def download(dataset_path: str, cache_dir: Path) -> bytes:
    cached = cache_dir / dataset_path
    if cached.exists():
        return cached.read_bytes()
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{dataset_path}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data


# --- edge-case transforms -------------------------------------------------------------------

def border_colour(arr: np.ndarray) -> np.ndarray:
    edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return np.median(edges, axis=0)


def background_mask(arr: np.ndarray) -> np.ndarray:
    """Flat, low-chroma pixels connected to the image border (studio backdrops, rotation fill)."""
    rgb = arr.astype(np.int16)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    grey = np.asarray(Image.fromarray(arr).convert("L").filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)
    gy, gx = np.gradient(grey)
    flat = (chroma < 28) & (np.hypot(gx, gy) < 6)
    mask = np.zeros_like(flat)
    mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1] = flat[0, :], flat[-1, :], flat[:, 0], flat[:, -1]
    for _ in range(max(arr.shape[:2])):  # grow from the border through flat pixels only
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        grown &= flat
        if (grown == mask).all():
            break
        mask = grown
    return mask


def make_backdrop(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    kind = rng.integers(3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == 0:  # wood grain
        base = np.array([120, 80, 45], np.float32)
        grain = np.sin(xx / w * 40 + np.sin(yy / h * 6) * 3) * 18
        img = base + grain[..., None]
    elif kind == 1:  # dark velvet
        base = np.array([30, 35, 70], np.float32)
        img = base + rng.normal(0, 8, (h, w, 1))
    else:  # warm linen with vignette
        base = np.array([215, 200, 175], np.float32)
        r = np.hypot(xx / w - 0.5, yy / h - 0.5)
        img = base - (r * 90)[..., None] + rng.normal(0, 6, (h, w, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def t_rotation(im: Image.Image, rng) -> Image.Image:
    angle = float(rng.choice([rng.uniform(35, 145), rng.uniform(-145, -35), 180.0]))
    fill = tuple(int(c) for c in border_colour(np.asarray(im)))
    return im.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill).resize(im.size, Image.LANCZOS)


def t_lighting(im: Image.Image, rng) -> Image.Image:
    arr = np.asarray(im, np.float32) / 255
    # darken bright photos and brighten dark ones, so the change is visible but the piece stays visible
    gamma = rng.uniform(1.6, 2.2) if arr.mean() > 0.45 else rng.uniform(0.45, 0.6)
    cast = np.array([1.12, 1.0, 0.82]) if rng.random() < 0.5 else np.array([0.85, 0.97, 1.15])  # warm or cool
    out = np.clip((arr ** gamma) * cast, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def t_colour(im: Image.Image, rng) -> Image.Image:
    """Change stone colours: shift the hue of saturated pixels outside the gold/yellow band."""
    hsv = np.asarray(im.convert("HSV"), np.float32) / 255
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    stones = (s > 0.25) & (v > 0.15) & ~((h > GOLD_HUE[0]) & (h < GOLD_HUE[1]))
    shift = rng.uniform(0.3, 0.6)
    h = np.where(stones, (h + shift) % 1.0, h)
    out = np.stack([h, s, v], axis=-1)
    return Image.fromarray((out * 255).astype(np.uint8), "HSV").convert("RGB")


def t_reflection(im: Image.Image, rng) -> Image.Image:
    arr = np.asarray(im, np.float32) / 255
    hgt, wid = arr.shape[:2]
    yy, xx = np.mgrid[0:hgt, 0:wid].astype(np.float32)
    glare = np.zeros((hgt, wid), np.float32)
    for _ in range(rng.integers(2, 5)):  # specular hot spots
        cy, cx = rng.uniform(0.2, 0.8) * hgt, rng.uniform(0.2, 0.8) * wid
        sigma = rng.uniform(0.03, 0.08) * max(hgt, wid)
        glare += rng.uniform(0.6, 0.95) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    # a soft diagonal streak, like light on glass or polished metal
    d = (xx / wid - yy / hgt) - rng.uniform(-0.3, 0.3)
    glare += 0.45 * np.exp(-(d**2) / (2 * 0.04**2))
    out = 1 - (1 - arr) * (1 - np.clip(glare, 0, 1)[..., None])  # screen blend
    return Image.fromarray((out * 255).astype(np.uint8))


def perspective_coeffs(src, dst):
    """Coefficients for PIL's PERSPECTIVE transform mapping output points dst -> input points src."""
    a = []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    return np.linalg.solve(np.array(a, np.float64), np.array(src, np.float64).reshape(8))


def t_angle(im: Image.Image, rng) -> Image.Image:
    w, h = im.size
    k = rng.uniform(0.18, 0.3)
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    if rng.random() < 0.5:  # tilt away at the top
        warped = [(w * k, h * k / 2), (w * (1 - k), h * k / 2), (w, h), (0, h)]
    else:  # turned to the side
        warped = [(0, 0), (w * (1 - k), h * k), (w * (1 - k), h * (1 - k)), (0, h)]
    fill = tuple(int(c) for c in border_colour(np.asarray(im)))
    coeffs = perspective_coeffs(corners, warped)
    return im.transform(im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=fill)


def t_background(im: Image.Image, rng) -> Image.Image:
    arr = np.asarray(im)
    mask = background_mask(arr)
    soft = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)),
                      np.float32)[..., None] / 255
    backdrop = make_backdrop(*arr.shape[:2], rng)
    out = arr * (1 - soft) + backdrop * soft
    return Image.fromarray(out.astype(np.uint8))


def t_distance(im: Image.Image, rng) -> Image.Image:
    w, h = im.size
    scale = rng.uniform(0.35, 0.5)
    small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    canvas = np.empty((h, w, 3), np.float32)
    canvas[:] = border_colour(np.asarray(im))
    canvas += rng.normal(0, 4, (h, w, 1))
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    x = int(rng.uniform(0.1, 0.9) * (w - small.width))
    y = int(rng.uniform(0.1, 0.9) * (h - small.height))
    out.paste(small, (x, y))
    return out


TRANSFORMS = {
    "none": lambda im, rng: im,
    "rotation": t_rotation,
    "lighting": t_lighting,
    "colour": t_colour,
    "reflection": t_reflection,
    "angle": t_angle,
    "background": t_background,
    "distance": t_distance,
}


# --- build --------------------------------------------------------------------------------

def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build(root: Path) -> None:
    cache = root / "data" / "raw" / REVISION[:12]
    products = read_rows(root / "data" / "products.csv")
    specs = read_rows(root / "eval" / "query_specs.csv")

    catalogue_rows = []
    for p in products:
        for n, source in enumerate(p["catalogue_sources"].split(";"), start=1):
            image_id = f"{p['product_id']}_{n:02d}"
            rel = Path("data") / "catalogue" / p["category"] / f"{image_id}.jpg"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(download(source, cache))
            catalogue_rows.append({
                "image_id": image_id, "product_id": p["product_id"], "project_id": p["project_id"],
                "category": p["category"], "subcategory": p["subcategory"], "image_path": rel.as_posix(),
            })
        print(f"catalogue  {p['product_id']}")
    write_rows(root / "data" / "catalogue.csv", list(catalogue_rows[0]), catalogue_rows)

    category_of = {p["product_id"]: p["category"] for p in products}
    query_rows = []
    for spec in specs:
        qid = spec["query_id"]
        im = Image.open(io.BytesIO(download(spec["source"], cache))).convert("RGB")
        im.thumbnail((MAX_QUERY_SIDE, MAX_QUERY_SIDE), Image.LANCZOS)
        seed = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16)
        out = TRANSFORMS[spec["transform"]](im, np.random.default_rng(seed))
        rel = Path("data") / "queries" / f"{qid}.jpg"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        out.save(root / rel, quality=90)
        first_expected = spec["expected_product_ids"].split(";")[0]
        query_rows.append({
            "query_id": qid, "query_path": rel.as_posix(), "category": category_of[first_expected],
            "expected_product_ids": spec["expected_product_ids"], "edge_case_tags": spec["edge_case_tags"],
        })
        print(f"query      {qid} {spec['transform']:<10} -> {spec['expected_product_ids']}")
    write_rows(root / "eval" / "queries.csv", list(query_rows[0]), query_rows)
    print(f"\n{len(catalogue_rows)} catalogue images, {len(query_rows)} queries")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT)
