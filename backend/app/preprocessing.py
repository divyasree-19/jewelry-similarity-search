"""Image preprocessing shared by catalogue ingestion and search queries.

Both paths must call `preprocess` so catalogue and query embeddings are always computed from images
prepared the same way. Any change to the output pixels must bump PREPROCESS_VERSION, and the catalogue
must then be re-ingested.
"""

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageCms, ImageFilter, ImageOps, UnidentifiedImageError

from app.config import Settings, get_settings

PREPROCESS_VERSION = "pp-v1"
# Versions of the derived variants (Stage 9). Bump one when its output changes; the main pipeline is unaffected.
GRAY_VERSION = "gray-v1"
EDGE_VERSION = "edge-v1"
TILES_VERSION = "tiles-v1"
TILE_SIZE = 224  # SigLIP 2 Base input size; tiles are sent at this size

WHITE = (255, 255, 255)


class ImagePreprocessError(ValueError):
    """Raised when an upload cannot be used as a search or catalogue image."""


@dataclass(frozen=True)
class PreprocessOptions:
    target_size: int = 512
    min_side: int = 128
    remove_background: bool = False
    rembg_model: str = "isnet-general-use"
    normalize_lighting: bool = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "PreprocessOptions":
        return cls(
            target_size=settings.preprocess_size,
            min_side=settings.preprocess_min_side,
            remove_background=settings.preprocess_remove_background,
            rembg_model=settings.preprocess_rembg_model,
            normalize_lighting=settings.preprocess_normalize_lighting,
        )

    @property
    def signature(self) -> str:
        """Identifies the exact pipeline, e.g. "pp-v1/512/bg:isnet-general-use/norm". Store it next to each vector."""
        parts = [PREPROCESS_VERSION, str(self.target_size)]
        if self.remove_background:
            parts.append(f"bg:{self.rembg_model}")
        if self.normalize_lighting:
            parts.append("norm")
        return "/".join(parts)


def preprocess(image_bytes: bytes, options: PreprocessOptions | None = None) -> Image.Image:
    """Decode an uploaded image and return a square RGB image of options.target_size pixels."""
    if options is None:
        options = PreprocessOptions.from_settings(get_settings())

    image = _decode(image_bytes)
    image = ImageOps.exif_transpose(image)

    if min(image.size) < options.min_side:
        raise ImagePreprocessError(
            f"Image is {image.width}x{image.height}; the shorter side must be at least {options.min_side} px"
        )

    image = _to_rgb(image)
    if options.remove_background:
        image = _flatten(remove_background(image, options.rembg_model))
    if options.normalize_lighting:
        # preserve_tone stretches luminance only, so gold stays gold and silver stays silver
        image = ImageOps.autocontrast(image, cutoff=1, preserve_tone=True)

    # Scale the long side to target_size and centre on white; the aspect ratio is kept
    return ImageOps.pad(image, (options.target_size, options.target_size), Image.Resampling.LANCZOS, color=WHITE)


def to_grayscale(image: Image.Image) -> Image.Image:
    """Colour-free copy of a preprocessed image, as 3 identical channels (SigLIP expects RGB). Same size."""
    return image.convert("L").convert("RGB")


def to_edge_map(image: Image.Image) -> Image.Image:
    """Dark outlines on white of a preprocessed image (3 channels, same size): shape only, no colour or shading."""
    gray = image.convert("L").filter(ImageFilter.GaussianBlur(1))
    edges = ImageOps.autocontrast(gray.filter(ImageFilter.FIND_EDGES), cutoff=1)
    return ImageOps.invert(edges).convert("RGB")


def make_tiles(image: Image.Image, grid: int) -> list[Image.Image]:
    """A centre crop (middle half of each side) followed by grid x grid tiles, row by row, each TILE_SIZE square.

    Returns 1 + grid * grid images. Works on the preprocessed square image.
    """
    if grid < 1:
        raise ValueError("grid must be at least 1")
    width, height = image.size
    crops = [(width // 4, height // 4, width * 3 // 4, height * 3 // 4)]
    for row in range(grid):
        for column in range(grid):
            crops.append((column * width // grid, row * height // grid,
                          (column + 1) * width // grid, (row + 1) * height // grid))
    return [image.crop(box).resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS) for box in crops]


def variant_signature(signature: str, variant: str, grid: int | None = None) -> str:
    """Signature of a derived feature, e.g. "pp-v1/512/gray-v1" or "pp-v1/512/tiles-v1/3x3"."""
    version = {"gray": GRAY_VERSION, "edge": EDGE_VERSION, "tiles": TILES_VERSION}[variant]
    return f"{signature}/{version}" + (f"/{grid}x{grid}" if variant == "tiles" else "")


def encode(image: Image.Image) -> bytes:
    """Lossless PNG bytes of a preprocessed image, for sending to the embedding service."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ImagePreprocessError("Image file is empty")
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Image.DecompressionBombError as exc:
        raise ImagePreprocessError("Image is too large") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImagePreprocessError("File is not a readable image") from exc
    return image


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert any mode to RGB, using the embedded ICC profile when present and flattening alpha onto white."""
    icc = image.info.get("icc_profile")
    if icc and image.mode in ("RGB", "RGBA", "CMYK"):
        try:
            source = ImageCms.ImageCmsProfile(BytesIO(icc))
            output_mode = "RGBA" if image.mode == "RGBA" else "RGB"
            image = ImageCms.profileToProfile(image, source, ImageCms.createProfile("sRGB"), outputMode=output_mode)
        except (ImageCms.PyCMSError, OSError):
            pass  # a broken profile is not worth rejecting the photo; fall back to a plain conversion

    if image.mode == "P" or image.mode == "PA" or image.mode == "LA" or "transparency" in image.info:
        image = image.convert("RGBA")
    if image.mode == "RGBA":
        return _flatten(image)
    if image.mode.startswith("I;16"):
        image = image.point(lambda value: value / 256, "L")  # 16-bit greyscale would clip in convert("RGB")
    return image.convert("RGB")


def _flatten(image: Image.Image) -> Image.Image:
    background = Image.new("RGB", image.size, WHITE)
    background.paste(image, mask=image.getchannel("A"))
    return background


_rembg_sessions: dict = {}


def remove_background(image: Image.Image, model: str) -> Image.Image:
    """Return an RGBA cut-out of the product (alpha 0 = background). Needs the optional rembg package."""
    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise RuntimeError(
            "PREPROCESS_REMOVE_BACKGROUND is on but rembg is not installed; pip install -r requirements-bg.txt"
        ) from exc
    if model not in _rembg_sessions:
        _rembg_sessions[model] = new_session(model)
    return remove(image, session=_rembg_sessions[model]).convert("RGBA")
