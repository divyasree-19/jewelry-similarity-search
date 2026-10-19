from io import BytesIO

import pytest
from PIL import Image

from app.preprocessing import PREPROCESS_VERSION, ImagePreprocessError, PreprocessOptions, encode, preprocess

OPTIONS = PreprocessOptions(target_size=256, min_side=128)


def to_bytes(image: Image.Image, fmt: str, **save_args) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=fmt, **save_args)
    return buffer.getvalue()


def test_output_is_square_rgb_and_keeps_aspect_ratio():
    # 400x200 red bar -> 256x128 bar centred between white bands
    result = preprocess(to_bytes(Image.new("RGB", (400, 200), (200, 0, 0)), "JPEG"), OPTIONS)
    assert result.mode == "RGB" and result.size == (256, 256)
    assert result.getpixel((128, 5)) == (255, 255, 255)
    assert result.getpixel((128, 250)) == (255, 255, 255)
    r, g, b = result.getpixel((128, 128))
    assert r > 180 and g < 30 and b < 30


def test_png_with_alpha_is_flattened_onto_white():
    image = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    image.paste((0, 0, 200, 255), (100, 100, 200, 200))
    result = preprocess(to_bytes(image, "PNG"), OPTIONS)
    assert result.mode == "RGB"
    assert result.getpixel((10, 10)) == (255, 255, 255)  # transparent, not black
    r, g, b = result.getpixel((128, 128))
    assert b > 180 and r < 30


def test_palette_png_with_transparency_is_flattened_onto_white():
    image = Image.new("P", (300, 300), 0)
    image.putpalette([0, 0, 0, 0, 200, 0] + [0] * 762)
    image.paste(1, (100, 100, 200, 200))
    result = preprocess(to_bytes(image, "PNG", transparency=0), OPTIONS)
    assert result.getpixel((10, 10)) == (255, 255, 255)
    assert result.getpixel((128, 128))[1] > 180


def test_exif_orientation_is_applied():
    # Stored as 400x200 landscape with the left half red; Orientation 6 means "rotate 90° clockwise to view"
    image = Image.new("RGB", (400, 200), (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, 200, 200))
    exif = Image.Exif()
    exif[0x0112] = 6
    result = preprocess(to_bytes(image, "JPEG", exif=exif), OPTIONS)

    # Viewed upright it is portrait with red on top: white bands left and right, not top and bottom
    assert result.getpixel((5, 128)) == (255, 255, 255)
    assert result.getpixel((128, 5)) != (255, 255, 255)
    top, bottom = result.getpixel((128, 40)), result.getpixel((128, 216))
    assert top[0] > 180 and top[2] < 60
    assert bottom[2] > 180 and bottom[0] < 60


@pytest.mark.parametrize("size", [(50, 50), (400, 100)])
def test_tiny_image_is_rejected(size):
    with pytest.raises(ImagePreprocessError, match="at least 128 px"):
        preprocess(to_bytes(Image.new("RGB", size, "white"), "PNG"), OPTIONS)


def test_cmyk_jpeg_is_converted_to_rgb():
    cyan = Image.new("CMYK", (300, 300), (255, 0, 0, 0))
    result = preprocess(to_bytes(cyan, "JPEG"), OPTIONS)
    assert result.mode == "RGB"
    r, g, b = result.getpixel((128, 128))
    assert r < 40 and g > 200 and b > 200


def test_16_bit_greyscale_is_not_blown_out():
    image = Image.new("I;16", (300, 300), 32768)
    result = preprocess(to_bytes(image, "PNG"), OPTIONS)
    assert 110 < result.getpixel((128, 128))[0] < 150


@pytest.mark.parametrize(
    "data",
    [b"", b"not an image at all", to_bytes(Image.new("RGB", (300, 300), "red"), "JPEG")[:200]],
    ids=["empty", "garbage", "truncated_jpeg"],
)
def test_corrupt_file_is_rejected(data):
    with pytest.raises(ImagePreprocessError):
        preprocess(data, OPTIONS)


def test_preprocess_is_deterministic_and_encode_round_trips():
    data = to_bytes(Image.new("RGB", (640, 480), (10, 120, 30)), "JPEG")
    first, second = preprocess(data, OPTIONS), preprocess(data, OPTIONS)
    assert first.tobytes() == second.tobytes()
    assert Image.open(BytesIO(encode(first))).tobytes() == first.tobytes()


def test_lighting_normalization_stretches_contrast():
    image = Image.new("RGB", (300, 300), (100, 100, 100))
    image.paste((140, 140, 140), (0, 0, 150, 300))
    options = PreprocessOptions(target_size=256, min_side=128, normalize_lighting=True)
    result = preprocess(to_bytes(image, "PNG"), options)
    assert result.getpixel((20, 128))[0] > 200
    assert result.getpixel((236, 128))[0] < 50


def test_signature_records_version_and_flags():
    assert PreprocessOptions().signature == f"{PREPROCESS_VERSION}/512"
    options = PreprocessOptions(remove_background=True, normalize_lighting=True)
    assert options.signature == f"{PREPROCESS_VERSION}/512/bg:isnet-general-use/norm"
