"""Integration test of POST /search/image over a small fixture catalogue.

Uses the real MongoDB (a throwaway database, dropped afterwards) and real FAISS indexes. The embedding
service is replaced by a deterministic fake that turns the preprocessed image into a vector, so the
test needs no model. Skipped when MongoDB is not reachable.
"""

import uuid
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from app import main
from app.config import get_settings
from app.embed_client import EmbeddingServiceError
from app.preprocessing import PreprocessOptions, encode, preprocess
from app.vector_index import CategoryIndexes

MONGO_URI = "mongodb://localhost:27017"


def fake_vector(png_bytes: bytes) -> list[float]:
    """8x8 thumbnail of the image, mean-centred: similar pictures give similar vectors."""
    pixels = np.asarray(Image.open(BytesIO(png_bytes)).convert("RGB").resize((8, 8)), dtype=np.float32).ravel()
    return (pixels - pixels.mean()).tolist()


class FakeEmbedClient:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def embed(self, image_bytes, filename, content_type):
        if self.fail:
            raise EmbeddingServiceError("Cannot reach embedding service")
        return fake_vector(image_bytes)

    async def health(self):
        return {"model": "fake"}

    async def aclose(self):
        pass


def picture(shape: str, colour: str, offset: int = 0, size=(300, 300)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    box = (60 + offset, 60, 240 + offset, 240)
    {"circle": draw.ellipse, "square": draw.rectangle}[shape](box, fill=colour)
    if shape == "bar":
        draw.rectangle((20, 130 + offset, 280, 170 + offset), fill=colour)
    return image


def jpeg(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


# image_id, product_id, project_id, category, subcategory, picture
CATALOGUE = [
    ("RG-1_a", "RG-1", "demo", "rings", "band", ("circle", "red", 0)),
    ("RG-1_b", "RG-1", "demo", "rings", "band", ("circle", "red", 30)),
    ("RG-2_a", "RG-2", "demo", "rings", "solitaire", ("square", "blue", 0)),
    ("RG-3_a", "RG-3", "other", "rings", "band", ("circle", "darkred", 10)),
    ("PD-1_a", "PD-1", "demo", "pendants", "heart", ("square", "green", 0)),
]


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    try:
        mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=500)
        mongo.admin.command("ping")
    except PyMongoError:
        pytest.skip("MongoDB is not running")

    db_name = f"jewellery_search_test_{uuid.uuid4().hex[:8]}"
    db = mongo[db_name]
    catalogue_dir = tmp_path / "catalogue"
    options = PreprocessOptions()
    store = CategoryIndexes("google/siglip2-base-patch16-224", "siglip2-base-v1", options.signature)
    files = {}

    for vector_id, (image_id, product_id, project_id, category, subcategory, (shape, colour, offset)) in enumerate(
        CATALOGUE, start=1
    ):
        path = catalogue_dir / category / f"{image_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        files[image_id] = jpeg(picture(shape, colour, offset))
        path.write_bytes(files[image_id])
        store.add(category, [vector_id], [fake_vector(encode(preprocess(files[image_id], options)))])
        db.images.insert_one({
            "_id": image_id, "product_id": product_id, "project_id": project_id, "category": category,
            "subcategory": subcategory, "image_path": str(path), "vector_id": vector_id,
        })
        db.products.update_one({"_id": product_id}, {"$set": {
            "project_id": project_id, "category": category, "subcategory": subcategory,
            "name": f"{colour} {shape}", "metal": ["gold"], "image_ids": [], "source": {"x": 1},
        }}, upsert=True)
    store.save(tmp_path / "faiss")

    categories = tmp_path / "categories.txt"
    categories.write_text("# test\nrings\npendants\nbangles\n")
    for key, value in {
        "MONGO_URI": MONGO_URI, "MONGO_DB": db_name, "EMBED_BACKEND": "siglip2_224",
        "FAISS_INDEX_DIR": str(tmp_path / "faiss"), "CATALOGUE_DIR": str(catalogue_dir),
        "CATEGORIES_FILE": str(categories), "QUERY_IMAGE_DIR": str(tmp_path / "uploads"), "MAX_UPLOAD_MB": "0.5",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()

    yield files, tmp_path

    get_settings.cache_clear()
    mongo.drop_database(db_name)
    mongo.close()


@pytest.fixture
def client(catalogue):
    with TestClient(main.app) as test_client:
        main.app.state.embed_client = FakeEmbedClient()
        yield test_client


def search(client, data: bytes, content_type="image/jpeg", **form):
    form.setdefault("category", "rings")
    return client.post("/search/image", files={"file": ("q.jpg", data, content_type)}, data=form)


def test_catalogue_image_returns_its_own_product_first(client, catalogue):
    files, _ = catalogue
    response = search(client, files["RG-2_a"])
    assert response.status_code == 200
    body = response.json()

    assert uuid.UUID(body["query_id"])
    assert body["model"] == "google/siglip2-base-patch16-224"
    top = body["results"][0]
    assert (top["product_id"], top["image_id"], top["category"]) == ("RG-2", "RG-2_a", "rings")
    assert top["similarity_score"] == pytest.approx(1.0, abs=1e-3)
    assert top["attributes"] == {"project_id": "demo", "subcategory": "solitaire", "name": "blue square", "metal": ["gold"]}


def test_one_result_per_product_with_its_best_image(client, catalogue):
    files, _ = catalogue
    body = search(client, files["RG-1_b"]).json()
    product_ids = [r["product_id"] for r in body["results"]]
    assert product_ids[0] == "RG-1" and body["results"][0]["image_id"] == "RG-1_b"
    assert sorted(product_ids) == ["RG-1", "RG-2", "RG-3"]  # RG-1_a collapsed into RG-1; no pendants
    scores = [r["similarity_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_filters_and_top_k(client, catalogue):
    files, _ = catalogue
    query = files["RG-1_a"]
    assert [r["product_id"] for r in search(client, query, project_id="other").json()["results"]] == ["RG-3"]
    assert [r["product_id"] for r in search(client, query, subcategory="Solitaire").json()["results"]] == ["RG-2"]
    assert len(search(client, query, top_k="1").json()["results"]) == 1
    assert search(client, query, project_id="nobody").json()["results"] == []


def test_image_urls_are_served_and_no_file_paths_leak(client, catalogue):
    files, tmp_path = catalogue
    response = search(client, files["RG-2_a"])
    assert str(tmp_path) not in response.text
    url = response.json()["results"][0]["image_url"]
    assert url == "/catalogue-images/rings/RG-2_a.jpg"
    served = client.get(url)
    assert served.status_code == 200 and served.content == files["RG-2_a"]
    for escape in ("..%2Fcategories.txt", "rings/..%2F..%2Fcategories.txt", "%2Fetc%2Fpasswd"):
        assert client.get(f"/catalogue-images/{escape}").status_code == 404
    assert client.get("/catalogue-images/rings/missing.jpg").status_code == 404


def test_uploads_are_not_stored_by_default(client, catalogue):
    files, tmp_path = catalogue
    search(client, files["RG-2_a"])
    assert not (tmp_path / "uploads").exists()


@pytest.mark.parametrize("category, status, message", [
    ("watches", 400, "Unknown category 'watches'"),
    ("bangles", 404, "No catalogue images are indexed for category 'bangles'"),
])
def test_unknown_category_and_empty_index_are_clear_errors(client, catalogue, category, status, message):
    files, _ = catalogue
    response = search(client, files["RG-2_a"], category=category)
    assert response.status_code == status
    assert message in response.json()["detail"]


def test_upload_validation(client, catalogue):
    files, _ = catalogue
    cases = [
        (search(client, b"GIF89a....", content_type="image/gif"), 415, "Unsupported file type"),
        (search(client, b"plain text", content_type="image/jpeg"), 415, "not a JPEG, PNG or WebP"),
        (search(client, b"", content_type="image/jpeg"), 400, "empty"),
        (search(client, jpeg(Image.effect_noise((1200, 1200), 100).convert("RGB"))), 413, "larger than 0.5 MB"),
        (search(client, files["RG-2_a"][:300]), 400, "not a readable image"),
        (search(client, jpeg(Image.new("RGB", (40, 40), "red"))), 400, "at least 128 px"),
        (search(client, files["RG-2_a"], top_k="0"), 400, "top_k must be between 1 and 50"),
        (search(client, files["RG-2_a"], top_k="ten"), 400, "top_k"),
        (client.post("/search/image", files={"file": ("q.jpg", files["RG-2_a"], "image/jpeg")}), 400, "category: Field required"),
    ]
    for response, status, message in cases:
        assert response.status_code == status, response.text
        assert message in response.json()["detail"]


def test_png_and_webp_uploads_are_accepted(client, catalogue):
    image = picture("square", "blue", 0)
    for fmt, content_type in (("PNG", "image/png"), ("WEBP", "image/webp")):
        buffer = BytesIO()
        image.save(buffer, format=fmt)
        response = search(client, buffer.getvalue(), content_type=content_type)
        assert response.status_code == 200 and response.json()["results"][0]["product_id"] == "RG-2"


def test_embedding_service_down_is_502(client, catalogue):
    files, _ = catalogue
    main.app.state.embed_client = FakeEmbedClient(fail=True)
    response = search(client, files["RG-2_a"])
    assert response.status_code == 502
    assert "Embedding service is unavailable" in response.json()["detail"]


def test_cors_allows_only_the_frontend_origin(client):
    allowed = client.options("/search/image", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    blocked = client.options("/search/image", headers={
        "Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in blocked.headers
