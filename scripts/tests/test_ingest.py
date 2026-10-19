import asyncio
from io import BytesIO

from PIL import Image

from scripts.ingest import Item, embed_items, make_plan, sha256_file  # puts backend/ on sys.path

from app.embed_client import EmbeddingServiceError  # noqa: E402  isort: skip
from app.preprocessing import PreprocessOptions  # noqa: E402  isort: skip

CURRENT = {"embedding_model": "m", "embedding_version": "v1", "preprocess_version": "pp-v1/512"}


def write_jpeg(path, colour="red", size=(200, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, format="JPEG")


def image(image_id, category="rings"):
    return {"_id": image_id, "category": category, "image_path": f"data/{image_id}.jpg"}


def indexed_doc(path, vector_id, **overrides):
    return {"vector_id": vector_id, "image_sha256": sha256_file(path), **CURRENT, **overrides}


def test_plan_keeps_up_to_date_images_and_embeds_the_rest(tmp_path):
    for name in ("same", "new", "changed", "moved", "old_settings", "lost_vector"):
        write_jpeg(tmp_path / "data" / f"{name}.jpg")
    existing = {name: indexed_doc(tmp_path / "data" / f"{name}.jpg", vid)
                for vid, name in enumerate(["same", "changed", "moved", "old_settings", "lost_vector"], start=1)}
    existing["changed"]["image_sha256"] = "different"
    existing["old_settings"]["preprocess_version"] = "pp-v0/224"
    indexed = {1: "rings", 2: "rings", 3: "pendants", 4: "rings", 99: "rings"}  # 5 missing, 99 orphaned

    images = [image(n) for n in ("same", "new", "changed", "moved", "old_settings", "lost_vector")]
    plan = make_plan(images, existing, indexed, CURRENT, tmp_path)

    assert plan.keep_ids == {1}
    assert {item.image_id: item.vector_id for item in plan.to_embed} == {
        "new": None, "changed": 2, "moved": 3, "old_settings": 4, "lost_vector": 5,
    }
    assert plan.failed == {}


def test_plan_reports_missing_file(tmp_path):
    plan = make_plan([image("missing")], {}, {}, CURRENT, tmp_path)
    assert plan.to_embed == [] and "cannot read data/missing.jpg" in plan.failed["missing"]


class FakeClient:
    async def embed(self, image_bytes, filename, content_type):
        assert content_type == "image/png" and Image.open(BytesIO(image_bytes)).size == (64, 64)
        if filename.startswith("down"):
            raise EmbeddingServiceError("HTTP 500")
        return [1.0, 0.0]


def test_embed_items_skips_bad_images_and_keeps_going(tmp_path):
    write_jpeg(tmp_path / "good.jpg")
    write_jpeg(tmp_path / "tiny.jpg", size=(10, 10))
    (tmp_path / "corrupt.jpg").write_bytes(b"not a jpeg")
    write_jpeg(tmp_path / "down.jpg")
    write_jpeg(tmp_path / "good2.jpg")
    items = [Item(name, "rings", tmp_path / f"{name}.jpg", "") for name in ("good", "tiny", "corrupt", "down", "good2")]
    items.append(Item("gone", "rings", tmp_path / "gone.jpg", ""))

    async def collect():
        options = PreprocessOptions(target_size=64, min_side=32)
        return [result async for result in embed_items(items, FakeClient(), options, batch_size=2)]

    results = {item.image_id: (vector, error) for item, vector, error in asyncio.run(collect())}
    assert results["good"] == ([1.0, 0.0], None)
    assert results["good2"] == ([1.0, 0.0], None)
    assert "at least 32 px" in results["tiny"][1]
    assert "not a readable image" in results["corrupt"][1]
    assert "HTTP 500" in results["down"][1]
    assert "No such file" in results["gone"][1]
