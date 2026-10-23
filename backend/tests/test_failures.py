"""Stage 12 failure tests: every dependency failure is a readable 503 or a reduced-mode answer, never a 500.

Uses the fixture catalogue from test_search.py (throwaway MongoDB database, real FAISS indexes, fake embedder).
Services that are "down" are real clients pointed at a closed port; "slow" services are fakes that sleep.
Skipped when MongoDB is not reachable, like test_search.py.
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pymongo import AsyncMongoClient, MongoClient

from app import main
from app.clients.llm_client import LLMClient
from app.clients.vlm_client import VLMClient
from app.config import PROJECT_ROOT, get_settings
from app.tagging.schema import QueryTags
from tests.test_search import (MONGO_URI, FakeEmbedClient, FakeLLMClient, catalogue, client, jpeg, picture,  # noqa: F401
                               search)

CLOSED_PORT = "http://127.0.0.1:9/v1"  # nothing listens on the discard port: connection refused at once


def text_search(client, text="blue square ring", **body):
    return client.post("/search/text", json={"text": text, **body})


def assert_readable_503(response, message):
    assert response.status_code == 503, response.text
    assert message in response.json()["detail"]


# ---- SigLIP (embedding service) down -----------------------------------------------------------

def test_siglip_down_is_503_for_image_and_text_search(client, catalogue):
    files, _ = catalogue
    main.app.state.embed_client = FakeEmbedClient(fail=True)
    assert_readable_503(search(client, files["RG-2_a"]), "Embedding service unavailable")

    async def fail(text):
        from app.embed_client import EmbeddingServiceError
        raise EmbeddingServiceError("Cannot reach embedding service")
    main.app.state.embed_client.embed_text = fail
    assert_readable_503(text_search(client), "Embedding service unavailable")


def test_siglip_down_during_category_detection_is_503(client, catalogue, monkeypatch):
    files, _ = catalogue

    async def rank(embed_client, vector, categories, scale):
        from app.embed_client import EmbeddingServiceError
        raise EmbeddingServiceError("Cannot reach embedding service")
    monkeypatch.setattr(main.app.state.category_detector, "rank", rank)
    response = client.post("/search/image", files={"file": ("q.jpg", files["RG-2_a"], "image/jpeg")})
    assert_readable_503(response, "Embedding service unavailable")


# ---- MongoDB down ------------------------------------------------------------------------------

def test_mongo_down_is_503(client, catalogue):
    files, _ = catalogue
    main.app.state.mongo = AsyncMongoClient("mongodb://127.0.0.1:9", serverSelectionTimeoutMS=200)
    assert_readable_503(search(client, files["RG-2_a"]), "Catalogue database is unavailable")
    assert_readable_503(text_search(client), "Catalogue database is unavailable")
    health = client.get("/health")
    assert health.status_code == 503 and health.json()["dependencies"]["mongodb"]["status"] == "error"


def test_backend_starts_with_mongo_down(monkeypatch, caplog):
    monkeypatch.setenv("MONGO_URI", "mongodb://127.0.0.1:9")
    monkeypatch.setenv("MONGO_TIMEOUT_MS", "200")
    get_settings.cache_clear()
    with caplog.at_level(logging.ERROR, logger="startup"), TestClient(main.app) as test_client:
        assert test_client.get("/health").status_code == 503
    get_settings.cache_clear()
    assert "could not create MongoDB indexes" in caplog.text


# ---- Qwen down or slow -------------------------------------------------------------------------

def use_unreachable_qwen():
    main.app.state.llm_client = LLMClient(CLOSED_PORT, "qwen", timeout=2)
    main.app.state.vlm_client = VLMClient(CLOSED_PORT, "qwen", timeout=2)


def test_qwen_down_image_search_runs_in_reduced_mode(client, catalogue):
    files, _ = catalogue
    use_unreachable_qwen()
    plain = search(client, files["RG-1_a"]).json()
    assert plain["reduced_mode"] is False and "warnings" not in plain

    refined = search(client, files["RG-1_a"], text="similar but in silver")
    assert refined.status_code == 200
    body = refined.json()
    assert body["reduced_mode"] is True
    assert body["warnings"][0] == "Reduced mode: Qwen is unavailable or slow; searched without text refinement"
    assert body["results"][0]["product_id"] == "RG-1"  # image-only order


def test_qwen_down_skips_query_tagging_and_vlm_judge(client, catalogue, monkeypatch):
    files, _ = catalogue
    use_unreachable_qwen()
    settings = main.app.state.settings
    monkeypatch.setattr(settings, "search_query_tagging", True)
    monkeypatch.setattr(settings, "feature_vlm_judge", True)
    response = search(client, files["RG-1_a"])
    assert response.status_code == 200
    body = response.json()
    assert body["reduced_mode"] is True and body["vlm_judge"]["failed"] == body["vlm_judge"]["top_n"]
    assert body["warnings"][0] == "Reduced mode: Qwen is unavailable or slow; searched without query tagging, VLM judge"
    assert body["results"][0]["product_id"] == "RG-1"


class SlowLLMClient(FakeLLMClient):
    async def parse_query(self, text):
        await asyncio.sleep(5)
        return await super().parse_query(text)


def test_qwen_slow_image_search_does_not_wait(client, catalogue, monkeypatch):
    files, _ = catalogue
    monkeypatch.setattr(main.app.state.settings, "llm_query_timeout_seconds", 0.2)
    main.app.state.llm_client = SlowLLMClient({"similar but in silver": QueryTags(metal=["silver"])})
    body = search(client, files["RG-1_a"], text="similar but in silver").json()
    assert body["reduced_mode"] is True and body["timings_ms"]["parse_text"] < 2000


def test_qwen_down_or_slow_text_search_is_503(client, catalogue, monkeypatch):
    use_unreachable_qwen()
    assert_readable_503(text_search(client), "The query parser (Qwen) is unavailable or too slow")
    monkeypatch.setattr(main.app.state.settings, "llm_query_timeout_seconds", 0.2)
    main.app.state.llm_client = SlowLLMClient({"blue square ring": QueryTags(category="rings")})
    assert_readable_503(text_search(client), "The query parser (Qwen) is unavailable or too slow")


# ---- data problems -----------------------------------------------------------------------------

def test_vector_without_mongo_record_is_skipped_and_logged(client, catalogue, caplog):
    files, _ = catalogue
    db = MongoClient(MONGO_URI)[main.app.state.settings.mongo_db]
    db.images.delete_one({"_id": "RG-3_a"})
    with caplog.at_level(logging.WARNING, logger="search"):
        response = search(client, files["RG-1_a"])
    assert response.status_code == 200
    assert "RG-3" not in [r["product_id"] for r in response.json()["results"]]
    assert "has no matching MongoDB image" in caplog.text


@pytest.mark.parametrize("data, content_type, status, message", [
    (b"\xff\xd8\xff\xe0 corrupt jpeg body", "image/jpeg", 400, "not a readable image"),
    (jpeg(Image.effect_noise((1200, 1200), 100).convert("RGB")), "image/jpeg", 413, "larger than 0.5 MB"),
    (b"%PDF-1.7 not an image", "application/pdf", 415, "Unsupported file type"),
    (b"GIF89a....", "image/jpeg", 415, "not a JPEG, PNG or WebP"),
])
def test_bad_uploads(client, catalogue, data, content_type, status, message):
    response = search(client, data, content_type=content_type)
    assert response.status_code == status and message in response.json()["detail"]


def test_empty_category_index_is_404(client, catalogue):
    files, _ = catalogue
    response = search(client, files["RG-2_a"], category="bangles")
    assert response.status_code == 404 and "No catalogue images are indexed" in response.json()["detail"]


# ---- index manifest does not match the settings --------------------------------------------------

def edit_manifest(tmp_path, change):
    path = next((tmp_path / "faiss").glob("*/manifest.json"))
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


def test_category_with_a_mismatched_index_file_is_refused(catalogue, caplog):
    files, tmp_path = catalogue
    edit_manifest(tmp_path, lambda m: m["categories"]["pendants"].update(count=99))
    with caplog.at_level(logging.ERROR), TestClient(main.app) as test_client:
        main.app.state.embed_client = FakeEmbedClient()
        main.app.state.llm_client = FakeLLMClient()
        refused = search(test_client, files["PD-1_a"], category="pendants")
        served = search(test_client, files["RG-2_a"], category="rings")
        health = test_client.get("/health").json()
    assert_readable_503(refused, "'pendants' index was built with a different model or preprocessing")
    assert served.status_code == 200 and served.json()["results"][0]["product_id"] == "RG-2"
    assert health["indexes"] == {"served": {"rings": 4}, "refused": ["pendants"]}
    assert "image index for 'pendants' NOT SERVED" in caplog.text


@pytest.mark.parametrize("key, value", [
    ("dimension", 768), ("preprocess_version", "pp-v0/224"), ("embedding_version", "old"), ("model", "other/model"),
])
def test_manifest_from_other_settings_refuses_every_category(catalogue, caplog, key, value):
    files, tmp_path = catalogue
    edit_manifest(tmp_path, lambda m: m.update({key: value}))
    with caplog.at_level(logging.ERROR), TestClient(main.app) as test_client:
        main.app.state.embed_client = FakeEmbedClient()
        response = search(test_client, files["RG-2_a"], category="rings")
        auto = test_client.post("/search/image", files={"file": ("q.jpg", files["RG-2_a"], "image/jpeg")})
    assert_readable_503(response, "'rings' index was built with a different model")
    assert_readable_503(auto, "No category index matches the current model and preprocessing settings")
    assert f"has {key}=" in caplog.text and "categories NOT served" in caplog.text


# ---- ingest is idempotent ------------------------------------------------------------------------

def test_running_ingest_twice_creates_no_duplicates(tmp_path, monkeypatch):
    try:
        mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=500)
        mongo.admin.command("ping")
    except Exception:
        pytest.skip("MongoDB is not running")
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts import ingest as ingest_script

    data = tmp_path / "data"
    rows = [("RG-1_a", "RG-1", "rings", ("circle", "red")), ("RG-1_b", "RG-1", "rings", ("circle", "orange")),
            ("RG-2_a", "RG-2", "rings", ("square", "blue")), ("PD-1_a", "PD-1", "pendants", ("square", "green"))]
    catalogue_lines = ["image_id,product_id,project_id,category,subcategory,image_path"]
    for image_id, product_id, category, (shape, colour) in rows:
        path = data / "catalogue" / category / f"{image_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg(picture(shape, colour)))
        catalogue_lines.append(f"{image_id},{product_id},demo,{category},band,{path.relative_to(tmp_path)}")
    (data / "catalogue.csv").write_text("\n".join(catalogue_lines) + "\n")
    (data / "products.csv").write_text(
        "product_id,project_id,category,subcategory,name,metal,gemstones,stone_colours,motifs,style,catalogue_sources\n"
        "RG-1,demo,rings,band,Ring one,gold,,,,,a.jpg\nRG-2,demo,rings,band,Ring two,gold,,,,,b.jpg\n"
        "PD-1,demo,pendants,band,Pendant,gold,,,,,c.jpg\n")

    db_name = f"jewellery_search_test_{uuid.uuid4().hex[:8]}"
    for key, value in {"MONGO_URI": MONGO_URI, "MONGO_DB": db_name, "EMBED_BACKEND": "siglip2_224",
                       "FAISS_INDEX_DIR": str(tmp_path / "faiss")}.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    monkeypatch.setattr(ingest_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ingest_script, "make_embed_client", lambda settings: FakeEmbedClient())
    args = argparse.Namespace(rebuild=False, verify=False, batch_size=2)

    def snapshot():
        images = list(mongo[db_name].images.find({}, {"vector_id": 1}))
        store = ingest_script.CategoryIndexes.load_or_create(
            tmp_path / "faiss", "google/siglip2-base-patch16-224", "siglip2-base-v1",
            ingest_script.PreprocessOptions().signature)
        return ({d["_id"]: d["vector_id"] for d in images}, store.counts(),
                mongo[db_name].products.count_documents({}))

    try:
        assert asyncio.run(ingest_script.ingest(args)) == 0
        first = snapshot()
        assert asyncio.run(ingest_script.ingest(args)) == 0
        second = snapshot()
    finally:
        get_settings.cache_clear()
        mongo.drop_database(db_name)
        mongo.close()

    vector_ids, counts, products = second
    assert first == second
    assert len(vector_ids) == 4 and len(set(vector_ids.values())) == 4
    assert counts == {"pendants": 1, "rings": 3} and products == 3
