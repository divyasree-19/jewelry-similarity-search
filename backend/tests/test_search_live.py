"""End-to-end search against the running stack: real SigLIP service, MongoDB and the ingested indexes.

Skipped unless /health is ok and scripts/ingest.py has been run. Runs with SEARCH_HYBRID=false: this checks
the image index itself. With hybrid on, a catalogue image whose own caption ranks low can lose first place
(BR-007_02, 70/71) - see the Stage 8 notes in the README.
"""

import csv
import os

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import PROJECT_ROOT, get_settings


@pytest.fixture(scope="module")
def client():
    # test the image index itself
    pinned = {"SEARCH_HYBRID": "false", "FEATURE_REGION_RERANK": "false", "SEARCH_RERANKER": "false"}
    previous = {key: os.environ.get(key) for key in pinned}
    os.environ.update(pinned)
    get_settings.cache_clear()
    try:
        yield from _client()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


def _client():
    with TestClient(main.app) as test_client:
        if test_client.get("/health").status_code != 200:
            pytest.skip("SigLIP service or MongoDB is not running")
        if main.app.state.indexes.get().total == 0:
            pytest.skip("No FAISS indexes; run scripts/ingest.py")
        yield test_client


def catalogue_rows() -> list[dict]:
    with (PROJECT_ROOT / "data" / "catalogue.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_every_catalogue_image_returns_its_own_product_first(client):
    misses = []
    for row in catalogue_rows():
        data = (PROJECT_ROOT / row["image_path"]).read_bytes()
        response = client.post(
            "/search/image", files={"file": ("q.jpg", data, "image/jpeg")}, data={"category": row["category"]}
        )
        assert response.status_code == 200, response.text
        top = response.json()["results"][0]
        if top["product_id"] != row["product_id"]:
            misses.append((row["image_id"], top["product_id"]))
    assert misses == []
