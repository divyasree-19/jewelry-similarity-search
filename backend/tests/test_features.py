"""Stage 9 features: preprocessing variants, stores, similarity, and the flags in /search/image."""

import numpy as np
import pytest
from PIL import Image

from app import main
from app.config import Settings
from app.preprocessing import PreprocessOptions, encode, make_tiles, preprocess, to_edge_map, to_grayscale, variant_signature
from app.search.features import (TileStore, VariantIndexCache, region_similarity, save_tiles, tile_dir,
                                 variant_root)
from app.vector_index import CategoryIndexes
from tests.test_search import CATALOGUE, FakeEmbedClient, catalogue, client, fake_vector, search  # noqa: F401

OPTIONS = PreprocessOptions()


# ---- preprocessing variants -------------------------------------------------------------------

def test_grayscale_is_three_channel_same_size_and_colourless():
    image = Image.new("RGB", (512, 512), (200, 30, 30))
    gray = to_grayscale(image)
    assert gray.mode == "RGB" and gray.size == image.size
    r, g, b = gray.getpixel((10, 10))
    assert r == g == b


def test_edge_map_is_three_channel_same_size():
    image = Image.new("RGB", (512, 512), "white")
    image.paste((0, 0, 0), (100, 100, 400, 400))
    edges = to_edge_map(image)
    assert edges.mode == "RGB" and edges.size == image.size
    assert edges.getpixel((250, 250)) == (255, 255, 255)  # flat area: no edge


@pytest.mark.parametrize("grid, count", [(2, 5), (3, 10)])
def test_tile_count_per_image(grid, count):
    tiles = make_tiles(Image.new("RGB", (512, 512), "white"), grid)
    assert len(tiles) == count and all(t.size == (224, 224) and t.mode == "RGB" for t in tiles)


def test_variant_signatures_leave_the_main_pipeline_version_alone():
    assert variant_signature(OPTIONS.signature, "gray") == f"{OPTIONS.signature}/gray-v1"
    assert variant_signature(OPTIONS.signature, "tiles", 3) == f"{OPTIONS.signature}/tiles-v1/3x3"
    assert OPTIONS.signature == "pp-v1/512"


# ---- region similarity ------------------------------------------------------------------------

def test_region_score_of_an_image_with_itself_is_the_highest():
    rng = np.random.default_rng(0)
    images = [rng.normal(size=(5, 16)) for _ in range(6)]
    for i, query in enumerate(images):
        scores = [region_similarity(query, other) for other in images]
        assert np.argmax(scores) == i and scores[i] == pytest.approx(1.0)


def test_region_score_takes_the_best_tile_per_query_tile():
    query = np.eye(2)                    # two query tiles: x and y
    candidate = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # contains both, in another order
    assert region_similarity(query, candidate) == pytest.approx(1.0)
    assert region_similarity(query, candidate[:1]) == pytest.approx(0.5)  # only y: x scores 0


# ---- flags in /search/image (fixture catalogue, fake embeddings) -------------------------------

class BatchEmbedClient(FakeEmbedClient):
    def __init__(self):
        super().__init__()
        self.single_calls = self.batch_calls = 0

    async def embed(self, image_bytes, filename, content_type):
        self.single_calls += 1
        return await super().embed(image_bytes, filename, content_type)

    async def embed_batch(self, images):
        self.batch_calls += 1
        return [fake_vector(data) for data, _, _ in images]


@pytest.fixture
def features_client(client, catalogue, tmp_path):
    """The Stage 8 test client plus gray and 2x2 tile stores built from the fixture catalogue."""
    files, _ = catalogue
    settings = main.app.state.settings
    gray = CategoryIndexes(settings.embed_model_name, settings.embed_version, variant_signature(OPTIONS.signature, "gray"))
    tiles = {}
    for vector_id, (image_id, _, _, category, _, _) in enumerate(CATALOGUE, start=1):
        image = preprocess(files[image_id], OPTIONS)
        gray.add(category, [vector_id], [fake_vector(encode(to_grayscale(image)))])
        vectors = np.asarray([fake_vector(encode(t)) for t in make_tiles(image, 2)], dtype=np.float32)
        ids = np.column_stack([np.full(len(vectors), vector_id), np.arange(len(vectors))])
        matrix, id_rows = tiles.get(category, (np.zeros((0, vectors.shape[1]), np.float32), np.zeros((0, 2), np.int64)))
        tiles[category] = (np.vstack([matrix, vectors]), np.vstack([id_rows, ids]))
    gray.save(variant_root(settings, "gray"))
    save_tiles(tmp_path / "tiles", 2, variant_signature(OPTIONS.signature, "tiles", 2), tiles, matrix.shape[1])

    main.app.state.embed_client = BatchEmbedClient()
    main.app.state.feature_indexes["gray"] = VariantIndexCache(settings, "gray", variant_signature(OPTIONS.signature, "gray"))
    main.app.state.tile_store = TileStore(tmp_path / "tiles", 2, variant_signature(OPTIONS.signature, "tiles", 2))
    return client


def ranked(response) -> list[tuple[str, float]]:
    return [(r["product_id"], r["score"]) for r in response.json()["results"]]


def test_flags_off_keeps_stage_8_results_and_single_embed_call(features_client, catalogue):
    files, _ = catalogue
    before = ranked(search(features_client, files["RG-1_b"]))
    embedder = main.app.state.embed_client
    assert embedder.single_calls == 1 and embedder.batch_calls == 0
    for flag in ("gray_retrieval", "gray_rerank", "edge_retrieval", "edge_rerank", "region_rerank"):
        assert getattr(main.app.state.settings, f"feature_{flag}") is False
    assert ranked(search(features_client, files["RG-1_b"])) == before


def test_gray_retrieval_adds_a_third_list(features_client, catalogue, monkeypatch):
    files, _ = catalogue
    monkeypatch.setattr(main.app.state.settings, "feature_gray_retrieval", True)
    body = search(features_client, files["RG-2_a"], debug="true").json()
    top = body["results"][0]["debug"]
    assert body["results"][0]["product_id"] == "RG-2"
    assert top["gray_rank"] == 1 and top["gray_score"] == pytest.approx(1.0, abs=1e-3)
    assert top["rrf_score"] == pytest.approx(2 / 61, abs=1e-6)  # image rank 1 + gray rank 1
    assert main.app.state.embed_client.batch_calls == 1 and main.app.state.embed_client.single_calls == 0


def test_gray_and_region_rerank_add_weighted_scores(features_client, catalogue, monkeypatch):
    files, _ = catalogue
    settings = main.app.state.settings
    monkeypatch.setattr(settings, "feature_gray_rerank", True)
    monkeypatch.setattr(settings, "feature_region_rerank", True)
    body = search(features_client, files["RG-2_a"], debug="true").json()
    debug = body["results"][0]["debug"]
    assert debug["rerank"]["gray"]["score"] == pytest.approx(1.0, abs=1e-3)
    assert debug["rerank"]["region"]["score"] == pytest.approx(1.0, abs=1e-3)  # an image against itself
    expected = debug["rrf_score"] + debug["boost_total"] + settings.feature_gray_rerank_weight * debug["rerank"]["gray"]["score"] \
        + settings.feature_region_weight * debug["rerank"]["region"]["score"]
    assert debug["stage9_score"] == pytest.approx(expected, abs=1e-4)
    assert body["debug"]["hybrid"]["rescored"] == {"gray": 4, "region": 4}  # 4 ring images, before 1-per-product
    assert main.app.state.embed_client.batch_calls == 1  # colour + gray + 5 tiles in one call


def test_missing_feature_stores_do_not_break_search(client, catalogue, monkeypatch):
    files, _ = catalogue
    main.app.state.embed_client = BatchEmbedClient()
    for flag in ("gray_retrieval", "gray_rerank", "region_rerank"):
        monkeypatch.setattr(main.app.state.settings, f"feature_{flag}", True)
    response = search(client, files["RG-2_a"])
    assert response.status_code == 200 and response.json()["results"][0]["product_id"] == "RG-2"


# ---- the real stores (skipped until scripts/build_feature_indexes.py has run) ------------------

def real_stores():
    settings = Settings()
    image = CategoryIndexes.load_or_create(settings.faiss_index_dir, settings.embed_model_name, settings.embed_version,
                                           OPTIONS.signature)
    if image.total == 0:
        pytest.skip("no image index")
    return settings, image


def test_gray_index_count_equals_image_index_count():
    settings, image = real_stores()
    gray = CategoryIndexes.load_or_create(variant_root(settings, "gray"), settings.embed_model_name,
                                          settings.embed_version, variant_signature(OPTIONS.signature, "gray"))
    if gray.total == 0:
        pytest.skip("gray index not built")
    assert gray.counts() == image.counts()
    for category in image.counts():
        assert set(gray.ids(category).tolist()) == set(image.ids(category).tolist())


@pytest.mark.parametrize("grid", [2, 3])
def test_tile_store_rows_match_catalogue_and_self_scores_highest(grid):
    settings, image = real_stores()
    store = TileStore(tile_dir(settings), grid, variant_signature(OPTIONS.signature, "tiles", grid))
    if not store.load():
        pytest.skip(f"{grid}x{grid} tile store not built")
    for category in image.counts():
        ids = store.ids(category)
        assert set(ids) == set(image.ids(category).tolist())
        tiles = {vector_id: store.tiles(category, vector_id) for vector_id in ids}
        assert all(len(t) == 1 + grid * grid for t in tiles.values())
        for vector_id, own in tiles.items():
            scores = {other: region_similarity(own, t) for other, t in tiles.items()}
            assert max(scores, key=scores.get) == vector_id
