import json

import numpy as np
import pytest

from app.vector_index import CategoryIndexes, IndexMismatchError, index_dir_name


def make_store() -> CategoryIndexes:
    return CategoryIndexes("google/siglip2-base-patch16-224", "siglip2-base-v1", "pp-v1/512")


def test_index_dir_name_drops_org_prefix():
    assert index_dir_name("google/siglip2-base-patch16-224", "v1") == "siglip2-base-patch16-224_v1"


def test_search_returns_cosine_similarity_of_unnormalized_vectors():
    store = make_store()
    store.add("rings", [7, 8], [[3.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    results = store.search("rings", [10.0, 0.0, 0.0], k=5)
    assert [vector_id for vector_id, _ in results] == [7, 8]
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(1 / np.sqrt(2))


def test_categories_are_separate_and_unknown_category_is_empty():
    store = make_store()
    store.add("rings", [1], [[1.0, 0.0]])
    store.add("earrings", [2], [[1.0, 0.0]])
    assert store.search("rings", [1.0, 0.0]) == [(1, pytest.approx(1.0))]
    assert store.search("bangles", [1.0, 0.0]) == []
    assert store.counts() == {"earrings": 1, "rings": 1}


def test_dimension_mismatch_is_rejected():
    store = make_store()
    store.add("rings", [1], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        store.add("rings", [2], [[1.0, 0.0, 0.0]])


def test_remove_and_empty_category_is_dropped():
    store = make_store()
    store.add("rings", [1, 2], [[1.0, 0.0], [0.0, 1.0]])
    assert store.remove("rings", [1]) == 1
    assert store.ids("rings").tolist() == [2]
    store.remove("rings", [2])
    assert "rings" not in store.indexes


def test_save_and_load_round_trip(tmp_path):
    store = make_store()
    store.add("rings", [1, 2], [[1.0, 0.0], [0.0, 1.0]])
    store.add("pendants", [3], [[0.6, 0.8]])
    directory = store.save(tmp_path)

    assert directory == tmp_path / "siglip2-base-patch16-224_siglip2-base-v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["dimension"] == 2
    assert manifest["preprocess_version"] == "pp-v1/512"
    assert manifest["categories"] == {"pendants": {"file": "pendants.index", "count": 1},
                                      "rings": {"file": "rings.index", "count": 2}}
    assert manifest["total"] == 3

    loaded = CategoryIndexes.load_or_create(tmp_path, *_args(store))
    assert loaded.counts() == {"pendants": 1, "rings": 2}
    assert loaded.id_to_category() == {1: "rings", 2: "rings", 3: "pendants"}


def test_save_deletes_index_files_of_removed_categories(tmp_path):
    store = make_store()
    store.add("rings", [1], [[1.0, 0.0]])
    store.add("pendants", [2], [[1.0, 0.0]])
    directory = store.save(tmp_path)
    store.remove("pendants", [2])
    store.save(tmp_path)
    assert sorted(p.name for p in directory.glob("*.index")) == ["rings.index"]


def test_load_rejects_other_preprocess_version(tmp_path):
    store = make_store()
    store.add("rings", [1], [[1.0, 0.0]])
    store.save(tmp_path)
    with pytest.raises(IndexMismatchError, match="--rebuild"):
        CategoryIndexes.load_or_create(tmp_path, store.model_name, store.embedding_version, "pp-v2/512")


def test_load_without_manifest_starts_empty(tmp_path):
    assert CategoryIndexes.load_or_create(tmp_path, "m", "v", "p").total == 0


def _args(store):
    return store.model_name, store.embedding_version, store.preprocess_version
