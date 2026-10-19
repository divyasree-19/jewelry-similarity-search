"""Stage 10 Part A: reranker scores, weights files, dedup, and the flags in /search/image."""

import json

import numpy as np
import pytest

from app import main
from app.config import PROJECT_ROOT
from app.search.dedup import best_image_per_product, mark_near_duplicates
from app.search.reranker import ModeWeights, combine, labels, load_weights, parse_weights, to_unit
from tests.test_search import catalogue, client, search  # noqa: F401

MODE = ModeWeights({"image": 0.5, "caption": 0.2, "gray": 0.0, "region": 0.3},
                   {"image": (0.6, 1.0), "caption": (0.0, 0.2), "region": (0.7, 1.0)})


# ---- weights files ----------------------------------------------------------------------------

@pytest.mark.parametrize("version", ["v1", "v2"])
def test_saved_weights_sum_to_one(version):
    weights = load_weights(PROJECT_ROOT / "config", version)
    for mode in ("image", "text"):
        assert sum(weights.for_mode(mode).weights.values()) == pytest.approx(1.0)


def test_tuned_file_records_its_tune_numbers():
    data = json.loads((PROJECT_ROOT / "config" / "reranker_weights_v2.json").read_text())
    for mode in ("image", "text"):
        assert data["tuned"][mode]["split"] == "tune" and "recall@5" in data["tuned"][mode]["tune_metrics"]


@pytest.mark.parametrize("weights, message", [
    ({"image": 0.5, "caption": 0.2}, "sum to 1"),
    ({"image": 1.2, "caption": -0.2}, "non-negative"),
])
def test_bad_weights_are_rejected(weights, message):
    block = {"weights": weights, "ranges": {"image": [0.6, 1.0], "caption": [0.0, 0.2]}}
    with pytest.raises(ValueError, match=message):
        parse_weights({"version": "x", "boost_scale": 1, "boost_cap": 0.1, "modes": {"image": block, "text": block}})


# ---- scores -----------------------------------------------------------------------------------

def test_signals_map_to_0_1_by_clipping():
    assert to_unit(0.5, 0.6, 1.0) == 0.0 and to_unit(1.2, 0.6, 1.0) == 1.0
    assert to_unit(0.8, 0.6, 1.0) == pytest.approx(0.5)


def test_missing_signal_is_dropped_and_weights_renormalized():
    score, detail = combine({"image": 0.8, "caption": 0.1, "gray": None, "region": None}, 0.0, MODE, 10, 0.1)
    # region missing: image 0.5/0.7, caption 0.2/0.7
    assert detail["image"]["weight"] == pytest.approx(0.5 / 0.7, abs=1e-4)
    assert detail["caption"]["weight"] == pytest.approx(0.2 / 0.7, abs=1e-4)
    assert detail["region"]["used"] is False and detail["region"]["weight"] == 0
    assert score == pytest.approx(0.5 / 0.7 * 0.5 + 0.2 / 0.7 * 0.5)


def test_image_signal_is_the_fallback_when_no_weighted_signal_exists():
    caption_only = ModeWeights({"image": 0.0, "caption": 1.0, "gray": 0.0, "region": 0.0},
                               {"image": (0.6, 1.0), "caption": (0.0, 0.2)})
    score, detail = combine({"image": 0.8, "caption": None}, 0.0, caption_only, 10, 0.1)
    assert score == pytest.approx(0.5) and detail["image"]["used"] and "note" in detail


def test_score_is_comparable_across_queries():
    # Same raw signals give the same score whatever the other candidates are: no per-query min-max
    a, _ = combine({"image": 0.9, "caption": 0.1, "region": 0.8}, 0, MODE, 10, 0.1)
    b, _ = combine({"image": 0.9, "caption": 0.1, "region": 0.8}, 0, MODE, 10, 0.1)
    assert a == b


@pytest.mark.parametrize("seed", range(5))
def test_scores_stay_within_0_1(seed):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        raw = {s: (None if rng.random() < 0.2 else float(rng.uniform(-1, 2))) for s in ("image", "caption", "region")}
        score, _ = combine(raw, float(rng.uniform(0, 1)), MODE, 10, 0.1)
        assert 0.0 <= score <= 1.0


def test_boost_points_are_capped():
    no_boost, _ = combine({"image": 0.8}, 0.0, MODE, 10, 0.1)
    huge_boost, detail = combine({"image": 0.8}, 5.0, MODE, 10, 0.1)
    assert detail["boost"]["points"] == 0.1 and huge_boost == pytest.approx(no_boost + 0.1)


def test_labels():
    _, detail = combine({"image": 0.97, "caption": 0.02, "region": None}, 0.002, MODE, 10, 0.1)
    assert labels(detail, {"gemstones": {"matched": ["ruby"]}}) == {"design": "high", "description": "low",
                                                                     "stones": "match"}


# ---- dedup ------------------------------------------------------------------------------------

def test_dedup_keeps_the_best_image_per_product():
    ranked = [("P1", "a"), ("P2", "b"), ("P1", "c"), ("P3", "d"), ("P2", "e")]
    assert best_image_per_product(ranked, lambda c: c[0]) == [("P1", "a"), ("P2", "b"), ("P3", "d")]


def test_near_duplicates_are_marked_not_removed():
    results = [{"vector_id": 1, "image_id": "A_01", "product_id": "A"},
               {"vector_id": 2, "image_id": "B_01", "product_id": "B"},
               {"vector_id": 3, "image_id": "C_01", "product_id": "C"}]
    same = np.array([1.0, 0.0])
    vectors = {1: same, 2: np.array([0.999, 0.0447]) / np.linalg.norm([0.999, 0.0447]), 3: np.array([0.0, 1.0])}
    mark_near_duplicates(results, vectors, threshold=0.98)
    assert len(results) == 3
    assert results[1]["possible_duplicate_of"]["product_id"] == "A"
    assert "possible_duplicate_of" not in results[0] and "possible_duplicate_of" not in results[2]


# ---- flags in /search/image -------------------------------------------------------------------

def ranked(response):
    return [(r["product_id"], r["similarity_score"]) for r in response.json()["results"]]


def test_flags_off_keep_previous_results(client, catalogue, monkeypatch):
    files, _ = catalogue
    settings = main.app.state.settings
    before = ranked(search(client, files["RG-1_b"]))  # SEARCH_RERANKER=false in the fixture
    monkeypatch.setattr(settings, "search_reranker", True)
    monkeypatch.setattr(settings, "reranker_modes", "text")  # on, but not for image queries
    assert ranked(search(client, files["RG-1_b"])) == before
    assert "reranker_weights_version" not in search(client, files["RG-1_b"]).json()


def test_reranker_on_gives_0_1_scores_with_signals(client, catalogue, monkeypatch):
    files, _ = catalogue
    settings = main.app.state.settings
    monkeypatch.setattr(settings, "search_reranker", True)
    monkeypatch.setattr(settings, "reranker_modes", "image,text")
    body = search(client, files["RG-2_a"], debug="true").json()
    assert body["reranker_weights_version"] == settings.reranker_weights_version
    assert "not a probability" in body["similarity_score"]
    assert body["results"][0]["product_id"] == "RG-2"
    for result in body["results"]:
        assert 0.0 <= result["similarity_score"] <= 1.0
        assert {"image", "caption", "gray", "region", "boost"} <= set(result["debug"]["signals"])
        assert "design" in result["debug"]["labels"]
        assert "vector_id" not in result and "_image_path" not in result
    scores = [r["similarity_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
