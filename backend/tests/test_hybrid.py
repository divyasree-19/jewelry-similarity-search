import numpy as np
import pytest

from app.config import Settings
from app.search.boosts import apply_boosts, boost_targets
from app.search.hybrid import Candidate, hybrid_search, rrf_merge
from app.search.query_analysis import CategoryDetector, plan_image_query
from app.tagging.schema import QueryTags
from app.vector_index import CategoryIndexes


def settings(**overrides) -> Settings:
    return Settings(**overrides)


# ---- reciprocal rank fusion -------------------------------------------------------------------

def test_rrf_sums_reciprocal_ranks_and_keeps_both_ranks():
    image = [(1, "rings", 0.9), (2, "rings", 0.8), (3, "rings", 0.7)]
    caption = [(3, "rings", 0.3), (1, "rings", 0.2), (4, "rings", 0.1)]
    merged = {c.vector_id: c for c in rrf_merge(image, caption, rrf_k=60)}

    assert merged[1].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert merged[3].rrf_score == pytest.approx(1 / 63 + 1 / 61)
    assert (merged[2].image_rank, merged[2].caption_rank) == (2, None)
    assert (merged[4].image_rank, merged[4].caption_rank, merged[4].caption_score) == (None, 3, 0.1)
    assert [c.vector_id for c in rrf_merge(image, caption)] == [1, 3, 2, 4]


def test_rrf_with_one_list_keeps_its_order():
    image = [(5, "rings", 0.9), (2, "rings", 0.8), (9, "rings", 0.7)]
    assert [c.vector_id for c in rrf_merge(image, [])] == [5, 2, 9]


def test_hybrid_search_without_captions_is_image_only():
    store = CategoryIndexes("m", "v", "p")
    store.add("rings", [1, 2], [[1.0, 0.0], [0.6, 0.8]])
    candidates, info = hybrid_search(store, None, ["rings"], [1.0, 0.0], k=10, rrf_k=60, use_captions=True)
    assert [c.vector_id for c in candidates] == [1, 2] and info["captions_used"] is False


# ---- boosts -----------------------------------------------------------------------------------

def image_plan(query_tags, **kwargs):
    from app.search.query_analysis import QueryPlan
    return QueryPlan(mode="image", category="rings", query_tags=query_tags, **kwargs)


def test_boosts_reorder_but_never_remove_results():
    # 0.0005 apart, about two RRF ranks each; the 0.003 boost lifts #5 past all four
    candidates = [Candidate(i, "rings", rrf_score=0.03 - i * 0.0005) for i in range(1, 6)]
    tags = {5: {"gemstones": ["ruby"], "motifs": ["flower"]}}  # the last candidate matches the query tags
    apply_boosts(candidates, tags, image_plan({"gemstones": ["ruby"], "motifs": ["flower"]}), settings())

    assert len(candidates) == 5
    ordered = sorted(candidates, key=lambda c: c.final_score, reverse=True)
    assert ordered[0].vector_id == 5
    assert ordered[0].boosts == {"gemstones": {"matched": ["ruby"], "boost": 0.002},
                                 "motifs": {"matched": ["flower"], "boost": 0.001}}
    assert all(c.boost == 0 for c in candidates if c.vector_id != 5)


def test_boost_is_capped():
    candidate = Candidate(1, "rings")
    many = [f"m{i}" for i in range(40)]
    plan = image_plan({"motifs": many})
    apply_boosts([candidate], {1: {"motifs": many}}, plan, settings(boost_cap=0.005))
    assert candidate.boost == pytest.approx(0.005)


def test_image_queries_never_use_stone_colour_or_metal():
    # The query image was tagged with red stones and gold, but a photo's colour is unreliable (colour_mismatch)
    plan = image_plan({"gemstones": ["ruby"], "stone_colors": ["red"], "metal": ["gold"]})
    assert boost_targets(plan) == {"gemstones": {"ruby"}}

    red = Candidate(1, "rings", rrf_score=0.02)
    green = Candidate(2, "rings", rrf_score=0.02)
    tags = {1: {"stone_colors": ["red"], "metal": ["gold"]}, 2: {"stone_colors": ["green"], "metal": ["silver"]}}
    apply_boosts([red, green], tags, plan, settings())
    assert red.boost == green.boost == 0  # the green (colour-mismatched) product is not penalised


async def test_plan_image_query_keeps_only_gemstones_and_motifs_from_query_tags():
    plan = await plan_image_query(
        settings=settings(), detector=CategoryDetector(), embed_client=None, vector=[1.0], user_category="rings",
        indexed=["rings"], query_tags={"category": "rings", "gemstones": ["ruby"], "stone_colors": ["red"],
                                       "metal": ["gold"], "motifs": [], "description": "x"},
    )
    assert plan.query_tags == {"category": "rings", "gemstones": ["ruby"]}
    assert plan.allow_stone_colors is False and plan.category_source == "user"


# ---- category detection -----------------------------------------------------------------------

class FakeTextEmbedder:
    """Category prompts map to fixed unit vectors; counts calls to check caching."""

    def __init__(self):
        self.calls = 0

    async def embed_text(self, text):
        self.calls += 1
        axis = {"ring": 0, "pair of earrings": 1, "necklace": 2}[text.rsplit(" a ", 1)[-1].removeprefix("a ")]
        vector = np.zeros(3)
        vector[axis] = 1.0
        return vector.tolist()


async def detect(vector, threshold=0.5):
    embedder = FakeTextEmbedder()
    plan = await plan_image_query(
        settings=settings(category_confidence_threshold=threshold), detector=CategoryDetector(),
        embed_client=embedder, vector=vector, user_category=None, indexed=["rings", "earrings", "necklaces"],
    )
    return plan, embedder


async def test_confident_detection_searches_the_detected_category():
    plan, _ = await detect([0.9, 0.1, 0.0])
    assert (plan.category, plan.category_source, plan.needs_category) == ("rings", "detected", False)
    assert plan.category_confidence > 0.99 and plan.search_categories == ["rings"]


async def test_low_confidence_asks_the_user_with_top_three():
    plan, _ = await detect([0.5, 0.5, 0.49])  # rings and earrings are a coin toss
    assert plan.needs_category is True and plan.category is None and plan.search_categories == []
    assert [c["category"] for c in plan.category_candidates] == ["rings", "earrings", "necklaces"]
    assert plan.category_confidence < 0.5


async def test_category_text_vectors_are_cached():
    detector, embedder = CategoryDetector(), FakeTextEmbedder()
    for _ in range(3):
        await detector.rank(embedder, [1.0, 0.0, 0.0], ["rings", "earrings"], 100.0)
    assert embedder.calls == 2 * 3  # 2 categories x 3 templates, embedded once


async def test_user_category_wins_over_text_and_detection():
    plan = await plan_image_query(
        settings=settings(), detector=CategoryDetector(), embed_client=None, vector=[1.0],
        user_category="earrings", indexed=["rings", "earrings"], parsed_text=QueryTags(category="rings"),
    )
    assert (plan.category, plan.category_source) == ("earrings", "user")


async def test_qwen_detection_falls_back_to_siglip_without_a_usable_category():
    embedder = FakeTextEmbedder()
    plan = await plan_image_query(
        settings=settings(category_detection="qwen"), detector=CategoryDetector(), embed_client=embedder,
        vector=[0.0, 0.0, 1.0], user_category=None, indexed=["rings", "earrings", "necklaces"],
        query_tags={"category": "bangles"},
    )
    assert plan.category == "necklaces" and "used SigLIP" in plan.notes[0]
