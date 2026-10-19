import json

import httpx
import pytest

from app.clients.llm_client import LLMClient, LLMError, category_written, colours_written, validate_query
from app.config import Settings
from app.search.boosts import apply_boosts, boost_targets
from app.search.hybrid import Candidate
from app.search.query_analysis import CategoryDetector, plan_image_query, plan_text_query
from app.tagging.schema import QueryTags


def llm(replies: list, seen: list) -> LLMClient:
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        reply = replies[len(seen) - 1]
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})
    return LLMClient("http://llm.test/v1", "qwen-test", transport=httpx.MockTransport(handler))


# ---- parse_query: validation and unknown values -----------------------------------------------

async def test_parse_query_validates_and_maps_synonyms():
    seen = []
    client = llm([json.dumps({"category": "Ring", "metal": ["yellow gold"], "gemstones": ["Rubies"],
                              "stone_colors": ["red"], "motifs": ["floral"], "style": None,
                              "semantic_query": "gold ring with a red ruby flower"})], seen)
    parsed = await client.parse_query("red ruby flower ring in gold")
    await client.aclose()

    assert parsed.tags == QueryTags(category="rings", metal=["gold"], gemstones=["ruby"], stone_colors=["red"],
                                    motifs=["flower"], semantic_query="gold ring with a red ruby flower")
    body = seen[0]
    assert body["temperature"] == 0 and body["response_format"] == {"type": "json_object"}
    assert body["messages"][1] == {"role": "user", "content": "red ruby flower ring in gold"}


def test_unknown_values_are_dropped_and_reported():
    parsed = validate_query({"category": "watches", "metal": ["unobtainium", "silver"], "motifs": ["dragon"],
                             "style": "cyberpunk"}, "silver dragon watch")
    assert parsed.tags.metal == ["silver"] and parsed.tags.motifs == [] and parsed.tags.style is None
    assert parsed.tags.category is None
    assert parsed.dropped == {"category": ["watches"], "metal": ["unobtainium"], "motifs": ["dragon"],
                              "style": ["cyberpunk"]}
    assert parsed.tags.semantic_query == "silver dragon watch"  # falls back to the raw text


async def test_invalid_json_is_retried_once_then_an_error():
    client = llm(["sure! {category: ring", json.dumps({"category": "rings", "semantic_query": "ring"})], [])
    assert (await client.parse_query("a ring")).attempts == 2
    await client.aclose()

    client = llm(["nope", "still nope"], [])
    with pytest.raises(LLMError, match="after a retry"):
        await client.parse_query("a ring")
    await client.aclose()


async def test_server_without_json_mode_falls_back_to_plain_requests():
    seen = []
    client = llm([httpx.Response(400, text="unknown field response_format"),
                  json.dumps({"semantic_query": "ring"})], seen)
    await client.parse_query("ring")
    await client.aclose()
    assert "response_format" in seen[0] and "response_format" not in seen[1]


# ---- colours only when written; category only when written ------------------------------------

@pytest.mark.parametrize("text, colours, expected", [
    ("red ruby ring", ["red"], ["red"]),
    ("ruby ring", ["red"], []),                      # the model guessed red from "ruby"
    ("earrings with dark red garnet", ["red"], ["red"]),
    ("grey stone necklace", ["grey"], ["grey"]),
    ("gray stone necklace", ["grey"], ["grey"]),       # synonym
    ("turquoise beads on leather", ["blue"], []),     # turquoise is the gemstone here
    ("gold ring", ["yellow"], []),                    # gold is the metal
])
def test_colours_written(text, colours, expected):
    assert colours_written(text, colours) == expected


def test_guessed_colour_and_category_are_dropped():
    parsed = validate_query({"category": "pendants", "gemstones": ["ruby"], "stone_colors": ["red"]},
                            "ruby with an elephant")
    assert parsed.tags.stone_colors == [] and parsed.tags.category is None
    assert parsed.dropped == {"stone_colors": ["red"], "category": ["pendants"]}


@pytest.mark.parametrize("text, category, expected", [
    ("red ruby ring", "rings", True), ("gold studs", "earrings", True), ("a nose pin", "nose_pins", True),
    ("evil eye with an elephant", "pendants", False), ("ring with a stone", "earrings", False),
])
def test_category_written(text, category, expected):
    assert category_written(text, category) is expected


async def test_text_plan_uses_colours_only_from_text():
    plan = await plan_text_query(settings=Settings(), detector=None, embed_client=None, vector=None,
                                 user_category=None, indexed=["rings"],
                                 parsed_text=QueryTags(category="rings", stone_colors=["red"], gemstones=["ruby"]))
    assert plan.allow_stone_colors is True and boost_targets(plan)["stone_colors"] == {"red"}

    plan = await plan_text_query(settings=Settings(), detector=None, embed_client=None, vector=None,
                                 user_category=None, indexed=["rings"],
                                 parsed_text=QueryTags(category="rings", gemstones=["ruby"]))
    assert plan.allow_stone_colors is False and "stone_colors" not in boost_targets(plan)


async def test_text_without_category_searches_all_or_asks():
    common = dict(detector=None, embed_client=None, vector=None, user_category=None, indexed=["rings", "earrings"],
                  parsed_text=QueryTags(metal=["silver"]))
    plan = await plan_text_query(settings=Settings(text_no_category="all"), **common)
    assert plan.search_categories == ["rings", "earrings"] and plan.needs_category is False


async def test_text_category_without_an_index_is_ignored():
    plan = await plan_text_query(settings=Settings(), detector=None, embed_client=None, vector=None,
                                 user_category=None, indexed=["rings", "bracelets"],
                                 parsed_text=QueryTags(category="bangles"))
    assert plan.search_categories == ["rings", "bracelets"] and "no catalogue images" in plan.notes[0]


# ---- image + text: metal override -------------------------------------------------------------

async def refine(text_tags: QueryTags, **settings_overrides):
    return await plan_image_query(
        settings=Settings(**settings_overrides), detector=CategoryDetector(), embed_client=None, vector=[1.0],
        user_category="necklaces", indexed=["necklaces"], query_tags={"gemstones": ["diamond"]},
        parsed_text=text_tags,
    )


async def test_in_silver_is_a_soft_boost_by_default():
    plan = await refine(QueryTags(metal=["silver"], semantic_query="silver jewellery"))
    assert plan.mode == "image_text" and plan.overrides == {"metal": ["silver"]} and plan.hard_filters == {}

    gold = Candidate(1, "necklaces", rrf_score=0.0330)  # the image's best match
    silver = Candidate(2, "necklaces", rrf_score=0.0320)
    apply_boosts([gold, silver], {1: {"metal": ["gold"]}, 2: {"metal": ["silver"]}}, plan, Settings())
    assert silver.final_score > gold.final_score and gold.boost == 0  # silver first, gold still there


async def test_metal_hard_filter_option():
    plan = await refine(QueryTags(metal=["silver"]), refine_metal_hard_filter=True)
    assert plan.hard_filters == {"metal": ["silver"]}


async def test_text_override_replaces_image_tags_and_allows_written_colours():
    plan = await refine(QueryTags(gemstones=["emerald"], stone_colors=["green"]))
    assert boost_targets(plan) == {"gemstones": {"emerald"}, "stone_colors": {"green"}}
