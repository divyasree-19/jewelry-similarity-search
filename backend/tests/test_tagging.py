import json

import httpx
import pytest
from pydantic import ValidationError

from app.caption_store import load_caption_store, new_caption_store
from app.clients.vlm_client import VLMClient, VLMError, normalize_tags, parse_json_object
from app.config import Settings
from app.tagging.prompts import RETRY_PROMPT
from app.tagging.schema import Tags
from app.tagging.vocab import ALLOWED, SYNONYMS, normalize
from app.vector_index import IndexMismatchError

GOOD = {"category": "ring", "metal": ["yellow gold"], "gemstones": ["diamonds"], "stone_colors": ["white"],
        "motifs": ["floral"], "style": "contemporary", "description": "A gold ring with a diamond flower."}


# ---- vocabulary and synonyms ------------------------------------------------------------------

def test_every_synonym_maps_to_an_allowed_value():
    for field, synonyms in SYNONYMS.items():
        bad = {raw: target for raw, target in synonyms.items() if target not in ALLOWED[field]}
        assert bad == {}, field


@pytest.mark.parametrize("field, raw, expected", [
    ("metal", "Yellow Gold", "gold"),
    ("metal", "sterling silver", "silver"),
    ("metal", "blackened metal", "black metal"),
    ("category", "Ring", "rings"),
    ("category", "nose pin", "nose_pins"),
    ("gemstones", "Diamonds", "diamond"),
    ("gemstones", "Rubies", "ruby"),
    ("motifs", "butterflies", "butterfly"),
    ("gemstones", "CZ", "cubic zirconia"),
    ("stone_colors", "light blue", "blue"),
    ("stone_colors", "Gray", "grey"),
    ("motifs", "hearts", "heart"),
    ("motifs", "crosses", "cross"),
    ("style", "Bohemian", "boho"),
    ("metal", "unobtainium", None),
    ("metal", None, None),
    ("metal", 42, None),
])
def test_normalize(field, raw, expected):
    assert normalize(field, raw) == expected


# ---- schema -----------------------------------------------------------------------------------

def test_schema_accepts_vocab_values():
    tags = Tags(category="rings", metal=["gold"], gemstones=["diamond"], stone_colors=["white"],
                motifs=["flower"], style="modern", description="  A gold   ring. ")
    assert tags.description == "A gold ring."


@pytest.mark.parametrize("override, message", [
    ({"metal": ["yellow gold"]}, "not allowed for metal"),  # synonyms must be mapped before validation
    ({"category": "watches"}, "not an allowed category"),
    ({"style": "funky"}, "not an allowed style"),
    ({"gemstones": ["diamond", "diamond"]}, "duplicate"),
    ({"description": "   "}, "description is empty"),
    ({"price": 10}, "Extra inputs are not permitted"),
])
def test_schema_rejects_values_outside_the_vocabulary(override, message):
    values = {"category": "rings", "metal": ["gold"], "style": "modern", "description": "A ring.", **override}
    with pytest.raises(ValidationError, match=message):
        Tags(**values)


# ---- normalizing raw VLM output ---------------------------------------------------------------

def test_normalize_tags_maps_synonyms():
    tags, dropped = normalize_tags(GOOD)
    assert tags.model_dump() == {"category": "rings", "metal": ["gold"], "gemstones": ["diamond"],
                                 "stone_colors": ["white"], "motifs": ["flower"], "style": "modern",
                                 "description": "A gold ring with a diamond flower."}
    assert dropped == {}


def test_normalize_tags_drops_unknown_values_and_reports_them():
    raw = {**GOOD, "metal": ["gold", "unobtainium", "yellow gold"], "motifs": ["dragon", "heart"],
           "style": "cyberpunk", "category": "watch", "extra_key": "ignored"}
    tags, dropped = normalize_tags(raw)
    assert tags.metal == ["gold"]  # "yellow gold" maps to the same value and is not duplicated
    assert tags.motifs == ["heart"]
    assert tags.style is None and tags.category is None
    assert dropped == {"metal": ["unobtainium"], "motifs": ["dragon"], "style": ["cyberpunk"], "category": ["watch"]}


def test_normalize_tags_accepts_strings_for_lists_and_lists_for_single_values():
    tags, _ = normalize_tags({**GOOD, "metal": "gold, silver", "style": ["classic", "modern"]})
    assert tags.metal == ["gold", "silver"] and tags.style == "classic"


def test_normalize_tags_requires_a_description():
    with pytest.raises(VLMError, match="validation"):
        normalize_tags({**GOOD, "description": None})


@pytest.mark.parametrize("text, expected", [
    (json.dumps(GOOD), GOOD),
    ("```json\n" + json.dumps(GOOD) + "\n```", GOOD),
    ("Here are the tags: " + json.dumps(GOOD) + " Hope this helps.", GOOD),
    ("I cannot tag this image.", None),
    ('{"category": "ring",', None),
    ("[1, 2]", None),
])
def test_parse_json_object(text, expected):
    assert parse_json_object(text) == expected


# ---- VLM client: invalid JSON and retry -------------------------------------------------------

def chat_reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def make_client(replies: list, seen: list) -> VLMClient:
    def handler(request):
        body = json.loads(request.content)
        seen.append((request, body))
        reply = replies[len(seen) - 1]
        return reply if isinstance(reply, httpx.Response) else chat_reply(reply)
    return VLMClient("http://vlm.test/v1", "qwen-test", api_key="secret", transport=httpx.MockTransport(handler))


async def test_tag_valid_first_time():
    seen = []
    client = make_client([json.dumps(GOOD)], seen)
    result = await client.tag(b"png-bytes")
    await client.aclose()

    request, body = seen[0]
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    assert body["model"] == "qwen-test" and body["temperature"] == 0
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert result.attempts == 1 and result.tags.metal == ["gold"]


async def test_invalid_json_is_retried_once():
    seen = []
    client = make_client(["Sorry, here you go: {broken", json.dumps(GOOD)], seen)
    result = await client.tag(b"png")
    await client.aclose()

    assert result.attempts == 2 and result.tags.category == "rings"
    retry_messages = seen[1][1]["messages"]
    assert retry_messages[1] == {"role": "assistant", "content": "Sorry, here you go: {broken"}
    assert retry_messages[2] == {"role": "user", "content": RETRY_PROMPT}


async def test_invalid_json_twice_is_an_error():
    client = make_client(["not json", "still not json"], [])
    with pytest.raises(VLMError, match="after a retry"):
        await client.tag(b"png")
    await client.aclose()


async def test_http_error_is_clear():
    client = make_client([httpx.Response(401, text='{"error":"Unauthorized"}')], [])
    with pytest.raises(VLMError, match="HTTP 401"):
        await client.tag(b"png")
    await client.aclose()


# ---- caption store ----------------------------------------------------------------------------

def test_caption_store_is_separate_and_refuses_another_tagger_version(tmp_path):
    settings = Settings(faiss_index_dir=tmp_path)
    store = new_caption_store(settings, "tags-p1-v1")
    store.add("rings", [5], [[1.0, 0.0]])
    directory = store.save(tmp_path / "captions")
    assert directory.parent == tmp_path / "captions"

    assert load_caption_store(settings, "tags-p1-v1").ids("rings").tolist() == [5]
    with pytest.raises(IndexMismatchError):
        load_caption_store(settings, "tags-p2-v1")
