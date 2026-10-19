"""Prompts for the tagging VLM. Changing the wording changes the tags, so bump PROMPT_VERSION."""

from app.tagging.vocab import CATEGORY, GEMSTONES, METAL, MOTIFS, STONE_COLORS, STYLE

PROMPT_VERSION = "1"


def _options(values) -> str:
    return ", ".join(sorted(values))


TAG_PROMPT = f"""You are tagging one jewellery catalogue photo for a search index.
Reply with ONLY a JSON object (no code fences, no other text) with exactly these keys:

"category": one of [{_options(CATEGORY)}]
"metal": list, from [{_options(METAL)}]; use "gold" for yellow gold
"gemstones": list, from [{_options(GEMSTONES)}]; [] if there are no stones
"stone_colors": list of stone colours, from [{_options(STONE_COLORS)}]; [] if there are no stones
"motifs": list of design motifs, preferably from [{_options(MOTIFS)}]
"style": one of [{_options(STYLE)}]
"description": one plain sentence (at most 30 words) describing the piece's shape, metal, stones and design

Only describe what is visible. If unsure about a value, leave it out."""

RETRY_PROMPT = "That was not a valid JSON object. Reply again with ONLY the JSON object, nothing else."


QUERY_PROMPT_VERSION = "1"

QUERY_PARSE_PROMPT = f"""You turn a shopper's jewellery search into JSON for a search engine.
Reply with ONLY a JSON object (no code fences, no other text) with exactly these keys:

"category": one of [{_options(CATEGORY)}], or null if the text does not say
"metal": list, from [{_options(METAL)}]; use "gold" for yellow gold
"gemstones": list, from [{_options(GEMSTONES)}]
"stone_colors": list, from [{_options(STONE_COLORS)}]; ONLY colours the shopper actually wrote
"motifs": list, preferably from [{_options(MOTIFS)}]
"style": one of [{_options(STYLE)}], or null
"semantic_query": a short English description of the item to look for, without words like
                  "show me", "similar", "but", "I want"; e.g. "gold ring with a red ruby flower"

Use [] or null for anything the shopper did not ask for. Do not guess colours from gemstone names."""


JUDGE_PROMPT_VERSION = "1"

JUDGE_PROMPT = """Image 1 is a shopper's photo. Image 2 is a catalogue product.
Are these the same jewellery design, ignoring stone colour, lighting, background and angle?
Judge only the design: shape, structure, motifs, setting style and the arrangement of stones.
Reply with ONLY a JSON object: {"same_design": true or false, "confidence": a number from 0 to 1, "reason": "at most 15 words"}"""
